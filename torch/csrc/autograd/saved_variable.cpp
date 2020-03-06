#include <torch/csrc/autograd/saved_variable.h>

#include <torch/csrc/autograd/edge.h>
#include <torch/csrc/autograd/function.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/autograd/anomaly_mode.h>

#include <ATen/Tensor.h>

#include <cstdint>
#include <list>
#include <memory>
#include <sstream>

#include <iostream>

using namespace std;

namespace torch { namespace autograd {

SavedVariable::SavedVariable(const Variable& variable, bool is_output, bool is_inplace_view) {
  if (variable.defined()) {
    was_default_constructed_ = false;
    output_nr_ = variable.output_nr();
    requires_grad_ = variable.requires_grad();
    has_grad_fn_ = !variable.is_leaf();
    is_inplace_view_ = is_inplace_view;
    // These copies are all shared_ptr copies, so slightly more expensive.
    // Do them here instead of in the init list in case data is undefined.
    data_ = variable.tensor_data();
    if (variable.is_leaf()) {
      grad_accumulator_ = impl::grad_accumulator(variable);
    } else if (!is_output) {
      grad_fn_ = variable.grad_fn();
    } else if (is_inplace_view) {
      weak_grad_fn_ = variable.grad_fn();
    }
    version_counter_ = impl::version_counter(variable);
    saved_version_ = version_counter_.current_version();
  }
}

Variable SavedVariable::unpack(std::shared_ptr<Node> saved_for) const {
  if (!data_.defined()) {
    if (!was_default_constructed_) {
      throw std::runtime_error(ERR_BACKWARD_TWICE);
    }
    return Variable();
  }

  auto grad_fn = is_inplace_view_ ? weak_grad_fn_.lock() : grad_fn_;
  if (has_grad_fn_ && !grad_fn) {
    if (!saved_for) {
      // If saving the grad_fn would create a circular reference, then it must
      // be passed in to the unpack function.
      throw std::runtime_error("No grad_fn for non-leaf saved variable");
    }
    grad_fn = std::move(saved_for);
  }

  if (saved_version_ != version_counter_.current_version()) {
    std::stringstream message;
    message << "one of the variables needed for gradient computation has been "
        "modified by an inplace operation: [" << data_.toString() << " "
        << data_.sizes() << "]";
    if (grad_fn) {
        message << ", which is output " << output_nr_
            << " of " << grad_fn->name() << ",";
    }
    message << " is at version " << version_counter_.current_version()
        << "; expected version " << saved_version_ << " instead.";
    if (!AnomalyMode::is_enabled()) {
        message << " Hint: enable anomaly detection to find the operation "
            "that failed to compute its gradient, with torch.autograd."
            "set_detect_anomaly(True).";
    }
    else {
        message << " Hint: the backtrace further above shows the operation "
            "that failed to compute its gradient. The variable in question "
            "was changed in there or anywhere later. Good luck!";
    }
    throw std::runtime_error(message.str());
  }

  // NB: saved views are unpacked as normal Variables (not views) even though
  // they still share the same storage. This works only because we never call
  // in-place functions on unpacked variables.
  Variable var;
  if (grad_fn) {
    var = make_variable(data_, Edge(std::move(grad_fn), output_nr_));
  } else {
    var = make_variable(data_, requires_grad_);
  }
  impl::set_version_counter(var, saved_version_);

  // If a Variable is a leaf (no grad_fn saved), and it requires_grad, then we
  // should have saved the grad accumulator. Even if the Variable no longer
  // alive, the accumulator should be kept alive by the references in the
  // graph).
  if (requires_grad_ && !var.grad_fn() && grad_accumulator_.expired())
    throw std::logic_error("No grad accumulator for a saved leaf!");
  impl::set_grad_accumulator(var, grad_accumulator_);

  return var;
}

void SavedVariable::copy_data_from(at::Tensor& otherData) {
  // Turn off grad so copy_ doesn't throw any errors.
  bool reqGrad = data_.requires_grad();
  data_.set_requires_grad(false);
  data_.copy_(otherData, /* non_blocking=*/ false);
  data_.set_requires_grad(reqGrad);
}

uint8_t* SavedVariable::serialize_to_blob() const {
  if(data_.is_sparse()) {
    cerr << "serialize_to_blob() - Cannot handle sparse tensors. Likely the graph "
	 << "will not be properly reconstructed."
	 << endl;
        // Return a valid buffer anyways so we don't cause any segfaults.
    uint8_t* errBuffer = reinterpret_cast<uint8_t*>(malloc(sizeof(size_t)));
    memset(errBuffer, 0, sizeof(size_t));
    return errBuffer;
  }

    // Copy the given tensor into CPU memory so we can directly access its memory
    // buffers.
  at::Tensor copied = data_.clone().detach().to(at::device(at::kCPU));

  uint8_t* buffer = reinterpret_cast<uint8_t*>(malloc(copied.nbytes() + sizeof(size_t)));
    // Pack in the size first, then the bytes.
  size_t* szEmplacement = reinterpret_cast<size_t*>(buffer);
  *szEmplacement = copied.nbytes();

  memcpy(&buffer[sizeof(size_t)], copied.data_ptr(), copied.nbytes());

  return buffer;
}

void SavedVariable::deserialize_from_blob(uint8_t* data) {
  if(data_.is_sparse()) {
    // Cannot reconstruct sparse tensors (currently). User should have been
    // warned when it was serialized.
    return;
  }

  size_t* dataSz = reinterpret_cast<size_t*>(data);
  if(*dataSz != data_.nbytes()) {
    cerr << "deserialize_from_blob() - Size mismatch. Expected " << *dataSz
	 << " from the provided buffer, but local tensor has size " << data_.nbytes()
	 << endl;
    return;
  }

  // Allocate a CPU tensor to use for copying purposes.
  at::Tensor ramTensor = data_.clone().detach().to(at::device(at::kCPU));
  memcpy(ramTensor.data_ptr(), &data[sizeof(size_t)], ramTensor.nbytes());

  // Copy the contents of the CPU tensor back into the original dest tensor.
  copy_data_from(ramTensor);
}

const char* ERR_BACKWARD_TWICE =
    "Trying to backward through the graph a second time, but the buffers have "
    "already been freed. Specify retain_graph=True when calling backward "
    "the first time.";

}} // namespace torch::autograd
