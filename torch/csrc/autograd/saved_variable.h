#pragma once

#include <torch/csrc/WindowsTorchApiMacro.h>

#include <ATen/ATen.h>

#include <cstdint>
#include <memory>

namespace torch { namespace autograd {

using Variable = at::Tensor;
struct Node;

TORCH_API extern const char* ERR_BACKWARD_TWICE;

/// A snapshot of a variable at a certain version. A `SavedVariable` stores
/// enough information to reconstruct a variable from a certain point in time.
class TORCH_API SavedVariable {
 public:
  SavedVariable() = default;
  SavedVariable(const Variable& variable, bool is_output, bool is_inplace_view=false);
  SavedVariable(SavedVariable&&) = default;
  SavedVariable& operator=(SavedVariable&&) = default;

  /// Reconstructs the saved variable. Pass `saved_for` as the gradient
  /// function if constructing the `SavedVariable` with it would have caused a
  /// circular reference.
  Variable unpack(std::shared_ptr<Node> saved_for = nullptr) const;

  void reset_data() {
    return data_.reset();
  }

  void reset_grad_function() {
    grad_fn_.reset();
  }

  // Does a direct copy of data from the given otherData. Does not increment version_history, so this can be used to
  // plop new data into an existing graph ready for backprop (in fact that is the intent). Asserts if otherData does
  // not have the exact size and dimensions as the local data_.
  void copy_data_from(at::Tensor& otherData);

  // Allocate a blob of memory and copy the contents of the state data from this variable to it, then return it.
  // The caller is responsible for freeing the memory.
  uint8_t* serialize_to_blob() const;

  // Restore the state of an external saved variable given the blob. Can optionally perform validation. Essentially
  // the reverse of copy_to_blob().
  void deserialize_from_blob(uint8_t* blob);

 private:
  at::Tensor data_;

  // The gradient function associated with this node. If has_grad_fn
  // is false, then this is a leaf node. Note that the grad_fn is not saved if
  // it would create a circular reference. In that case, the grad_fn must be
  // passed in to the unpack function when reconstructing the Variable.
  std::shared_ptr<Node> grad_fn_;
  // Weak version of grad_fn_ that prevents leaks in rebase_history() for
  // inplace views.
  std::weak_ptr<Node> weak_grad_fn_;
  std::weak_ptr<Node> grad_accumulator_;
  c10::VariableVersion version_counter_;

  uint32_t saved_version_ = 0;
  uint32_t output_nr_ = 0;
  bool was_default_constructed_ = true;
  bool requires_grad_ = false;
  bool has_grad_fn_ = false;
  bool is_inplace_view_ = false;
};
}} // namespace torch::autograd
