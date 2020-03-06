#include <torch/torch.h>
#include <iostream>
#include <queue>

using namespace std;

std::shared_ptr<torch::autograd::Node> tensorToNode(torch::Tensor& tensor) {
  torch::autograd::AutogradMeta* meta =
      dynamic_cast<torch::autograd::AutogradMeta*>(tensor.unsafeGetTensorImpl()->autograd_meta());
  return meta->grad_fn_;
}

void recurseGraph(std::shared_ptr<torch::autograd::Node> node, queue<queue<uint8_t*>>& fullstk) {
  if(!node) return;
  cout << node->name();
  if(node->can_serialize_variables()) {
    cout << " [serializable]";
    queue<uint8_t*> blobq = node->serialize_variables();
    size_t blobsz = *reinterpret_cast<size_t*>(blobq.front());
    cout << " qsz:" << blobq.size() << " 1st blobsz:" <<blobsz;
    fullstk.push(blobq);
  }
  cout << endl;
  for(auto next : node->next_edges()) {
    recurseGraph(next.function, fullstk);
  }
}

void recurseRestore(std::shared_ptr<torch::autograd::Node> node, queue<queue<uint8_t*>>& fullstk) {
  if(!node) return;
  if(node->can_serialize_variables()) {
    node->deserialize_variables(fullstk.front());
    fullstk.pop();
  }
  for(auto next : node->next_edges()) {
    recurseGraph(next.function, fullstk);
  }
}

torch::Tensor forward(torch::Tensor& input) {
  auto t = input * input;
  return t.softmax(0);
}

int main() {
  torch::Tensor a = torch::tensor({1.,2.,3.},
                                  torch::device(torch::kCPU).dtype(torch::kFloat).requires_grad(true));

  torch::Tensor b = torch::tensor({1., 1., 1.},
                                  torch::device(torch::kCPU).dtype(torch::kFloat).requires_grad(true));

  torch::Tensor za = forward(a);
  queue<queue<uint8_t*>> fullstk;
  recurseGraph(tensorToNode(za), fullstk);

  za.backward(torch::tensor({0.,1.,0.}, torch::device(a.device()).dtype(torch::kFloat)));

  torch::Tensor zb = forward(b);
  zb.backward(torch::tensor({0.,1.,0.}, torch::device(b.device()).dtype(torch::kFloat)), true);
  cout << "After initial computation: " << endl << a.grad() << endl << b.grad() << endl << endl;

  recurseRestore(tensorToNode(zb), fullstk);
  memcpy(b.data_ptr(), a.data_ptr(), b.nbytes());

  cout << "b after deserialize: " << endl << b << endl << endl;
  b.grad().zero_();
  zb.backward(torch::tensor({0.,1.,0.}, torch::device(b.device()).dtype(torch::kFloat)), true);
  cout << "After b copy: " << endl << a.grad() << endl << b.grad() << endl << endl;
}
