#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include <iostream>
#include <vector>
#include <math.h>

using namespace tensorflow;
using namespace std;

REGISTER_OP("BpMllGrad")
  .Input("grad: float")
  .Input("logits: float")
  .Input("labels: float")
  .Output("grad_input: float")
  .Output("grad_labels: float");
float f_signoid(float numero)
{
    return numero;//(float)1 / (1 + exp( -1 * numero));
}

class BpMllGradOp : public OpKernel {
  public:
  explicit BpMllGradOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    DCHECK_EQ(3, context->num_inputs());
    // Grab the input tensor
    // get the gradient tensor
    const Tensor& grad_tensor = context->input(0);

    const Tensor& input_tensor = context->input(1);
    
    const Tensor& input2_tensor = context->input(2);
    
    // create input shape (inferred from the additional attribute `n`)
    TensorShape input_shape = input_tensor.shape();
    TensorShape input2_shape = input2_tensor.shape();
    
    //DCHECK_EQ(input_shape.dim_size(0), input2_shape.dim_size(1));
    //DCHECK_EQ(input2_shape.dim_size(0), grad_tensor.shape().dim_size(0));
    
    // create output tensors
    Tensor* grad_input = NULL;
    Tensor* grad_input2 = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_shape, &grad_input));
    //OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({1}), &grad_input));
    OP_REQUIRES_OK(context, context->allocate_output(1, input2_shape, &grad_input2));
    //OP_REQUIRES_OK(context, context->allocate_output(1, TensorShape({1}), &grad_input2));

    //auto grad = grad_tensor.matrix<float>();
    auto input = input_tensor.matrix<float>();
    auto labels = input2_tensor.matrix<float>();
    auto grad_input_tensor = grad_input->matrix<float>();
    auto grad_input2_tensor = grad_input2->matrix<float>();
    int labels_size = 103;
    float total_error = 0;
    for (int i = 0; i < 128; i++) {
      // separate labels
      vector<float> y_belong;
      vector<float> y_not_belong;
      //cout << (i+1) << ". Labels: [";
      for (int j = labels_size * i; j < labels_size * (i + 1); j++) {
        //cout << labels(j) << ", ";
        if (labels(j) == 1.0) {
          
          y_belong.push_back(input(j));
        }
        else
          y_not_belong.push_back(input(j));
      }
      //cout << "] \n";
      // calulcate error
      //cout << (i + 1) << ". Output [";
      for (int j = labels_size * i; j < labels_size * (i + 1); j++) {
        //float delta_error_function = (1 + input(j)) * (1 - input(j)); // tanh
        float delta_error_function = input(j) * (1 - input(j)); //sigmoid
        float probability_label = ((float)1 / (y_belong.size() * y_not_belong.size()));
        //cout << roundf(input(j) * 1000)/1000 << ", ";
        if (labels(j) == 1.0) {
          
          float temp = 0;
          for (int l = 0; l < y_not_belong.size(); l++) {
            temp += exp(-1 * (input(j) - y_not_belong[l]));
          }
          grad_input_tensor(j) = -1 * probability_label * temp * delta_error_function;
          //cout << roundf(grad_input_tensor(j) * 10000)/10000 << ", ";
        }
        else {

          float temp = 0;
          for (int k = 0; k < y_belong.size(); k++)
            temp += exp(-1 * (y_belong[k] - input(j)));
          grad_input_tensor(j) = 1 * probability_label * temp * delta_error_function;
          //cout << roundf(grad_input_tensor(j) * 10000)/10000 << ", ";
        }
        //cout << grad_input_tensor(j) << ", ";
      }
      //cout << "]\n";
    }
    
  }
};

REGISTER_KERNEL_BUILDER(Name("BpMllGrad").Device(DEVICE_CPU), BpMllGradOp);