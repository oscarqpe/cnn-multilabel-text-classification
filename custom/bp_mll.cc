#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include <iostream>
#include <vector>
#include <math.h>

using namespace tensorflow;
using namespace std;

REGISTER_OP("BpMll")
	.Input("logits: float")
	.Input("labels: float")
	.Output("output: float");
float f_signoid(float numero)
{
    return numero;//(float)1 / (1 + exp( -1 * numero));
}
class BpMllOp : public OpKernel {
	public:
	explicit BpMllOp(OpKernelConstruction* context) : OpKernel(context) {}

	void Compute(OpKernelContext* context) override {
		// Grab the input tensor
		const Tensor& input_tensor = context->input(0);
		auto input = input_tensor.matrix<float>();
		const Tensor& input2_tensor = context->input(1);
		auto labels = input2_tensor.matrix<float>();
		int labels_size = 103;
		float total_error = 0;
		//cout << "[";
		for (int i = 0; i < 128; i++) {
			// separate labels
			vector<float> y_belong;
			vector<float> y_not_belong;
			for (int j = labels_size * i; j < labels_size * (i + 1); j++)
				if (labels(j) == 1.0)
					y_belong.push_back(input(j));
				else
					y_not_belong.push_back(input(j));
			float error_i = 0;
			for (int k = 0; k < y_belong.size(); k++)
				for (int l = 0; l < y_not_belong.size(); l++) {
					//cout << y_belong[k] << " - " << y_not_belong[l] << " = " << y_belong[k] - y_not_belong[l] << "-> " << exp(-1 * (y_belong[k] - y_not_belong[l]));
					//cout << exp(-1 * (y_belong[k] - y_not_belong[l])) << ", ";
					error_i += exp(-1 * (y_belong[k] - y_not_belong[l]));
				}
			float probability = ((float)1 / (y_belong.size() * y_not_belong.size()));
			//cout << "Probability: " << probability << endl;
			total_error += probability * error_i;
			//cout << "Error " << i + 1 << ": " << error_i << endl;
			//cout << error_i << ", ";
			//cout << "Error total: " << total_error << endl;
			//cout << endl;
		}
		//cout << "] \n";
		//cout << "Error total: " << total_error << endl;
		Tensor* output_tensor = NULL;
		OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({1}), 
			&output_tensor));
		auto output = output_tensor->flat<float>();
		output(0) = total_error;
	}
};

REGISTER_KERNEL_BUILDER(Name("BpMll").Device(DEVICE_CPU), BpMllOp);