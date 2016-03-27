#include <algorithm>
#include <vector>

#include "caffe/layers/oplu_layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ void OPLUForward(const int n, const Dtype* in, Dtype* out) {
  CUDA_KERNEL_LOOP2(index, n) {
    if (in[index] > in[index+1]) {
       out[index] = in[index];
       out[index+1] = in[index+1];
     }
    else {
       out[index+1] = in[index];
       out[index] = in[index+1];  
    }
  }
}


template <typename Dtype>
void OPLULayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int count = bottom[0]->count();

  // NOLINT_NEXT_LINE(whitespace/operators)
  OPLUForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, bottom_data, top_data);
  CUDA_POST_KERNEL_CHECK;
}


template <typename Dtype>
__global__ void OPLUBackward(const int n, const Dtype* in_diff,
    const Dtype* out_data, const Dtype* in_data,
    Dtype* out_diff) {

    CUDA_KERNEL_LOOP2(index, n) {
    if (in_data[index] > in_data[index+1]) {
       out_diff[index] = in_diff[index];
       out_diff[index+1] = in_diff[index+1];
     }
    else {
       out_diff[index+1] = in_diff[index];
       out_diff[index] = in_diff[index+1];  
    }
 }
}


template <typename Dtype>
void OPLULayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* bottom_data = bottom[0]->gpu_data();
    const Dtype* top_diff = top[0]->gpu_diff();
    const Dtype* top_data = top[0]->gpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const int count = bottom[0]->count();

    // NOLINT_NEXT_LINE(whitespace/operators)
    OPLUBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, top_diff, top_data, bottom_data, bottom_diff);
    CUDA_POST_KERNEL_CHECK;
  }
}


INSTANTIATE_LAYER_GPU_FUNCS(OPLULayer);


}  // namespace caffe
