#include <algorithm>
#include <vector>

#include "caffe/layers/oplu_layer.hpp"

namespace caffe {

template <typename Dtype>
void OPLULayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int count = bottom[0]->count();

  for (int i = 0; i < count; i = i + 2) {
     int j = i + 1;
     
     if (bottom_data[i] > bottom_data[j]) {
	top_data[i] = bottom_data[i];
	top_data[j] = bottom_data[j];
     }
     else {
	top_data[i] = bottom_data[j];
	top_data[j] = bottom_data[i];
     }
  }


  /*int ndata = bottom[0]->shape(0);
  int nhidden = bottom[0]->shape(1);

  for (int k = 0; k < ndata; ++k) {
   for (int n = 1; n <= nhidden/2; ++n) {
     //int i = k*nhidden + n;
     int i2 = k*nhidden + n*2;
     int i1 = i2 - 1;

     if (bottom_data[i1] > bottom_data[i2]) {
	top_data[i1] = bottom_data[i1];
	top_data[i2] = bottom_data[i2];
     } // if
     else {
	top_data[i1] = bottom_data[i2];
	top_data[i2] = bottom_data[i1];
     } // else     
    } // for n  
  } // for k */
 
}


template <typename Dtype>
void OPLULayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* bottom_data = bottom[0]->cpu_data();
    const Dtype* top_data = top[0]->cpu_data();
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const int count = bottom[0]->count();

  for (int i = 0; i < count; i = i + 2) {
     int j = i + 1;
     
     if (bottom_data[i] > bottom_data[j]) {
	bottom_diff[i] = top_diff[i];
	bottom_diff[j] = top_diff[j];
     }
     else {
	bottom_diff[i] = top_diff[j];
	bottom_diff[j] = top_diff[i];
     } 
  }


  /*
  int ndata = bottom[0]->shape(0);
  int nhidden = bottom[0]->shape(1);
    
  for (int k = 0; k < ndata; ++k) {
   for (int n = 1; n <= nhidden/2; ++n) {
     //int i = k*nhidden + n;
     int i2 = k*nhidden + n*2;
     int i1 = i2 - 1;

     if (bottom_data[i1] > bottom_data[i2]) {
	bottom_diff[i1] = top_diff[i1];
	bottom_diff[i2] = top_diff[i2];
     }
     else {
	bottom_diff[i1] = top_diff[i2];
	bottom_diff[i2] = top_diff[i1];
     } // else     

    } // for n  
  } //for k */


  }
}


#ifdef CPU_ONLY
STUB_GPU(OPLULayer);
#endif

INSTANTIATE_CLASS(OPLULayer);
REGISTER_LAYER_CLASS(OPLU);

}  // namespace caffe
