#include <math/TensorMath.hpp>
#include <stdexcept>

namespace{
    /*Ensures `a` and `b` have the same exact shape*/
    void _ensure_same_shape(Tensor& a, Tensor& b){
        if(a.shape != b.shape){ throw std::runtime_error("Error: Tensor shape mismatch"); }
    }

    void _ensure_same_device(Tensor& a, Tensor& b, DeviceType type){
        auto a_device_type = a.getDevice()->get_type();
        auto b_device_type = b.getDevice()->get_type();
        if(a_device_type != type || a_device_type != b_device_type){ throw std::runtime_error("Error: Unexpected device"); }
    }

    /*Expands the shape adding outer dimensions of length one with strides of zeros */
    Shape _expand_shape_no_strides(const Shape& a){
        //Create the new shape consisting of all dimensions of length 1
        std::array<size_t,TENSOR_MAX_DIM> _temp;
        _temp.fill(1);
        Shape new_shape = Shape(_temp);
        //Manually make all the strides zero
        for(size_t i = 0; i < TENSOR_MAX_DIM; i++){
            new_shape._strides[i] = 0;
        }
        //Now append the `a` shape which is something like (A,B,C...,M,0,0,0,0) to the new shape as ->(1,1,1,...,1,A,B,C,...,M)
        for(size_t i = 0; i < a.n_dim; i++){
            new_shape._shapes[TENSOR_MAX_DIM - a.n_dim + i] = a._shapes[i];
            new_shape._strides[TENSOR_MAX_DIM - a.n_dim + i] = a._strides[i]; 
        }
        //And return the expanded shape
        return new_shape;
    }
    /*Expands the shape adding outer dimensions of lenght 1, but strides are kept correct*/
    Shape _expand_shape_with_strides(const Shape& a){
        //FIrst make a copy of a._shape in a freshly new array
        std::array<size_t, TENSOR_MAX_DIM> _new_shape;
        _new_shape.fill(1);
        for(size_t i = 0; i < a.n_dim; i++){
            _new_shape[TENSOR_MAX_DIM - a.n_dim + i] = a._shapes[i];
        }
        //Shape constructor also initializes strides, so is okay
        return Shape(_new_shape);
    }

    /*Matmul subroutine: Broadcast shapes `a` and `b`, ignoring the two innermost dimensions*/
    void _input_broadcast(Shape& a, Shape& b){
        //A small validation: Are they expanded?
        if(a.n_dim != TENSOR_MAX_DIM || a.n_dim != b.n_dim){ throw std::runtime_error("Error: Shapes are not expanded"); }
        
        for(size_t i = TENSOR_MAX_DIM - 2; i-->0;){
            //If one of them is one, then broadcast the max shape between both
            if(a._shapes[i] == 1 || b._shapes[i] == 1){
                size_t max = std::max(a._shapes[i], b._shapes[i]);
                //Before broadcasting, we need sure to put a zero stride on the dimension affected
                if( a._shapes[i] == max) b._strides[i] = 0; 
                else a._strides[i] = 0;
                a._shapes[i] = max;
                b._shapes[i] = max;
            }else{
                //If neither one of them is one, then THEY SHOULD BE EQUAL, or the broadcast is impossible
                if( a._shapes[i] != b._shapes[i] ){ throw std::runtime_error("Error: Bad broadcasting"); }
            }
        }
    }

    /*Check the already broadcasted and expanded shapes of the inputs and output tensors*/
    void _check_matmul_shapes(Shape& a, Shape& b, Shape& out){
        //Clasic matrix multiplication check on two last dimensions
        if( 
            a._shapes[TENSOR_MAX_DIM - 1] != b._shapes[TENSOR_MAX_DIM - 2] || 
            a._shapes[TENSOR_MAX_DIM - 2] != out._shapes[TENSOR_MAX_DIM - 2] ||
            b._shapes[TENSOR_MAX_DIM - 1] != out._shapes[TENSOR_MAX_DIM - 1]
        ){
            throw std::exception("Error: Unexpected shapes. Expected inputs (...,M,K),(...,K,N) and output (...,M,N)");
        }
        //Now check if the rest of the shapes are equal
        for(size_t i = TENSOR_MAX_DIM - 2; i --> 0;){
            if(a._shapes[i] != b._shapes[i]){ throw std::exception("Error: Outer dimensions are supposed to be equal. Bad broadcasting"); }
            if(a._shapes[i] != out._shapes[i]){ throw std::exception("Error: Outer dimensions are supposed to be equal. Bad output expansion"); }
        }
    }

    void _check_conv2D_shapes(Shape& a, Shape& b, Shape& out){

    }

};

void TensorMath::add(Tensor& a, Tensor&b, Tensor& out){
    _ensure_same_shape(a,b);
    _ensure_same_device(a,b,DeviceType::CPU);
    _ensure_same_shape(b,out);
    _ensure_same_device(b,out,DeviceType::CPU);
    
    auto device = a.getDevice();
    device->math->add(a.data(),b.data(),out.data(),a.shape.n_elements);

}

void TensorMath::sub(Tensor& a, Tensor&b, Tensor& out){
    _ensure_same_shape(a,b);
    _ensure_same_device(a,b,DeviceType::CPU);
    _ensure_same_shape(b,out);
    _ensure_same_device(b,out,DeviceType::CPU);

}

void TensorMath::scale(Tensor& a, float n, Tensor& out){
    _ensure_same_shape(a,out);
    _ensure_same_device(a,out,DeviceType::CPU);

}

void TensorMath::multiply(Tensor& a, Tensor& b, Tensor& out){
    _ensure_same_shape(a,b);
    _ensure_same_device(a,b,DeviceType::CPU);
    _ensure_same_shape(b,out);
    _ensure_same_device(b,out,DeviceType::CPU);

}

/*Matmul routine*/
void TensorMath::matmul(Tensor& a , Tensor& b , Tensor& out){
    
    _ensure_same_device(a,b,DeviceType::CPU);
    _ensure_same_device(b,out,DeviceType::CPU);

    auto device = a.getDevice();

    Shape _expanded_a = _expand_shape_no_strides(a.shape);
    Shape _expanded_b = _expand_shape_no_strides(b.shape);
    _input_broadcast(_expanded_a,_expanded_b);
    
    Shape _virtual_expanded_out = _expand_shape_with_strides(out.shape);
    _check_matmul_shapes(_expanded_a,_expanded_b,_virtual_expanded_out);

    //Now call the real matmul

    std::printf("--- MATMUL METADATA SANITY CHECK ---\n");
    std::printf("A Shape:   [%zu, %zu, %zu, %zu] | Strides: [%zu, %zu, %zu, %zu]\n",
            _expanded_a[0], _expanded_a[1], _expanded_a[2], _expanded_a[3],
            _expanded_a._strides[0], _expanded_a._strides[1], _expanded_a._strides[2], _expanded_a._strides[3]);
    std::printf("B Shape:   [%zu, %zu, %zu, %zu] | Strides: [%zu, %zu, %zu, %zu]\n",
            _expanded_b[0], _expanded_b[1], _expanded_b[2], _expanded_b[3],
            _expanded_b._strides[0], _expanded_b._strides[1], _expanded_b._strides[2], _expanded_b._strides[3]);
    std::printf("Out Shape: [%zu, %zu, %zu, %zu] | Strides: [%zu, %zu, %zu, %zu]\n",
            _virtual_expanded_out[0], _virtual_expanded_out[1], _virtual_expanded_out[2], _virtual_expanded_out[3],
            _virtual_expanded_out._strides[0], _virtual_expanded_out._strides[1], _virtual_expanded_out._strides[2], _virtual_expanded_out._strides[3]);

    device->math->matmul(
        a.data(),_expanded_a,
        b.data(),_expanded_b,
        out.data(), _virtual_expanded_out
    );

    //Finish
}
/*
Convolution operation.
Expects 4D tensors. No exceptions allowed.
*/
void TensorMath::conv2D(Tensor& in, Tensor& ker, Tensor& out, size_t stride, size_t padding){
    //Device guards
    _ensure_same_device(in,ker,DeviceType::CPU);
    _ensure_same_device(ker,out,DeviceType::CPU);
    //Ensure all tensors are EXACTLY 4D
    if(in.shape.n_dim != 4 || ker.shape.n_dim != 4 || out.shape.n_dim != 4){ throw std::runtime_error("Error: Tensors must be 4-dimensional"); }
    
    //Should we implement all guards here?

    auto device = in.getDevice();
    
    //just call the underlying math function
    device->math->conv2D(
        in.data(),in.shape,
        ker.data(),ker.shape,
        out.data(),out.shape,
        {stride,stride},
        {padding,padding}
    );

}


void TensorMath::conv3D(Tensor& in, Tensor& ker, Tensor& out, size_t stride, size_t padding){}

void TensorMath::reLU(Tensor& a, Tensor& out){}

void TensorMath::sigmoid(Tensor& a, Tensor& out){}

void TensorMath::tanh(Tensor& a, Tensor& out){}

void TensorMath::softmax(Tensor& a, Tensor& out){}

void TensorMath::sum(Tensor& a, size_t axis, Tensor& out){}

Tensor TensorMath::reshape(Tensor& a, std::array<size_t, TENSOR_MAX_DIM> new_shape){
    Shape target_shape(new_shape);
    if(target_shape.n_elements != a.shape.n_elements){
        throw std::runtime_error("Error: reshape requires the same number of elements");
    }
    Tensor out = Tensor(new_shape, a.getDevice());
    for(size_t i = 0; i < a.shape.n_elements; i++){
        out.data()[i] = a.data()[i];
    }
    return out;
}

Tensor TensorMath::transpose(Tensor& a){    
    if(a.shape.n_dim < 2){
        return Tensor(a, a.getDevice());
    }
    std::array<size_t, TENSOR_MAX_DIM> new_shape_arr = a.shape._shapes;
    size_t last = a.shape.n_dim - 1;
    size_t prev = a.shape.n_dim - 2;
    size_t tmp = new_shape_arr[prev];
    new_shape_arr[prev] = new_shape_arr[last];
    new_shape_arr[last] = tmp;
    Tensor out(new_shape_arr, a.getDevice());
    std::array<size_t, TENSOR_MAX_DIM> coords;
    coords.fill(0);

    auto index_from_coords = [](const Shape& shape, const std::array<size_t, TENSOR_MAX_DIM>& c){
        size_t idx = 0;
        for(size_t i = 0; i < shape.n_dim; i++){
            idx += c[i] * shape._strides[i];
        }
        return idx;
    };

    for(size_t processed = 0; processed < out.shape.n_elements; processed++){
        std::array<size_t, TENSOR_MAX_DIM> src_coords = coords;
        size_t tmp_coord = src_coords[prev];
        src_coords[prev] = src_coords[last];
        src_coords[last] = tmp_coord;

        size_t src_idx = index_from_coords(a.shape, src_coords);
        size_t dst_idx = index_from_coords(out.shape, coords);

        out.data()[dst_idx] = a.data()[src_idx];

        for(size_t i = out.shape.n_dim; i-- > 0;){
            coords[i]++;
            if(coords[i] < out.shape[i]){
                break;
            }
            coords[i] = 0;
        }
    }
    return out;
}