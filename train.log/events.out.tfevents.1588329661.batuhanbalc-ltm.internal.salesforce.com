       �K"	  @/���Abrain.Event:2EIg���     �L9l	
�^/���A"��

anchor_inputPlaceholder*
dtype0*/
_output_shapes
:���������
*$
shape:���������

�
positive_inputPlaceholder*
dtype0*/
_output_shapes
:���������
*$
shape:���������

�
negative_inputPlaceholder*
dtype0*/
_output_shapes
:���������
*$
shape:���������

�
.conv1/weights/Initializer/random_uniform/shapeConst*%
valueB"             * 
_class
loc:@conv1/weights*
dtype0*
_output_shapes
:
�
,conv1/weights/Initializer/random_uniform/minConst*
valueB
 *�Er�* 
_class
loc:@conv1/weights*
dtype0*
_output_shapes
: 
�
,conv1/weights/Initializer/random_uniform/maxConst*
valueB
 *�Er=* 
_class
loc:@conv1/weights*
dtype0*
_output_shapes
: 
�
6conv1/weights/Initializer/random_uniform/RandomUniformRandomUniform.conv1/weights/Initializer/random_uniform/shape*
dtype0*&
_output_shapes
: *

seed *
T0* 
_class
loc:@conv1/weights*
seed2 
�
,conv1/weights/Initializer/random_uniform/subSub,conv1/weights/Initializer/random_uniform/max,conv1/weights/Initializer/random_uniform/min*
T0* 
_class
loc:@conv1/weights*
_output_shapes
: 
�
,conv1/weights/Initializer/random_uniform/mulMul6conv1/weights/Initializer/random_uniform/RandomUniform,conv1/weights/Initializer/random_uniform/sub*
T0* 
_class
loc:@conv1/weights*&
_output_shapes
: 
�
(conv1/weights/Initializer/random_uniformAdd,conv1/weights/Initializer/random_uniform/mul,conv1/weights/Initializer/random_uniform/min*
T0* 
_class
loc:@conv1/weights*&
_output_shapes
: 
�
conv1/weights
VariableV2*
shared_name * 
_class
loc:@conv1/weights*
	container *
shape: *
dtype0*&
_output_shapes
: 
�
conv1/weights/AssignAssignconv1/weights(conv1/weights/Initializer/random_uniform*
validate_shape(*&
_output_shapes
: *
use_locking(*
T0* 
_class
loc:@conv1/weights
�
conv1/weights/readIdentityconv1/weights*
T0* 
_class
loc:@conv1/weights*&
_output_shapes
: 
�
conv1/biases/Initializer/zerosConst*
valueB *    *
_class
loc:@conv1/biases*
dtype0*
_output_shapes
: 
�
conv1/biases
VariableV2*
_class
loc:@conv1/biases*
	container *
shape: *
dtype0*
_output_shapes
: *
shared_name 
�
conv1/biases/AssignAssignconv1/biasesconv1/biases/Initializer/zeros*
use_locking(*
T0*
_class
loc:@conv1/biases*
validate_shape(*
_output_shapes
: 
q
conv1/biases/readIdentityconv1/biases*
T0*
_class
loc:@conv1/biases*
_output_shapes
: 
j
model/conv1/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
�
model/conv1/Conv2DConv2Danchor_inputconv1/weights/read*/
_output_shapes
:���������
 *
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME
�
model/conv1/BiasAddBiasAddmodel/conv1/Conv2Dconv1/biases/read*
T0*
data_formatNHWC*/
_output_shapes
:���������
 
m
model/conv1/conv1/ReluRelumodel/conv1/BiasAdd*
T0*/
_output_shapes
:���������
 
�
model/conv1/MaxPool2D/MaxPoolMaxPoolmodel/conv1/conv1/Relu*
ksize
*
paddingSAME*/
_output_shapes
:���������
 *
T0*
data_formatNHWC*
strides

�
.conv2/weights/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*%
valueB"          @   * 
_class
loc:@conv2/weights
�
,conv2/weights/Initializer/random_uniform/minConst*
valueB
 *��L�* 
_class
loc:@conv2/weights*
dtype0*
_output_shapes
: 
�
,conv2/weights/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *
valueB
 *��L=* 
_class
loc:@conv2/weights
�
6conv2/weights/Initializer/random_uniform/RandomUniformRandomUniform.conv2/weights/Initializer/random_uniform/shape*
T0* 
_class
loc:@conv2/weights*
seed2 *
dtype0*&
_output_shapes
: @*

seed 
�
,conv2/weights/Initializer/random_uniform/subSub,conv2/weights/Initializer/random_uniform/max,conv2/weights/Initializer/random_uniform/min*
_output_shapes
: *
T0* 
_class
loc:@conv2/weights
�
,conv2/weights/Initializer/random_uniform/mulMul6conv2/weights/Initializer/random_uniform/RandomUniform,conv2/weights/Initializer/random_uniform/sub*&
_output_shapes
: @*
T0* 
_class
loc:@conv2/weights
�
(conv2/weights/Initializer/random_uniformAdd,conv2/weights/Initializer/random_uniform/mul,conv2/weights/Initializer/random_uniform/min*
T0* 
_class
loc:@conv2/weights*&
_output_shapes
: @
�
conv2/weights
VariableV2*
dtype0*&
_output_shapes
: @*
shared_name * 
_class
loc:@conv2/weights*
	container *
shape: @
�
conv2/weights/AssignAssignconv2/weights(conv2/weights/Initializer/random_uniform*
validate_shape(*&
_output_shapes
: @*
use_locking(*
T0* 
_class
loc:@conv2/weights
�
conv2/weights/readIdentityconv2/weights*&
_output_shapes
: @*
T0* 
_class
loc:@conv2/weights
�
conv2/biases/Initializer/zerosConst*
valueB@*    *
_class
loc:@conv2/biases*
dtype0*
_output_shapes
:@
�
conv2/biases
VariableV2*
dtype0*
_output_shapes
:@*
shared_name *
_class
loc:@conv2/biases*
	container *
shape:@
�
conv2/biases/AssignAssignconv2/biasesconv2/biases/Initializer/zeros*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*
_class
loc:@conv2/biases
q
conv2/biases/readIdentityconv2/biases*
T0*
_class
loc:@conv2/biases*
_output_shapes
:@
j
model/conv2/dilation_rateConst*
dtype0*
_output_shapes
:*
valueB"      
�
model/conv2/Conv2DConv2Dmodel/conv1/MaxPool2D/MaxPoolconv2/weights/read*/
_output_shapes
:���������
@*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME
�
model/conv2/BiasAddBiasAddmodel/conv2/Conv2Dconv2/biases/read*
T0*
data_formatNHWC*/
_output_shapes
:���������
@
m
model/conv2/conv2/ReluRelumodel/conv2/BiasAdd*/
_output_shapes
:���������
@*
T0
�
model/conv2/MaxPool2D/MaxPoolMaxPoolmodel/conv2/conv2/Relu*
ksize
*
paddingSAME*/
_output_shapes
:���������@*
T0*
data_formatNHWC*
strides

�
.conv3/weights/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*%
valueB"      @   �   * 
_class
loc:@conv3/weights
�
,conv3/weights/Initializer/random_uniform/minConst*
valueB
 *�[q�* 
_class
loc:@conv3/weights*
dtype0*
_output_shapes
: 
�
,conv3/weights/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *
valueB
 *�[q=* 
_class
loc:@conv3/weights
�
6conv3/weights/Initializer/random_uniform/RandomUniformRandomUniform.conv3/weights/Initializer/random_uniform/shape*
T0* 
_class
loc:@conv3/weights*
seed2 *
dtype0*'
_output_shapes
:@�*

seed 
�
,conv3/weights/Initializer/random_uniform/subSub,conv3/weights/Initializer/random_uniform/max,conv3/weights/Initializer/random_uniform/min*
T0* 
_class
loc:@conv3/weights*
_output_shapes
: 
�
,conv3/weights/Initializer/random_uniform/mulMul6conv3/weights/Initializer/random_uniform/RandomUniform,conv3/weights/Initializer/random_uniform/sub*
T0* 
_class
loc:@conv3/weights*'
_output_shapes
:@�
�
(conv3/weights/Initializer/random_uniformAdd,conv3/weights/Initializer/random_uniform/mul,conv3/weights/Initializer/random_uniform/min*
T0* 
_class
loc:@conv3/weights*'
_output_shapes
:@�
�
conv3/weights
VariableV2* 
_class
loc:@conv3/weights*
	container *
shape:@�*
dtype0*'
_output_shapes
:@�*
shared_name 
�
conv3/weights/AssignAssignconv3/weights(conv3/weights/Initializer/random_uniform*
use_locking(*
T0* 
_class
loc:@conv3/weights*
validate_shape(*'
_output_shapes
:@�
�
conv3/weights/readIdentityconv3/weights*
T0* 
_class
loc:@conv3/weights*'
_output_shapes
:@�
�
conv3/biases/Initializer/zerosConst*
valueB�*    *
_class
loc:@conv3/biases*
dtype0*
_output_shapes	
:�
�
conv3/biases
VariableV2*
shared_name *
_class
loc:@conv3/biases*
	container *
shape:�*
dtype0*
_output_shapes	
:�
�
conv3/biases/AssignAssignconv3/biasesconv3/biases/Initializer/zeros*
T0*
_class
loc:@conv3/biases*
validate_shape(*
_output_shapes	
:�*
use_locking(
r
conv3/biases/readIdentityconv3/biases*
T0*
_class
loc:@conv3/biases*
_output_shapes	
:�
j
model/conv3/dilation_rateConst*
dtype0*
_output_shapes
:*
valueB"      
�
model/conv3/Conv2DConv2Dmodel/conv2/MaxPool2D/MaxPoolconv3/weights/read*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*0
_output_shapes
:����������*
	dilations
*
T0
�
model/conv3/BiasAddBiasAddmodel/conv3/Conv2Dconv3/biases/read*
T0*
data_formatNHWC*0
_output_shapes
:����������
n
model/conv3/conv3/ReluRelumodel/conv3/BiasAdd*
T0*0
_output_shapes
:����������
�
model/conv3/MaxPool2D/MaxPoolMaxPoolmodel/conv3/conv3/Relu*0
_output_shapes
:����������*
T0*
data_formatNHWC*
strides
*
ksize
*
paddingSAME
�
.conv4/weights/Initializer/random_uniform/shapeConst*%
valueB"      �      * 
_class
loc:@conv4/weights*
dtype0*
_output_shapes
:
�
,conv4/weights/Initializer/random_uniform/minConst*
valueB
 *   �* 
_class
loc:@conv4/weights*
dtype0*
_output_shapes
: 
�
,conv4/weights/Initializer/random_uniform/maxConst*
valueB
 *   >* 
_class
loc:@conv4/weights*
dtype0*
_output_shapes
: 
�
6conv4/weights/Initializer/random_uniform/RandomUniformRandomUniform.conv4/weights/Initializer/random_uniform/shape*
dtype0*(
_output_shapes
:��*

seed *
T0* 
_class
loc:@conv4/weights*
seed2 
�
,conv4/weights/Initializer/random_uniform/subSub,conv4/weights/Initializer/random_uniform/max,conv4/weights/Initializer/random_uniform/min*
T0* 
_class
loc:@conv4/weights*
_output_shapes
: 
�
,conv4/weights/Initializer/random_uniform/mulMul6conv4/weights/Initializer/random_uniform/RandomUniform,conv4/weights/Initializer/random_uniform/sub*
T0* 
_class
loc:@conv4/weights*(
_output_shapes
:��
�
(conv4/weights/Initializer/random_uniformAdd,conv4/weights/Initializer/random_uniform/mul,conv4/weights/Initializer/random_uniform/min*
T0* 
_class
loc:@conv4/weights*(
_output_shapes
:��
�
conv4/weights
VariableV2*
shape:��*
dtype0*(
_output_shapes
:��*
shared_name * 
_class
loc:@conv4/weights*
	container 
�
conv4/weights/AssignAssignconv4/weights(conv4/weights/Initializer/random_uniform*
validate_shape(*(
_output_shapes
:��*
use_locking(*
T0* 
_class
loc:@conv4/weights
�
conv4/weights/readIdentityconv4/weights*
T0* 
_class
loc:@conv4/weights*(
_output_shapes
:��
�
conv4/biases/Initializer/zerosConst*
valueB�*    *
_class
loc:@conv4/biases*
dtype0*
_output_shapes	
:�
�
conv4/biases
VariableV2*
dtype0*
_output_shapes	
:�*
shared_name *
_class
loc:@conv4/biases*
	container *
shape:�
�
conv4/biases/AssignAssignconv4/biasesconv4/biases/Initializer/zeros*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0*
_class
loc:@conv4/biases
r
conv4/biases/readIdentityconv4/biases*
T0*
_class
loc:@conv4/biases*
_output_shapes	
:�
j
model/conv4/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
�
model/conv4/Conv2DConv2Dmodel/conv3/MaxPool2D/MaxPoolconv4/weights/read*0
_output_shapes
:����������*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME
�
model/conv4/BiasAddBiasAddmodel/conv4/Conv2Dconv4/biases/read*
T0*
data_formatNHWC*0
_output_shapes
:����������
n
model/conv4/conv4/ReluRelumodel/conv4/BiasAdd*0
_output_shapes
:����������*
T0
�
model/conv4/MaxPool2D/MaxPoolMaxPoolmodel/conv4/conv4/Relu*
T0*
data_formatNHWC*
strides
*
ksize
*
paddingSAME*0
_output_shapes
:����������
�
.conv5/weights/Initializer/random_uniform/shapeConst*%
valueB"            * 
_class
loc:@conv5/weights*
dtype0*
_output_shapes
:
�
,conv5/weights/Initializer/random_uniform/minConst*
valueB
 *���* 
_class
loc:@conv5/weights*
dtype0*
_output_shapes
: 
�
,conv5/weights/Initializer/random_uniform/maxConst*
valueB
 *��>* 
_class
loc:@conv5/weights*
dtype0*
_output_shapes
: 
�
6conv5/weights/Initializer/random_uniform/RandomUniformRandomUniform.conv5/weights/Initializer/random_uniform/shape*
dtype0*'
_output_shapes
:�*

seed *
T0* 
_class
loc:@conv5/weights*
seed2 
�
,conv5/weights/Initializer/random_uniform/subSub,conv5/weights/Initializer/random_uniform/max,conv5/weights/Initializer/random_uniform/min*
T0* 
_class
loc:@conv5/weights*
_output_shapes
: 
�
,conv5/weights/Initializer/random_uniform/mulMul6conv5/weights/Initializer/random_uniform/RandomUniform,conv5/weights/Initializer/random_uniform/sub*
T0* 
_class
loc:@conv5/weights*'
_output_shapes
:�
�
(conv5/weights/Initializer/random_uniformAdd,conv5/weights/Initializer/random_uniform/mul,conv5/weights/Initializer/random_uniform/min*
T0* 
_class
loc:@conv5/weights*'
_output_shapes
:�
�
conv5/weights
VariableV2*
	container *
shape:�*
dtype0*'
_output_shapes
:�*
shared_name * 
_class
loc:@conv5/weights
�
conv5/weights/AssignAssignconv5/weights(conv5/weights/Initializer/random_uniform*
T0* 
_class
loc:@conv5/weights*
validate_shape(*'
_output_shapes
:�*
use_locking(
�
conv5/weights/readIdentityconv5/weights*
T0* 
_class
loc:@conv5/weights*'
_output_shapes
:�
�
conv5/biases/Initializer/zerosConst*
valueB*    *
_class
loc:@conv5/biases*
dtype0*
_output_shapes
:
�
conv5/biases
VariableV2*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name *
_class
loc:@conv5/biases
�
conv5/biases/AssignAssignconv5/biasesconv5/biases/Initializer/zeros*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*
_class
loc:@conv5/biases
q
conv5/biases/readIdentityconv5/biases*
T0*
_class
loc:@conv5/biases*
_output_shapes
:
j
model/conv5/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
�
model/conv5/Conv2DConv2Dmodel/conv4/MaxPool2D/MaxPoolconv5/weights/read*/
_output_shapes
:���������*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME
�
model/conv5/BiasAddBiasAddmodel/conv5/Conv2Dconv5/biases/read*
T0*
data_formatNHWC*/
_output_shapes
:���������
�
model/conv5/MaxPool2D/MaxPoolMaxPoolmodel/conv5/BiasAdd*
ksize
*
paddingSAME*/
_output_shapes
:���������*
T0*
data_formatNHWC*
strides

x
model/Flatten/flatten/ShapeShapemodel/conv5/MaxPool2D/MaxPool*
T0*
out_type0*
_output_shapes
:
s
)model/Flatten/flatten/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
u
+model/Flatten/flatten/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
u
+model/Flatten/flatten/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
�
#model/Flatten/flatten/strided_sliceStridedSlicemodel/Flatten/flatten/Shape)model/Flatten/flatten/strided_slice/stack+model/Flatten/flatten/strided_slice/stack_1+model/Flatten/flatten/strided_slice/stack_2*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
Index0*
T0
p
%model/Flatten/flatten/Reshape/shape/1Const*
valueB :
���������*
dtype0*
_output_shapes
: 
�
#model/Flatten/flatten/Reshape/shapePack#model/Flatten/flatten/strided_slice%model/Flatten/flatten/Reshape/shape/1*
T0*

axis *
N*
_output_shapes
:
�
model/Flatten/flatten/ReshapeReshapemodel/conv5/MaxPool2D/MaxPool#model/Flatten/flatten/Reshape/shape*'
_output_shapes
:���������*
T0*
Tshape0
l
model_1/conv1/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
�
model_1/conv1/Conv2DConv2Dpositive_inputconv1/weights/read*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*/
_output_shapes
:���������
 *
	dilations

�
model_1/conv1/BiasAddBiasAddmodel_1/conv1/Conv2Dconv1/biases/read*
T0*
data_formatNHWC*/
_output_shapes
:���������
 
q
model_1/conv1/conv1/ReluRelumodel_1/conv1/BiasAdd*
T0*/
_output_shapes
:���������
 
�
model_1/conv1/MaxPool2D/MaxPoolMaxPoolmodel_1/conv1/conv1/Relu*
T0*
data_formatNHWC*
strides
*
ksize
*
paddingSAME*/
_output_shapes
:���������
 
l
model_1/conv2/dilation_rateConst*
dtype0*
_output_shapes
:*
valueB"      
�
model_1/conv2/Conv2DConv2Dmodel_1/conv1/MaxPool2D/MaxPoolconv2/weights/read*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME*/
_output_shapes
:���������
@*
	dilations

�
model_1/conv2/BiasAddBiasAddmodel_1/conv2/Conv2Dconv2/biases/read*
T0*
data_formatNHWC*/
_output_shapes
:���������
@
q
model_1/conv2/conv2/ReluRelumodel_1/conv2/BiasAdd*
T0*/
_output_shapes
:���������
@
�
model_1/conv2/MaxPool2D/MaxPoolMaxPoolmodel_1/conv2/conv2/Relu*
T0*
data_formatNHWC*
strides
*
ksize
*
paddingSAME*/
_output_shapes
:���������@
l
model_1/conv3/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
�
model_1/conv3/Conv2DConv2Dmodel_1/conv2/MaxPool2D/MaxPoolconv3/weights/read*
paddingSAME*0
_output_shapes
:����������*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
�
model_1/conv3/BiasAddBiasAddmodel_1/conv3/Conv2Dconv3/biases/read*
T0*
data_formatNHWC*0
_output_shapes
:����������
r
model_1/conv3/conv3/ReluRelumodel_1/conv3/BiasAdd*
T0*0
_output_shapes
:����������
�
model_1/conv3/MaxPool2D/MaxPoolMaxPoolmodel_1/conv3/conv3/Relu*
T0*
data_formatNHWC*
strides
*
ksize
*
paddingSAME*0
_output_shapes
:����������
l
model_1/conv4/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
�
model_1/conv4/Conv2DConv2Dmodel_1/conv3/MaxPool2D/MaxPoolconv4/weights/read*
paddingSAME*0
_output_shapes
:����������*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
�
model_1/conv4/BiasAddBiasAddmodel_1/conv4/Conv2Dconv4/biases/read*
T0*
data_formatNHWC*0
_output_shapes
:����������
r
model_1/conv4/conv4/ReluRelumodel_1/conv4/BiasAdd*
T0*0
_output_shapes
:����������
�
model_1/conv4/MaxPool2D/MaxPoolMaxPoolmodel_1/conv4/conv4/Relu*0
_output_shapes
:����������*
T0*
data_formatNHWC*
strides
*
ksize
*
paddingSAME
l
model_1/conv5/dilation_rateConst*
dtype0*
_output_shapes
:*
valueB"      
�
model_1/conv5/Conv2DConv2Dmodel_1/conv4/MaxPool2D/MaxPoolconv5/weights/read*/
_output_shapes
:���������*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME
�
model_1/conv5/BiasAddBiasAddmodel_1/conv5/Conv2Dconv5/biases/read*
T0*
data_formatNHWC*/
_output_shapes
:���������
�
model_1/conv5/MaxPool2D/MaxPoolMaxPoolmodel_1/conv5/BiasAdd*
ksize
*
paddingSAME*/
_output_shapes
:���������*
T0*
data_formatNHWC*
strides

|
model_1/Flatten/flatten/ShapeShapemodel_1/conv5/MaxPool2D/MaxPool*
T0*
out_type0*
_output_shapes
:
u
+model_1/Flatten/flatten/strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB: 
w
-model_1/Flatten/flatten/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
w
-model_1/Flatten/flatten/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
�
%model_1/Flatten/flatten/strided_sliceStridedSlicemodel_1/Flatten/flatten/Shape+model_1/Flatten/flatten/strided_slice/stack-model_1/Flatten/flatten/strided_slice/stack_1-model_1/Flatten/flatten/strided_slice/stack_2*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
T0*
Index0
r
'model_1/Flatten/flatten/Reshape/shape/1Const*
dtype0*
_output_shapes
: *
valueB :
���������
�
%model_1/Flatten/flatten/Reshape/shapePack%model_1/Flatten/flatten/strided_slice'model_1/Flatten/flatten/Reshape/shape/1*
T0*

axis *
N*
_output_shapes
:
�
model_1/Flatten/flatten/ReshapeReshapemodel_1/conv5/MaxPool2D/MaxPool%model_1/Flatten/flatten/Reshape/shape*'
_output_shapes
:���������*
T0*
Tshape0
l
model_2/conv1/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
�
model_2/conv1/Conv2DConv2Dnegative_inputconv1/weights/read*/
_output_shapes
:���������
 *
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME
�
model_2/conv1/BiasAddBiasAddmodel_2/conv1/Conv2Dconv1/biases/read*
T0*
data_formatNHWC*/
_output_shapes
:���������
 
q
model_2/conv1/conv1/ReluRelumodel_2/conv1/BiasAdd*
T0*/
_output_shapes
:���������
 
�
model_2/conv1/MaxPool2D/MaxPoolMaxPoolmodel_2/conv1/conv1/Relu*/
_output_shapes
:���������
 *
T0*
data_formatNHWC*
strides
*
ksize
*
paddingSAME
l
model_2/conv2/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
�
model_2/conv2/Conv2DConv2Dmodel_2/conv1/MaxPool2D/MaxPoolconv2/weights/read*
paddingSAME*/
_output_shapes
:���������
@*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(
�
model_2/conv2/BiasAddBiasAddmodel_2/conv2/Conv2Dconv2/biases/read*
T0*
data_formatNHWC*/
_output_shapes
:���������
@
q
model_2/conv2/conv2/ReluRelumodel_2/conv2/BiasAdd*
T0*/
_output_shapes
:���������
@
�
model_2/conv2/MaxPool2D/MaxPoolMaxPoolmodel_2/conv2/conv2/Relu*
ksize
*
paddingSAME*/
_output_shapes
:���������@*
T0*
data_formatNHWC*
strides

l
model_2/conv3/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
�
model_2/conv3/Conv2DConv2Dmodel_2/conv2/MaxPool2D/MaxPoolconv3/weights/read*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*0
_output_shapes
:����������
�
model_2/conv3/BiasAddBiasAddmodel_2/conv3/Conv2Dconv3/biases/read*
T0*
data_formatNHWC*0
_output_shapes
:����������
r
model_2/conv3/conv3/ReluRelumodel_2/conv3/BiasAdd*0
_output_shapes
:����������*
T0
�
model_2/conv3/MaxPool2D/MaxPoolMaxPoolmodel_2/conv3/conv3/Relu*0
_output_shapes
:����������*
T0*
data_formatNHWC*
strides
*
ksize
*
paddingSAME
l
model_2/conv4/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
�
model_2/conv4/Conv2DConv2Dmodel_2/conv3/MaxPool2D/MaxPoolconv4/weights/read*
paddingSAME*0
_output_shapes
:����������*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
�
model_2/conv4/BiasAddBiasAddmodel_2/conv4/Conv2Dconv4/biases/read*
data_formatNHWC*0
_output_shapes
:����������*
T0
r
model_2/conv4/conv4/ReluRelumodel_2/conv4/BiasAdd*
T0*0
_output_shapes
:����������
�
model_2/conv4/MaxPool2D/MaxPoolMaxPoolmodel_2/conv4/conv4/Relu*0
_output_shapes
:����������*
T0*
data_formatNHWC*
strides
*
ksize
*
paddingSAME
l
model_2/conv5/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
�
model_2/conv5/Conv2DConv2Dmodel_2/conv4/MaxPool2D/MaxPoolconv5/weights/read*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*/
_output_shapes
:���������
�
model_2/conv5/BiasAddBiasAddmodel_2/conv5/Conv2Dconv5/biases/read*
T0*
data_formatNHWC*/
_output_shapes
:���������
�
model_2/conv5/MaxPool2D/MaxPoolMaxPoolmodel_2/conv5/BiasAdd*/
_output_shapes
:���������*
T0*
data_formatNHWC*
strides
*
ksize
*
paddingSAME
|
model_2/Flatten/flatten/ShapeShapemodel_2/conv5/MaxPool2D/MaxPool*
_output_shapes
:*
T0*
out_type0
u
+model_2/Flatten/flatten/strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB: 
w
-model_2/Flatten/flatten/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
w
-model_2/Flatten/flatten/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
�
%model_2/Flatten/flatten/strided_sliceStridedSlicemodel_2/Flatten/flatten/Shape+model_2/Flatten/flatten/strided_slice/stack-model_2/Flatten/flatten/strided_slice/stack_1-model_2/Flatten/flatten/strided_slice/stack_2*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
Index0*
T0
r
'model_2/Flatten/flatten/Reshape/shape/1Const*
valueB :
���������*
dtype0*
_output_shapes
: 
�
%model_2/Flatten/flatten/Reshape/shapePack%model_2/Flatten/flatten/strided_slice'model_2/Flatten/flatten/Reshape/shape/1*
T0*

axis *
N*
_output_shapes
:
�
model_2/Flatten/flatten/ReshapeReshapemodel_2/conv5/MaxPool2D/MaxPool%model_2/Flatten/flatten/Reshape/shape*'
_output_shapes
:���������*
T0*
Tshape0
|
subSubmodel/Flatten/flatten/Reshapemodel_1/Flatten/flatten/Reshape*
T0*'
_output_shapes
:���������
J
Pow/yConst*
valueB
 *   @*
dtype0*
_output_shapes
: 
H
PowPowsubPow/y*
T0*'
_output_shapes
:���������
W
Sum/reduction_indicesConst*
value	B :*
dtype0*
_output_shapes
: 
u
SumSumPowSum/reduction_indices*
T0*'
_output_shapes
:���������*
	keep_dims(*

Tidx0
C
SqrtSqrtSum*
T0*'
_output_shapes
:���������
~
sub_1Submodel/Flatten/flatten/Reshapemodel_2/Flatten/flatten/Reshape*'
_output_shapes
:���������*
T0
L
Pow_1/yConst*
valueB
 *   @*
dtype0*
_output_shapes
: 
N
Pow_1Powsub_1Pow_1/y*
T0*'
_output_shapes
:���������
Y
Sum_1/reduction_indicesConst*
value	B :*
dtype0*
_output_shapes
: 
{
Sum_1SumPow_1Sum_1/reduction_indices*
T0*'
_output_shapes
:���������*
	keep_dims(*

Tidx0
G
Sqrt_1SqrtSum_1*'
_output_shapes
:���������*
T0
L
sub_2SubSqrtSqrt_1*
T0*'
_output_shapes
:���������
J
add/yConst*
valueB
 *   ?*
dtype0*
_output_shapes
: 
J
addAddsub_2add/y*
T0*'
_output_shapes
:���������
N
	Maximum/yConst*
valueB
 *    *
dtype0*
_output_shapes
: 
T
MaximumMaximumadd	Maximum/y*
T0*'
_output_shapes
:���������
V
ConstConst*
dtype0*
_output_shapes
:*
valueB"       
Z
MeanMeanMaximumConst*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
X
Variable/initial_valueConst*
value	B : *
dtype0*
_output_shapes
: 
l
Variable
VariableV2*
dtype0*
_output_shapes
: *
	container *
shape: *
shared_name 
�
Variable/AssignAssignVariableVariable/initial_value*
use_locking(*
T0*
_class
loc:@Variable*
validate_shape(*
_output_shapes
: 
a
Variable/readIdentityVariable*
T0*
_class
loc:@Variable*
_output_shapes
: 
R
gradients/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
X
gradients/grad_ys_0Const*
dtype0*
_output_shapes
: *
valueB
 *  �?
o
gradients/FillFillgradients/Shapegradients/grad_ys_0*
T0*

index_type0*
_output_shapes
: 
r
!gradients/Mean_grad/Reshape/shapeConst*
dtype0*
_output_shapes
:*
valueB"      
�
gradients/Mean_grad/ReshapeReshapegradients/Fill!gradients/Mean_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes

:
`
gradients/Mean_grad/ShapeShapeMaximum*
T0*
out_type0*
_output_shapes
:
�
gradients/Mean_grad/TileTilegradients/Mean_grad/Reshapegradients/Mean_grad/Shape*
T0*'
_output_shapes
:���������*

Tmultiples0
b
gradients/Mean_grad/Shape_1ShapeMaximum*
T0*
out_type0*
_output_shapes
:
^
gradients/Mean_grad/Shape_2Const*
valueB *
dtype0*
_output_shapes
: 
c
gradients/Mean_grad/ConstConst*
dtype0*
_output_shapes
:*
valueB: 
�
gradients/Mean_grad/ProdProdgradients/Mean_grad/Shape_1gradients/Mean_grad/Const*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
e
gradients/Mean_grad/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
�
gradients/Mean_grad/Prod_1Prodgradients/Mean_grad/Shape_2gradients/Mean_grad/Const_1*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
_
gradients/Mean_grad/Maximum/yConst*
dtype0*
_output_shapes
: *
value	B :
�
gradients/Mean_grad/MaximumMaximumgradients/Mean_grad/Prod_1gradients/Mean_grad/Maximum/y*
T0*
_output_shapes
: 
�
gradients/Mean_grad/floordivFloorDivgradients/Mean_grad/Prodgradients/Mean_grad/Maximum*
T0*
_output_shapes
: 
~
gradients/Mean_grad/CastCastgradients/Mean_grad/floordiv*

SrcT0*
Truncate( *
_output_shapes
: *

DstT0
�
gradients/Mean_grad/truedivRealDivgradients/Mean_grad/Tilegradients/Mean_grad/Cast*
T0*'
_output_shapes
:���������
_
gradients/Maximum_grad/ShapeShapeadd*
T0*
out_type0*
_output_shapes
:
a
gradients/Maximum_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
y
gradients/Maximum_grad/Shape_2Shapegradients/Mean_grad/truediv*
T0*
out_type0*
_output_shapes
:
g
"gradients/Maximum_grad/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
gradients/Maximum_grad/zerosFillgradients/Maximum_grad/Shape_2"gradients/Maximum_grad/zeros/Const*
T0*

index_type0*'
_output_shapes
:���������
u
#gradients/Maximum_grad/GreaterEqualGreaterEqualadd	Maximum/y*
T0*'
_output_shapes
:���������
�
,gradients/Maximum_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/Maximum_grad/Shapegradients/Maximum_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
gradients/Maximum_grad/SelectSelect#gradients/Maximum_grad/GreaterEqualgradients/Mean_grad/truedivgradients/Maximum_grad/zeros*'
_output_shapes
:���������*
T0
�
gradients/Maximum_grad/Select_1Select#gradients/Maximum_grad/GreaterEqualgradients/Maximum_grad/zerosgradients/Mean_grad/truediv*
T0*'
_output_shapes
:���������
�
gradients/Maximum_grad/SumSumgradients/Maximum_grad/Select,gradients/Maximum_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
gradients/Maximum_grad/ReshapeReshapegradients/Maximum_grad/Sumgradients/Maximum_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
gradients/Maximum_grad/Sum_1Sumgradients/Maximum_grad/Select_1.gradients/Maximum_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
 gradients/Maximum_grad/Reshape_1Reshapegradients/Maximum_grad/Sum_1gradients/Maximum_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
s
'gradients/Maximum_grad/tuple/group_depsNoOp^gradients/Maximum_grad/Reshape!^gradients/Maximum_grad/Reshape_1
�
/gradients/Maximum_grad/tuple/control_dependencyIdentitygradients/Maximum_grad/Reshape(^gradients/Maximum_grad/tuple/group_deps*'
_output_shapes
:���������*
T0*1
_class'
%#loc:@gradients/Maximum_grad/Reshape
�
1gradients/Maximum_grad/tuple/control_dependency_1Identity gradients/Maximum_grad/Reshape_1(^gradients/Maximum_grad/tuple/group_deps*
T0*3
_class)
'%loc:@gradients/Maximum_grad/Reshape_1*
_output_shapes
: 
]
gradients/add_grad/ShapeShapesub_2*
T0*
out_type0*
_output_shapes
:
]
gradients/add_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
�
(gradients/add_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_grad/Shapegradients/add_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
gradients/add_grad/SumSum/gradients/Maximum_grad/tuple/control_dependency(gradients/add_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
gradients/add_grad/ReshapeReshapegradients/add_grad/Sumgradients/add_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
gradients/add_grad/Sum_1Sum/gradients/Maximum_grad/tuple/control_dependency*gradients/add_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
gradients/add_grad/Reshape_1Reshapegradients/add_grad/Sum_1gradients/add_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
g
#gradients/add_grad/tuple/group_depsNoOp^gradients/add_grad/Reshape^gradients/add_grad/Reshape_1
�
+gradients/add_grad/tuple/control_dependencyIdentitygradients/add_grad/Reshape$^gradients/add_grad/tuple/group_deps*
T0*-
_class#
!loc:@gradients/add_grad/Reshape*'
_output_shapes
:���������
�
-gradients/add_grad/tuple/control_dependency_1Identitygradients/add_grad/Reshape_1$^gradients/add_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/add_grad/Reshape_1*
_output_shapes
: 
^
gradients/sub_2_grad/ShapeShapeSqrt*
T0*
out_type0*
_output_shapes
:
b
gradients/sub_2_grad/Shape_1ShapeSqrt_1*
_output_shapes
:*
T0*
out_type0
�
*gradients/sub_2_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/sub_2_grad/Shapegradients/sub_2_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
gradients/sub_2_grad/SumSum+gradients/add_grad/tuple/control_dependency*gradients/sub_2_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
gradients/sub_2_grad/ReshapeReshapegradients/sub_2_grad/Sumgradients/sub_2_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
gradients/sub_2_grad/Sum_1Sum+gradients/add_grad/tuple/control_dependency,gradients/sub_2_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
^
gradients/sub_2_grad/NegNeggradients/sub_2_grad/Sum_1*
T0*
_output_shapes
:
�
gradients/sub_2_grad/Reshape_1Reshapegradients/sub_2_grad/Neggradients/sub_2_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:���������
m
%gradients/sub_2_grad/tuple/group_depsNoOp^gradients/sub_2_grad/Reshape^gradients/sub_2_grad/Reshape_1
�
-gradients/sub_2_grad/tuple/control_dependencyIdentitygradients/sub_2_grad/Reshape&^gradients/sub_2_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/sub_2_grad/Reshape*'
_output_shapes
:���������
�
/gradients/sub_2_grad/tuple/control_dependency_1Identitygradients/sub_2_grad/Reshape_1&^gradients/sub_2_grad/tuple/group_deps*'
_output_shapes
:���������*
T0*1
_class'
%#loc:@gradients/sub_2_grad/Reshape_1
�
gradients/Sqrt_grad/SqrtGradSqrtGradSqrt-gradients/sub_2_grad/tuple/control_dependency*
T0*'
_output_shapes
:���������
�
gradients/Sqrt_1_grad/SqrtGradSqrtGradSqrt_1/gradients/sub_2_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:���������
[
gradients/Sum_grad/ShapeShapePow*
T0*
out_type0*
_output_shapes
:
�
gradients/Sum_grad/SizeConst*
value	B :*+
_class!
loc:@gradients/Sum_grad/Shape*
dtype0*
_output_shapes
: 
�
gradients/Sum_grad/addAddSum/reduction_indicesgradients/Sum_grad/Size*
_output_shapes
: *
T0*+
_class!
loc:@gradients/Sum_grad/Shape
�
gradients/Sum_grad/modFloorModgradients/Sum_grad/addgradients/Sum_grad/Size*
_output_shapes
: *
T0*+
_class!
loc:@gradients/Sum_grad/Shape
�
gradients/Sum_grad/Shape_1Const*
dtype0*
_output_shapes
: *
valueB *+
_class!
loc:@gradients/Sum_grad/Shape
�
gradients/Sum_grad/range/startConst*
value	B : *+
_class!
loc:@gradients/Sum_grad/Shape*
dtype0*
_output_shapes
: 
�
gradients/Sum_grad/range/deltaConst*
dtype0*
_output_shapes
: *
value	B :*+
_class!
loc:@gradients/Sum_grad/Shape
�
gradients/Sum_grad/rangeRangegradients/Sum_grad/range/startgradients/Sum_grad/Sizegradients/Sum_grad/range/delta*+
_class!
loc:@gradients/Sum_grad/Shape*
_output_shapes
:*

Tidx0
�
gradients/Sum_grad/Fill/valueConst*
dtype0*
_output_shapes
: *
value	B :*+
_class!
loc:@gradients/Sum_grad/Shape
�
gradients/Sum_grad/FillFillgradients/Sum_grad/Shape_1gradients/Sum_grad/Fill/value*
_output_shapes
: *
T0*

index_type0*+
_class!
loc:@gradients/Sum_grad/Shape
�
 gradients/Sum_grad/DynamicStitchDynamicStitchgradients/Sum_grad/rangegradients/Sum_grad/modgradients/Sum_grad/Shapegradients/Sum_grad/Fill*
T0*+
_class!
loc:@gradients/Sum_grad/Shape*
N*
_output_shapes
:
�
gradients/Sum_grad/Maximum/yConst*
value	B :*+
_class!
loc:@gradients/Sum_grad/Shape*
dtype0*
_output_shapes
: 
�
gradients/Sum_grad/MaximumMaximum gradients/Sum_grad/DynamicStitchgradients/Sum_grad/Maximum/y*
T0*+
_class!
loc:@gradients/Sum_grad/Shape*
_output_shapes
:
�
gradients/Sum_grad/floordivFloorDivgradients/Sum_grad/Shapegradients/Sum_grad/Maximum*
T0*+
_class!
loc:@gradients/Sum_grad/Shape*
_output_shapes
:
�
gradients/Sum_grad/ReshapeReshapegradients/Sqrt_grad/SqrtGrad gradients/Sum_grad/DynamicStitch*
T0*
Tshape0*0
_output_shapes
:������������������
�
gradients/Sum_grad/TileTilegradients/Sum_grad/Reshapegradients/Sum_grad/floordiv*

Tmultiples0*
T0*'
_output_shapes
:���������
_
gradients/Sum_1_grad/ShapeShapePow_1*
T0*
out_type0*
_output_shapes
:
�
gradients/Sum_1_grad/SizeConst*
value	B :*-
_class#
!loc:@gradients/Sum_1_grad/Shape*
dtype0*
_output_shapes
: 
�
gradients/Sum_1_grad/addAddSum_1/reduction_indicesgradients/Sum_1_grad/Size*
T0*-
_class#
!loc:@gradients/Sum_1_grad/Shape*
_output_shapes
: 
�
gradients/Sum_1_grad/modFloorModgradients/Sum_1_grad/addgradients/Sum_1_grad/Size*
T0*-
_class#
!loc:@gradients/Sum_1_grad/Shape*
_output_shapes
: 
�
gradients/Sum_1_grad/Shape_1Const*
valueB *-
_class#
!loc:@gradients/Sum_1_grad/Shape*
dtype0*
_output_shapes
: 
�
 gradients/Sum_1_grad/range/startConst*
value	B : *-
_class#
!loc:@gradients/Sum_1_grad/Shape*
dtype0*
_output_shapes
: 
�
 gradients/Sum_1_grad/range/deltaConst*
value	B :*-
_class#
!loc:@gradients/Sum_1_grad/Shape*
dtype0*
_output_shapes
: 
�
gradients/Sum_1_grad/rangeRange gradients/Sum_1_grad/range/startgradients/Sum_1_grad/Size gradients/Sum_1_grad/range/delta*

Tidx0*-
_class#
!loc:@gradients/Sum_1_grad/Shape*
_output_shapes
:
�
gradients/Sum_1_grad/Fill/valueConst*
value	B :*-
_class#
!loc:@gradients/Sum_1_grad/Shape*
dtype0*
_output_shapes
: 
�
gradients/Sum_1_grad/FillFillgradients/Sum_1_grad/Shape_1gradients/Sum_1_grad/Fill/value*
T0*

index_type0*-
_class#
!loc:@gradients/Sum_1_grad/Shape*
_output_shapes
: 
�
"gradients/Sum_1_grad/DynamicStitchDynamicStitchgradients/Sum_1_grad/rangegradients/Sum_1_grad/modgradients/Sum_1_grad/Shapegradients/Sum_1_grad/Fill*
T0*-
_class#
!loc:@gradients/Sum_1_grad/Shape*
N*
_output_shapes
:
�
gradients/Sum_1_grad/Maximum/yConst*
value	B :*-
_class#
!loc:@gradients/Sum_1_grad/Shape*
dtype0*
_output_shapes
: 
�
gradients/Sum_1_grad/MaximumMaximum"gradients/Sum_1_grad/DynamicStitchgradients/Sum_1_grad/Maximum/y*
T0*-
_class#
!loc:@gradients/Sum_1_grad/Shape*
_output_shapes
:
�
gradients/Sum_1_grad/floordivFloorDivgradients/Sum_1_grad/Shapegradients/Sum_1_grad/Maximum*
T0*-
_class#
!loc:@gradients/Sum_1_grad/Shape*
_output_shapes
:
�
gradients/Sum_1_grad/ReshapeReshapegradients/Sqrt_1_grad/SqrtGrad"gradients/Sum_1_grad/DynamicStitch*
T0*
Tshape0*0
_output_shapes
:������������������
�
gradients/Sum_1_grad/TileTilegradients/Sum_1_grad/Reshapegradients/Sum_1_grad/floordiv*

Tmultiples0*
T0*'
_output_shapes
:���������
[
gradients/Pow_grad/ShapeShapesub*
_output_shapes
:*
T0*
out_type0
]
gradients/Pow_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
�
(gradients/Pow_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/Pow_grad/Shapegradients/Pow_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
o
gradients/Pow_grad/mulMulgradients/Sum_grad/TilePow/y*
T0*'
_output_shapes
:���������
]
gradients/Pow_grad/sub/yConst*
dtype0*
_output_shapes
: *
valueB
 *  �?
_
gradients/Pow_grad/subSubPow/ygradients/Pow_grad/sub/y*
T0*
_output_shapes
: 
l
gradients/Pow_grad/PowPowsubgradients/Pow_grad/sub*'
_output_shapes
:���������*
T0
�
gradients/Pow_grad/mul_1Mulgradients/Pow_grad/mulgradients/Pow_grad/Pow*
T0*'
_output_shapes
:���������
�
gradients/Pow_grad/SumSumgradients/Pow_grad/mul_1(gradients/Pow_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
gradients/Pow_grad/ReshapeReshapegradients/Pow_grad/Sumgradients/Pow_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
a
gradients/Pow_grad/Greater/yConst*
dtype0*
_output_shapes
: *
valueB
 *    
z
gradients/Pow_grad/GreaterGreatersubgradients/Pow_grad/Greater/y*
T0*'
_output_shapes
:���������
e
"gradients/Pow_grad/ones_like/ShapeShapesub*
_output_shapes
:*
T0*
out_type0
g
"gradients/Pow_grad/ones_like/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *  �?
�
gradients/Pow_grad/ones_likeFill"gradients/Pow_grad/ones_like/Shape"gradients/Pow_grad/ones_like/Const*
T0*

index_type0*'
_output_shapes
:���������
�
gradients/Pow_grad/SelectSelectgradients/Pow_grad/Greatersubgradients/Pow_grad/ones_like*'
_output_shapes
:���������*
T0
j
gradients/Pow_grad/LogLoggradients/Pow_grad/Select*
T0*'
_output_shapes
:���������
a
gradients/Pow_grad/zeros_like	ZerosLikesub*
T0*'
_output_shapes
:���������
�
gradients/Pow_grad/Select_1Selectgradients/Pow_grad/Greatergradients/Pow_grad/Loggradients/Pow_grad/zeros_like*'
_output_shapes
:���������*
T0
o
gradients/Pow_grad/mul_2Mulgradients/Sum_grad/TilePow*
T0*'
_output_shapes
:���������
�
gradients/Pow_grad/mul_3Mulgradients/Pow_grad/mul_2gradients/Pow_grad/Select_1*
T0*'
_output_shapes
:���������
�
gradients/Pow_grad/Sum_1Sumgradients/Pow_grad/mul_3*gradients/Pow_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
gradients/Pow_grad/Reshape_1Reshapegradients/Pow_grad/Sum_1gradients/Pow_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
g
#gradients/Pow_grad/tuple/group_depsNoOp^gradients/Pow_grad/Reshape^gradients/Pow_grad/Reshape_1
�
+gradients/Pow_grad/tuple/control_dependencyIdentitygradients/Pow_grad/Reshape$^gradients/Pow_grad/tuple/group_deps*
T0*-
_class#
!loc:@gradients/Pow_grad/Reshape*'
_output_shapes
:���������
�
-gradients/Pow_grad/tuple/control_dependency_1Identitygradients/Pow_grad/Reshape_1$^gradients/Pow_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/Pow_grad/Reshape_1*
_output_shapes
: 
_
gradients/Pow_1_grad/ShapeShapesub_1*
T0*
out_type0*
_output_shapes
:
_
gradients/Pow_1_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
�
*gradients/Pow_1_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/Pow_1_grad/Shapegradients/Pow_1_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
u
gradients/Pow_1_grad/mulMulgradients/Sum_1_grad/TilePow_1/y*
T0*'
_output_shapes
:���������
_
gradients/Pow_1_grad/sub/yConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
e
gradients/Pow_1_grad/subSubPow_1/ygradients/Pow_1_grad/sub/y*
_output_shapes
: *
T0
r
gradients/Pow_1_grad/PowPowsub_1gradients/Pow_1_grad/sub*
T0*'
_output_shapes
:���������
�
gradients/Pow_1_grad/mul_1Mulgradients/Pow_1_grad/mulgradients/Pow_1_grad/Pow*'
_output_shapes
:���������*
T0
�
gradients/Pow_1_grad/SumSumgradients/Pow_1_grad/mul_1*gradients/Pow_1_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
gradients/Pow_1_grad/ReshapeReshapegradients/Pow_1_grad/Sumgradients/Pow_1_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
c
gradients/Pow_1_grad/Greater/yConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
gradients/Pow_1_grad/GreaterGreatersub_1gradients/Pow_1_grad/Greater/y*'
_output_shapes
:���������*
T0
i
$gradients/Pow_1_grad/ones_like/ShapeShapesub_1*
T0*
out_type0*
_output_shapes
:
i
$gradients/Pow_1_grad/ones_like/ConstConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
gradients/Pow_1_grad/ones_likeFill$gradients/Pow_1_grad/ones_like/Shape$gradients/Pow_1_grad/ones_like/Const*
T0*

index_type0*'
_output_shapes
:���������
�
gradients/Pow_1_grad/SelectSelectgradients/Pow_1_grad/Greatersub_1gradients/Pow_1_grad/ones_like*
T0*'
_output_shapes
:���������
n
gradients/Pow_1_grad/LogLoggradients/Pow_1_grad/Select*'
_output_shapes
:���������*
T0
e
gradients/Pow_1_grad/zeros_like	ZerosLikesub_1*
T0*'
_output_shapes
:���������
�
gradients/Pow_1_grad/Select_1Selectgradients/Pow_1_grad/Greatergradients/Pow_1_grad/Loggradients/Pow_1_grad/zeros_like*
T0*'
_output_shapes
:���������
u
gradients/Pow_1_grad/mul_2Mulgradients/Sum_1_grad/TilePow_1*'
_output_shapes
:���������*
T0
�
gradients/Pow_1_grad/mul_3Mulgradients/Pow_1_grad/mul_2gradients/Pow_1_grad/Select_1*'
_output_shapes
:���������*
T0
�
gradients/Pow_1_grad/Sum_1Sumgradients/Pow_1_grad/mul_3,gradients/Pow_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
gradients/Pow_1_grad/Reshape_1Reshapegradients/Pow_1_grad/Sum_1gradients/Pow_1_grad/Shape_1*
_output_shapes
: *
T0*
Tshape0
m
%gradients/Pow_1_grad/tuple/group_depsNoOp^gradients/Pow_1_grad/Reshape^gradients/Pow_1_grad/Reshape_1
�
-gradients/Pow_1_grad/tuple/control_dependencyIdentitygradients/Pow_1_grad/Reshape&^gradients/Pow_1_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/Pow_1_grad/Reshape*'
_output_shapes
:���������
�
/gradients/Pow_1_grad/tuple/control_dependency_1Identitygradients/Pow_1_grad/Reshape_1&^gradients/Pow_1_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/Pow_1_grad/Reshape_1*
_output_shapes
: 
u
gradients/sub_grad/ShapeShapemodel/Flatten/flatten/Reshape*
T0*
out_type0*
_output_shapes
:
y
gradients/sub_grad/Shape_1Shapemodel_1/Flatten/flatten/Reshape*
T0*
out_type0*
_output_shapes
:
�
(gradients/sub_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/sub_grad/Shapegradients/sub_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
gradients/sub_grad/SumSum+gradients/Pow_grad/tuple/control_dependency(gradients/sub_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
gradients/sub_grad/ReshapeReshapegradients/sub_grad/Sumgradients/sub_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
gradients/sub_grad/Sum_1Sum+gradients/Pow_grad/tuple/control_dependency*gradients/sub_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
Z
gradients/sub_grad/NegNeggradients/sub_grad/Sum_1*
_output_shapes
:*
T0
�
gradients/sub_grad/Reshape_1Reshapegradients/sub_grad/Neggradients/sub_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:���������
g
#gradients/sub_grad/tuple/group_depsNoOp^gradients/sub_grad/Reshape^gradients/sub_grad/Reshape_1
�
+gradients/sub_grad/tuple/control_dependencyIdentitygradients/sub_grad/Reshape$^gradients/sub_grad/tuple/group_deps*
T0*-
_class#
!loc:@gradients/sub_grad/Reshape*'
_output_shapes
:���������
�
-gradients/sub_grad/tuple/control_dependency_1Identitygradients/sub_grad/Reshape_1$^gradients/sub_grad/tuple/group_deps*'
_output_shapes
:���������*
T0*/
_class%
#!loc:@gradients/sub_grad/Reshape_1
w
gradients/sub_1_grad/ShapeShapemodel/Flatten/flatten/Reshape*
T0*
out_type0*
_output_shapes
:
{
gradients/sub_1_grad/Shape_1Shapemodel_2/Flatten/flatten/Reshape*
T0*
out_type0*
_output_shapes
:
�
*gradients/sub_1_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/sub_1_grad/Shapegradients/sub_1_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
gradients/sub_1_grad/SumSum-gradients/Pow_1_grad/tuple/control_dependency*gradients/sub_1_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
gradients/sub_1_grad/ReshapeReshapegradients/sub_1_grad/Sumgradients/sub_1_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
gradients/sub_1_grad/Sum_1Sum-gradients/Pow_1_grad/tuple/control_dependency,gradients/sub_1_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
^
gradients/sub_1_grad/NegNeggradients/sub_1_grad/Sum_1*
_output_shapes
:*
T0
�
gradients/sub_1_grad/Reshape_1Reshapegradients/sub_1_grad/Neggradients/sub_1_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:���������
m
%gradients/sub_1_grad/tuple/group_depsNoOp^gradients/sub_1_grad/Reshape^gradients/sub_1_grad/Reshape_1
�
-gradients/sub_1_grad/tuple/control_dependencyIdentitygradients/sub_1_grad/Reshape&^gradients/sub_1_grad/tuple/group_deps*'
_output_shapes
:���������*
T0*/
_class%
#!loc:@gradients/sub_1_grad/Reshape
�
/gradients/sub_1_grad/tuple/control_dependency_1Identitygradients/sub_1_grad/Reshape_1&^gradients/sub_1_grad/tuple/group_deps*'
_output_shapes
:���������*
T0*1
_class'
%#loc:@gradients/sub_1_grad/Reshape_1
�
4gradients/model_1/Flatten/flatten/Reshape_grad/ShapeShapemodel_1/conv5/MaxPool2D/MaxPool*
T0*
out_type0*
_output_shapes
:
�
6gradients/model_1/Flatten/flatten/Reshape_grad/ReshapeReshape-gradients/sub_grad/tuple/control_dependency_14gradients/model_1/Flatten/flatten/Reshape_grad/Shape*
T0*
Tshape0*/
_output_shapes
:���������
�
gradients/AddNAddN+gradients/sub_grad/tuple/control_dependency-gradients/sub_1_grad/tuple/control_dependency*
T0*-
_class#
!loc:@gradients/sub_grad/Reshape*
N*'
_output_shapes
:���������
�
2gradients/model/Flatten/flatten/Reshape_grad/ShapeShapemodel/conv5/MaxPool2D/MaxPool*
T0*
out_type0*
_output_shapes
:
�
4gradients/model/Flatten/flatten/Reshape_grad/ReshapeReshapegradients/AddN2gradients/model/Flatten/flatten/Reshape_grad/Shape*/
_output_shapes
:���������*
T0*
Tshape0
�
4gradients/model_2/Flatten/flatten/Reshape_grad/ShapeShapemodel_2/conv5/MaxPool2D/MaxPool*
_output_shapes
:*
T0*
out_type0
�
6gradients/model_2/Flatten/flatten/Reshape_grad/ReshapeReshape/gradients/sub_1_grad/tuple/control_dependency_14gradients/model_2/Flatten/flatten/Reshape_grad/Shape*
T0*
Tshape0*/
_output_shapes
:���������
�
:gradients/model_1/conv5/MaxPool2D/MaxPool_grad/MaxPoolGradMaxPoolGradmodel_1/conv5/BiasAddmodel_1/conv5/MaxPool2D/MaxPool6gradients/model_1/Flatten/flatten/Reshape_grad/Reshape*
ksize
*
paddingSAME*/
_output_shapes
:���������*
T0*
data_formatNHWC*
strides

�
8gradients/model/conv5/MaxPool2D/MaxPool_grad/MaxPoolGradMaxPoolGradmodel/conv5/BiasAddmodel/conv5/MaxPool2D/MaxPool4gradients/model/Flatten/flatten/Reshape_grad/Reshape*
T0*
data_formatNHWC*
strides
*
ksize
*
paddingSAME*/
_output_shapes
:���������
�
:gradients/model_2/conv5/MaxPool2D/MaxPool_grad/MaxPoolGradMaxPoolGradmodel_2/conv5/BiasAddmodel_2/conv5/MaxPool2D/MaxPool6gradients/model_2/Flatten/flatten/Reshape_grad/Reshape*
T0*
data_formatNHWC*
strides
*
ksize
*
paddingSAME*/
_output_shapes
:���������
�
0gradients/model_1/conv5/BiasAdd_grad/BiasAddGradBiasAddGrad:gradients/model_1/conv5/MaxPool2D/MaxPool_grad/MaxPoolGrad*
data_formatNHWC*
_output_shapes
:*
T0
�
5gradients/model_1/conv5/BiasAdd_grad/tuple/group_depsNoOp1^gradients/model_1/conv5/BiasAdd_grad/BiasAddGrad;^gradients/model_1/conv5/MaxPool2D/MaxPool_grad/MaxPoolGrad
�
=gradients/model_1/conv5/BiasAdd_grad/tuple/control_dependencyIdentity:gradients/model_1/conv5/MaxPool2D/MaxPool_grad/MaxPoolGrad6^gradients/model_1/conv5/BiasAdd_grad/tuple/group_deps*/
_output_shapes
:���������*
T0*M
_classC
A?loc:@gradients/model_1/conv5/MaxPool2D/MaxPool_grad/MaxPoolGrad
�
?gradients/model_1/conv5/BiasAdd_grad/tuple/control_dependency_1Identity0gradients/model_1/conv5/BiasAdd_grad/BiasAddGrad6^gradients/model_1/conv5/BiasAdd_grad/tuple/group_deps*
_output_shapes
:*
T0*C
_class9
75loc:@gradients/model_1/conv5/BiasAdd_grad/BiasAddGrad
�
.gradients/model/conv5/BiasAdd_grad/BiasAddGradBiasAddGrad8gradients/model/conv5/MaxPool2D/MaxPool_grad/MaxPoolGrad*
T0*
data_formatNHWC*
_output_shapes
:
�
3gradients/model/conv5/BiasAdd_grad/tuple/group_depsNoOp/^gradients/model/conv5/BiasAdd_grad/BiasAddGrad9^gradients/model/conv5/MaxPool2D/MaxPool_grad/MaxPoolGrad
�
;gradients/model/conv5/BiasAdd_grad/tuple/control_dependencyIdentity8gradients/model/conv5/MaxPool2D/MaxPool_grad/MaxPoolGrad4^gradients/model/conv5/BiasAdd_grad/tuple/group_deps*
T0*K
_classA
?=loc:@gradients/model/conv5/MaxPool2D/MaxPool_grad/MaxPoolGrad*/
_output_shapes
:���������
�
=gradients/model/conv5/BiasAdd_grad/tuple/control_dependency_1Identity.gradients/model/conv5/BiasAdd_grad/BiasAddGrad4^gradients/model/conv5/BiasAdd_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients/model/conv5/BiasAdd_grad/BiasAddGrad*
_output_shapes
:
�
0gradients/model_2/conv5/BiasAdd_grad/BiasAddGradBiasAddGrad:gradients/model_2/conv5/MaxPool2D/MaxPool_grad/MaxPoolGrad*
T0*
data_formatNHWC*
_output_shapes
:
�
5gradients/model_2/conv5/BiasAdd_grad/tuple/group_depsNoOp1^gradients/model_2/conv5/BiasAdd_grad/BiasAddGrad;^gradients/model_2/conv5/MaxPool2D/MaxPool_grad/MaxPoolGrad
�
=gradients/model_2/conv5/BiasAdd_grad/tuple/control_dependencyIdentity:gradients/model_2/conv5/MaxPool2D/MaxPool_grad/MaxPoolGrad6^gradients/model_2/conv5/BiasAdd_grad/tuple/group_deps*/
_output_shapes
:���������*
T0*M
_classC
A?loc:@gradients/model_2/conv5/MaxPool2D/MaxPool_grad/MaxPoolGrad
�
?gradients/model_2/conv5/BiasAdd_grad/tuple/control_dependency_1Identity0gradients/model_2/conv5/BiasAdd_grad/BiasAddGrad6^gradients/model_2/conv5/BiasAdd_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/model_2/conv5/BiasAdd_grad/BiasAddGrad*
_output_shapes
:
�
*gradients/model_1/conv5/Conv2D_grad/ShapeNShapeNmodel_1/conv4/MaxPool2D/MaxPoolconv5/weights/read*
N* 
_output_shapes
::*
T0*
out_type0
�
7gradients/model_1/conv5/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput*gradients/model_1/conv5/Conv2D_grad/ShapeNconv5/weights/read=gradients/model_1/conv5/BiasAdd_grad/tuple/control_dependency*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*0
_output_shapes
:����������*
	dilations

�
8gradients/model_1/conv5/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFiltermodel_1/conv4/MaxPool2D/MaxPool,gradients/model_1/conv5/Conv2D_grad/ShapeN:1=gradients/model_1/conv5/BiasAdd_grad/tuple/control_dependency*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*'
_output_shapes
:�
�
4gradients/model_1/conv5/Conv2D_grad/tuple/group_depsNoOp9^gradients/model_1/conv5/Conv2D_grad/Conv2DBackpropFilter8^gradients/model_1/conv5/Conv2D_grad/Conv2DBackpropInput
�
<gradients/model_1/conv5/Conv2D_grad/tuple/control_dependencyIdentity7gradients/model_1/conv5/Conv2D_grad/Conv2DBackpropInput5^gradients/model_1/conv5/Conv2D_grad/tuple/group_deps*
T0*J
_class@
><loc:@gradients/model_1/conv5/Conv2D_grad/Conv2DBackpropInput*0
_output_shapes
:����������
�
>gradients/model_1/conv5/Conv2D_grad/tuple/control_dependency_1Identity8gradients/model_1/conv5/Conv2D_grad/Conv2DBackpropFilter5^gradients/model_1/conv5/Conv2D_grad/tuple/group_deps*
T0*K
_classA
?=loc:@gradients/model_1/conv5/Conv2D_grad/Conv2DBackpropFilter*'
_output_shapes
:�
�
(gradients/model/conv5/Conv2D_grad/ShapeNShapeNmodel/conv4/MaxPool2D/MaxPoolconv5/weights/read*
T0*
out_type0*
N* 
_output_shapes
::
�
5gradients/model/conv5/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput(gradients/model/conv5/Conv2D_grad/ShapeNconv5/weights/read;gradients/model/conv5/BiasAdd_grad/tuple/control_dependency*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME*0
_output_shapes
:����������*
	dilations

�
6gradients/model/conv5/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFiltermodel/conv4/MaxPool2D/MaxPool*gradients/model/conv5/Conv2D_grad/ShapeN:1;gradients/model/conv5/BiasAdd_grad/tuple/control_dependency*'
_output_shapes
:�*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME
�
2gradients/model/conv5/Conv2D_grad/tuple/group_depsNoOp7^gradients/model/conv5/Conv2D_grad/Conv2DBackpropFilter6^gradients/model/conv5/Conv2D_grad/Conv2DBackpropInput
�
:gradients/model/conv5/Conv2D_grad/tuple/control_dependencyIdentity5gradients/model/conv5/Conv2D_grad/Conv2DBackpropInput3^gradients/model/conv5/Conv2D_grad/tuple/group_deps*
T0*H
_class>
<:loc:@gradients/model/conv5/Conv2D_grad/Conv2DBackpropInput*0
_output_shapes
:����������
�
<gradients/model/conv5/Conv2D_grad/tuple/control_dependency_1Identity6gradients/model/conv5/Conv2D_grad/Conv2DBackpropFilter3^gradients/model/conv5/Conv2D_grad/tuple/group_deps*
T0*I
_class?
=;loc:@gradients/model/conv5/Conv2D_grad/Conv2DBackpropFilter*'
_output_shapes
:�
�
*gradients/model_2/conv5/Conv2D_grad/ShapeNShapeNmodel_2/conv4/MaxPool2D/MaxPoolconv5/weights/read*
N* 
_output_shapes
::*
T0*
out_type0
�
7gradients/model_2/conv5/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput*gradients/model_2/conv5/Conv2D_grad/ShapeNconv5/weights/read=gradients/model_2/conv5/BiasAdd_grad/tuple/control_dependency*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*0
_output_shapes
:����������*
	dilations
*
T0
�
8gradients/model_2/conv5/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFiltermodel_2/conv4/MaxPool2D/MaxPool,gradients/model_2/conv5/Conv2D_grad/ShapeN:1=gradients/model_2/conv5/BiasAdd_grad/tuple/control_dependency*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*'
_output_shapes
:�*
	dilations

�
4gradients/model_2/conv5/Conv2D_grad/tuple/group_depsNoOp9^gradients/model_2/conv5/Conv2D_grad/Conv2DBackpropFilter8^gradients/model_2/conv5/Conv2D_grad/Conv2DBackpropInput
�
<gradients/model_2/conv5/Conv2D_grad/tuple/control_dependencyIdentity7gradients/model_2/conv5/Conv2D_grad/Conv2DBackpropInput5^gradients/model_2/conv5/Conv2D_grad/tuple/group_deps*
T0*J
_class@
><loc:@gradients/model_2/conv5/Conv2D_grad/Conv2DBackpropInput*0
_output_shapes
:����������
�
>gradients/model_2/conv5/Conv2D_grad/tuple/control_dependency_1Identity8gradients/model_2/conv5/Conv2D_grad/Conv2DBackpropFilter5^gradients/model_2/conv5/Conv2D_grad/tuple/group_deps*
T0*K
_classA
?=loc:@gradients/model_2/conv5/Conv2D_grad/Conv2DBackpropFilter*'
_output_shapes
:�
�
gradients/AddN_1AddN?gradients/model_1/conv5/BiasAdd_grad/tuple/control_dependency_1=gradients/model/conv5/BiasAdd_grad/tuple/control_dependency_1?gradients/model_2/conv5/BiasAdd_grad/tuple/control_dependency_1*
N*
_output_shapes
:*
T0*C
_class9
75loc:@gradients/model_1/conv5/BiasAdd_grad/BiasAddGrad
�
:gradients/model_1/conv4/MaxPool2D/MaxPool_grad/MaxPoolGradMaxPoolGradmodel_1/conv4/conv4/Relumodel_1/conv4/MaxPool2D/MaxPool<gradients/model_1/conv5/Conv2D_grad/tuple/control_dependency*
T0*
data_formatNHWC*
strides
*
ksize
*
paddingSAME*0
_output_shapes
:����������
�
8gradients/model/conv4/MaxPool2D/MaxPool_grad/MaxPoolGradMaxPoolGradmodel/conv4/conv4/Relumodel/conv4/MaxPool2D/MaxPool:gradients/model/conv5/Conv2D_grad/tuple/control_dependency*
T0*
data_formatNHWC*
strides
*
ksize
*
paddingSAME*0
_output_shapes
:����������
�
:gradients/model_2/conv4/MaxPool2D/MaxPool_grad/MaxPoolGradMaxPoolGradmodel_2/conv4/conv4/Relumodel_2/conv4/MaxPool2D/MaxPool<gradients/model_2/conv5/Conv2D_grad/tuple/control_dependency*0
_output_shapes
:����������*
T0*
data_formatNHWC*
strides
*
ksize
*
paddingSAME
�
gradients/AddN_2AddN>gradients/model_1/conv5/Conv2D_grad/tuple/control_dependency_1<gradients/model/conv5/Conv2D_grad/tuple/control_dependency_1>gradients/model_2/conv5/Conv2D_grad/tuple/control_dependency_1*
T0*K
_classA
?=loc:@gradients/model_1/conv5/Conv2D_grad/Conv2DBackpropFilter*
N*'
_output_shapes
:�
�
0gradients/model_1/conv4/conv4/Relu_grad/ReluGradReluGrad:gradients/model_1/conv4/MaxPool2D/MaxPool_grad/MaxPoolGradmodel_1/conv4/conv4/Relu*0
_output_shapes
:����������*
T0
�
.gradients/model/conv4/conv4/Relu_grad/ReluGradReluGrad8gradients/model/conv4/MaxPool2D/MaxPool_grad/MaxPoolGradmodel/conv4/conv4/Relu*
T0*0
_output_shapes
:����������
�
0gradients/model_2/conv4/conv4/Relu_grad/ReluGradReluGrad:gradients/model_2/conv4/MaxPool2D/MaxPool_grad/MaxPoolGradmodel_2/conv4/conv4/Relu*
T0*0
_output_shapes
:����������
�
0gradients/model_1/conv4/BiasAdd_grad/BiasAddGradBiasAddGrad0gradients/model_1/conv4/conv4/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes	
:�
�
5gradients/model_1/conv4/BiasAdd_grad/tuple/group_depsNoOp1^gradients/model_1/conv4/BiasAdd_grad/BiasAddGrad1^gradients/model_1/conv4/conv4/Relu_grad/ReluGrad
�
=gradients/model_1/conv4/BiasAdd_grad/tuple/control_dependencyIdentity0gradients/model_1/conv4/conv4/Relu_grad/ReluGrad6^gradients/model_1/conv4/BiasAdd_grad/tuple/group_deps*0
_output_shapes
:����������*
T0*C
_class9
75loc:@gradients/model_1/conv4/conv4/Relu_grad/ReluGrad
�
?gradients/model_1/conv4/BiasAdd_grad/tuple/control_dependency_1Identity0gradients/model_1/conv4/BiasAdd_grad/BiasAddGrad6^gradients/model_1/conv4/BiasAdd_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/model_1/conv4/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:�
�
.gradients/model/conv4/BiasAdd_grad/BiasAddGradBiasAddGrad.gradients/model/conv4/conv4/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes	
:�
�
3gradients/model/conv4/BiasAdd_grad/tuple/group_depsNoOp/^gradients/model/conv4/BiasAdd_grad/BiasAddGrad/^gradients/model/conv4/conv4/Relu_grad/ReluGrad
�
;gradients/model/conv4/BiasAdd_grad/tuple/control_dependencyIdentity.gradients/model/conv4/conv4/Relu_grad/ReluGrad4^gradients/model/conv4/BiasAdd_grad/tuple/group_deps*0
_output_shapes
:����������*
T0*A
_class7
53loc:@gradients/model/conv4/conv4/Relu_grad/ReluGrad
�
=gradients/model/conv4/BiasAdd_grad/tuple/control_dependency_1Identity.gradients/model/conv4/BiasAdd_grad/BiasAddGrad4^gradients/model/conv4/BiasAdd_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients/model/conv4/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:�
�
0gradients/model_2/conv4/BiasAdd_grad/BiasAddGradBiasAddGrad0gradients/model_2/conv4/conv4/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes	
:�
�
5gradients/model_2/conv4/BiasAdd_grad/tuple/group_depsNoOp1^gradients/model_2/conv4/BiasAdd_grad/BiasAddGrad1^gradients/model_2/conv4/conv4/Relu_grad/ReluGrad
�
=gradients/model_2/conv4/BiasAdd_grad/tuple/control_dependencyIdentity0gradients/model_2/conv4/conv4/Relu_grad/ReluGrad6^gradients/model_2/conv4/BiasAdd_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/model_2/conv4/conv4/Relu_grad/ReluGrad*0
_output_shapes
:����������
�
?gradients/model_2/conv4/BiasAdd_grad/tuple/control_dependency_1Identity0gradients/model_2/conv4/BiasAdd_grad/BiasAddGrad6^gradients/model_2/conv4/BiasAdd_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/model_2/conv4/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:�
�
*gradients/model_1/conv4/Conv2D_grad/ShapeNShapeNmodel_1/conv3/MaxPool2D/MaxPoolconv4/weights/read*
N* 
_output_shapes
::*
T0*
out_type0
�
7gradients/model_1/conv4/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput*gradients/model_1/conv4/Conv2D_grad/ShapeNconv4/weights/read=gradients/model_1/conv4/BiasAdd_grad/tuple/control_dependency*0
_output_shapes
:����������*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME
�
8gradients/model_1/conv4/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFiltermodel_1/conv3/MaxPool2D/MaxPool,gradients/model_1/conv4/Conv2D_grad/ShapeN:1=gradients/model_1/conv4/BiasAdd_grad/tuple/control_dependency*(
_output_shapes
:��*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME
�
4gradients/model_1/conv4/Conv2D_grad/tuple/group_depsNoOp9^gradients/model_1/conv4/Conv2D_grad/Conv2DBackpropFilter8^gradients/model_1/conv4/Conv2D_grad/Conv2DBackpropInput
�
<gradients/model_1/conv4/Conv2D_grad/tuple/control_dependencyIdentity7gradients/model_1/conv4/Conv2D_grad/Conv2DBackpropInput5^gradients/model_1/conv4/Conv2D_grad/tuple/group_deps*
T0*J
_class@
><loc:@gradients/model_1/conv4/Conv2D_grad/Conv2DBackpropInput*0
_output_shapes
:����������
�
>gradients/model_1/conv4/Conv2D_grad/tuple/control_dependency_1Identity8gradients/model_1/conv4/Conv2D_grad/Conv2DBackpropFilter5^gradients/model_1/conv4/Conv2D_grad/tuple/group_deps*
T0*K
_classA
?=loc:@gradients/model_1/conv4/Conv2D_grad/Conv2DBackpropFilter*(
_output_shapes
:��
�
(gradients/model/conv4/Conv2D_grad/ShapeNShapeNmodel/conv3/MaxPool2D/MaxPoolconv4/weights/read*
N* 
_output_shapes
::*
T0*
out_type0
�
5gradients/model/conv4/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput(gradients/model/conv4/Conv2D_grad/ShapeNconv4/weights/read;gradients/model/conv4/BiasAdd_grad/tuple/control_dependency*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*0
_output_shapes
:����������
�
6gradients/model/conv4/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFiltermodel/conv3/MaxPool2D/MaxPool*gradients/model/conv4/Conv2D_grad/ShapeN:1;gradients/model/conv4/BiasAdd_grad/tuple/control_dependency*(
_output_shapes
:��*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME
�
2gradients/model/conv4/Conv2D_grad/tuple/group_depsNoOp7^gradients/model/conv4/Conv2D_grad/Conv2DBackpropFilter6^gradients/model/conv4/Conv2D_grad/Conv2DBackpropInput
�
:gradients/model/conv4/Conv2D_grad/tuple/control_dependencyIdentity5gradients/model/conv4/Conv2D_grad/Conv2DBackpropInput3^gradients/model/conv4/Conv2D_grad/tuple/group_deps*
T0*H
_class>
<:loc:@gradients/model/conv4/Conv2D_grad/Conv2DBackpropInput*0
_output_shapes
:����������
�
<gradients/model/conv4/Conv2D_grad/tuple/control_dependency_1Identity6gradients/model/conv4/Conv2D_grad/Conv2DBackpropFilter3^gradients/model/conv4/Conv2D_grad/tuple/group_deps*
T0*I
_class?
=;loc:@gradients/model/conv4/Conv2D_grad/Conv2DBackpropFilter*(
_output_shapes
:��
�
*gradients/model_2/conv4/Conv2D_grad/ShapeNShapeNmodel_2/conv3/MaxPool2D/MaxPoolconv4/weights/read*
N* 
_output_shapes
::*
T0*
out_type0
�
7gradients/model_2/conv4/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput*gradients/model_2/conv4/Conv2D_grad/ShapeNconv4/weights/read=gradients/model_2/conv4/BiasAdd_grad/tuple/control_dependency*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*0
_output_shapes
:����������
�
8gradients/model_2/conv4/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFiltermodel_2/conv3/MaxPool2D/MaxPool,gradients/model_2/conv4/Conv2D_grad/ShapeN:1=gradients/model_2/conv4/BiasAdd_grad/tuple/control_dependency*
paddingSAME*(
_output_shapes
:��*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
�
4gradients/model_2/conv4/Conv2D_grad/tuple/group_depsNoOp9^gradients/model_2/conv4/Conv2D_grad/Conv2DBackpropFilter8^gradients/model_2/conv4/Conv2D_grad/Conv2DBackpropInput
�
<gradients/model_2/conv4/Conv2D_grad/tuple/control_dependencyIdentity7gradients/model_2/conv4/Conv2D_grad/Conv2DBackpropInput5^gradients/model_2/conv4/Conv2D_grad/tuple/group_deps*0
_output_shapes
:����������*
T0*J
_class@
><loc:@gradients/model_2/conv4/Conv2D_grad/Conv2DBackpropInput
�
>gradients/model_2/conv4/Conv2D_grad/tuple/control_dependency_1Identity8gradients/model_2/conv4/Conv2D_grad/Conv2DBackpropFilter5^gradients/model_2/conv4/Conv2D_grad/tuple/group_deps*
T0*K
_classA
?=loc:@gradients/model_2/conv4/Conv2D_grad/Conv2DBackpropFilter*(
_output_shapes
:��
�
gradients/AddN_3AddN?gradients/model_1/conv4/BiasAdd_grad/tuple/control_dependency_1=gradients/model/conv4/BiasAdd_grad/tuple/control_dependency_1?gradients/model_2/conv4/BiasAdd_grad/tuple/control_dependency_1*
T0*C
_class9
75loc:@gradients/model_1/conv4/BiasAdd_grad/BiasAddGrad*
N*
_output_shapes	
:�
�
:gradients/model_1/conv3/MaxPool2D/MaxPool_grad/MaxPoolGradMaxPoolGradmodel_1/conv3/conv3/Relumodel_1/conv3/MaxPool2D/MaxPool<gradients/model_1/conv4/Conv2D_grad/tuple/control_dependency*0
_output_shapes
:����������*
T0*
data_formatNHWC*
strides
*
ksize
*
paddingSAME
�
8gradients/model/conv3/MaxPool2D/MaxPool_grad/MaxPoolGradMaxPoolGradmodel/conv3/conv3/Relumodel/conv3/MaxPool2D/MaxPool:gradients/model/conv4/Conv2D_grad/tuple/control_dependency*
data_formatNHWC*
strides
*
ksize
*
paddingSAME*0
_output_shapes
:����������*
T0
�
:gradients/model_2/conv3/MaxPool2D/MaxPool_grad/MaxPoolGradMaxPoolGradmodel_2/conv3/conv3/Relumodel_2/conv3/MaxPool2D/MaxPool<gradients/model_2/conv4/Conv2D_grad/tuple/control_dependency*
data_formatNHWC*
strides
*
ksize
*
paddingSAME*0
_output_shapes
:����������*
T0
�
gradients/AddN_4AddN>gradients/model_1/conv4/Conv2D_grad/tuple/control_dependency_1<gradients/model/conv4/Conv2D_grad/tuple/control_dependency_1>gradients/model_2/conv4/Conv2D_grad/tuple/control_dependency_1*
N*(
_output_shapes
:��*
T0*K
_classA
?=loc:@gradients/model_1/conv4/Conv2D_grad/Conv2DBackpropFilter
�
0gradients/model_1/conv3/conv3/Relu_grad/ReluGradReluGrad:gradients/model_1/conv3/MaxPool2D/MaxPool_grad/MaxPoolGradmodel_1/conv3/conv3/Relu*0
_output_shapes
:����������*
T0
�
.gradients/model/conv3/conv3/Relu_grad/ReluGradReluGrad8gradients/model/conv3/MaxPool2D/MaxPool_grad/MaxPoolGradmodel/conv3/conv3/Relu*
T0*0
_output_shapes
:����������
�
0gradients/model_2/conv3/conv3/Relu_grad/ReluGradReluGrad:gradients/model_2/conv3/MaxPool2D/MaxPool_grad/MaxPoolGradmodel_2/conv3/conv3/Relu*
T0*0
_output_shapes
:����������
�
0gradients/model_1/conv3/BiasAdd_grad/BiasAddGradBiasAddGrad0gradients/model_1/conv3/conv3/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes	
:�
�
5gradients/model_1/conv3/BiasAdd_grad/tuple/group_depsNoOp1^gradients/model_1/conv3/BiasAdd_grad/BiasAddGrad1^gradients/model_1/conv3/conv3/Relu_grad/ReluGrad
�
=gradients/model_1/conv3/BiasAdd_grad/tuple/control_dependencyIdentity0gradients/model_1/conv3/conv3/Relu_grad/ReluGrad6^gradients/model_1/conv3/BiasAdd_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/model_1/conv3/conv3/Relu_grad/ReluGrad*0
_output_shapes
:����������
�
?gradients/model_1/conv3/BiasAdd_grad/tuple/control_dependency_1Identity0gradients/model_1/conv3/BiasAdd_grad/BiasAddGrad6^gradients/model_1/conv3/BiasAdd_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/model_1/conv3/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:�
�
.gradients/model/conv3/BiasAdd_grad/BiasAddGradBiasAddGrad.gradients/model/conv3/conv3/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes	
:�
�
3gradients/model/conv3/BiasAdd_grad/tuple/group_depsNoOp/^gradients/model/conv3/BiasAdd_grad/BiasAddGrad/^gradients/model/conv3/conv3/Relu_grad/ReluGrad
�
;gradients/model/conv3/BiasAdd_grad/tuple/control_dependencyIdentity.gradients/model/conv3/conv3/Relu_grad/ReluGrad4^gradients/model/conv3/BiasAdd_grad/tuple/group_deps*0
_output_shapes
:����������*
T0*A
_class7
53loc:@gradients/model/conv3/conv3/Relu_grad/ReluGrad
�
=gradients/model/conv3/BiasAdd_grad/tuple/control_dependency_1Identity.gradients/model/conv3/BiasAdd_grad/BiasAddGrad4^gradients/model/conv3/BiasAdd_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients/model/conv3/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:�
�
0gradients/model_2/conv3/BiasAdd_grad/BiasAddGradBiasAddGrad0gradients/model_2/conv3/conv3/Relu_grad/ReluGrad*
data_formatNHWC*
_output_shapes	
:�*
T0
�
5gradients/model_2/conv3/BiasAdd_grad/tuple/group_depsNoOp1^gradients/model_2/conv3/BiasAdd_grad/BiasAddGrad1^gradients/model_2/conv3/conv3/Relu_grad/ReluGrad
�
=gradients/model_2/conv3/BiasAdd_grad/tuple/control_dependencyIdentity0gradients/model_2/conv3/conv3/Relu_grad/ReluGrad6^gradients/model_2/conv3/BiasAdd_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/model_2/conv3/conv3/Relu_grad/ReluGrad*0
_output_shapes
:����������
�
?gradients/model_2/conv3/BiasAdd_grad/tuple/control_dependency_1Identity0gradients/model_2/conv3/BiasAdd_grad/BiasAddGrad6^gradients/model_2/conv3/BiasAdd_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/model_2/conv3/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:�
�
*gradients/model_1/conv3/Conv2D_grad/ShapeNShapeNmodel_1/conv2/MaxPool2D/MaxPoolconv3/weights/read*
T0*
out_type0*
N* 
_output_shapes
::
�
7gradients/model_1/conv3/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput*gradients/model_1/conv3/Conv2D_grad/ShapeNconv3/weights/read=gradients/model_1/conv3/BiasAdd_grad/tuple/control_dependency*/
_output_shapes
:���������@*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME
�
8gradients/model_1/conv3/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFiltermodel_1/conv2/MaxPool2D/MaxPool,gradients/model_1/conv3/Conv2D_grad/ShapeN:1=gradients/model_1/conv3/BiasAdd_grad/tuple/control_dependency*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*'
_output_shapes
:@�
�
4gradients/model_1/conv3/Conv2D_grad/tuple/group_depsNoOp9^gradients/model_1/conv3/Conv2D_grad/Conv2DBackpropFilter8^gradients/model_1/conv3/Conv2D_grad/Conv2DBackpropInput
�
<gradients/model_1/conv3/Conv2D_grad/tuple/control_dependencyIdentity7gradients/model_1/conv3/Conv2D_grad/Conv2DBackpropInput5^gradients/model_1/conv3/Conv2D_grad/tuple/group_deps*
T0*J
_class@
><loc:@gradients/model_1/conv3/Conv2D_grad/Conv2DBackpropInput*/
_output_shapes
:���������@
�
>gradients/model_1/conv3/Conv2D_grad/tuple/control_dependency_1Identity8gradients/model_1/conv3/Conv2D_grad/Conv2DBackpropFilter5^gradients/model_1/conv3/Conv2D_grad/tuple/group_deps*
T0*K
_classA
?=loc:@gradients/model_1/conv3/Conv2D_grad/Conv2DBackpropFilter*'
_output_shapes
:@�
�
(gradients/model/conv3/Conv2D_grad/ShapeNShapeNmodel/conv2/MaxPool2D/MaxPoolconv3/weights/read*
T0*
out_type0*
N* 
_output_shapes
::
�
5gradients/model/conv3/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput(gradients/model/conv3/Conv2D_grad/ShapeNconv3/weights/read;gradients/model/conv3/BiasAdd_grad/tuple/control_dependency*
paddingSAME*/
_output_shapes
:���������@*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
�
6gradients/model/conv3/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFiltermodel/conv2/MaxPool2D/MaxPool*gradients/model/conv3/Conv2D_grad/ShapeN:1;gradients/model/conv3/BiasAdd_grad/tuple/control_dependency*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*'
_output_shapes
:@�
�
2gradients/model/conv3/Conv2D_grad/tuple/group_depsNoOp7^gradients/model/conv3/Conv2D_grad/Conv2DBackpropFilter6^gradients/model/conv3/Conv2D_grad/Conv2DBackpropInput
�
:gradients/model/conv3/Conv2D_grad/tuple/control_dependencyIdentity5gradients/model/conv3/Conv2D_grad/Conv2DBackpropInput3^gradients/model/conv3/Conv2D_grad/tuple/group_deps*/
_output_shapes
:���������@*
T0*H
_class>
<:loc:@gradients/model/conv3/Conv2D_grad/Conv2DBackpropInput
�
<gradients/model/conv3/Conv2D_grad/tuple/control_dependency_1Identity6gradients/model/conv3/Conv2D_grad/Conv2DBackpropFilter3^gradients/model/conv3/Conv2D_grad/tuple/group_deps*
T0*I
_class?
=;loc:@gradients/model/conv3/Conv2D_grad/Conv2DBackpropFilter*'
_output_shapes
:@�
�
*gradients/model_2/conv3/Conv2D_grad/ShapeNShapeNmodel_2/conv2/MaxPool2D/MaxPoolconv3/weights/read*
T0*
out_type0*
N* 
_output_shapes
::
�
7gradients/model_2/conv3/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput*gradients/model_2/conv3/Conv2D_grad/ShapeNconv3/weights/read=gradients/model_2/conv3/BiasAdd_grad/tuple/control_dependency*/
_output_shapes
:���������@*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME
�
8gradients/model_2/conv3/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFiltermodel_2/conv2/MaxPool2D/MaxPool,gradients/model_2/conv3/Conv2D_grad/ShapeN:1=gradients/model_2/conv3/BiasAdd_grad/tuple/control_dependency*
paddingSAME*'
_output_shapes
:@�*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(
�
4gradients/model_2/conv3/Conv2D_grad/tuple/group_depsNoOp9^gradients/model_2/conv3/Conv2D_grad/Conv2DBackpropFilter8^gradients/model_2/conv3/Conv2D_grad/Conv2DBackpropInput
�
<gradients/model_2/conv3/Conv2D_grad/tuple/control_dependencyIdentity7gradients/model_2/conv3/Conv2D_grad/Conv2DBackpropInput5^gradients/model_2/conv3/Conv2D_grad/tuple/group_deps*
T0*J
_class@
><loc:@gradients/model_2/conv3/Conv2D_grad/Conv2DBackpropInput*/
_output_shapes
:���������@
�
>gradients/model_2/conv3/Conv2D_grad/tuple/control_dependency_1Identity8gradients/model_2/conv3/Conv2D_grad/Conv2DBackpropFilter5^gradients/model_2/conv3/Conv2D_grad/tuple/group_deps*
T0*K
_classA
?=loc:@gradients/model_2/conv3/Conv2D_grad/Conv2DBackpropFilter*'
_output_shapes
:@�
�
gradients/AddN_5AddN?gradients/model_1/conv3/BiasAdd_grad/tuple/control_dependency_1=gradients/model/conv3/BiasAdd_grad/tuple/control_dependency_1?gradients/model_2/conv3/BiasAdd_grad/tuple/control_dependency_1*
T0*C
_class9
75loc:@gradients/model_1/conv3/BiasAdd_grad/BiasAddGrad*
N*
_output_shapes	
:�
�
:gradients/model_1/conv2/MaxPool2D/MaxPool_grad/MaxPoolGradMaxPoolGradmodel_1/conv2/conv2/Relumodel_1/conv2/MaxPool2D/MaxPool<gradients/model_1/conv3/Conv2D_grad/tuple/control_dependency*
data_formatNHWC*
strides
*
ksize
*
paddingSAME*/
_output_shapes
:���������
@*
T0
�
8gradients/model/conv2/MaxPool2D/MaxPool_grad/MaxPoolGradMaxPoolGradmodel/conv2/conv2/Relumodel/conv2/MaxPool2D/MaxPool:gradients/model/conv3/Conv2D_grad/tuple/control_dependency*
T0*
data_formatNHWC*
strides
*
ksize
*
paddingSAME*/
_output_shapes
:���������
@
�
:gradients/model_2/conv2/MaxPool2D/MaxPool_grad/MaxPoolGradMaxPoolGradmodel_2/conv2/conv2/Relumodel_2/conv2/MaxPool2D/MaxPool<gradients/model_2/conv3/Conv2D_grad/tuple/control_dependency*
T0*
data_formatNHWC*
strides
*
ksize
*
paddingSAME*/
_output_shapes
:���������
@
�
gradients/AddN_6AddN>gradients/model_1/conv3/Conv2D_grad/tuple/control_dependency_1<gradients/model/conv3/Conv2D_grad/tuple/control_dependency_1>gradients/model_2/conv3/Conv2D_grad/tuple/control_dependency_1*
T0*K
_classA
?=loc:@gradients/model_1/conv3/Conv2D_grad/Conv2DBackpropFilter*
N*'
_output_shapes
:@�
�
0gradients/model_1/conv2/conv2/Relu_grad/ReluGradReluGrad:gradients/model_1/conv2/MaxPool2D/MaxPool_grad/MaxPoolGradmodel_1/conv2/conv2/Relu*
T0*/
_output_shapes
:���������
@
�
.gradients/model/conv2/conv2/Relu_grad/ReluGradReluGrad8gradients/model/conv2/MaxPool2D/MaxPool_grad/MaxPoolGradmodel/conv2/conv2/Relu*
T0*/
_output_shapes
:���������
@
�
0gradients/model_2/conv2/conv2/Relu_grad/ReluGradReluGrad:gradients/model_2/conv2/MaxPool2D/MaxPool_grad/MaxPoolGradmodel_2/conv2/conv2/Relu*
T0*/
_output_shapes
:���������
@
�
0gradients/model_1/conv2/BiasAdd_grad/BiasAddGradBiasAddGrad0gradients/model_1/conv2/conv2/Relu_grad/ReluGrad*
data_formatNHWC*
_output_shapes
:@*
T0
�
5gradients/model_1/conv2/BiasAdd_grad/tuple/group_depsNoOp1^gradients/model_1/conv2/BiasAdd_grad/BiasAddGrad1^gradients/model_1/conv2/conv2/Relu_grad/ReluGrad
�
=gradients/model_1/conv2/BiasAdd_grad/tuple/control_dependencyIdentity0gradients/model_1/conv2/conv2/Relu_grad/ReluGrad6^gradients/model_1/conv2/BiasAdd_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/model_1/conv2/conv2/Relu_grad/ReluGrad*/
_output_shapes
:���������
@
�
?gradients/model_1/conv2/BiasAdd_grad/tuple/control_dependency_1Identity0gradients/model_1/conv2/BiasAdd_grad/BiasAddGrad6^gradients/model_1/conv2/BiasAdd_grad/tuple/group_deps*
_output_shapes
:@*
T0*C
_class9
75loc:@gradients/model_1/conv2/BiasAdd_grad/BiasAddGrad
�
.gradients/model/conv2/BiasAdd_grad/BiasAddGradBiasAddGrad.gradients/model/conv2/conv2/Relu_grad/ReluGrad*
data_formatNHWC*
_output_shapes
:@*
T0
�
3gradients/model/conv2/BiasAdd_grad/tuple/group_depsNoOp/^gradients/model/conv2/BiasAdd_grad/BiasAddGrad/^gradients/model/conv2/conv2/Relu_grad/ReluGrad
�
;gradients/model/conv2/BiasAdd_grad/tuple/control_dependencyIdentity.gradients/model/conv2/conv2/Relu_grad/ReluGrad4^gradients/model/conv2/BiasAdd_grad/tuple/group_deps*/
_output_shapes
:���������
@*
T0*A
_class7
53loc:@gradients/model/conv2/conv2/Relu_grad/ReluGrad
�
=gradients/model/conv2/BiasAdd_grad/tuple/control_dependency_1Identity.gradients/model/conv2/BiasAdd_grad/BiasAddGrad4^gradients/model/conv2/BiasAdd_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients/model/conv2/BiasAdd_grad/BiasAddGrad*
_output_shapes
:@
�
0gradients/model_2/conv2/BiasAdd_grad/BiasAddGradBiasAddGrad0gradients/model_2/conv2/conv2/Relu_grad/ReluGrad*
data_formatNHWC*
_output_shapes
:@*
T0
�
5gradients/model_2/conv2/BiasAdd_grad/tuple/group_depsNoOp1^gradients/model_2/conv2/BiasAdd_grad/BiasAddGrad1^gradients/model_2/conv2/conv2/Relu_grad/ReluGrad
�
=gradients/model_2/conv2/BiasAdd_grad/tuple/control_dependencyIdentity0gradients/model_2/conv2/conv2/Relu_grad/ReluGrad6^gradients/model_2/conv2/BiasAdd_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/model_2/conv2/conv2/Relu_grad/ReluGrad*/
_output_shapes
:���������
@
�
?gradients/model_2/conv2/BiasAdd_grad/tuple/control_dependency_1Identity0gradients/model_2/conv2/BiasAdd_grad/BiasAddGrad6^gradients/model_2/conv2/BiasAdd_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/model_2/conv2/BiasAdd_grad/BiasAddGrad*
_output_shapes
:@
�
*gradients/model_1/conv2/Conv2D_grad/ShapeNShapeNmodel_1/conv1/MaxPool2D/MaxPoolconv2/weights/read*
N* 
_output_shapes
::*
T0*
out_type0
�
7gradients/model_1/conv2/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput*gradients/model_1/conv2/Conv2D_grad/ShapeNconv2/weights/read=gradients/model_1/conv2/BiasAdd_grad/tuple/control_dependency*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*/
_output_shapes
:���������
 
�
8gradients/model_1/conv2/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFiltermodel_1/conv1/MaxPool2D/MaxPool,gradients/model_1/conv2/Conv2D_grad/ShapeN:1=gradients/model_1/conv2/BiasAdd_grad/tuple/control_dependency*&
_output_shapes
: @*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME
�
4gradients/model_1/conv2/Conv2D_grad/tuple/group_depsNoOp9^gradients/model_1/conv2/Conv2D_grad/Conv2DBackpropFilter8^gradients/model_1/conv2/Conv2D_grad/Conv2DBackpropInput
�
<gradients/model_1/conv2/Conv2D_grad/tuple/control_dependencyIdentity7gradients/model_1/conv2/Conv2D_grad/Conv2DBackpropInput5^gradients/model_1/conv2/Conv2D_grad/tuple/group_deps*
T0*J
_class@
><loc:@gradients/model_1/conv2/Conv2D_grad/Conv2DBackpropInput*/
_output_shapes
:���������
 
�
>gradients/model_1/conv2/Conv2D_grad/tuple/control_dependency_1Identity8gradients/model_1/conv2/Conv2D_grad/Conv2DBackpropFilter5^gradients/model_1/conv2/Conv2D_grad/tuple/group_deps*&
_output_shapes
: @*
T0*K
_classA
?=loc:@gradients/model_1/conv2/Conv2D_grad/Conv2DBackpropFilter
�
(gradients/model/conv2/Conv2D_grad/ShapeNShapeNmodel/conv1/MaxPool2D/MaxPoolconv2/weights/read*
T0*
out_type0*
N* 
_output_shapes
::
�
5gradients/model/conv2/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput(gradients/model/conv2/Conv2D_grad/ShapeNconv2/weights/read;gradients/model/conv2/BiasAdd_grad/tuple/control_dependency*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*/
_output_shapes
:���������
 
�
6gradients/model/conv2/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFiltermodel/conv1/MaxPool2D/MaxPool*gradients/model/conv2/Conv2D_grad/ShapeN:1;gradients/model/conv2/BiasAdd_grad/tuple/control_dependency*
paddingSAME*&
_output_shapes
: @*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
�
2gradients/model/conv2/Conv2D_grad/tuple/group_depsNoOp7^gradients/model/conv2/Conv2D_grad/Conv2DBackpropFilter6^gradients/model/conv2/Conv2D_grad/Conv2DBackpropInput
�
:gradients/model/conv2/Conv2D_grad/tuple/control_dependencyIdentity5gradients/model/conv2/Conv2D_grad/Conv2DBackpropInput3^gradients/model/conv2/Conv2D_grad/tuple/group_deps*
T0*H
_class>
<:loc:@gradients/model/conv2/Conv2D_grad/Conv2DBackpropInput*/
_output_shapes
:���������
 
�
<gradients/model/conv2/Conv2D_grad/tuple/control_dependency_1Identity6gradients/model/conv2/Conv2D_grad/Conv2DBackpropFilter3^gradients/model/conv2/Conv2D_grad/tuple/group_deps*&
_output_shapes
: @*
T0*I
_class?
=;loc:@gradients/model/conv2/Conv2D_grad/Conv2DBackpropFilter
�
*gradients/model_2/conv2/Conv2D_grad/ShapeNShapeNmodel_2/conv1/MaxPool2D/MaxPoolconv2/weights/read*
T0*
out_type0*
N* 
_output_shapes
::
�
7gradients/model_2/conv2/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput*gradients/model_2/conv2/Conv2D_grad/ShapeNconv2/weights/read=gradients/model_2/conv2/BiasAdd_grad/tuple/control_dependency*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME*/
_output_shapes
:���������
 
�
8gradients/model_2/conv2/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFiltermodel_2/conv1/MaxPool2D/MaxPool,gradients/model_2/conv2/Conv2D_grad/ShapeN:1=gradients/model_2/conv2/BiasAdd_grad/tuple/control_dependency*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*&
_output_shapes
: @
�
4gradients/model_2/conv2/Conv2D_grad/tuple/group_depsNoOp9^gradients/model_2/conv2/Conv2D_grad/Conv2DBackpropFilter8^gradients/model_2/conv2/Conv2D_grad/Conv2DBackpropInput
�
<gradients/model_2/conv2/Conv2D_grad/tuple/control_dependencyIdentity7gradients/model_2/conv2/Conv2D_grad/Conv2DBackpropInput5^gradients/model_2/conv2/Conv2D_grad/tuple/group_deps*/
_output_shapes
:���������
 *
T0*J
_class@
><loc:@gradients/model_2/conv2/Conv2D_grad/Conv2DBackpropInput
�
>gradients/model_2/conv2/Conv2D_grad/tuple/control_dependency_1Identity8gradients/model_2/conv2/Conv2D_grad/Conv2DBackpropFilter5^gradients/model_2/conv2/Conv2D_grad/tuple/group_deps*&
_output_shapes
: @*
T0*K
_classA
?=loc:@gradients/model_2/conv2/Conv2D_grad/Conv2DBackpropFilter
�
gradients/AddN_7AddN?gradients/model_1/conv2/BiasAdd_grad/tuple/control_dependency_1=gradients/model/conv2/BiasAdd_grad/tuple/control_dependency_1?gradients/model_2/conv2/BiasAdd_grad/tuple/control_dependency_1*
T0*C
_class9
75loc:@gradients/model_1/conv2/BiasAdd_grad/BiasAddGrad*
N*
_output_shapes
:@
�
:gradients/model_1/conv1/MaxPool2D/MaxPool_grad/MaxPoolGradMaxPoolGradmodel_1/conv1/conv1/Relumodel_1/conv1/MaxPool2D/MaxPool<gradients/model_1/conv2/Conv2D_grad/tuple/control_dependency*/
_output_shapes
:���������
 *
T0*
data_formatNHWC*
strides
*
ksize
*
paddingSAME
�
8gradients/model/conv1/MaxPool2D/MaxPool_grad/MaxPoolGradMaxPoolGradmodel/conv1/conv1/Relumodel/conv1/MaxPool2D/MaxPool:gradients/model/conv2/Conv2D_grad/tuple/control_dependency*
T0*
data_formatNHWC*
strides
*
ksize
*
paddingSAME*/
_output_shapes
:���������
 
�
:gradients/model_2/conv1/MaxPool2D/MaxPool_grad/MaxPoolGradMaxPoolGradmodel_2/conv1/conv1/Relumodel_2/conv1/MaxPool2D/MaxPool<gradients/model_2/conv2/Conv2D_grad/tuple/control_dependency*/
_output_shapes
:���������
 *
T0*
data_formatNHWC*
strides
*
ksize
*
paddingSAME
�
gradients/AddN_8AddN>gradients/model_1/conv2/Conv2D_grad/tuple/control_dependency_1<gradients/model/conv2/Conv2D_grad/tuple/control_dependency_1>gradients/model_2/conv2/Conv2D_grad/tuple/control_dependency_1*
T0*K
_classA
?=loc:@gradients/model_1/conv2/Conv2D_grad/Conv2DBackpropFilter*
N*&
_output_shapes
: @
�
0gradients/model_1/conv1/conv1/Relu_grad/ReluGradReluGrad:gradients/model_1/conv1/MaxPool2D/MaxPool_grad/MaxPoolGradmodel_1/conv1/conv1/Relu*/
_output_shapes
:���������
 *
T0
�
.gradients/model/conv1/conv1/Relu_grad/ReluGradReluGrad8gradients/model/conv1/MaxPool2D/MaxPool_grad/MaxPoolGradmodel/conv1/conv1/Relu*
T0*/
_output_shapes
:���������
 
�
0gradients/model_2/conv1/conv1/Relu_grad/ReluGradReluGrad:gradients/model_2/conv1/MaxPool2D/MaxPool_grad/MaxPoolGradmodel_2/conv1/conv1/Relu*
T0*/
_output_shapes
:���������
 
�
0gradients/model_1/conv1/BiasAdd_grad/BiasAddGradBiasAddGrad0gradients/model_1/conv1/conv1/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes
: 
�
5gradients/model_1/conv1/BiasAdd_grad/tuple/group_depsNoOp1^gradients/model_1/conv1/BiasAdd_grad/BiasAddGrad1^gradients/model_1/conv1/conv1/Relu_grad/ReluGrad
�
=gradients/model_1/conv1/BiasAdd_grad/tuple/control_dependencyIdentity0gradients/model_1/conv1/conv1/Relu_grad/ReluGrad6^gradients/model_1/conv1/BiasAdd_grad/tuple/group_deps*/
_output_shapes
:���������
 *
T0*C
_class9
75loc:@gradients/model_1/conv1/conv1/Relu_grad/ReluGrad
�
?gradients/model_1/conv1/BiasAdd_grad/tuple/control_dependency_1Identity0gradients/model_1/conv1/BiasAdd_grad/BiasAddGrad6^gradients/model_1/conv1/BiasAdd_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/model_1/conv1/BiasAdd_grad/BiasAddGrad*
_output_shapes
: 
�
.gradients/model/conv1/BiasAdd_grad/BiasAddGradBiasAddGrad.gradients/model/conv1/conv1/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes
: 
�
3gradients/model/conv1/BiasAdd_grad/tuple/group_depsNoOp/^gradients/model/conv1/BiasAdd_grad/BiasAddGrad/^gradients/model/conv1/conv1/Relu_grad/ReluGrad
�
;gradients/model/conv1/BiasAdd_grad/tuple/control_dependencyIdentity.gradients/model/conv1/conv1/Relu_grad/ReluGrad4^gradients/model/conv1/BiasAdd_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients/model/conv1/conv1/Relu_grad/ReluGrad*/
_output_shapes
:���������
 
�
=gradients/model/conv1/BiasAdd_grad/tuple/control_dependency_1Identity.gradients/model/conv1/BiasAdd_grad/BiasAddGrad4^gradients/model/conv1/BiasAdd_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients/model/conv1/BiasAdd_grad/BiasAddGrad*
_output_shapes
: 
�
0gradients/model_2/conv1/BiasAdd_grad/BiasAddGradBiasAddGrad0gradients/model_2/conv1/conv1/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes
: 
�
5gradients/model_2/conv1/BiasAdd_grad/tuple/group_depsNoOp1^gradients/model_2/conv1/BiasAdd_grad/BiasAddGrad1^gradients/model_2/conv1/conv1/Relu_grad/ReluGrad
�
=gradients/model_2/conv1/BiasAdd_grad/tuple/control_dependencyIdentity0gradients/model_2/conv1/conv1/Relu_grad/ReluGrad6^gradients/model_2/conv1/BiasAdd_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/model_2/conv1/conv1/Relu_grad/ReluGrad*/
_output_shapes
:���������
 
�
?gradients/model_2/conv1/BiasAdd_grad/tuple/control_dependency_1Identity0gradients/model_2/conv1/BiasAdd_grad/BiasAddGrad6^gradients/model_2/conv1/BiasAdd_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/model_2/conv1/BiasAdd_grad/BiasAddGrad*
_output_shapes
: 
�
*gradients/model_1/conv1/Conv2D_grad/ShapeNShapeNpositive_inputconv1/weights/read*
T0*
out_type0*
N* 
_output_shapes
::
�
7gradients/model_1/conv1/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput*gradients/model_1/conv1/Conv2D_grad/ShapeNconv1/weights/read=gradients/model_1/conv1/BiasAdd_grad/tuple/control_dependency*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*/
_output_shapes
:���������
*
	dilations

�
8gradients/model_1/conv1/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterpositive_input,gradients/model_1/conv1/Conv2D_grad/ShapeN:1=gradients/model_1/conv1/BiasAdd_grad/tuple/control_dependency*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*&
_output_shapes
: 
�
4gradients/model_1/conv1/Conv2D_grad/tuple/group_depsNoOp9^gradients/model_1/conv1/Conv2D_grad/Conv2DBackpropFilter8^gradients/model_1/conv1/Conv2D_grad/Conv2DBackpropInput
�
<gradients/model_1/conv1/Conv2D_grad/tuple/control_dependencyIdentity7gradients/model_1/conv1/Conv2D_grad/Conv2DBackpropInput5^gradients/model_1/conv1/Conv2D_grad/tuple/group_deps*
T0*J
_class@
><loc:@gradients/model_1/conv1/Conv2D_grad/Conv2DBackpropInput*/
_output_shapes
:���������

�
>gradients/model_1/conv1/Conv2D_grad/tuple/control_dependency_1Identity8gradients/model_1/conv1/Conv2D_grad/Conv2DBackpropFilter5^gradients/model_1/conv1/Conv2D_grad/tuple/group_deps*
T0*K
_classA
?=loc:@gradients/model_1/conv1/Conv2D_grad/Conv2DBackpropFilter*&
_output_shapes
: 
�
(gradients/model/conv1/Conv2D_grad/ShapeNShapeNanchor_inputconv1/weights/read*
T0*
out_type0*
N* 
_output_shapes
::
�
5gradients/model/conv1/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput(gradients/model/conv1/Conv2D_grad/ShapeNconv1/weights/read;gradients/model/conv1/BiasAdd_grad/tuple/control_dependency*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*/
_output_shapes
:���������
*
	dilations
*
T0
�
6gradients/model/conv1/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilteranchor_input*gradients/model/conv1/Conv2D_grad/ShapeN:1;gradients/model/conv1/BiasAdd_grad/tuple/control_dependency*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*&
_output_shapes
: 
�
2gradients/model/conv1/Conv2D_grad/tuple/group_depsNoOp7^gradients/model/conv1/Conv2D_grad/Conv2DBackpropFilter6^gradients/model/conv1/Conv2D_grad/Conv2DBackpropInput
�
:gradients/model/conv1/Conv2D_grad/tuple/control_dependencyIdentity5gradients/model/conv1/Conv2D_grad/Conv2DBackpropInput3^gradients/model/conv1/Conv2D_grad/tuple/group_deps*
T0*H
_class>
<:loc:@gradients/model/conv1/Conv2D_grad/Conv2DBackpropInput*/
_output_shapes
:���������

�
<gradients/model/conv1/Conv2D_grad/tuple/control_dependency_1Identity6gradients/model/conv1/Conv2D_grad/Conv2DBackpropFilter3^gradients/model/conv1/Conv2D_grad/tuple/group_deps*
T0*I
_class?
=;loc:@gradients/model/conv1/Conv2D_grad/Conv2DBackpropFilter*&
_output_shapes
: 
�
*gradients/model_2/conv1/Conv2D_grad/ShapeNShapeNnegative_inputconv1/weights/read*
T0*
out_type0*
N* 
_output_shapes
::
�
7gradients/model_2/conv1/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput*gradients/model_2/conv1/Conv2D_grad/ShapeNconv1/weights/read=gradients/model_2/conv1/BiasAdd_grad/tuple/control_dependency*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*/
_output_shapes
:���������

�
8gradients/model_2/conv1/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilternegative_input,gradients/model_2/conv1/Conv2D_grad/ShapeN:1=gradients/model_2/conv1/BiasAdd_grad/tuple/control_dependency*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*&
_output_shapes
: *
	dilations

�
4gradients/model_2/conv1/Conv2D_grad/tuple/group_depsNoOp9^gradients/model_2/conv1/Conv2D_grad/Conv2DBackpropFilter8^gradients/model_2/conv1/Conv2D_grad/Conv2DBackpropInput
�
<gradients/model_2/conv1/Conv2D_grad/tuple/control_dependencyIdentity7gradients/model_2/conv1/Conv2D_grad/Conv2DBackpropInput5^gradients/model_2/conv1/Conv2D_grad/tuple/group_deps*
T0*J
_class@
><loc:@gradients/model_2/conv1/Conv2D_grad/Conv2DBackpropInput*/
_output_shapes
:���������

�
>gradients/model_2/conv1/Conv2D_grad/tuple/control_dependency_1Identity8gradients/model_2/conv1/Conv2D_grad/Conv2DBackpropFilter5^gradients/model_2/conv1/Conv2D_grad/tuple/group_deps*
T0*K
_classA
?=loc:@gradients/model_2/conv1/Conv2D_grad/Conv2DBackpropFilter*&
_output_shapes
: 
�
gradients/AddN_9AddN?gradients/model_1/conv1/BiasAdd_grad/tuple/control_dependency_1=gradients/model/conv1/BiasAdd_grad/tuple/control_dependency_1?gradients/model_2/conv1/BiasAdd_grad/tuple/control_dependency_1*
T0*C
_class9
75loc:@gradients/model_1/conv1/BiasAdd_grad/BiasAddGrad*
N*
_output_shapes
: 
�
gradients/AddN_10AddN>gradients/model_1/conv1/Conv2D_grad/tuple/control_dependency_1<gradients/model/conv1/Conv2D_grad/tuple/control_dependency_1>gradients/model_2/conv1/Conv2D_grad/tuple/control_dependency_1*
T0*K
_classA
?=loc:@gradients/model_1/conv1/Conv2D_grad/Conv2DBackpropFilter*
N*&
_output_shapes
: 
�
8conv1/weights/Momentum/Initializer/zeros/shape_as_tensorConst* 
_class
loc:@conv1/weights*%
valueB"             *
dtype0*
_output_shapes
:
�
.conv1/weights/Momentum/Initializer/zeros/ConstConst* 
_class
loc:@conv1/weights*
valueB
 *    *
dtype0*
_output_shapes
: 
�
(conv1/weights/Momentum/Initializer/zerosFill8conv1/weights/Momentum/Initializer/zeros/shape_as_tensor.conv1/weights/Momentum/Initializer/zeros/Const*
T0* 
_class
loc:@conv1/weights*

index_type0*&
_output_shapes
: 
�
conv1/weights/Momentum
VariableV2*
	container *
shape: *
dtype0*&
_output_shapes
: *
shared_name * 
_class
loc:@conv1/weights
�
conv1/weights/Momentum/AssignAssignconv1/weights/Momentum(conv1/weights/Momentum/Initializer/zeros*
T0* 
_class
loc:@conv1/weights*
validate_shape(*&
_output_shapes
: *
use_locking(
�
conv1/weights/Momentum/readIdentityconv1/weights/Momentum*
T0* 
_class
loc:@conv1/weights*&
_output_shapes
: 
�
'conv1/biases/Momentum/Initializer/zerosConst*
_class
loc:@conv1/biases*
valueB *    *
dtype0*
_output_shapes
: 
�
conv1/biases/Momentum
VariableV2*
dtype0*
_output_shapes
: *
shared_name *
_class
loc:@conv1/biases*
	container *
shape: 
�
conv1/biases/Momentum/AssignAssignconv1/biases/Momentum'conv1/biases/Momentum/Initializer/zeros*
T0*
_class
loc:@conv1/biases*
validate_shape(*
_output_shapes
: *
use_locking(
�
conv1/biases/Momentum/readIdentityconv1/biases/Momentum*
T0*
_class
loc:@conv1/biases*
_output_shapes
: 
�
8conv2/weights/Momentum/Initializer/zeros/shape_as_tensorConst* 
_class
loc:@conv2/weights*%
valueB"          @   *
dtype0*
_output_shapes
:
�
.conv2/weights/Momentum/Initializer/zeros/ConstConst* 
_class
loc:@conv2/weights*
valueB
 *    *
dtype0*
_output_shapes
: 
�
(conv2/weights/Momentum/Initializer/zerosFill8conv2/weights/Momentum/Initializer/zeros/shape_as_tensor.conv2/weights/Momentum/Initializer/zeros/Const*
T0* 
_class
loc:@conv2/weights*

index_type0*&
_output_shapes
: @
�
conv2/weights/Momentum
VariableV2*
dtype0*&
_output_shapes
: @*
shared_name * 
_class
loc:@conv2/weights*
	container *
shape: @
�
conv2/weights/Momentum/AssignAssignconv2/weights/Momentum(conv2/weights/Momentum/Initializer/zeros*
use_locking(*
T0* 
_class
loc:@conv2/weights*
validate_shape(*&
_output_shapes
: @
�
conv2/weights/Momentum/readIdentityconv2/weights/Momentum*
T0* 
_class
loc:@conv2/weights*&
_output_shapes
: @
�
'conv2/biases/Momentum/Initializer/zerosConst*
_class
loc:@conv2/biases*
valueB@*    *
dtype0*
_output_shapes
:@
�
conv2/biases/Momentum
VariableV2*
	container *
shape:@*
dtype0*
_output_shapes
:@*
shared_name *
_class
loc:@conv2/biases
�
conv2/biases/Momentum/AssignAssignconv2/biases/Momentum'conv2/biases/Momentum/Initializer/zeros*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*
_class
loc:@conv2/biases
�
conv2/biases/Momentum/readIdentityconv2/biases/Momentum*
T0*
_class
loc:@conv2/biases*
_output_shapes
:@
�
8conv3/weights/Momentum/Initializer/zeros/shape_as_tensorConst* 
_class
loc:@conv3/weights*%
valueB"      @   �   *
dtype0*
_output_shapes
:
�
.conv3/weights/Momentum/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: * 
_class
loc:@conv3/weights*
valueB
 *    
�
(conv3/weights/Momentum/Initializer/zerosFill8conv3/weights/Momentum/Initializer/zeros/shape_as_tensor.conv3/weights/Momentum/Initializer/zeros/Const*
T0* 
_class
loc:@conv3/weights*

index_type0*'
_output_shapes
:@�
�
conv3/weights/Momentum
VariableV2*
	container *
shape:@�*
dtype0*'
_output_shapes
:@�*
shared_name * 
_class
loc:@conv3/weights
�
conv3/weights/Momentum/AssignAssignconv3/weights/Momentum(conv3/weights/Momentum/Initializer/zeros*
T0* 
_class
loc:@conv3/weights*
validate_shape(*'
_output_shapes
:@�*
use_locking(
�
conv3/weights/Momentum/readIdentityconv3/weights/Momentum*
T0* 
_class
loc:@conv3/weights*'
_output_shapes
:@�
�
'conv3/biases/Momentum/Initializer/zerosConst*
_class
loc:@conv3/biases*
valueB�*    *
dtype0*
_output_shapes	
:�
�
conv3/biases/Momentum
VariableV2*
shared_name *
_class
loc:@conv3/biases*
	container *
shape:�*
dtype0*
_output_shapes	
:�
�
conv3/biases/Momentum/AssignAssignconv3/biases/Momentum'conv3/biases/Momentum/Initializer/zeros*
use_locking(*
T0*
_class
loc:@conv3/biases*
validate_shape(*
_output_shapes	
:�
�
conv3/biases/Momentum/readIdentityconv3/biases/Momentum*
T0*
_class
loc:@conv3/biases*
_output_shapes	
:�
�
8conv4/weights/Momentum/Initializer/zeros/shape_as_tensorConst* 
_class
loc:@conv4/weights*%
valueB"      �      *
dtype0*
_output_shapes
:
�
.conv4/weights/Momentum/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: * 
_class
loc:@conv4/weights*
valueB
 *    
�
(conv4/weights/Momentum/Initializer/zerosFill8conv4/weights/Momentum/Initializer/zeros/shape_as_tensor.conv4/weights/Momentum/Initializer/zeros/Const*(
_output_shapes
:��*
T0* 
_class
loc:@conv4/weights*

index_type0
�
conv4/weights/Momentum
VariableV2*
dtype0*(
_output_shapes
:��*
shared_name * 
_class
loc:@conv4/weights*
	container *
shape:��
�
conv4/weights/Momentum/AssignAssignconv4/weights/Momentum(conv4/weights/Momentum/Initializer/zeros*
T0* 
_class
loc:@conv4/weights*
validate_shape(*(
_output_shapes
:��*
use_locking(
�
conv4/weights/Momentum/readIdentityconv4/weights/Momentum*
T0* 
_class
loc:@conv4/weights*(
_output_shapes
:��
�
'conv4/biases/Momentum/Initializer/zerosConst*
dtype0*
_output_shapes	
:�*
_class
loc:@conv4/biases*
valueB�*    
�
conv4/biases/Momentum
VariableV2*
shared_name *
_class
loc:@conv4/biases*
	container *
shape:�*
dtype0*
_output_shapes	
:�
�
conv4/biases/Momentum/AssignAssignconv4/biases/Momentum'conv4/biases/Momentum/Initializer/zeros*
T0*
_class
loc:@conv4/biases*
validate_shape(*
_output_shapes	
:�*
use_locking(
�
conv4/biases/Momentum/readIdentityconv4/biases/Momentum*
_output_shapes	
:�*
T0*
_class
loc:@conv4/biases
�
8conv5/weights/Momentum/Initializer/zeros/shape_as_tensorConst* 
_class
loc:@conv5/weights*%
valueB"            *
dtype0*
_output_shapes
:
�
.conv5/weights/Momentum/Initializer/zeros/ConstConst* 
_class
loc:@conv5/weights*
valueB
 *    *
dtype0*
_output_shapes
: 
�
(conv5/weights/Momentum/Initializer/zerosFill8conv5/weights/Momentum/Initializer/zeros/shape_as_tensor.conv5/weights/Momentum/Initializer/zeros/Const*
T0* 
_class
loc:@conv5/weights*

index_type0*'
_output_shapes
:�
�
conv5/weights/Momentum
VariableV2*
dtype0*'
_output_shapes
:�*
shared_name * 
_class
loc:@conv5/weights*
	container *
shape:�
�
conv5/weights/Momentum/AssignAssignconv5/weights/Momentum(conv5/weights/Momentum/Initializer/zeros*
T0* 
_class
loc:@conv5/weights*
validate_shape(*'
_output_shapes
:�*
use_locking(
�
conv5/weights/Momentum/readIdentityconv5/weights/Momentum*
T0* 
_class
loc:@conv5/weights*'
_output_shapes
:�
�
'conv5/biases/Momentum/Initializer/zerosConst*
_class
loc:@conv5/biases*
valueB*    *
dtype0*
_output_shapes
:
�
conv5/biases/Momentum
VariableV2*
shape:*
dtype0*
_output_shapes
:*
shared_name *
_class
loc:@conv5/biases*
	container 
�
conv5/biases/Momentum/AssignAssignconv5/biases/Momentum'conv5/biases/Momentum/Initializer/zeros*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*
_class
loc:@conv5/biases
�
conv5/biases/Momentum/readIdentityconv5/biases/Momentum*
T0*
_class
loc:@conv5/biases*
_output_shapes
:
[
Momentum/learning_rateConst*
dtype0*
_output_shapes
: *
valueB
 *
�#<
V
Momentum/momentumConst*
valueB
 *�p}?*
dtype0*
_output_shapes
: 
�
+Momentum/update_conv1/weights/ApplyMomentumApplyMomentumconv1/weightsconv1/weights/MomentumMomentum/learning_rategradients/AddN_10Momentum/momentum*
T0* 
_class
loc:@conv1/weights*
use_nesterov(*&
_output_shapes
: *
use_locking( 
�
*Momentum/update_conv1/biases/ApplyMomentumApplyMomentumconv1/biasesconv1/biases/MomentumMomentum/learning_rategradients/AddN_9Momentum/momentum*
T0*
_class
loc:@conv1/biases*
use_nesterov(*
_output_shapes
: *
use_locking( 
�
+Momentum/update_conv2/weights/ApplyMomentumApplyMomentumconv2/weightsconv2/weights/MomentumMomentum/learning_rategradients/AddN_8Momentum/momentum*
use_locking( *
T0* 
_class
loc:@conv2/weights*
use_nesterov(*&
_output_shapes
: @
�
*Momentum/update_conv2/biases/ApplyMomentumApplyMomentumconv2/biasesconv2/biases/MomentumMomentum/learning_rategradients/AddN_7Momentum/momentum*
use_locking( *
T0*
_class
loc:@conv2/biases*
use_nesterov(*
_output_shapes
:@
�
+Momentum/update_conv3/weights/ApplyMomentumApplyMomentumconv3/weightsconv3/weights/MomentumMomentum/learning_rategradients/AddN_6Momentum/momentum*
T0* 
_class
loc:@conv3/weights*
use_nesterov(*'
_output_shapes
:@�*
use_locking( 
�
*Momentum/update_conv3/biases/ApplyMomentumApplyMomentumconv3/biasesconv3/biases/MomentumMomentum/learning_rategradients/AddN_5Momentum/momentum*
use_nesterov(*
_output_shapes	
:�*
use_locking( *
T0*
_class
loc:@conv3/biases
�
+Momentum/update_conv4/weights/ApplyMomentumApplyMomentumconv4/weightsconv4/weights/MomentumMomentum/learning_rategradients/AddN_4Momentum/momentum*
use_locking( *
T0* 
_class
loc:@conv4/weights*
use_nesterov(*(
_output_shapes
:��
�
*Momentum/update_conv4/biases/ApplyMomentumApplyMomentumconv4/biasesconv4/biases/MomentumMomentum/learning_rategradients/AddN_3Momentum/momentum*
use_nesterov(*
_output_shapes	
:�*
use_locking( *
T0*
_class
loc:@conv4/biases
�
+Momentum/update_conv5/weights/ApplyMomentumApplyMomentumconv5/weightsconv5/weights/MomentumMomentum/learning_rategradients/AddN_2Momentum/momentum*
use_locking( *
T0* 
_class
loc:@conv5/weights*
use_nesterov(*'
_output_shapes
:�
�
*Momentum/update_conv5/biases/ApplyMomentumApplyMomentumconv5/biasesconv5/biases/MomentumMomentum/learning_rategradients/AddN_1Momentum/momentum*
T0*
_class
loc:@conv5/biases*
use_nesterov(*
_output_shapes
:*
use_locking( 
�
Momentum/updateNoOp+^Momentum/update_conv1/biases/ApplyMomentum,^Momentum/update_conv1/weights/ApplyMomentum+^Momentum/update_conv2/biases/ApplyMomentum,^Momentum/update_conv2/weights/ApplyMomentum+^Momentum/update_conv3/biases/ApplyMomentum,^Momentum/update_conv3/weights/ApplyMomentum+^Momentum/update_conv4/biases/ApplyMomentum,^Momentum/update_conv4/weights/ApplyMomentum+^Momentum/update_conv5/biases/ApplyMomentum,^Momentum/update_conv5/weights/ApplyMomentum

Momentum/valueConst^Momentum/update*
dtype0*
_output_shapes
: *
value	B :*
_class
loc:@Variable
�
Momentum	AssignAddVariableMomentum/value*
use_locking( *
T0*
_class
loc:@Variable*
_output_shapes
: 
Y
save/filename/inputConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
n
save/filenamePlaceholderWithDefaultsave/filename/input*
dtype0*
_output_shapes
: *
shape: 
e

save/ConstPlaceholderWithDefaultsave/filename*
dtype0*
_output_shapes
: *
shape: 
�
save/SaveV2/tensor_namesConst*�
value�B�BVariableBconv1/biasesBconv1/biases/MomentumBconv1/weightsBconv1/weights/MomentumBconv2/biasesBconv2/biases/MomentumBconv2/weightsBconv2/weights/MomentumBconv3/biasesBconv3/biases/MomentumBconv3/weightsBconv3/weights/MomentumBconv4/biasesBconv4/biases/MomentumBconv4/weightsBconv4/weights/MomentumBconv5/biasesBconv5/biases/MomentumBconv5/weightsBconv5/weights/Momentum*
dtype0*
_output_shapes
:
�
save/SaveV2/shape_and_slicesConst*
dtype0*
_output_shapes
:*=
value4B2B B B B B B B B B B B B B B B B B B B B B 
�
save/SaveV2SaveV2
save/Constsave/SaveV2/tensor_namessave/SaveV2/shape_and_slicesVariableconv1/biasesconv1/biases/Momentumconv1/weightsconv1/weights/Momentumconv2/biasesconv2/biases/Momentumconv2/weightsconv2/weights/Momentumconv3/biasesconv3/biases/Momentumconv3/weightsconv3/weights/Momentumconv4/biasesconv4/biases/Momentumconv4/weightsconv4/weights/Momentumconv5/biasesconv5/biases/Momentumconv5/weightsconv5/weights/Momentum*#
dtypes
2
}
save/control_dependencyIdentity
save/Const^save/SaveV2*
_output_shapes
: *
T0*
_class
loc:@save/Const
�
save/RestoreV2/tensor_namesConst"/device:CPU:0*�
value�B�BVariableBconv1/biasesBconv1/biases/MomentumBconv1/weightsBconv1/weights/MomentumBconv2/biasesBconv2/biases/MomentumBconv2/weightsBconv2/weights/MomentumBconv3/biasesBconv3/biases/MomentumBconv3/weightsBconv3/weights/MomentumBconv4/biasesBconv4/biases/MomentumBconv4/weightsBconv4/weights/MomentumBconv5/biasesBconv5/biases/MomentumBconv5/weightsBconv5/weights/Momentum*
dtype0*
_output_shapes
:
�
save/RestoreV2/shape_and_slicesConst"/device:CPU:0*=
value4B2B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:
�
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices"/device:CPU:0*h
_output_shapesV
T:::::::::::::::::::::*#
dtypes
2
�
save/AssignAssignVariablesave/RestoreV2*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*
_class
loc:@Variable
�
save/Assign_1Assignconv1/biasessave/RestoreV2:1*
use_locking(*
T0*
_class
loc:@conv1/biases*
validate_shape(*
_output_shapes
: 
�
save/Assign_2Assignconv1/biases/Momentumsave/RestoreV2:2*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*
_class
loc:@conv1/biases
�
save/Assign_3Assignconv1/weightssave/RestoreV2:3*
use_locking(*
T0* 
_class
loc:@conv1/weights*
validate_shape(*&
_output_shapes
: 
�
save/Assign_4Assignconv1/weights/Momentumsave/RestoreV2:4*
use_locking(*
T0* 
_class
loc:@conv1/weights*
validate_shape(*&
_output_shapes
: 
�
save/Assign_5Assignconv2/biasessave/RestoreV2:5*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*
_class
loc:@conv2/biases
�
save/Assign_6Assignconv2/biases/Momentumsave/RestoreV2:6*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*
_class
loc:@conv2/biases
�
save/Assign_7Assignconv2/weightssave/RestoreV2:7*
T0* 
_class
loc:@conv2/weights*
validate_shape(*&
_output_shapes
: @*
use_locking(
�
save/Assign_8Assignconv2/weights/Momentumsave/RestoreV2:8*
validate_shape(*&
_output_shapes
: @*
use_locking(*
T0* 
_class
loc:@conv2/weights
�
save/Assign_9Assignconv3/biasessave/RestoreV2:9*
T0*
_class
loc:@conv3/biases*
validate_shape(*
_output_shapes	
:�*
use_locking(
�
save/Assign_10Assignconv3/biases/Momentumsave/RestoreV2:10*
use_locking(*
T0*
_class
loc:@conv3/biases*
validate_shape(*
_output_shapes	
:�
�
save/Assign_11Assignconv3/weightssave/RestoreV2:11*
use_locking(*
T0* 
_class
loc:@conv3/weights*
validate_shape(*'
_output_shapes
:@�
�
save/Assign_12Assignconv3/weights/Momentumsave/RestoreV2:12*
use_locking(*
T0* 
_class
loc:@conv3/weights*
validate_shape(*'
_output_shapes
:@�
�
save/Assign_13Assignconv4/biasessave/RestoreV2:13*
T0*
_class
loc:@conv4/biases*
validate_shape(*
_output_shapes	
:�*
use_locking(
�
save/Assign_14Assignconv4/biases/Momentumsave/RestoreV2:14*
use_locking(*
T0*
_class
loc:@conv4/biases*
validate_shape(*
_output_shapes	
:�
�
save/Assign_15Assignconv4/weightssave/RestoreV2:15*
validate_shape(*(
_output_shapes
:��*
use_locking(*
T0* 
_class
loc:@conv4/weights
�
save/Assign_16Assignconv4/weights/Momentumsave/RestoreV2:16*
validate_shape(*(
_output_shapes
:��*
use_locking(*
T0* 
_class
loc:@conv4/weights
�
save/Assign_17Assignconv5/biasessave/RestoreV2:17*
T0*
_class
loc:@conv5/biases*
validate_shape(*
_output_shapes
:*
use_locking(
�
save/Assign_18Assignconv5/biases/Momentumsave/RestoreV2:18*
use_locking(*
T0*
_class
loc:@conv5/biases*
validate_shape(*
_output_shapes
:
�
save/Assign_19Assignconv5/weightssave/RestoreV2:19*
use_locking(*
T0* 
_class
loc:@conv5/weights*
validate_shape(*'
_output_shapes
:�
�
save/Assign_20Assignconv5/weights/Momentumsave/RestoreV2:20*
use_locking(*
T0* 
_class
loc:@conv5/weights*
validate_shape(*'
_output_shapes
:�
�
save/restore_allNoOp^save/Assign^save/Assign_1^save/Assign_10^save/Assign_11^save/Assign_12^save/Assign_13^save/Assign_14^save/Assign_15^save/Assign_16^save/Assign_17^save/Assign_18^save/Assign_19^save/Assign_2^save/Assign_20^save/Assign_3^save/Assign_4^save/Assign_5^save/Assign_6^save/Assign_7^save/Assign_8^save/Assign_9
�
initNoOp^Variable/Assign^conv1/biases/Assign^conv1/biases/Momentum/Assign^conv1/weights/Assign^conv1/weights/Momentum/Assign^conv2/biases/Assign^conv2/biases/Momentum/Assign^conv2/weights/Assign^conv2/weights/Momentum/Assign^conv3/biases/Assign^conv3/biases/Momentum/Assign^conv3/weights/Assign^conv3/weights/Momentum/Assign^conv4/biases/Assign^conv4/biases/Momentum/Assign^conv4/weights/Assign^conv4/weights/Momentum/Assign^conv5/biases/Assign^conv5/biases/Momentum/Assign^conv5/weights/Assign^conv5/weights/Momentum/Assign
N
	step/tagsConst*
valueB
 Bstep*
dtype0*
_output_shapes
: 
P
stepScalarSummary	step/tagsVariable/read*
_output_shapes
: *
T0
N
	loss/tagsConst*
valueB
 Bloss*
dtype0*
_output_shapes
: 
G
lossScalarSummary	loss/tagsMean*
T0*
_output_shapes
: 
c
conv1/weights_1/tagConst* 
valueB Bconv1/weights_1*
dtype0*
_output_shapes
: 
m
conv1/weights_1HistogramSummaryconv1/weights_1/tagconv1/weights/read*
T0*
_output_shapes
: 
a
conv1/biases_1/tagConst*
valueB Bconv1/biases_1*
dtype0*
_output_shapes
: 
j
conv1/biases_1HistogramSummaryconv1/biases_1/tagconv1/biases/read*
_output_shapes
: *
T0
c
conv2/weights_1/tagConst*
dtype0*
_output_shapes
: * 
valueB Bconv2/weights_1
m
conv2/weights_1HistogramSummaryconv2/weights_1/tagconv2/weights/read*
T0*
_output_shapes
: 
a
conv2/biases_1/tagConst*
dtype0*
_output_shapes
: *
valueB Bconv2/biases_1
j
conv2/biases_1HistogramSummaryconv2/biases_1/tagconv2/biases/read*
T0*
_output_shapes
: 
c
conv3/weights_1/tagConst* 
valueB Bconv3/weights_1*
dtype0*
_output_shapes
: 
m
conv3/weights_1HistogramSummaryconv3/weights_1/tagconv3/weights/read*
_output_shapes
: *
T0
a
conv3/biases_1/tagConst*
dtype0*
_output_shapes
: *
valueB Bconv3/biases_1
j
conv3/biases_1HistogramSummaryconv3/biases_1/tagconv3/biases/read*
T0*
_output_shapes
: 
c
conv4/weights_1/tagConst* 
valueB Bconv4/weights_1*
dtype0*
_output_shapes
: 
m
conv4/weights_1HistogramSummaryconv4/weights_1/tagconv4/weights/read*
_output_shapes
: *
T0
a
conv4/biases_1/tagConst*
valueB Bconv4/biases_1*
dtype0*
_output_shapes
: 
j
conv4/biases_1HistogramSummaryconv4/biases_1/tagconv4/biases/read*
_output_shapes
: *
T0
c
conv5/weights_1/tagConst* 
valueB Bconv5/weights_1*
dtype0*
_output_shapes
: 
m
conv5/weights_1HistogramSummaryconv5/weights_1/tagconv5/weights/read*
_output_shapes
: *
T0
a
conv5/biases_1/tagConst*
valueB Bconv5/biases_1*
dtype0*
_output_shapes
: 
j
conv5/biases_1HistogramSummaryconv5/biases_1/tagconv5/biases/read*
T0*
_output_shapes
: 
�
Merge/MergeSummaryMergeSummarysteplossconv1/weights_1conv1/biases_1conv2/weights_1conv2/biases_1conv3/weights_1conv3/biases_1conv4/weights_1conv4/biases_1conv5/weights_1conv5/biases_1*
N*
_output_shapes
: "Kٲ�(     ��{	3�`/���AJ��
�,�,
:
Add
x"T
y"T
z"T"
Ttype:
2	
W
AddN
inputs"T*N
sum"T"
Nint(0"!
Ttype:
2	��
�
ApplyMomentum
var"T�
accum"T�
lr"T	
grad"T
momentum"T
out"T�" 
Ttype:
2	"
use_lockingbool( "
use_nesterovbool( 
x
Assign
ref"T�

value"T

output_ref"T�"	
Ttype"
validate_shapebool("
use_lockingbool(�
s
	AssignAdd
ref"T�

value"T

output_ref"T�" 
Ttype:
2	"
use_lockingbool( 
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
~
BiasAddGrad
out_backprop"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
R
BroadcastGradientArgs
s0"T
s1"T
r0"T
r1"T"
Ttype0:
2	
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
8
Const
output"dtype"
valuetensor"
dtypetype
�
Conv2D

input"T
filter"T
output"T"
Ttype:
2"
strides	list(int)"
use_cudnn_on_gpubool(""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

�
Conv2DBackpropFilter

input"T
filter_sizes
out_backprop"T
output"T"
Ttype:
2"
strides	list(int)"
use_cudnn_on_gpubool(""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

�
Conv2DBackpropInput
input_sizes
filter"T
out_backprop"T
output"T"
Ttype:
2"
strides	list(int)"
use_cudnn_on_gpubool(""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

S
DynamicStitch
indices*N
data"T*N
merged"T"
Nint(0"	
Ttype
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
?
FloorDiv
x"T
y"T
z"T"
Ttype:
2	
9
FloorMod
x"T
y"T
z"T"
Ttype:

2	
=
Greater
x"T
y"T
z
"
Ttype:
2	
B
GreaterEqual
x"T
y"T
z
"
Ttype:
2	
V
HistogramSummary
tag
values"T
summary"
Ttype0:
2	
.
Identity

input"T
output"T"	
Ttype
,
Log
x"T
y"T"
Ttype:

2
�
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0""
paddingstring:
SAMEVALID":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
�
MaxPoolGrad

orig_input"T
orig_output"T	
grad"T
output"T"
ksize	list(int)(0"
strides	list(int)(0""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW"
Ttype0:
2	
;
Maximum
x"T
y"T
z"T"
Ttype:

2	�
�
Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
8
MergeSummary
inputs*N
summary"
Nint(0
=
Mul
x"T
y"T
z"T"
Ttype:
2	�
.
Neg
x"T
y"T"
Ttype:

2	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
X
PlaceholderWithDefault
input"dtype
output"dtype"
dtypetype"
shapeshape
6
Pow
x"T
y"T
z"T"
Ttype:

2	
�
Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
~
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	�
a
Range
start"Tidx
limit"Tidx
delta"Tidx
output"Tidx"
Tidxtype0:	
2	
>
RealDiv
x"T
y"T
z"T"
Ttype:
2	
E
Relu
features"T
activations"T"
Ttype:
2	
V
ReluGrad
	gradients"T
features"T
	backprops"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
P
ScalarSummary
tags
values"T
summary"
Ttype:
2	
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
e
ShapeN
input"T*N
output"out_type*N"
Nint(0"	
Ttype"
out_typetype0:
2	
-
Sqrt
x"T
y"T"
Ttype:

2
:
SqrtGrad
y"T
dy"T
z"T"
Ttype:

2
�
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
:
Sub
x"T
y"T
z"T"
Ttype:
2	
�
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
c
Tile

input"T
	multiples"
Tmultiples
output"T"	
Ttype"

Tmultiplestype0:
2	
s

VariableV2
ref"dtype�"
shapeshape"
dtypetype"
	containerstring "
shared_namestring �
&
	ZerosLike
x"T
y"T"	
Ttype*1.13.12b'v1.13.0-rc2-5-g6612da8951'��

anchor_inputPlaceholder*
dtype0*/
_output_shapes
:���������
*$
shape:���������

�
positive_inputPlaceholder*
dtype0*/
_output_shapes
:���������
*$
shape:���������

�
negative_inputPlaceholder*
dtype0*/
_output_shapes
:���������
*$
shape:���������

�
.conv1/weights/Initializer/random_uniform/shapeConst* 
_class
loc:@conv1/weights*%
valueB"             *
dtype0*
_output_shapes
:
�
,conv1/weights/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: * 
_class
loc:@conv1/weights*
valueB
 *�Er�
�
,conv1/weights/Initializer/random_uniform/maxConst* 
_class
loc:@conv1/weights*
valueB
 *�Er=*
dtype0*
_output_shapes
: 
�
6conv1/weights/Initializer/random_uniform/RandomUniformRandomUniform.conv1/weights/Initializer/random_uniform/shape*
T0* 
_class
loc:@conv1/weights*
seed2 *
dtype0*&
_output_shapes
: *

seed 
�
,conv1/weights/Initializer/random_uniform/subSub,conv1/weights/Initializer/random_uniform/max,conv1/weights/Initializer/random_uniform/min*
T0* 
_class
loc:@conv1/weights*
_output_shapes
: 
�
,conv1/weights/Initializer/random_uniform/mulMul6conv1/weights/Initializer/random_uniform/RandomUniform,conv1/weights/Initializer/random_uniform/sub*&
_output_shapes
: *
T0* 
_class
loc:@conv1/weights
�
(conv1/weights/Initializer/random_uniformAdd,conv1/weights/Initializer/random_uniform/mul,conv1/weights/Initializer/random_uniform/min*
T0* 
_class
loc:@conv1/weights*&
_output_shapes
: 
�
conv1/weights
VariableV2*
	container *
shape: *
dtype0*&
_output_shapes
: *
shared_name * 
_class
loc:@conv1/weights
�
conv1/weights/AssignAssignconv1/weights(conv1/weights/Initializer/random_uniform*
validate_shape(*&
_output_shapes
: *
use_locking(*
T0* 
_class
loc:@conv1/weights
�
conv1/weights/readIdentityconv1/weights*
T0* 
_class
loc:@conv1/weights*&
_output_shapes
: 
�
conv1/biases/Initializer/zerosConst*
_class
loc:@conv1/biases*
valueB *    *
dtype0*
_output_shapes
: 
�
conv1/biases
VariableV2*
shared_name *
_class
loc:@conv1/biases*
	container *
shape: *
dtype0*
_output_shapes
: 
�
conv1/biases/AssignAssignconv1/biasesconv1/biases/Initializer/zeros*
use_locking(*
T0*
_class
loc:@conv1/biases*
validate_shape(*
_output_shapes
: 
q
conv1/biases/readIdentityconv1/biases*
T0*
_class
loc:@conv1/biases*
_output_shapes
: 
j
model/conv1/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
�
model/conv1/Conv2DConv2Danchor_inputconv1/weights/read*
paddingSAME*/
_output_shapes
:���������
 *
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
�
model/conv1/BiasAddBiasAddmodel/conv1/Conv2Dconv1/biases/read*
T0*
data_formatNHWC*/
_output_shapes
:���������
 
m
model/conv1/conv1/ReluRelumodel/conv1/BiasAdd*
T0*/
_output_shapes
:���������
 
�
model/conv1/MaxPool2D/MaxPoolMaxPoolmodel/conv1/conv1/Relu*
T0*
data_formatNHWC*
strides
*
ksize
*
paddingSAME*/
_output_shapes
:���������
 
�
.conv2/weights/Initializer/random_uniform/shapeConst* 
_class
loc:@conv2/weights*%
valueB"          @   *
dtype0*
_output_shapes
:
�
,conv2/weights/Initializer/random_uniform/minConst* 
_class
loc:@conv2/weights*
valueB
 *��L�*
dtype0*
_output_shapes
: 
�
,conv2/weights/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: * 
_class
loc:@conv2/weights*
valueB
 *��L=
�
6conv2/weights/Initializer/random_uniform/RandomUniformRandomUniform.conv2/weights/Initializer/random_uniform/shape*
T0* 
_class
loc:@conv2/weights*
seed2 *
dtype0*&
_output_shapes
: @*

seed 
�
,conv2/weights/Initializer/random_uniform/subSub,conv2/weights/Initializer/random_uniform/max,conv2/weights/Initializer/random_uniform/min*
T0* 
_class
loc:@conv2/weights*
_output_shapes
: 
�
,conv2/weights/Initializer/random_uniform/mulMul6conv2/weights/Initializer/random_uniform/RandomUniform,conv2/weights/Initializer/random_uniform/sub*&
_output_shapes
: @*
T0* 
_class
loc:@conv2/weights
�
(conv2/weights/Initializer/random_uniformAdd,conv2/weights/Initializer/random_uniform/mul,conv2/weights/Initializer/random_uniform/min*
T0* 
_class
loc:@conv2/weights*&
_output_shapes
: @
�
conv2/weights
VariableV2*
shared_name * 
_class
loc:@conv2/weights*
	container *
shape: @*
dtype0*&
_output_shapes
: @
�
conv2/weights/AssignAssignconv2/weights(conv2/weights/Initializer/random_uniform*
validate_shape(*&
_output_shapes
: @*
use_locking(*
T0* 
_class
loc:@conv2/weights
�
conv2/weights/readIdentityconv2/weights*
T0* 
_class
loc:@conv2/weights*&
_output_shapes
: @
�
conv2/biases/Initializer/zerosConst*
dtype0*
_output_shapes
:@*
_class
loc:@conv2/biases*
valueB@*    
�
conv2/biases
VariableV2*
dtype0*
_output_shapes
:@*
shared_name *
_class
loc:@conv2/biases*
	container *
shape:@
�
conv2/biases/AssignAssignconv2/biasesconv2/biases/Initializer/zeros*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*
_class
loc:@conv2/biases
q
conv2/biases/readIdentityconv2/biases*
T0*
_class
loc:@conv2/biases*
_output_shapes
:@
j
model/conv2/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
�
model/conv2/Conv2DConv2Dmodel/conv1/MaxPool2D/MaxPoolconv2/weights/read*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*/
_output_shapes
:���������
@*
	dilations

�
model/conv2/BiasAddBiasAddmodel/conv2/Conv2Dconv2/biases/read*
T0*
data_formatNHWC*/
_output_shapes
:���������
@
m
model/conv2/conv2/ReluRelumodel/conv2/BiasAdd*
T0*/
_output_shapes
:���������
@
�
model/conv2/MaxPool2D/MaxPoolMaxPoolmodel/conv2/conv2/Relu*
ksize
*
paddingSAME*/
_output_shapes
:���������@*
T0*
strides
*
data_formatNHWC
�
.conv3/weights/Initializer/random_uniform/shapeConst* 
_class
loc:@conv3/weights*%
valueB"      @   �   *
dtype0*
_output_shapes
:
�
,conv3/weights/Initializer/random_uniform/minConst* 
_class
loc:@conv3/weights*
valueB
 *�[q�*
dtype0*
_output_shapes
: 
�
,conv3/weights/Initializer/random_uniform/maxConst* 
_class
loc:@conv3/weights*
valueB
 *�[q=*
dtype0*
_output_shapes
: 
�
6conv3/weights/Initializer/random_uniform/RandomUniformRandomUniform.conv3/weights/Initializer/random_uniform/shape*
T0* 
_class
loc:@conv3/weights*
seed2 *
dtype0*'
_output_shapes
:@�*

seed 
�
,conv3/weights/Initializer/random_uniform/subSub,conv3/weights/Initializer/random_uniform/max,conv3/weights/Initializer/random_uniform/min*
T0* 
_class
loc:@conv3/weights*
_output_shapes
: 
�
,conv3/weights/Initializer/random_uniform/mulMul6conv3/weights/Initializer/random_uniform/RandomUniform,conv3/weights/Initializer/random_uniform/sub*
T0* 
_class
loc:@conv3/weights*'
_output_shapes
:@�
�
(conv3/weights/Initializer/random_uniformAdd,conv3/weights/Initializer/random_uniform/mul,conv3/weights/Initializer/random_uniform/min*
T0* 
_class
loc:@conv3/weights*'
_output_shapes
:@�
�
conv3/weights
VariableV2*
shared_name * 
_class
loc:@conv3/weights*
	container *
shape:@�*
dtype0*'
_output_shapes
:@�
�
conv3/weights/AssignAssignconv3/weights(conv3/weights/Initializer/random_uniform*
T0* 
_class
loc:@conv3/weights*
validate_shape(*'
_output_shapes
:@�*
use_locking(
�
conv3/weights/readIdentityconv3/weights*'
_output_shapes
:@�*
T0* 
_class
loc:@conv3/weights
�
conv3/biases/Initializer/zerosConst*
_class
loc:@conv3/biases*
valueB�*    *
dtype0*
_output_shapes	
:�
�
conv3/biases
VariableV2*
_class
loc:@conv3/biases*
	container *
shape:�*
dtype0*
_output_shapes	
:�*
shared_name 
�
conv3/biases/AssignAssignconv3/biasesconv3/biases/Initializer/zeros*
use_locking(*
T0*
_class
loc:@conv3/biases*
validate_shape(*
_output_shapes	
:�
r
conv3/biases/readIdentityconv3/biases*
T0*
_class
loc:@conv3/biases*
_output_shapes	
:�
j
model/conv3/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
�
model/conv3/Conv2DConv2Dmodel/conv2/MaxPool2D/MaxPoolconv3/weights/read*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*0
_output_shapes
:����������
�
model/conv3/BiasAddBiasAddmodel/conv3/Conv2Dconv3/biases/read*
T0*
data_formatNHWC*0
_output_shapes
:����������
n
model/conv3/conv3/ReluRelumodel/conv3/BiasAdd*
T0*0
_output_shapes
:����������
�
model/conv3/MaxPool2D/MaxPoolMaxPoolmodel/conv3/conv3/Relu*
ksize
*
paddingSAME*0
_output_shapes
:����������*
T0*
data_formatNHWC*
strides

�
.conv4/weights/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:* 
_class
loc:@conv4/weights*%
valueB"      �      
�
,conv4/weights/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: * 
_class
loc:@conv4/weights*
valueB
 *   �
�
,conv4/weights/Initializer/random_uniform/maxConst* 
_class
loc:@conv4/weights*
valueB
 *   >*
dtype0*
_output_shapes
: 
�
6conv4/weights/Initializer/random_uniform/RandomUniformRandomUniform.conv4/weights/Initializer/random_uniform/shape* 
_class
loc:@conv4/weights*
seed2 *
dtype0*(
_output_shapes
:��*

seed *
T0
�
,conv4/weights/Initializer/random_uniform/subSub,conv4/weights/Initializer/random_uniform/max,conv4/weights/Initializer/random_uniform/min*
T0* 
_class
loc:@conv4/weights*
_output_shapes
: 
�
,conv4/weights/Initializer/random_uniform/mulMul6conv4/weights/Initializer/random_uniform/RandomUniform,conv4/weights/Initializer/random_uniform/sub*(
_output_shapes
:��*
T0* 
_class
loc:@conv4/weights
�
(conv4/weights/Initializer/random_uniformAdd,conv4/weights/Initializer/random_uniform/mul,conv4/weights/Initializer/random_uniform/min*
T0* 
_class
loc:@conv4/weights*(
_output_shapes
:��
�
conv4/weights
VariableV2*
shape:��*
dtype0*(
_output_shapes
:��*
shared_name * 
_class
loc:@conv4/weights*
	container 
�
conv4/weights/AssignAssignconv4/weights(conv4/weights/Initializer/random_uniform*
T0* 
_class
loc:@conv4/weights*
validate_shape(*(
_output_shapes
:��*
use_locking(
�
conv4/weights/readIdentityconv4/weights*
T0* 
_class
loc:@conv4/weights*(
_output_shapes
:��
�
conv4/biases/Initializer/zerosConst*
_class
loc:@conv4/biases*
valueB�*    *
dtype0*
_output_shapes	
:�
�
conv4/biases
VariableV2*
dtype0*
_output_shapes	
:�*
shared_name *
_class
loc:@conv4/biases*
	container *
shape:�
�
conv4/biases/AssignAssignconv4/biasesconv4/biases/Initializer/zeros*
_output_shapes	
:�*
use_locking(*
T0*
_class
loc:@conv4/biases*
validate_shape(
r
conv4/biases/readIdentityconv4/biases*
T0*
_class
loc:@conv4/biases*
_output_shapes	
:�
j
model/conv4/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
�
model/conv4/Conv2DConv2Dmodel/conv3/MaxPool2D/MaxPoolconv4/weights/read*
paddingSAME*0
_output_shapes
:����������*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
�
model/conv4/BiasAddBiasAddmodel/conv4/Conv2Dconv4/biases/read*
T0*
data_formatNHWC*0
_output_shapes
:����������
n
model/conv4/conv4/ReluRelumodel/conv4/BiasAdd*
T0*0
_output_shapes
:����������
�
model/conv4/MaxPool2D/MaxPoolMaxPoolmodel/conv4/conv4/Relu*0
_output_shapes
:����������*
T0*
data_formatNHWC*
strides
*
ksize
*
paddingSAME
�
.conv5/weights/Initializer/random_uniform/shapeConst* 
_class
loc:@conv5/weights*%
valueB"            *
dtype0*
_output_shapes
:
�
,conv5/weights/Initializer/random_uniform/minConst* 
_class
loc:@conv5/weights*
valueB
 *���*
dtype0*
_output_shapes
: 
�
,conv5/weights/Initializer/random_uniform/maxConst* 
_class
loc:@conv5/weights*
valueB
 *��>*
dtype0*
_output_shapes
: 
�
6conv5/weights/Initializer/random_uniform/RandomUniformRandomUniform.conv5/weights/Initializer/random_uniform/shape*
seed2 *
dtype0*'
_output_shapes
:�*

seed *
T0* 
_class
loc:@conv5/weights
�
,conv5/weights/Initializer/random_uniform/subSub,conv5/weights/Initializer/random_uniform/max,conv5/weights/Initializer/random_uniform/min*
_output_shapes
: *
T0* 
_class
loc:@conv5/weights
�
,conv5/weights/Initializer/random_uniform/mulMul6conv5/weights/Initializer/random_uniform/RandomUniform,conv5/weights/Initializer/random_uniform/sub*
T0* 
_class
loc:@conv5/weights*'
_output_shapes
:�
�
(conv5/weights/Initializer/random_uniformAdd,conv5/weights/Initializer/random_uniform/mul,conv5/weights/Initializer/random_uniform/min* 
_class
loc:@conv5/weights*'
_output_shapes
:�*
T0
�
conv5/weights
VariableV2*
dtype0*'
_output_shapes
:�*
shared_name * 
_class
loc:@conv5/weights*
	container *
shape:�
�
conv5/weights/AssignAssignconv5/weights(conv5/weights/Initializer/random_uniform*
T0* 
_class
loc:@conv5/weights*
validate_shape(*'
_output_shapes
:�*
use_locking(
�
conv5/weights/readIdentityconv5/weights*
T0* 
_class
loc:@conv5/weights*'
_output_shapes
:�
�
conv5/biases/Initializer/zerosConst*
_class
loc:@conv5/biases*
valueB*    *
dtype0*
_output_shapes
:
�
conv5/biases
VariableV2*
dtype0*
_output_shapes
:*
shared_name *
_class
loc:@conv5/biases*
	container *
shape:
�
conv5/biases/AssignAssignconv5/biasesconv5/biases/Initializer/zeros*
use_locking(*
T0*
_class
loc:@conv5/biases*
validate_shape(*
_output_shapes
:
q
conv5/biases/readIdentityconv5/biases*
_class
loc:@conv5/biases*
_output_shapes
:*
T0
j
model/conv5/dilation_rateConst*
dtype0*
_output_shapes
:*
valueB"      
�
model/conv5/Conv2DConv2Dmodel/conv4/MaxPool2D/MaxPoolconv5/weights/read*/
_output_shapes
:���������*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME
�
model/conv5/BiasAddBiasAddmodel/conv5/Conv2Dconv5/biases/read*
T0*
data_formatNHWC*/
_output_shapes
:���������
�
model/conv5/MaxPool2D/MaxPoolMaxPoolmodel/conv5/BiasAdd*
T0*
strides
*
data_formatNHWC*
ksize
*
paddingSAME*/
_output_shapes
:���������
x
model/Flatten/flatten/ShapeShapemodel/conv5/MaxPool2D/MaxPool*
T0*
out_type0*
_output_shapes
:
s
)model/Flatten/flatten/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
u
+model/Flatten/flatten/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB:
u
+model/Flatten/flatten/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
�
#model/Flatten/flatten/strided_sliceStridedSlicemodel/Flatten/flatten/Shape)model/Flatten/flatten/strided_slice/stack+model/Flatten/flatten/strided_slice/stack_1+model/Flatten/flatten/strided_slice/stack_2*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
T0*
Index0
p
%model/Flatten/flatten/Reshape/shape/1Const*
_output_shapes
: *
valueB :
���������*
dtype0
�
#model/Flatten/flatten/Reshape/shapePack#model/Flatten/flatten/strided_slice%model/Flatten/flatten/Reshape/shape/1*
T0*

axis *
N*
_output_shapes
:
�
model/Flatten/flatten/ReshapeReshapemodel/conv5/MaxPool2D/MaxPool#model/Flatten/flatten/Reshape/shape*
T0*
Tshape0*'
_output_shapes
:���������
l
model_1/conv1/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
�
model_1/conv1/Conv2DConv2Dpositive_inputconv1/weights/read*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*/
_output_shapes
:���������
 
�
model_1/conv1/BiasAddBiasAddmodel_1/conv1/Conv2Dconv1/biases/read*/
_output_shapes
:���������
 *
T0*
data_formatNHWC
q
model_1/conv1/conv1/ReluRelumodel_1/conv1/BiasAdd*/
_output_shapes
:���������
 *
T0
�
model_1/conv1/MaxPool2D/MaxPoolMaxPoolmodel_1/conv1/conv1/Relu*/
_output_shapes
:���������
 *
T0*
strides
*
data_formatNHWC*
ksize
*
paddingSAME
l
model_1/conv2/dilation_rateConst*
dtype0*
_output_shapes
:*
valueB"      
�
model_1/conv2/Conv2DConv2Dmodel_1/conv1/MaxPool2D/MaxPoolconv2/weights/read*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*/
_output_shapes
:���������
@
�
model_1/conv2/BiasAddBiasAddmodel_1/conv2/Conv2Dconv2/biases/read*/
_output_shapes
:���������
@*
T0*
data_formatNHWC
q
model_1/conv2/conv2/ReluRelumodel_1/conv2/BiasAdd*/
_output_shapes
:���������
@*
T0
�
model_1/conv2/MaxPool2D/MaxPoolMaxPoolmodel_1/conv2/conv2/Relu*
T0*
data_formatNHWC*
strides
*
ksize
*
paddingSAME*/
_output_shapes
:���������@
l
model_1/conv3/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
�
model_1/conv3/Conv2DConv2Dmodel_1/conv2/MaxPool2D/MaxPoolconv3/weights/read*
use_cudnn_on_gpu(*
paddingSAME*0
_output_shapes
:����������*
	dilations
*
T0*
data_formatNHWC*
strides

�
model_1/conv3/BiasAddBiasAddmodel_1/conv3/Conv2Dconv3/biases/read*
T0*
data_formatNHWC*0
_output_shapes
:����������
r
model_1/conv3/conv3/ReluRelumodel_1/conv3/BiasAdd*0
_output_shapes
:����������*
T0
�
model_1/conv3/MaxPool2D/MaxPoolMaxPoolmodel_1/conv3/conv3/Relu*
T0*
data_formatNHWC*
strides
*
ksize
*
paddingSAME*0
_output_shapes
:����������
l
model_1/conv4/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
�
model_1/conv4/Conv2DConv2Dmodel_1/conv3/MaxPool2D/MaxPoolconv4/weights/read*0
_output_shapes
:����������*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME
�
model_1/conv4/BiasAddBiasAddmodel_1/conv4/Conv2Dconv4/biases/read*
data_formatNHWC*0
_output_shapes
:����������*
T0
r
model_1/conv4/conv4/ReluRelumodel_1/conv4/BiasAdd*
T0*0
_output_shapes
:����������
�
model_1/conv4/MaxPool2D/MaxPoolMaxPoolmodel_1/conv4/conv4/Relu*
ksize
*
paddingSAME*0
_output_shapes
:����������*
T0*
data_formatNHWC*
strides

l
model_1/conv5/dilation_rateConst*
_output_shapes
:*
valueB"      *
dtype0
�
model_1/conv5/Conv2DConv2Dmodel_1/conv4/MaxPool2D/MaxPoolconv5/weights/read*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*/
_output_shapes
:���������
�
model_1/conv5/BiasAddBiasAddmodel_1/conv5/Conv2Dconv5/biases/read*
T0*
data_formatNHWC*/
_output_shapes
:���������
�
model_1/conv5/MaxPool2D/MaxPoolMaxPoolmodel_1/conv5/BiasAdd*
T0*
data_formatNHWC*
strides
*
ksize
*
paddingSAME*/
_output_shapes
:���������
|
model_1/Flatten/flatten/ShapeShapemodel_1/conv5/MaxPool2D/MaxPool*
T0*
out_type0*
_output_shapes
:
u
+model_1/Flatten/flatten/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
w
-model_1/Flatten/flatten/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
w
-model_1/Flatten/flatten/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
�
%model_1/Flatten/flatten/strided_sliceStridedSlicemodel_1/Flatten/flatten/Shape+model_1/Flatten/flatten/strided_slice/stack-model_1/Flatten/flatten/strided_slice/stack_1-model_1/Flatten/flatten/strided_slice/stack_2*
Index0*
T0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: 
r
'model_1/Flatten/flatten/Reshape/shape/1Const*
valueB :
���������*
dtype0*
_output_shapes
: 
�
%model_1/Flatten/flatten/Reshape/shapePack%model_1/Flatten/flatten/strided_slice'model_1/Flatten/flatten/Reshape/shape/1*
_output_shapes
:*
T0*

axis *
N
�
model_1/Flatten/flatten/ReshapeReshapemodel_1/conv5/MaxPool2D/MaxPool%model_1/Flatten/flatten/Reshape/shape*
Tshape0*'
_output_shapes
:���������*
T0
l
model_2/conv1/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
�
model_2/conv1/Conv2DConv2Dnegative_inputconv1/weights/read*
paddingSAME*/
_output_shapes
:���������
 *
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(
�
model_2/conv1/BiasAddBiasAddmodel_2/conv1/Conv2Dconv1/biases/read*
data_formatNHWC*/
_output_shapes
:���������
 *
T0
q
model_2/conv1/conv1/ReluRelumodel_2/conv1/BiasAdd*/
_output_shapes
:���������
 *
T0
�
model_2/conv1/MaxPool2D/MaxPoolMaxPoolmodel_2/conv1/conv1/Relu*/
_output_shapes
:���������
 *
T0*
data_formatNHWC*
strides
*
ksize
*
paddingSAME
l
model_2/conv2/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
�
model_2/conv2/Conv2DConv2Dmodel_2/conv1/MaxPool2D/MaxPoolconv2/weights/read*
paddingSAME*/
_output_shapes
:���������
@*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(
�
model_2/conv2/BiasAddBiasAddmodel_2/conv2/Conv2Dconv2/biases/read*/
_output_shapes
:���������
@*
T0*
data_formatNHWC
q
model_2/conv2/conv2/ReluRelumodel_2/conv2/BiasAdd*/
_output_shapes
:���������
@*
T0
�
model_2/conv2/MaxPool2D/MaxPoolMaxPoolmodel_2/conv2/conv2/Relu*
T0*
data_formatNHWC*
strides
*
ksize
*
paddingSAME*/
_output_shapes
:���������@
l
model_2/conv3/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
�
model_2/conv3/Conv2DConv2Dmodel_2/conv2/MaxPool2D/MaxPoolconv3/weights/read*
paddingSAME*0
_output_shapes
:����������*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
�
model_2/conv3/BiasAddBiasAddmodel_2/conv3/Conv2Dconv3/biases/read*
data_formatNHWC*0
_output_shapes
:����������*
T0
r
model_2/conv3/conv3/ReluRelumodel_2/conv3/BiasAdd*
T0*0
_output_shapes
:����������
�
model_2/conv3/MaxPool2D/MaxPoolMaxPoolmodel_2/conv3/conv3/Relu*
ksize
*
paddingSAME*0
_output_shapes
:����������*
T0*
data_formatNHWC*
strides

l
model_2/conv4/dilation_rateConst*
_output_shapes
:*
valueB"      *
dtype0
�
model_2/conv4/Conv2DConv2Dmodel_2/conv3/MaxPool2D/MaxPoolconv4/weights/read*0
_output_shapes
:����������*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME
�
model_2/conv4/BiasAddBiasAddmodel_2/conv4/Conv2Dconv4/biases/read*
data_formatNHWC*0
_output_shapes
:����������*
T0
r
model_2/conv4/conv4/ReluRelumodel_2/conv4/BiasAdd*0
_output_shapes
:����������*
T0
�
model_2/conv4/MaxPool2D/MaxPoolMaxPoolmodel_2/conv4/conv4/Relu*
paddingSAME*0
_output_shapes
:����������*
T0*
data_formatNHWC*
strides
*
ksize

l
model_2/conv5/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
�
model_2/conv5/Conv2DConv2Dmodel_2/conv4/MaxPool2D/MaxPoolconv5/weights/read*
use_cudnn_on_gpu(*
paddingSAME*/
_output_shapes
:���������*
	dilations
*
T0*
data_formatNHWC*
strides

�
model_2/conv5/BiasAddBiasAddmodel_2/conv5/Conv2Dconv5/biases/read*
T0*
data_formatNHWC*/
_output_shapes
:���������
�
model_2/conv5/MaxPool2D/MaxPoolMaxPoolmodel_2/conv5/BiasAdd*
ksize
*
paddingSAME*/
_output_shapes
:���������*
T0*
data_formatNHWC*
strides

|
model_2/Flatten/flatten/ShapeShapemodel_2/conv5/MaxPool2D/MaxPool*
_output_shapes
:*
T0*
out_type0
u
+model_2/Flatten/flatten/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
w
-model_2/Flatten/flatten/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
w
-model_2/Flatten/flatten/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
�
%model_2/Flatten/flatten/strided_sliceStridedSlicemodel_2/Flatten/flatten/Shape+model_2/Flatten/flatten/strided_slice/stack-model_2/Flatten/flatten/strided_slice/stack_1-model_2/Flatten/flatten/strided_slice/stack_2*
Index0*
T0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: 
r
'model_2/Flatten/flatten/Reshape/shape/1Const*
_output_shapes
: *
valueB :
���������*
dtype0
�
%model_2/Flatten/flatten/Reshape/shapePack%model_2/Flatten/flatten/strided_slice'model_2/Flatten/flatten/Reshape/shape/1*
_output_shapes
:*
T0*

axis *
N
�
model_2/Flatten/flatten/ReshapeReshapemodel_2/conv5/MaxPool2D/MaxPool%model_2/Flatten/flatten/Reshape/shape*'
_output_shapes
:���������*
T0*
Tshape0
|
subSubmodel/Flatten/flatten/Reshapemodel_1/Flatten/flatten/Reshape*'
_output_shapes
:���������*
T0
J
Pow/yConst*
valueB
 *   @*
dtype0*
_output_shapes
: 
H
PowPowsubPow/y*
T0*'
_output_shapes
:���������
W
Sum/reduction_indicesConst*
value	B :*
dtype0*
_output_shapes
: 
u
SumSumPowSum/reduction_indices*'
_output_shapes
:���������*

Tidx0*
	keep_dims(*
T0
C
SqrtSqrtSum*
T0*'
_output_shapes
:���������
~
sub_1Submodel/Flatten/flatten/Reshapemodel_2/Flatten/flatten/Reshape*'
_output_shapes
:���������*
T0
L
Pow_1/yConst*
valueB
 *   @*
dtype0*
_output_shapes
: 
N
Pow_1Powsub_1Pow_1/y*'
_output_shapes
:���������*
T0
Y
Sum_1/reduction_indicesConst*
value	B :*
dtype0*
_output_shapes
: 
{
Sum_1SumPow_1Sum_1/reduction_indices*'
_output_shapes
:���������*

Tidx0*
	keep_dims(*
T0
G
Sqrt_1SqrtSum_1*
T0*'
_output_shapes
:���������
L
sub_2SubSqrtSqrt_1*
T0*'
_output_shapes
:���������
J
add/yConst*
dtype0*
_output_shapes
: *
valueB
 *   ?
J
addAddsub_2add/y*
T0*'
_output_shapes
:���������
N
	Maximum/yConst*
valueB
 *    *
dtype0*
_output_shapes
: 
T
MaximumMaximumadd	Maximum/y*
T0*'
_output_shapes
:���������
V
ConstConst*
valueB"       *
dtype0*
_output_shapes
:
Z
MeanMeanMaximumConst*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
X
Variable/initial_valueConst*
value	B : *
dtype0*
_output_shapes
: 
l
Variable
VariableV2*
dtype0*
	container *
_output_shapes
: *
shape: *
shared_name 
�
Variable/AssignAssignVariableVariable/initial_value*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*
_class
loc:@Variable
a
Variable/readIdentityVariable*
T0*
_class
loc:@Variable*
_output_shapes
: 
R
gradients/ShapeConst*
_output_shapes
: *
valueB *
dtype0
X
gradients/grad_ys_0Const*
_output_shapes
: *
valueB
 *  �?*
dtype0
o
gradients/FillFillgradients/Shapegradients/grad_ys_0*
_output_shapes
: *
T0*

index_type0
r
!gradients/Mean_grad/Reshape/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
�
gradients/Mean_grad/ReshapeReshapegradients/Fill!gradients/Mean_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes

:
`
gradients/Mean_grad/ShapeShapeMaximum*
T0*
out_type0*
_output_shapes
:
�
gradients/Mean_grad/TileTilegradients/Mean_grad/Reshapegradients/Mean_grad/Shape*

Tmultiples0*
T0*'
_output_shapes
:���������
b
gradients/Mean_grad/Shape_1ShapeMaximum*
out_type0*
_output_shapes
:*
T0
^
gradients/Mean_grad/Shape_2Const*
valueB *
dtype0*
_output_shapes
: 
c
gradients/Mean_grad/ConstConst*
dtype0*
_output_shapes
:*
valueB: 
�
gradients/Mean_grad/ProdProdgradients/Mean_grad/Shape_1gradients/Mean_grad/Const*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
e
gradients/Mean_grad/Const_1Const*
dtype0*
_output_shapes
:*
valueB: 
�
gradients/Mean_grad/Prod_1Prodgradients/Mean_grad/Shape_2gradients/Mean_grad/Const_1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
_
gradients/Mean_grad/Maximum/yConst*
value	B :*
dtype0*
_output_shapes
: 
�
gradients/Mean_grad/MaximumMaximumgradients/Mean_grad/Prod_1gradients/Mean_grad/Maximum/y*
T0*
_output_shapes
: 
�
gradients/Mean_grad/floordivFloorDivgradients/Mean_grad/Prodgradients/Mean_grad/Maximum*
T0*
_output_shapes
: 
~
gradients/Mean_grad/CastCastgradients/Mean_grad/floordiv*
Truncate( *

DstT0*
_output_shapes
: *

SrcT0
�
gradients/Mean_grad/truedivRealDivgradients/Mean_grad/Tilegradients/Mean_grad/Cast*
T0*'
_output_shapes
:���������
_
gradients/Maximum_grad/ShapeShapeadd*
out_type0*
_output_shapes
:*
T0
a
gradients/Maximum_grad/Shape_1Const*
_output_shapes
: *
valueB *
dtype0
y
gradients/Maximum_grad/Shape_2Shapegradients/Mean_grad/truediv*
T0*
out_type0*
_output_shapes
:
g
"gradients/Maximum_grad/zeros/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    
�
gradients/Maximum_grad/zerosFillgradients/Maximum_grad/Shape_2"gradients/Maximum_grad/zeros/Const*'
_output_shapes
:���������*
T0*

index_type0
u
#gradients/Maximum_grad/GreaterEqualGreaterEqualadd	Maximum/y*'
_output_shapes
:���������*
T0
�
,gradients/Maximum_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/Maximum_grad/Shapegradients/Maximum_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
gradients/Maximum_grad/SelectSelect#gradients/Maximum_grad/GreaterEqualgradients/Mean_grad/truedivgradients/Maximum_grad/zeros*'
_output_shapes
:���������*
T0
�
gradients/Maximum_grad/Select_1Select#gradients/Maximum_grad/GreaterEqualgradients/Maximum_grad/zerosgradients/Mean_grad/truediv*
T0*'
_output_shapes
:���������
�
gradients/Maximum_grad/SumSumgradients/Maximum_grad/Select,gradients/Maximum_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
gradients/Maximum_grad/ReshapeReshapegradients/Maximum_grad/Sumgradients/Maximum_grad/Shape*
Tshape0*'
_output_shapes
:���������*
T0
�
gradients/Maximum_grad/Sum_1Sumgradients/Maximum_grad/Select_1.gradients/Maximum_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
�
 gradients/Maximum_grad/Reshape_1Reshapegradients/Maximum_grad/Sum_1gradients/Maximum_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
s
'gradients/Maximum_grad/tuple/group_depsNoOp^gradients/Maximum_grad/Reshape!^gradients/Maximum_grad/Reshape_1
�
/gradients/Maximum_grad/tuple/control_dependencyIdentitygradients/Maximum_grad/Reshape(^gradients/Maximum_grad/tuple/group_deps*'
_output_shapes
:���������*
T0*1
_class'
%#loc:@gradients/Maximum_grad/Reshape
�
1gradients/Maximum_grad/tuple/control_dependency_1Identity gradients/Maximum_grad/Reshape_1(^gradients/Maximum_grad/tuple/group_deps*
_output_shapes
: *
T0*3
_class)
'%loc:@gradients/Maximum_grad/Reshape_1
]
gradients/add_grad/ShapeShapesub_2*
T0*
out_type0*
_output_shapes
:
]
gradients/add_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
�
(gradients/add_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_grad/Shapegradients/add_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
gradients/add_grad/SumSum/gradients/Maximum_grad/tuple/control_dependency(gradients/add_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
�
gradients/add_grad/ReshapeReshapegradients/add_grad/Sumgradients/add_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
gradients/add_grad/Sum_1Sum/gradients/Maximum_grad/tuple/control_dependency*gradients/add_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
gradients/add_grad/Reshape_1Reshapegradients/add_grad/Sum_1gradients/add_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
g
#gradients/add_grad/tuple/group_depsNoOp^gradients/add_grad/Reshape^gradients/add_grad/Reshape_1
�
+gradients/add_grad/tuple/control_dependencyIdentitygradients/add_grad/Reshape$^gradients/add_grad/tuple/group_deps*-
_class#
!loc:@gradients/add_grad/Reshape*'
_output_shapes
:���������*
T0
�
-gradients/add_grad/tuple/control_dependency_1Identitygradients/add_grad/Reshape_1$^gradients/add_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/add_grad/Reshape_1*
_output_shapes
: 
^
gradients/sub_2_grad/ShapeShapeSqrt*
_output_shapes
:*
T0*
out_type0
b
gradients/sub_2_grad/Shape_1ShapeSqrt_1*
T0*
out_type0*
_output_shapes
:
�
*gradients/sub_2_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/sub_2_grad/Shapegradients/sub_2_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
gradients/sub_2_grad/SumSum+gradients/add_grad/tuple/control_dependency*gradients/sub_2_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
�
gradients/sub_2_grad/ReshapeReshapegradients/sub_2_grad/Sumgradients/sub_2_grad/Shape*'
_output_shapes
:���������*
T0*
Tshape0
�
gradients/sub_2_grad/Sum_1Sum+gradients/add_grad/tuple/control_dependency,gradients/sub_2_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
^
gradients/sub_2_grad/NegNeggradients/sub_2_grad/Sum_1*
_output_shapes
:*
T0
�
gradients/sub_2_grad/Reshape_1Reshapegradients/sub_2_grad/Neggradients/sub_2_grad/Shape_1*'
_output_shapes
:���������*
T0*
Tshape0
m
%gradients/sub_2_grad/tuple/group_depsNoOp^gradients/sub_2_grad/Reshape^gradients/sub_2_grad/Reshape_1
�
-gradients/sub_2_grad/tuple/control_dependencyIdentitygradients/sub_2_grad/Reshape&^gradients/sub_2_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/sub_2_grad/Reshape*'
_output_shapes
:���������
�
/gradients/sub_2_grad/tuple/control_dependency_1Identitygradients/sub_2_grad/Reshape_1&^gradients/sub_2_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/sub_2_grad/Reshape_1*'
_output_shapes
:���������
�
gradients/Sqrt_grad/SqrtGradSqrtGradSqrt-gradients/sub_2_grad/tuple/control_dependency*
T0*'
_output_shapes
:���������
�
gradients/Sqrt_1_grad/SqrtGradSqrtGradSqrt_1/gradients/sub_2_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:���������
[
gradients/Sum_grad/ShapeShapePow*
T0*
out_type0*
_output_shapes
:
�
gradients/Sum_grad/SizeConst*
dtype0*
_output_shapes
: *+
_class!
loc:@gradients/Sum_grad/Shape*
value	B :
�
gradients/Sum_grad/addAddSum/reduction_indicesgradients/Sum_grad/Size*+
_class!
loc:@gradients/Sum_grad/Shape*
_output_shapes
: *
T0
�
gradients/Sum_grad/modFloorModgradients/Sum_grad/addgradients/Sum_grad/Size*
_output_shapes
: *
T0*+
_class!
loc:@gradients/Sum_grad/Shape
�
gradients/Sum_grad/Shape_1Const*+
_class!
loc:@gradients/Sum_grad/Shape*
valueB *
dtype0*
_output_shapes
: 
�
gradients/Sum_grad/range/startConst*+
_class!
loc:@gradients/Sum_grad/Shape*
value	B : *
dtype0*
_output_shapes
: 
�
gradients/Sum_grad/range/deltaConst*
_output_shapes
: *+
_class!
loc:@gradients/Sum_grad/Shape*
value	B :*
dtype0
�
gradients/Sum_grad/rangeRangegradients/Sum_grad/range/startgradients/Sum_grad/Sizegradients/Sum_grad/range/delta*+
_class!
loc:@gradients/Sum_grad/Shape*
_output_shapes
:*

Tidx0
�
gradients/Sum_grad/Fill/valueConst*
_output_shapes
: *+
_class!
loc:@gradients/Sum_grad/Shape*
value	B :*
dtype0
�
gradients/Sum_grad/FillFillgradients/Sum_grad/Shape_1gradients/Sum_grad/Fill/value*
T0*+
_class!
loc:@gradients/Sum_grad/Shape*

index_type0*
_output_shapes
: 
�
 gradients/Sum_grad/DynamicStitchDynamicStitchgradients/Sum_grad/rangegradients/Sum_grad/modgradients/Sum_grad/Shapegradients/Sum_grad/Fill*
N*
_output_shapes
:*
T0*+
_class!
loc:@gradients/Sum_grad/Shape
�
gradients/Sum_grad/Maximum/yConst*+
_class!
loc:@gradients/Sum_grad/Shape*
value	B :*
dtype0*
_output_shapes
: 
�
gradients/Sum_grad/MaximumMaximum gradients/Sum_grad/DynamicStitchgradients/Sum_grad/Maximum/y*
T0*+
_class!
loc:@gradients/Sum_grad/Shape*
_output_shapes
:
�
gradients/Sum_grad/floordivFloorDivgradients/Sum_grad/Shapegradients/Sum_grad/Maximum*
T0*+
_class!
loc:@gradients/Sum_grad/Shape*
_output_shapes
:
�
gradients/Sum_grad/ReshapeReshapegradients/Sqrt_grad/SqrtGrad gradients/Sum_grad/DynamicStitch*
T0*
Tshape0*0
_output_shapes
:������������������
�
gradients/Sum_grad/TileTilegradients/Sum_grad/Reshapegradients/Sum_grad/floordiv*

Tmultiples0*
T0*'
_output_shapes
:���������
_
gradients/Sum_1_grad/ShapeShapePow_1*
out_type0*
_output_shapes
:*
T0
�
gradients/Sum_1_grad/SizeConst*-
_class#
!loc:@gradients/Sum_1_grad/Shape*
value	B :*
dtype0*
_output_shapes
: 
�
gradients/Sum_1_grad/addAddSum_1/reduction_indicesgradients/Sum_1_grad/Size*-
_class#
!loc:@gradients/Sum_1_grad/Shape*
_output_shapes
: *
T0
�
gradients/Sum_1_grad/modFloorModgradients/Sum_1_grad/addgradients/Sum_1_grad/Size*
_output_shapes
: *
T0*-
_class#
!loc:@gradients/Sum_1_grad/Shape
�
gradients/Sum_1_grad/Shape_1Const*-
_class#
!loc:@gradients/Sum_1_grad/Shape*
valueB *
dtype0*
_output_shapes
: 
�
 gradients/Sum_1_grad/range/startConst*
dtype0*
_output_shapes
: *-
_class#
!loc:@gradients/Sum_1_grad/Shape*
value	B : 
�
 gradients/Sum_1_grad/range/deltaConst*-
_class#
!loc:@gradients/Sum_1_grad/Shape*
value	B :*
dtype0*
_output_shapes
: 
�
gradients/Sum_1_grad/rangeRange gradients/Sum_1_grad/range/startgradients/Sum_1_grad/Size gradients/Sum_1_grad/range/delta*
_output_shapes
:*

Tidx0*-
_class#
!loc:@gradients/Sum_1_grad/Shape
�
gradients/Sum_1_grad/Fill/valueConst*-
_class#
!loc:@gradients/Sum_1_grad/Shape*
value	B :*
dtype0*
_output_shapes
: 
�
gradients/Sum_1_grad/FillFillgradients/Sum_1_grad/Shape_1gradients/Sum_1_grad/Fill/value*
_output_shapes
: *
T0*-
_class#
!loc:@gradients/Sum_1_grad/Shape*

index_type0
�
"gradients/Sum_1_grad/DynamicStitchDynamicStitchgradients/Sum_1_grad/rangegradients/Sum_1_grad/modgradients/Sum_1_grad/Shapegradients/Sum_1_grad/Fill*
T0*-
_class#
!loc:@gradients/Sum_1_grad/Shape*
N*
_output_shapes
:
�
gradients/Sum_1_grad/Maximum/yConst*-
_class#
!loc:@gradients/Sum_1_grad/Shape*
value	B :*
dtype0*
_output_shapes
: 
�
gradients/Sum_1_grad/MaximumMaximum"gradients/Sum_1_grad/DynamicStitchgradients/Sum_1_grad/Maximum/y*
T0*-
_class#
!loc:@gradients/Sum_1_grad/Shape*
_output_shapes
:
�
gradients/Sum_1_grad/floordivFloorDivgradients/Sum_1_grad/Shapegradients/Sum_1_grad/Maximum*
T0*-
_class#
!loc:@gradients/Sum_1_grad/Shape*
_output_shapes
:
�
gradients/Sum_1_grad/ReshapeReshapegradients/Sqrt_1_grad/SqrtGrad"gradients/Sum_1_grad/DynamicStitch*
T0*
Tshape0*0
_output_shapes
:������������������
�
gradients/Sum_1_grad/TileTilegradients/Sum_1_grad/Reshapegradients/Sum_1_grad/floordiv*

Tmultiples0*
T0*'
_output_shapes
:���������
[
gradients/Pow_grad/ShapeShapesub*
T0*
out_type0*
_output_shapes
:
]
gradients/Pow_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
�
(gradients/Pow_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/Pow_grad/Shapegradients/Pow_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
o
gradients/Pow_grad/mulMulgradients/Sum_grad/TilePow/y*
T0*'
_output_shapes
:���������
]
gradients/Pow_grad/sub/yConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
_
gradients/Pow_grad/subSubPow/ygradients/Pow_grad/sub/y*
T0*
_output_shapes
: 
l
gradients/Pow_grad/PowPowsubgradients/Pow_grad/sub*
T0*'
_output_shapes
:���������
�
gradients/Pow_grad/mul_1Mulgradients/Pow_grad/mulgradients/Pow_grad/Pow*
T0*'
_output_shapes
:���������
�
gradients/Pow_grad/SumSumgradients/Pow_grad/mul_1(gradients/Pow_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
�
gradients/Pow_grad/ReshapeReshapegradients/Pow_grad/Sumgradients/Pow_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
a
gradients/Pow_grad/Greater/yConst*
valueB
 *    *
dtype0*
_output_shapes
: 
z
gradients/Pow_grad/GreaterGreatersubgradients/Pow_grad/Greater/y*
T0*'
_output_shapes
:���������
e
"gradients/Pow_grad/ones_like/ShapeShapesub*
T0*
out_type0*
_output_shapes
:
g
"gradients/Pow_grad/ones_like/ConstConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
gradients/Pow_grad/ones_likeFill"gradients/Pow_grad/ones_like/Shape"gradients/Pow_grad/ones_like/Const*
T0*

index_type0*'
_output_shapes
:���������
�
gradients/Pow_grad/SelectSelectgradients/Pow_grad/Greatersubgradients/Pow_grad/ones_like*
T0*'
_output_shapes
:���������
j
gradients/Pow_grad/LogLoggradients/Pow_grad/Select*'
_output_shapes
:���������*
T0
a
gradients/Pow_grad/zeros_like	ZerosLikesub*
T0*'
_output_shapes
:���������
�
gradients/Pow_grad/Select_1Selectgradients/Pow_grad/Greatergradients/Pow_grad/Loggradients/Pow_grad/zeros_like*
T0*'
_output_shapes
:���������
o
gradients/Pow_grad/mul_2Mulgradients/Sum_grad/TilePow*
T0*'
_output_shapes
:���������
�
gradients/Pow_grad/mul_3Mulgradients/Pow_grad/mul_2gradients/Pow_grad/Select_1*
T0*'
_output_shapes
:���������
�
gradients/Pow_grad/Sum_1Sumgradients/Pow_grad/mul_3*gradients/Pow_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
�
gradients/Pow_grad/Reshape_1Reshapegradients/Pow_grad/Sum_1gradients/Pow_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
g
#gradients/Pow_grad/tuple/group_depsNoOp^gradients/Pow_grad/Reshape^gradients/Pow_grad/Reshape_1
�
+gradients/Pow_grad/tuple/control_dependencyIdentitygradients/Pow_grad/Reshape$^gradients/Pow_grad/tuple/group_deps*
T0*-
_class#
!loc:@gradients/Pow_grad/Reshape*'
_output_shapes
:���������
�
-gradients/Pow_grad/tuple/control_dependency_1Identitygradients/Pow_grad/Reshape_1$^gradients/Pow_grad/tuple/group_deps*
_output_shapes
: *
T0*/
_class%
#!loc:@gradients/Pow_grad/Reshape_1
_
gradients/Pow_1_grad/ShapeShapesub_1*
T0*
out_type0*
_output_shapes
:
_
gradients/Pow_1_grad/Shape_1Const*
dtype0*
_output_shapes
: *
valueB 
�
*gradients/Pow_1_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/Pow_1_grad/Shapegradients/Pow_1_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
u
gradients/Pow_1_grad/mulMulgradients/Sum_1_grad/TilePow_1/y*
T0*'
_output_shapes
:���������
_
gradients/Pow_1_grad/sub/yConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
e
gradients/Pow_1_grad/subSubPow_1/ygradients/Pow_1_grad/sub/y*
T0*
_output_shapes
: 
r
gradients/Pow_1_grad/PowPowsub_1gradients/Pow_1_grad/sub*'
_output_shapes
:���������*
T0
�
gradients/Pow_1_grad/mul_1Mulgradients/Pow_1_grad/mulgradients/Pow_1_grad/Pow*
T0*'
_output_shapes
:���������
�
gradients/Pow_1_grad/SumSumgradients/Pow_1_grad/mul_1*gradients/Pow_1_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
�
gradients/Pow_1_grad/ReshapeReshapegradients/Pow_1_grad/Sumgradients/Pow_1_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
c
gradients/Pow_1_grad/Greater/yConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
gradients/Pow_1_grad/GreaterGreatersub_1gradients/Pow_1_grad/Greater/y*
T0*'
_output_shapes
:���������
i
$gradients/Pow_1_grad/ones_like/ShapeShapesub_1*
_output_shapes
:*
T0*
out_type0
i
$gradients/Pow_1_grad/ones_like/ConstConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
gradients/Pow_1_grad/ones_likeFill$gradients/Pow_1_grad/ones_like/Shape$gradients/Pow_1_grad/ones_like/Const*
T0*

index_type0*'
_output_shapes
:���������
�
gradients/Pow_1_grad/SelectSelectgradients/Pow_1_grad/Greatersub_1gradients/Pow_1_grad/ones_like*
T0*'
_output_shapes
:���������
n
gradients/Pow_1_grad/LogLoggradients/Pow_1_grad/Select*'
_output_shapes
:���������*
T0
e
gradients/Pow_1_grad/zeros_like	ZerosLikesub_1*
T0*'
_output_shapes
:���������
�
gradients/Pow_1_grad/Select_1Selectgradients/Pow_1_grad/Greatergradients/Pow_1_grad/Loggradients/Pow_1_grad/zeros_like*'
_output_shapes
:���������*
T0
u
gradients/Pow_1_grad/mul_2Mulgradients/Sum_1_grad/TilePow_1*'
_output_shapes
:���������*
T0
�
gradients/Pow_1_grad/mul_3Mulgradients/Pow_1_grad/mul_2gradients/Pow_1_grad/Select_1*'
_output_shapes
:���������*
T0
�
gradients/Pow_1_grad/Sum_1Sumgradients/Pow_1_grad/mul_3,gradients/Pow_1_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
�
gradients/Pow_1_grad/Reshape_1Reshapegradients/Pow_1_grad/Sum_1gradients/Pow_1_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
m
%gradients/Pow_1_grad/tuple/group_depsNoOp^gradients/Pow_1_grad/Reshape^gradients/Pow_1_grad/Reshape_1
�
-gradients/Pow_1_grad/tuple/control_dependencyIdentitygradients/Pow_1_grad/Reshape&^gradients/Pow_1_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/Pow_1_grad/Reshape*'
_output_shapes
:���������
�
/gradients/Pow_1_grad/tuple/control_dependency_1Identitygradients/Pow_1_grad/Reshape_1&^gradients/Pow_1_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/Pow_1_grad/Reshape_1*
_output_shapes
: 
u
gradients/sub_grad/ShapeShapemodel/Flatten/flatten/Reshape*
T0*
out_type0*
_output_shapes
:
y
gradients/sub_grad/Shape_1Shapemodel_1/Flatten/flatten/Reshape*
T0*
out_type0*
_output_shapes
:
�
(gradients/sub_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/sub_grad/Shapegradients/sub_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
gradients/sub_grad/SumSum+gradients/Pow_grad/tuple/control_dependency(gradients/sub_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
gradients/sub_grad/ReshapeReshapegradients/sub_grad/Sumgradients/sub_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
gradients/sub_grad/Sum_1Sum+gradients/Pow_grad/tuple/control_dependency*gradients/sub_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
Z
gradients/sub_grad/NegNeggradients/sub_grad/Sum_1*
_output_shapes
:*
T0
�
gradients/sub_grad/Reshape_1Reshapegradients/sub_grad/Neggradients/sub_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:���������
g
#gradients/sub_grad/tuple/group_depsNoOp^gradients/sub_grad/Reshape^gradients/sub_grad/Reshape_1
�
+gradients/sub_grad/tuple/control_dependencyIdentitygradients/sub_grad/Reshape$^gradients/sub_grad/tuple/group_deps*
T0*-
_class#
!loc:@gradients/sub_grad/Reshape*'
_output_shapes
:���������
�
-gradients/sub_grad/tuple/control_dependency_1Identitygradients/sub_grad/Reshape_1$^gradients/sub_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/sub_grad/Reshape_1*'
_output_shapes
:���������
w
gradients/sub_1_grad/ShapeShapemodel/Flatten/flatten/Reshape*
T0*
out_type0*
_output_shapes
:
{
gradients/sub_1_grad/Shape_1Shapemodel_2/Flatten/flatten/Reshape*
_output_shapes
:*
T0*
out_type0
�
*gradients/sub_1_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/sub_1_grad/Shapegradients/sub_1_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
gradients/sub_1_grad/SumSum-gradients/Pow_1_grad/tuple/control_dependency*gradients/sub_1_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
gradients/sub_1_grad/ReshapeReshapegradients/sub_1_grad/Sumgradients/sub_1_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
gradients/sub_1_grad/Sum_1Sum-gradients/Pow_1_grad/tuple/control_dependency,gradients/sub_1_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
^
gradients/sub_1_grad/NegNeggradients/sub_1_grad/Sum_1*
T0*
_output_shapes
:
�
gradients/sub_1_grad/Reshape_1Reshapegradients/sub_1_grad/Neggradients/sub_1_grad/Shape_1*'
_output_shapes
:���������*
T0*
Tshape0
m
%gradients/sub_1_grad/tuple/group_depsNoOp^gradients/sub_1_grad/Reshape^gradients/sub_1_grad/Reshape_1
�
-gradients/sub_1_grad/tuple/control_dependencyIdentitygradients/sub_1_grad/Reshape&^gradients/sub_1_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/sub_1_grad/Reshape*'
_output_shapes
:���������
�
/gradients/sub_1_grad/tuple/control_dependency_1Identitygradients/sub_1_grad/Reshape_1&^gradients/sub_1_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/sub_1_grad/Reshape_1*'
_output_shapes
:���������
�
4gradients/model_1/Flatten/flatten/Reshape_grad/ShapeShapemodel_1/conv5/MaxPool2D/MaxPool*
T0*
out_type0*
_output_shapes
:
�
6gradients/model_1/Flatten/flatten/Reshape_grad/ReshapeReshape-gradients/sub_grad/tuple/control_dependency_14gradients/model_1/Flatten/flatten/Reshape_grad/Shape*
T0*
Tshape0*/
_output_shapes
:���������
�
gradients/AddNAddN+gradients/sub_grad/tuple/control_dependency-gradients/sub_1_grad/tuple/control_dependency*
N*'
_output_shapes
:���������*
T0*-
_class#
!loc:@gradients/sub_grad/Reshape
�
2gradients/model/Flatten/flatten/Reshape_grad/ShapeShapemodel/conv5/MaxPool2D/MaxPool*
T0*
out_type0*
_output_shapes
:
�
4gradients/model/Flatten/flatten/Reshape_grad/ReshapeReshapegradients/AddN2gradients/model/Flatten/flatten/Reshape_grad/Shape*
T0*
Tshape0*/
_output_shapes
:���������
�
4gradients/model_2/Flatten/flatten/Reshape_grad/ShapeShapemodel_2/conv5/MaxPool2D/MaxPool*
T0*
out_type0*
_output_shapes
:
�
6gradients/model_2/Flatten/flatten/Reshape_grad/ReshapeReshape/gradients/sub_1_grad/tuple/control_dependency_14gradients/model_2/Flatten/flatten/Reshape_grad/Shape*/
_output_shapes
:���������*
T0*
Tshape0
�
:gradients/model_1/conv5/MaxPool2D/MaxPool_grad/MaxPoolGradMaxPoolGradmodel_1/conv5/BiasAddmodel_1/conv5/MaxPool2D/MaxPool6gradients/model_1/Flatten/flatten/Reshape_grad/Reshape*/
_output_shapes
:���������*
T0*
data_formatNHWC*
strides
*
ksize
*
paddingSAME
�
8gradients/model/conv5/MaxPool2D/MaxPool_grad/MaxPoolGradMaxPoolGradmodel/conv5/BiasAddmodel/conv5/MaxPool2D/MaxPool4gradients/model/Flatten/flatten/Reshape_grad/Reshape*
ksize
*
paddingSAME*/
_output_shapes
:���������*
T0*
data_formatNHWC*
strides

�
:gradients/model_2/conv5/MaxPool2D/MaxPool_grad/MaxPoolGradMaxPoolGradmodel_2/conv5/BiasAddmodel_2/conv5/MaxPool2D/MaxPool6gradients/model_2/Flatten/flatten/Reshape_grad/Reshape*
T0*
data_formatNHWC*
strides
*
ksize
*
paddingSAME*/
_output_shapes
:���������
�
0gradients/model_1/conv5/BiasAdd_grad/BiasAddGradBiasAddGrad:gradients/model_1/conv5/MaxPool2D/MaxPool_grad/MaxPoolGrad*
data_formatNHWC*
_output_shapes
:*
T0
�
5gradients/model_1/conv5/BiasAdd_grad/tuple/group_depsNoOp1^gradients/model_1/conv5/BiasAdd_grad/BiasAddGrad;^gradients/model_1/conv5/MaxPool2D/MaxPool_grad/MaxPoolGrad
�
=gradients/model_1/conv5/BiasAdd_grad/tuple/control_dependencyIdentity:gradients/model_1/conv5/MaxPool2D/MaxPool_grad/MaxPoolGrad6^gradients/model_1/conv5/BiasAdd_grad/tuple/group_deps*
T0*M
_classC
A?loc:@gradients/model_1/conv5/MaxPool2D/MaxPool_grad/MaxPoolGrad*/
_output_shapes
:���������
�
?gradients/model_1/conv5/BiasAdd_grad/tuple/control_dependency_1Identity0gradients/model_1/conv5/BiasAdd_grad/BiasAddGrad6^gradients/model_1/conv5/BiasAdd_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/model_1/conv5/BiasAdd_grad/BiasAddGrad*
_output_shapes
:
�
.gradients/model/conv5/BiasAdd_grad/BiasAddGradBiasAddGrad8gradients/model/conv5/MaxPool2D/MaxPool_grad/MaxPoolGrad*
T0*
data_formatNHWC*
_output_shapes
:
�
3gradients/model/conv5/BiasAdd_grad/tuple/group_depsNoOp/^gradients/model/conv5/BiasAdd_grad/BiasAddGrad9^gradients/model/conv5/MaxPool2D/MaxPool_grad/MaxPoolGrad
�
;gradients/model/conv5/BiasAdd_grad/tuple/control_dependencyIdentity8gradients/model/conv5/MaxPool2D/MaxPool_grad/MaxPoolGrad4^gradients/model/conv5/BiasAdd_grad/tuple/group_deps*
T0*K
_classA
?=loc:@gradients/model/conv5/MaxPool2D/MaxPool_grad/MaxPoolGrad*/
_output_shapes
:���������
�
=gradients/model/conv5/BiasAdd_grad/tuple/control_dependency_1Identity.gradients/model/conv5/BiasAdd_grad/BiasAddGrad4^gradients/model/conv5/BiasAdd_grad/tuple/group_deps*
_output_shapes
:*
T0*A
_class7
53loc:@gradients/model/conv5/BiasAdd_grad/BiasAddGrad
�
0gradients/model_2/conv5/BiasAdd_grad/BiasAddGradBiasAddGrad:gradients/model_2/conv5/MaxPool2D/MaxPool_grad/MaxPoolGrad*
T0*
data_formatNHWC*
_output_shapes
:
�
5gradients/model_2/conv5/BiasAdd_grad/tuple/group_depsNoOp1^gradients/model_2/conv5/BiasAdd_grad/BiasAddGrad;^gradients/model_2/conv5/MaxPool2D/MaxPool_grad/MaxPoolGrad
�
=gradients/model_2/conv5/BiasAdd_grad/tuple/control_dependencyIdentity:gradients/model_2/conv5/MaxPool2D/MaxPool_grad/MaxPoolGrad6^gradients/model_2/conv5/BiasAdd_grad/tuple/group_deps*/
_output_shapes
:���������*
T0*M
_classC
A?loc:@gradients/model_2/conv5/MaxPool2D/MaxPool_grad/MaxPoolGrad
�
?gradients/model_2/conv5/BiasAdd_grad/tuple/control_dependency_1Identity0gradients/model_2/conv5/BiasAdd_grad/BiasAddGrad6^gradients/model_2/conv5/BiasAdd_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/model_2/conv5/BiasAdd_grad/BiasAddGrad*
_output_shapes
:
�
*gradients/model_1/conv5/Conv2D_grad/ShapeNShapeNmodel_1/conv4/MaxPool2D/MaxPoolconv5/weights/read*
T0*
out_type0*
N* 
_output_shapes
::
�
7gradients/model_1/conv5/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput*gradients/model_1/conv5/Conv2D_grad/ShapeNconv5/weights/read=gradients/model_1/conv5/BiasAdd_grad/tuple/control_dependency*
paddingSAME*0
_output_shapes
:����������*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(
�
8gradients/model_1/conv5/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFiltermodel_1/conv4/MaxPool2D/MaxPool,gradients/model_1/conv5/Conv2D_grad/ShapeN:1=gradients/model_1/conv5/BiasAdd_grad/tuple/control_dependency*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*'
_output_shapes
:�
�
4gradients/model_1/conv5/Conv2D_grad/tuple/group_depsNoOp9^gradients/model_1/conv5/Conv2D_grad/Conv2DBackpropFilter8^gradients/model_1/conv5/Conv2D_grad/Conv2DBackpropInput
�
<gradients/model_1/conv5/Conv2D_grad/tuple/control_dependencyIdentity7gradients/model_1/conv5/Conv2D_grad/Conv2DBackpropInput5^gradients/model_1/conv5/Conv2D_grad/tuple/group_deps*
T0*J
_class@
><loc:@gradients/model_1/conv5/Conv2D_grad/Conv2DBackpropInput*0
_output_shapes
:����������
�
>gradients/model_1/conv5/Conv2D_grad/tuple/control_dependency_1Identity8gradients/model_1/conv5/Conv2D_grad/Conv2DBackpropFilter5^gradients/model_1/conv5/Conv2D_grad/tuple/group_deps*
T0*K
_classA
?=loc:@gradients/model_1/conv5/Conv2D_grad/Conv2DBackpropFilter*'
_output_shapes
:�
�
(gradients/model/conv5/Conv2D_grad/ShapeNShapeNmodel/conv4/MaxPool2D/MaxPoolconv5/weights/read*
T0*
out_type0*
N* 
_output_shapes
::
�
5gradients/model/conv5/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput(gradients/model/conv5/Conv2D_grad/ShapeNconv5/weights/read;gradients/model/conv5/BiasAdd_grad/tuple/control_dependency*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*0
_output_shapes
:����������*
	dilations

�
6gradients/model/conv5/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFiltermodel/conv4/MaxPool2D/MaxPool*gradients/model/conv5/Conv2D_grad/ShapeN:1;gradients/model/conv5/BiasAdd_grad/tuple/control_dependency*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*'
_output_shapes
:�*
	dilations
*
T0
�
2gradients/model/conv5/Conv2D_grad/tuple/group_depsNoOp7^gradients/model/conv5/Conv2D_grad/Conv2DBackpropFilter6^gradients/model/conv5/Conv2D_grad/Conv2DBackpropInput
�
:gradients/model/conv5/Conv2D_grad/tuple/control_dependencyIdentity5gradients/model/conv5/Conv2D_grad/Conv2DBackpropInput3^gradients/model/conv5/Conv2D_grad/tuple/group_deps*
T0*H
_class>
<:loc:@gradients/model/conv5/Conv2D_grad/Conv2DBackpropInput*0
_output_shapes
:����������
�
<gradients/model/conv5/Conv2D_grad/tuple/control_dependency_1Identity6gradients/model/conv5/Conv2D_grad/Conv2DBackpropFilter3^gradients/model/conv5/Conv2D_grad/tuple/group_deps*'
_output_shapes
:�*
T0*I
_class?
=;loc:@gradients/model/conv5/Conv2D_grad/Conv2DBackpropFilter
�
*gradients/model_2/conv5/Conv2D_grad/ShapeNShapeNmodel_2/conv4/MaxPool2D/MaxPoolconv5/weights/read*
T0*
out_type0*
N* 
_output_shapes
::
�
7gradients/model_2/conv5/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput*gradients/model_2/conv5/Conv2D_grad/ShapeNconv5/weights/read=gradients/model_2/conv5/BiasAdd_grad/tuple/control_dependency*0
_output_shapes
:����������*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME
�
8gradients/model_2/conv5/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFiltermodel_2/conv4/MaxPool2D/MaxPool,gradients/model_2/conv5/Conv2D_grad/ShapeN:1=gradients/model_2/conv5/BiasAdd_grad/tuple/control_dependency*
paddingSAME*'
_output_shapes
:�*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
�
4gradients/model_2/conv5/Conv2D_grad/tuple/group_depsNoOp9^gradients/model_2/conv5/Conv2D_grad/Conv2DBackpropFilter8^gradients/model_2/conv5/Conv2D_grad/Conv2DBackpropInput
�
<gradients/model_2/conv5/Conv2D_grad/tuple/control_dependencyIdentity7gradients/model_2/conv5/Conv2D_grad/Conv2DBackpropInput5^gradients/model_2/conv5/Conv2D_grad/tuple/group_deps*
T0*J
_class@
><loc:@gradients/model_2/conv5/Conv2D_grad/Conv2DBackpropInput*0
_output_shapes
:����������
�
>gradients/model_2/conv5/Conv2D_grad/tuple/control_dependency_1Identity8gradients/model_2/conv5/Conv2D_grad/Conv2DBackpropFilter5^gradients/model_2/conv5/Conv2D_grad/tuple/group_deps*
T0*K
_classA
?=loc:@gradients/model_2/conv5/Conv2D_grad/Conv2DBackpropFilter*'
_output_shapes
:�
�
gradients/AddN_1AddN?gradients/model_1/conv5/BiasAdd_grad/tuple/control_dependency_1=gradients/model/conv5/BiasAdd_grad/tuple/control_dependency_1?gradients/model_2/conv5/BiasAdd_grad/tuple/control_dependency_1*
N*
_output_shapes
:*
T0*C
_class9
75loc:@gradients/model_1/conv5/BiasAdd_grad/BiasAddGrad
�
:gradients/model_1/conv4/MaxPool2D/MaxPool_grad/MaxPoolGradMaxPoolGradmodel_1/conv4/conv4/Relumodel_1/conv4/MaxPool2D/MaxPool<gradients/model_1/conv5/Conv2D_grad/tuple/control_dependency*
ksize
*
paddingSAME*0
_output_shapes
:����������*
T0*
data_formatNHWC*
strides

�
8gradients/model/conv4/MaxPool2D/MaxPool_grad/MaxPoolGradMaxPoolGradmodel/conv4/conv4/Relumodel/conv4/MaxPool2D/MaxPool:gradients/model/conv5/Conv2D_grad/tuple/control_dependency*0
_output_shapes
:����������*
T0*
strides
*
data_formatNHWC*
ksize
*
paddingSAME
�
:gradients/model_2/conv4/MaxPool2D/MaxPool_grad/MaxPoolGradMaxPoolGradmodel_2/conv4/conv4/Relumodel_2/conv4/MaxPool2D/MaxPool<gradients/model_2/conv5/Conv2D_grad/tuple/control_dependency*
T0*
data_formatNHWC*
strides
*
ksize
*
paddingSAME*0
_output_shapes
:����������
�
gradients/AddN_2AddN>gradients/model_1/conv5/Conv2D_grad/tuple/control_dependency_1<gradients/model/conv5/Conv2D_grad/tuple/control_dependency_1>gradients/model_2/conv5/Conv2D_grad/tuple/control_dependency_1*
N*'
_output_shapes
:�*
T0*K
_classA
?=loc:@gradients/model_1/conv5/Conv2D_grad/Conv2DBackpropFilter
�
0gradients/model_1/conv4/conv4/Relu_grad/ReluGradReluGrad:gradients/model_1/conv4/MaxPool2D/MaxPool_grad/MaxPoolGradmodel_1/conv4/conv4/Relu*
T0*0
_output_shapes
:����������
�
.gradients/model/conv4/conv4/Relu_grad/ReluGradReluGrad8gradients/model/conv4/MaxPool2D/MaxPool_grad/MaxPoolGradmodel/conv4/conv4/Relu*
T0*0
_output_shapes
:����������
�
0gradients/model_2/conv4/conv4/Relu_grad/ReluGradReluGrad:gradients/model_2/conv4/MaxPool2D/MaxPool_grad/MaxPoolGradmodel_2/conv4/conv4/Relu*0
_output_shapes
:����������*
T0
�
0gradients/model_1/conv4/BiasAdd_grad/BiasAddGradBiasAddGrad0gradients/model_1/conv4/conv4/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes	
:�
�
5gradients/model_1/conv4/BiasAdd_grad/tuple/group_depsNoOp1^gradients/model_1/conv4/BiasAdd_grad/BiasAddGrad1^gradients/model_1/conv4/conv4/Relu_grad/ReluGrad
�
=gradients/model_1/conv4/BiasAdd_grad/tuple/control_dependencyIdentity0gradients/model_1/conv4/conv4/Relu_grad/ReluGrad6^gradients/model_1/conv4/BiasAdd_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/model_1/conv4/conv4/Relu_grad/ReluGrad*0
_output_shapes
:����������
�
?gradients/model_1/conv4/BiasAdd_grad/tuple/control_dependency_1Identity0gradients/model_1/conv4/BiasAdd_grad/BiasAddGrad6^gradients/model_1/conv4/BiasAdd_grad/tuple/group_deps*
_output_shapes	
:�*
T0*C
_class9
75loc:@gradients/model_1/conv4/BiasAdd_grad/BiasAddGrad
�
.gradients/model/conv4/BiasAdd_grad/BiasAddGradBiasAddGrad.gradients/model/conv4/conv4/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes	
:�
�
3gradients/model/conv4/BiasAdd_grad/tuple/group_depsNoOp/^gradients/model/conv4/BiasAdd_grad/BiasAddGrad/^gradients/model/conv4/conv4/Relu_grad/ReluGrad
�
;gradients/model/conv4/BiasAdd_grad/tuple/control_dependencyIdentity.gradients/model/conv4/conv4/Relu_grad/ReluGrad4^gradients/model/conv4/BiasAdd_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients/model/conv4/conv4/Relu_grad/ReluGrad*0
_output_shapes
:����������
�
=gradients/model/conv4/BiasAdd_grad/tuple/control_dependency_1Identity.gradients/model/conv4/BiasAdd_grad/BiasAddGrad4^gradients/model/conv4/BiasAdd_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients/model/conv4/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:�
�
0gradients/model_2/conv4/BiasAdd_grad/BiasAddGradBiasAddGrad0gradients/model_2/conv4/conv4/Relu_grad/ReluGrad*
data_formatNHWC*
_output_shapes	
:�*
T0
�
5gradients/model_2/conv4/BiasAdd_grad/tuple/group_depsNoOp1^gradients/model_2/conv4/BiasAdd_grad/BiasAddGrad1^gradients/model_2/conv4/conv4/Relu_grad/ReluGrad
�
=gradients/model_2/conv4/BiasAdd_grad/tuple/control_dependencyIdentity0gradients/model_2/conv4/conv4/Relu_grad/ReluGrad6^gradients/model_2/conv4/BiasAdd_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/model_2/conv4/conv4/Relu_grad/ReluGrad*0
_output_shapes
:����������
�
?gradients/model_2/conv4/BiasAdd_grad/tuple/control_dependency_1Identity0gradients/model_2/conv4/BiasAdd_grad/BiasAddGrad6^gradients/model_2/conv4/BiasAdd_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/model_2/conv4/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:�
�
*gradients/model_1/conv4/Conv2D_grad/ShapeNShapeNmodel_1/conv3/MaxPool2D/MaxPoolconv4/weights/read*
T0*
out_type0*
N* 
_output_shapes
::
�
7gradients/model_1/conv4/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput*gradients/model_1/conv4/Conv2D_grad/ShapeNconv4/weights/read=gradients/model_1/conv4/BiasAdd_grad/tuple/control_dependency*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*0
_output_shapes
:����������
�
8gradients/model_1/conv4/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFiltermodel_1/conv3/MaxPool2D/MaxPool,gradients/model_1/conv4/Conv2D_grad/ShapeN:1=gradients/model_1/conv4/BiasAdd_grad/tuple/control_dependency*
paddingSAME*(
_output_shapes
:��*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
�
4gradients/model_1/conv4/Conv2D_grad/tuple/group_depsNoOp9^gradients/model_1/conv4/Conv2D_grad/Conv2DBackpropFilter8^gradients/model_1/conv4/Conv2D_grad/Conv2DBackpropInput
�
<gradients/model_1/conv4/Conv2D_grad/tuple/control_dependencyIdentity7gradients/model_1/conv4/Conv2D_grad/Conv2DBackpropInput5^gradients/model_1/conv4/Conv2D_grad/tuple/group_deps*
T0*J
_class@
><loc:@gradients/model_1/conv4/Conv2D_grad/Conv2DBackpropInput*0
_output_shapes
:����������
�
>gradients/model_1/conv4/Conv2D_grad/tuple/control_dependency_1Identity8gradients/model_1/conv4/Conv2D_grad/Conv2DBackpropFilter5^gradients/model_1/conv4/Conv2D_grad/tuple/group_deps*
T0*K
_classA
?=loc:@gradients/model_1/conv4/Conv2D_grad/Conv2DBackpropFilter*(
_output_shapes
:��
�
(gradients/model/conv4/Conv2D_grad/ShapeNShapeNmodel/conv3/MaxPool2D/MaxPoolconv4/weights/read*
N* 
_output_shapes
::*
T0*
out_type0
�
5gradients/model/conv4/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput(gradients/model/conv4/Conv2D_grad/ShapeNconv4/weights/read;gradients/model/conv4/BiasAdd_grad/tuple/control_dependency*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*0
_output_shapes
:����������
�
6gradients/model/conv4/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFiltermodel/conv3/MaxPool2D/MaxPool*gradients/model/conv4/Conv2D_grad/ShapeN:1;gradients/model/conv4/BiasAdd_grad/tuple/control_dependency*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME*(
_output_shapes
:��
�
2gradients/model/conv4/Conv2D_grad/tuple/group_depsNoOp7^gradients/model/conv4/Conv2D_grad/Conv2DBackpropFilter6^gradients/model/conv4/Conv2D_grad/Conv2DBackpropInput
�
:gradients/model/conv4/Conv2D_grad/tuple/control_dependencyIdentity5gradients/model/conv4/Conv2D_grad/Conv2DBackpropInput3^gradients/model/conv4/Conv2D_grad/tuple/group_deps*
T0*H
_class>
<:loc:@gradients/model/conv4/Conv2D_grad/Conv2DBackpropInput*0
_output_shapes
:����������
�
<gradients/model/conv4/Conv2D_grad/tuple/control_dependency_1Identity6gradients/model/conv4/Conv2D_grad/Conv2DBackpropFilter3^gradients/model/conv4/Conv2D_grad/tuple/group_deps*
T0*I
_class?
=;loc:@gradients/model/conv4/Conv2D_grad/Conv2DBackpropFilter*(
_output_shapes
:��
�
*gradients/model_2/conv4/Conv2D_grad/ShapeNShapeNmodel_2/conv3/MaxPool2D/MaxPoolconv4/weights/read*
N* 
_output_shapes
::*
T0*
out_type0
�
7gradients/model_2/conv4/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput*gradients/model_2/conv4/Conv2D_grad/ShapeNconv4/weights/read=gradients/model_2/conv4/BiasAdd_grad/tuple/control_dependency*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*0
_output_shapes
:����������*
	dilations

�
8gradients/model_2/conv4/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFiltermodel_2/conv3/MaxPool2D/MaxPool,gradients/model_2/conv4/Conv2D_grad/ShapeN:1=gradients/model_2/conv4/BiasAdd_grad/tuple/control_dependency*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME*(
_output_shapes
:��
�
4gradients/model_2/conv4/Conv2D_grad/tuple/group_depsNoOp9^gradients/model_2/conv4/Conv2D_grad/Conv2DBackpropFilter8^gradients/model_2/conv4/Conv2D_grad/Conv2DBackpropInput
�
<gradients/model_2/conv4/Conv2D_grad/tuple/control_dependencyIdentity7gradients/model_2/conv4/Conv2D_grad/Conv2DBackpropInput5^gradients/model_2/conv4/Conv2D_grad/tuple/group_deps*0
_output_shapes
:����������*
T0*J
_class@
><loc:@gradients/model_2/conv4/Conv2D_grad/Conv2DBackpropInput
�
>gradients/model_2/conv4/Conv2D_grad/tuple/control_dependency_1Identity8gradients/model_2/conv4/Conv2D_grad/Conv2DBackpropFilter5^gradients/model_2/conv4/Conv2D_grad/tuple/group_deps*
T0*K
_classA
?=loc:@gradients/model_2/conv4/Conv2D_grad/Conv2DBackpropFilter*(
_output_shapes
:��
�
gradients/AddN_3AddN?gradients/model_1/conv4/BiasAdd_grad/tuple/control_dependency_1=gradients/model/conv4/BiasAdd_grad/tuple/control_dependency_1?gradients/model_2/conv4/BiasAdd_grad/tuple/control_dependency_1*
T0*C
_class9
75loc:@gradients/model_1/conv4/BiasAdd_grad/BiasAddGrad*
N*
_output_shapes	
:�
�
:gradients/model_1/conv3/MaxPool2D/MaxPool_grad/MaxPoolGradMaxPoolGradmodel_1/conv3/conv3/Relumodel_1/conv3/MaxPool2D/MaxPool<gradients/model_1/conv4/Conv2D_grad/tuple/control_dependency*0
_output_shapes
:����������*
T0*
strides
*
data_formatNHWC*
ksize
*
paddingSAME
�
8gradients/model/conv3/MaxPool2D/MaxPool_grad/MaxPoolGradMaxPoolGradmodel/conv3/conv3/Relumodel/conv3/MaxPool2D/MaxPool:gradients/model/conv4/Conv2D_grad/tuple/control_dependency*0
_output_shapes
:����������*
T0*
data_formatNHWC*
strides
*
ksize
*
paddingSAME
�
:gradients/model_2/conv3/MaxPool2D/MaxPool_grad/MaxPoolGradMaxPoolGradmodel_2/conv3/conv3/Relumodel_2/conv3/MaxPool2D/MaxPool<gradients/model_2/conv4/Conv2D_grad/tuple/control_dependency*
ksize
*
paddingSAME*0
_output_shapes
:����������*
T0*
data_formatNHWC*
strides

�
gradients/AddN_4AddN>gradients/model_1/conv4/Conv2D_grad/tuple/control_dependency_1<gradients/model/conv4/Conv2D_grad/tuple/control_dependency_1>gradients/model_2/conv4/Conv2D_grad/tuple/control_dependency_1*
T0*K
_classA
?=loc:@gradients/model_1/conv4/Conv2D_grad/Conv2DBackpropFilter*
N*(
_output_shapes
:��
�
0gradients/model_1/conv3/conv3/Relu_grad/ReluGradReluGrad:gradients/model_1/conv3/MaxPool2D/MaxPool_grad/MaxPoolGradmodel_1/conv3/conv3/Relu*
T0*0
_output_shapes
:����������
�
.gradients/model/conv3/conv3/Relu_grad/ReluGradReluGrad8gradients/model/conv3/MaxPool2D/MaxPool_grad/MaxPoolGradmodel/conv3/conv3/Relu*
T0*0
_output_shapes
:����������
�
0gradients/model_2/conv3/conv3/Relu_grad/ReluGradReluGrad:gradients/model_2/conv3/MaxPool2D/MaxPool_grad/MaxPoolGradmodel_2/conv3/conv3/Relu*
T0*0
_output_shapes
:����������
�
0gradients/model_1/conv3/BiasAdd_grad/BiasAddGradBiasAddGrad0gradients/model_1/conv3/conv3/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes	
:�
�
5gradients/model_1/conv3/BiasAdd_grad/tuple/group_depsNoOp1^gradients/model_1/conv3/BiasAdd_grad/BiasAddGrad1^gradients/model_1/conv3/conv3/Relu_grad/ReluGrad
�
=gradients/model_1/conv3/BiasAdd_grad/tuple/control_dependencyIdentity0gradients/model_1/conv3/conv3/Relu_grad/ReluGrad6^gradients/model_1/conv3/BiasAdd_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/model_1/conv3/conv3/Relu_grad/ReluGrad*0
_output_shapes
:����������
�
?gradients/model_1/conv3/BiasAdd_grad/tuple/control_dependency_1Identity0gradients/model_1/conv3/BiasAdd_grad/BiasAddGrad6^gradients/model_1/conv3/BiasAdd_grad/tuple/group_deps*
_output_shapes	
:�*
T0*C
_class9
75loc:@gradients/model_1/conv3/BiasAdd_grad/BiasAddGrad
�
.gradients/model/conv3/BiasAdd_grad/BiasAddGradBiasAddGrad.gradients/model/conv3/conv3/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes	
:�
�
3gradients/model/conv3/BiasAdd_grad/tuple/group_depsNoOp/^gradients/model/conv3/BiasAdd_grad/BiasAddGrad/^gradients/model/conv3/conv3/Relu_grad/ReluGrad
�
;gradients/model/conv3/BiasAdd_grad/tuple/control_dependencyIdentity.gradients/model/conv3/conv3/Relu_grad/ReluGrad4^gradients/model/conv3/BiasAdd_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients/model/conv3/conv3/Relu_grad/ReluGrad*0
_output_shapes
:����������
�
=gradients/model/conv3/BiasAdd_grad/tuple/control_dependency_1Identity.gradients/model/conv3/BiasAdd_grad/BiasAddGrad4^gradients/model/conv3/BiasAdd_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients/model/conv3/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:�
�
0gradients/model_2/conv3/BiasAdd_grad/BiasAddGradBiasAddGrad0gradients/model_2/conv3/conv3/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes	
:�
�
5gradients/model_2/conv3/BiasAdd_grad/tuple/group_depsNoOp1^gradients/model_2/conv3/BiasAdd_grad/BiasAddGrad1^gradients/model_2/conv3/conv3/Relu_grad/ReluGrad
�
=gradients/model_2/conv3/BiasAdd_grad/tuple/control_dependencyIdentity0gradients/model_2/conv3/conv3/Relu_grad/ReluGrad6^gradients/model_2/conv3/BiasAdd_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/model_2/conv3/conv3/Relu_grad/ReluGrad*0
_output_shapes
:����������
�
?gradients/model_2/conv3/BiasAdd_grad/tuple/control_dependency_1Identity0gradients/model_2/conv3/BiasAdd_grad/BiasAddGrad6^gradients/model_2/conv3/BiasAdd_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/model_2/conv3/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:�
�
*gradients/model_1/conv3/Conv2D_grad/ShapeNShapeNmodel_1/conv2/MaxPool2D/MaxPoolconv3/weights/read*
N* 
_output_shapes
::*
T0*
out_type0
�
7gradients/model_1/conv3/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput*gradients/model_1/conv3/Conv2D_grad/ShapeNconv3/weights/read=gradients/model_1/conv3/BiasAdd_grad/tuple/control_dependency*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME*/
_output_shapes
:���������@
�
8gradients/model_1/conv3/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFiltermodel_1/conv2/MaxPool2D/MaxPool,gradients/model_1/conv3/Conv2D_grad/ShapeN:1=gradients/model_1/conv3/BiasAdd_grad/tuple/control_dependency*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME*'
_output_shapes
:@�
�
4gradients/model_1/conv3/Conv2D_grad/tuple/group_depsNoOp9^gradients/model_1/conv3/Conv2D_grad/Conv2DBackpropFilter8^gradients/model_1/conv3/Conv2D_grad/Conv2DBackpropInput
�
<gradients/model_1/conv3/Conv2D_grad/tuple/control_dependencyIdentity7gradients/model_1/conv3/Conv2D_grad/Conv2DBackpropInput5^gradients/model_1/conv3/Conv2D_grad/tuple/group_deps*
T0*J
_class@
><loc:@gradients/model_1/conv3/Conv2D_grad/Conv2DBackpropInput*/
_output_shapes
:���������@
�
>gradients/model_1/conv3/Conv2D_grad/tuple/control_dependency_1Identity8gradients/model_1/conv3/Conv2D_grad/Conv2DBackpropFilter5^gradients/model_1/conv3/Conv2D_grad/tuple/group_deps*
T0*K
_classA
?=loc:@gradients/model_1/conv3/Conv2D_grad/Conv2DBackpropFilter*'
_output_shapes
:@�
�
(gradients/model/conv3/Conv2D_grad/ShapeNShapeNmodel/conv2/MaxPool2D/MaxPoolconv3/weights/read*
T0*
out_type0*
N* 
_output_shapes
::
�
5gradients/model/conv3/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput(gradients/model/conv3/Conv2D_grad/ShapeNconv3/weights/read;gradients/model/conv3/BiasAdd_grad/tuple/control_dependency*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME*/
_output_shapes
:���������@*
	dilations

�
6gradients/model/conv3/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFiltermodel/conv2/MaxPool2D/MaxPool*gradients/model/conv3/Conv2D_grad/ShapeN:1;gradients/model/conv3/BiasAdd_grad/tuple/control_dependency*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME*'
_output_shapes
:@�
�
2gradients/model/conv3/Conv2D_grad/tuple/group_depsNoOp7^gradients/model/conv3/Conv2D_grad/Conv2DBackpropFilter6^gradients/model/conv3/Conv2D_grad/Conv2DBackpropInput
�
:gradients/model/conv3/Conv2D_grad/tuple/control_dependencyIdentity5gradients/model/conv3/Conv2D_grad/Conv2DBackpropInput3^gradients/model/conv3/Conv2D_grad/tuple/group_deps*
T0*H
_class>
<:loc:@gradients/model/conv3/Conv2D_grad/Conv2DBackpropInput*/
_output_shapes
:���������@
�
<gradients/model/conv3/Conv2D_grad/tuple/control_dependency_1Identity6gradients/model/conv3/Conv2D_grad/Conv2DBackpropFilter3^gradients/model/conv3/Conv2D_grad/tuple/group_deps*
T0*I
_class?
=;loc:@gradients/model/conv3/Conv2D_grad/Conv2DBackpropFilter*'
_output_shapes
:@�
�
*gradients/model_2/conv3/Conv2D_grad/ShapeNShapeNmodel_2/conv2/MaxPool2D/MaxPoolconv3/weights/read*
T0*
out_type0*
N* 
_output_shapes
::
�
7gradients/model_2/conv3/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput*gradients/model_2/conv3/Conv2D_grad/ShapeNconv3/weights/read=gradients/model_2/conv3/BiasAdd_grad/tuple/control_dependency*
paddingSAME*/
_output_shapes
:���������@*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(
�
8gradients/model_2/conv3/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFiltermodel_2/conv2/MaxPool2D/MaxPool,gradients/model_2/conv3/Conv2D_grad/ShapeN:1=gradients/model_2/conv3/BiasAdd_grad/tuple/control_dependency*
paddingSAME*'
_output_shapes
:@�*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
�
4gradients/model_2/conv3/Conv2D_grad/tuple/group_depsNoOp9^gradients/model_2/conv3/Conv2D_grad/Conv2DBackpropFilter8^gradients/model_2/conv3/Conv2D_grad/Conv2DBackpropInput
�
<gradients/model_2/conv3/Conv2D_grad/tuple/control_dependencyIdentity7gradients/model_2/conv3/Conv2D_grad/Conv2DBackpropInput5^gradients/model_2/conv3/Conv2D_grad/tuple/group_deps*/
_output_shapes
:���������@*
T0*J
_class@
><loc:@gradients/model_2/conv3/Conv2D_grad/Conv2DBackpropInput
�
>gradients/model_2/conv3/Conv2D_grad/tuple/control_dependency_1Identity8gradients/model_2/conv3/Conv2D_grad/Conv2DBackpropFilter5^gradients/model_2/conv3/Conv2D_grad/tuple/group_deps*'
_output_shapes
:@�*
T0*K
_classA
?=loc:@gradients/model_2/conv3/Conv2D_grad/Conv2DBackpropFilter
�
gradients/AddN_5AddN?gradients/model_1/conv3/BiasAdd_grad/tuple/control_dependency_1=gradients/model/conv3/BiasAdd_grad/tuple/control_dependency_1?gradients/model_2/conv3/BiasAdd_grad/tuple/control_dependency_1*
T0*C
_class9
75loc:@gradients/model_1/conv3/BiasAdd_grad/BiasAddGrad*
N*
_output_shapes	
:�
�
:gradients/model_1/conv2/MaxPool2D/MaxPool_grad/MaxPoolGradMaxPoolGradmodel_1/conv2/conv2/Relumodel_1/conv2/MaxPool2D/MaxPool<gradients/model_1/conv3/Conv2D_grad/tuple/control_dependency*
T0*
data_formatNHWC*
strides
*
ksize
*
paddingSAME*/
_output_shapes
:���������
@
�
8gradients/model/conv2/MaxPool2D/MaxPool_grad/MaxPoolGradMaxPoolGradmodel/conv2/conv2/Relumodel/conv2/MaxPool2D/MaxPool:gradients/model/conv3/Conv2D_grad/tuple/control_dependency*
T0*
data_formatNHWC*
strides
*
ksize
*
paddingSAME*/
_output_shapes
:���������
@
�
:gradients/model_2/conv2/MaxPool2D/MaxPool_grad/MaxPoolGradMaxPoolGradmodel_2/conv2/conv2/Relumodel_2/conv2/MaxPool2D/MaxPool<gradients/model_2/conv3/Conv2D_grad/tuple/control_dependency*/
_output_shapes
:���������
@*
T0*
data_formatNHWC*
strides
*
ksize
*
paddingSAME
�
gradients/AddN_6AddN>gradients/model_1/conv3/Conv2D_grad/tuple/control_dependency_1<gradients/model/conv3/Conv2D_grad/tuple/control_dependency_1>gradients/model_2/conv3/Conv2D_grad/tuple/control_dependency_1*
N*'
_output_shapes
:@�*
T0*K
_classA
?=loc:@gradients/model_1/conv3/Conv2D_grad/Conv2DBackpropFilter
�
0gradients/model_1/conv2/conv2/Relu_grad/ReluGradReluGrad:gradients/model_1/conv2/MaxPool2D/MaxPool_grad/MaxPoolGradmodel_1/conv2/conv2/Relu*
T0*/
_output_shapes
:���������
@
�
.gradients/model/conv2/conv2/Relu_grad/ReluGradReluGrad8gradients/model/conv2/MaxPool2D/MaxPool_grad/MaxPoolGradmodel/conv2/conv2/Relu*
T0*/
_output_shapes
:���������
@
�
0gradients/model_2/conv2/conv2/Relu_grad/ReluGradReluGrad:gradients/model_2/conv2/MaxPool2D/MaxPool_grad/MaxPoolGradmodel_2/conv2/conv2/Relu*
T0*/
_output_shapes
:���������
@
�
0gradients/model_1/conv2/BiasAdd_grad/BiasAddGradBiasAddGrad0gradients/model_1/conv2/conv2/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes
:@
�
5gradients/model_1/conv2/BiasAdd_grad/tuple/group_depsNoOp1^gradients/model_1/conv2/BiasAdd_grad/BiasAddGrad1^gradients/model_1/conv2/conv2/Relu_grad/ReluGrad
�
=gradients/model_1/conv2/BiasAdd_grad/tuple/control_dependencyIdentity0gradients/model_1/conv2/conv2/Relu_grad/ReluGrad6^gradients/model_1/conv2/BiasAdd_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/model_1/conv2/conv2/Relu_grad/ReluGrad*/
_output_shapes
:���������
@
�
?gradients/model_1/conv2/BiasAdd_grad/tuple/control_dependency_1Identity0gradients/model_1/conv2/BiasAdd_grad/BiasAddGrad6^gradients/model_1/conv2/BiasAdd_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/model_1/conv2/BiasAdd_grad/BiasAddGrad*
_output_shapes
:@
�
.gradients/model/conv2/BiasAdd_grad/BiasAddGradBiasAddGrad.gradients/model/conv2/conv2/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes
:@
�
3gradients/model/conv2/BiasAdd_grad/tuple/group_depsNoOp/^gradients/model/conv2/BiasAdd_grad/BiasAddGrad/^gradients/model/conv2/conv2/Relu_grad/ReluGrad
�
;gradients/model/conv2/BiasAdd_grad/tuple/control_dependencyIdentity.gradients/model/conv2/conv2/Relu_grad/ReluGrad4^gradients/model/conv2/BiasAdd_grad/tuple/group_deps*/
_output_shapes
:���������
@*
T0*A
_class7
53loc:@gradients/model/conv2/conv2/Relu_grad/ReluGrad
�
=gradients/model/conv2/BiasAdd_grad/tuple/control_dependency_1Identity.gradients/model/conv2/BiasAdd_grad/BiasAddGrad4^gradients/model/conv2/BiasAdd_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients/model/conv2/BiasAdd_grad/BiasAddGrad*
_output_shapes
:@
�
0gradients/model_2/conv2/BiasAdd_grad/BiasAddGradBiasAddGrad0gradients/model_2/conv2/conv2/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes
:@
�
5gradients/model_2/conv2/BiasAdd_grad/tuple/group_depsNoOp1^gradients/model_2/conv2/BiasAdd_grad/BiasAddGrad1^gradients/model_2/conv2/conv2/Relu_grad/ReluGrad
�
=gradients/model_2/conv2/BiasAdd_grad/tuple/control_dependencyIdentity0gradients/model_2/conv2/conv2/Relu_grad/ReluGrad6^gradients/model_2/conv2/BiasAdd_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/model_2/conv2/conv2/Relu_grad/ReluGrad*/
_output_shapes
:���������
@
�
?gradients/model_2/conv2/BiasAdd_grad/tuple/control_dependency_1Identity0gradients/model_2/conv2/BiasAdd_grad/BiasAddGrad6^gradients/model_2/conv2/BiasAdd_grad/tuple/group_deps*
_output_shapes
:@*
T0*C
_class9
75loc:@gradients/model_2/conv2/BiasAdd_grad/BiasAddGrad
�
*gradients/model_1/conv2/Conv2D_grad/ShapeNShapeNmodel_1/conv1/MaxPool2D/MaxPoolconv2/weights/read*
T0*
out_type0*
N* 
_output_shapes
::
�
7gradients/model_1/conv2/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput*gradients/model_1/conv2/Conv2D_grad/ShapeNconv2/weights/read=gradients/model_1/conv2/BiasAdd_grad/tuple/control_dependency*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*/
_output_shapes
:���������
 
�
8gradients/model_1/conv2/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFiltermodel_1/conv1/MaxPool2D/MaxPool,gradients/model_1/conv2/Conv2D_grad/ShapeN:1=gradients/model_1/conv2/BiasAdd_grad/tuple/control_dependency*
paddingSAME*&
_output_shapes
: @*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
�
4gradients/model_1/conv2/Conv2D_grad/tuple/group_depsNoOp9^gradients/model_1/conv2/Conv2D_grad/Conv2DBackpropFilter8^gradients/model_1/conv2/Conv2D_grad/Conv2DBackpropInput
�
<gradients/model_1/conv2/Conv2D_grad/tuple/control_dependencyIdentity7gradients/model_1/conv2/Conv2D_grad/Conv2DBackpropInput5^gradients/model_1/conv2/Conv2D_grad/tuple/group_deps*/
_output_shapes
:���������
 *
T0*J
_class@
><loc:@gradients/model_1/conv2/Conv2D_grad/Conv2DBackpropInput
�
>gradients/model_1/conv2/Conv2D_grad/tuple/control_dependency_1Identity8gradients/model_1/conv2/Conv2D_grad/Conv2DBackpropFilter5^gradients/model_1/conv2/Conv2D_grad/tuple/group_deps*
T0*K
_classA
?=loc:@gradients/model_1/conv2/Conv2D_grad/Conv2DBackpropFilter*&
_output_shapes
: @
�
(gradients/model/conv2/Conv2D_grad/ShapeNShapeNmodel/conv1/MaxPool2D/MaxPoolconv2/weights/read*
T0*
out_type0*
N* 
_output_shapes
::
�
5gradients/model/conv2/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput(gradients/model/conv2/Conv2D_grad/ShapeNconv2/weights/read;gradients/model/conv2/BiasAdd_grad/tuple/control_dependency*/
_output_shapes
:���������
 *
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME
�
6gradients/model/conv2/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFiltermodel/conv1/MaxPool2D/MaxPool*gradients/model/conv2/Conv2D_grad/ShapeN:1;gradients/model/conv2/BiasAdd_grad/tuple/control_dependency*&
_output_shapes
: @*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME
�
2gradients/model/conv2/Conv2D_grad/tuple/group_depsNoOp7^gradients/model/conv2/Conv2D_grad/Conv2DBackpropFilter6^gradients/model/conv2/Conv2D_grad/Conv2DBackpropInput
�
:gradients/model/conv2/Conv2D_grad/tuple/control_dependencyIdentity5gradients/model/conv2/Conv2D_grad/Conv2DBackpropInput3^gradients/model/conv2/Conv2D_grad/tuple/group_deps*
T0*H
_class>
<:loc:@gradients/model/conv2/Conv2D_grad/Conv2DBackpropInput*/
_output_shapes
:���������
 
�
<gradients/model/conv2/Conv2D_grad/tuple/control_dependency_1Identity6gradients/model/conv2/Conv2D_grad/Conv2DBackpropFilter3^gradients/model/conv2/Conv2D_grad/tuple/group_deps*&
_output_shapes
: @*
T0*I
_class?
=;loc:@gradients/model/conv2/Conv2D_grad/Conv2DBackpropFilter
�
*gradients/model_2/conv2/Conv2D_grad/ShapeNShapeNmodel_2/conv1/MaxPool2D/MaxPoolconv2/weights/read*
T0*
out_type0*
N* 
_output_shapes
::
�
7gradients/model_2/conv2/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput*gradients/model_2/conv2/Conv2D_grad/ShapeNconv2/weights/read=gradients/model_2/conv2/BiasAdd_grad/tuple/control_dependency*/
_output_shapes
:���������
 *
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME
�
8gradients/model_2/conv2/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFiltermodel_2/conv1/MaxPool2D/MaxPool,gradients/model_2/conv2/Conv2D_grad/ShapeN:1=gradients/model_2/conv2/BiasAdd_grad/tuple/control_dependency*
paddingSAME*&
_output_shapes
: @*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(
�
4gradients/model_2/conv2/Conv2D_grad/tuple/group_depsNoOp9^gradients/model_2/conv2/Conv2D_grad/Conv2DBackpropFilter8^gradients/model_2/conv2/Conv2D_grad/Conv2DBackpropInput
�
<gradients/model_2/conv2/Conv2D_grad/tuple/control_dependencyIdentity7gradients/model_2/conv2/Conv2D_grad/Conv2DBackpropInput5^gradients/model_2/conv2/Conv2D_grad/tuple/group_deps*
T0*J
_class@
><loc:@gradients/model_2/conv2/Conv2D_grad/Conv2DBackpropInput*/
_output_shapes
:���������
 
�
>gradients/model_2/conv2/Conv2D_grad/tuple/control_dependency_1Identity8gradients/model_2/conv2/Conv2D_grad/Conv2DBackpropFilter5^gradients/model_2/conv2/Conv2D_grad/tuple/group_deps*
T0*K
_classA
?=loc:@gradients/model_2/conv2/Conv2D_grad/Conv2DBackpropFilter*&
_output_shapes
: @
�
gradients/AddN_7AddN?gradients/model_1/conv2/BiasAdd_grad/tuple/control_dependency_1=gradients/model/conv2/BiasAdd_grad/tuple/control_dependency_1?gradients/model_2/conv2/BiasAdd_grad/tuple/control_dependency_1*
N*
_output_shapes
:@*
T0*C
_class9
75loc:@gradients/model_1/conv2/BiasAdd_grad/BiasAddGrad
�
:gradients/model_1/conv1/MaxPool2D/MaxPool_grad/MaxPoolGradMaxPoolGradmodel_1/conv1/conv1/Relumodel_1/conv1/MaxPool2D/MaxPool<gradients/model_1/conv2/Conv2D_grad/tuple/control_dependency*
T0*
data_formatNHWC*
strides
*
ksize
*
paddingSAME*/
_output_shapes
:���������
 
�
8gradients/model/conv1/MaxPool2D/MaxPool_grad/MaxPoolGradMaxPoolGradmodel/conv1/conv1/Relumodel/conv1/MaxPool2D/MaxPool:gradients/model/conv2/Conv2D_grad/tuple/control_dependency*
T0*
strides
*
data_formatNHWC*
ksize
*
paddingSAME*/
_output_shapes
:���������
 
�
:gradients/model_2/conv1/MaxPool2D/MaxPool_grad/MaxPoolGradMaxPoolGradmodel_2/conv1/conv1/Relumodel_2/conv1/MaxPool2D/MaxPool<gradients/model_2/conv2/Conv2D_grad/tuple/control_dependency*
ksize
*
paddingSAME*/
_output_shapes
:���������
 *
T0*
strides
*
data_formatNHWC
�
gradients/AddN_8AddN>gradients/model_1/conv2/Conv2D_grad/tuple/control_dependency_1<gradients/model/conv2/Conv2D_grad/tuple/control_dependency_1>gradients/model_2/conv2/Conv2D_grad/tuple/control_dependency_1*
N*&
_output_shapes
: @*
T0*K
_classA
?=loc:@gradients/model_1/conv2/Conv2D_grad/Conv2DBackpropFilter
�
0gradients/model_1/conv1/conv1/Relu_grad/ReluGradReluGrad:gradients/model_1/conv1/MaxPool2D/MaxPool_grad/MaxPoolGradmodel_1/conv1/conv1/Relu*
T0*/
_output_shapes
:���������
 
�
.gradients/model/conv1/conv1/Relu_grad/ReluGradReluGrad8gradients/model/conv1/MaxPool2D/MaxPool_grad/MaxPoolGradmodel/conv1/conv1/Relu*/
_output_shapes
:���������
 *
T0
�
0gradients/model_2/conv1/conv1/Relu_grad/ReluGradReluGrad:gradients/model_2/conv1/MaxPool2D/MaxPool_grad/MaxPoolGradmodel_2/conv1/conv1/Relu*
T0*/
_output_shapes
:���������
 
�
0gradients/model_1/conv1/BiasAdd_grad/BiasAddGradBiasAddGrad0gradients/model_1/conv1/conv1/Relu_grad/ReluGrad*
data_formatNHWC*
_output_shapes
: *
T0
�
5gradients/model_1/conv1/BiasAdd_grad/tuple/group_depsNoOp1^gradients/model_1/conv1/BiasAdd_grad/BiasAddGrad1^gradients/model_1/conv1/conv1/Relu_grad/ReluGrad
�
=gradients/model_1/conv1/BiasAdd_grad/tuple/control_dependencyIdentity0gradients/model_1/conv1/conv1/Relu_grad/ReluGrad6^gradients/model_1/conv1/BiasAdd_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/model_1/conv1/conv1/Relu_grad/ReluGrad*/
_output_shapes
:���������
 
�
?gradients/model_1/conv1/BiasAdd_grad/tuple/control_dependency_1Identity0gradients/model_1/conv1/BiasAdd_grad/BiasAddGrad6^gradients/model_1/conv1/BiasAdd_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/model_1/conv1/BiasAdd_grad/BiasAddGrad*
_output_shapes
: 
�
.gradients/model/conv1/BiasAdd_grad/BiasAddGradBiasAddGrad.gradients/model/conv1/conv1/Relu_grad/ReluGrad*
data_formatNHWC*
_output_shapes
: *
T0
�
3gradients/model/conv1/BiasAdd_grad/tuple/group_depsNoOp/^gradients/model/conv1/BiasAdd_grad/BiasAddGrad/^gradients/model/conv1/conv1/Relu_grad/ReluGrad
�
;gradients/model/conv1/BiasAdd_grad/tuple/control_dependencyIdentity.gradients/model/conv1/conv1/Relu_grad/ReluGrad4^gradients/model/conv1/BiasAdd_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients/model/conv1/conv1/Relu_grad/ReluGrad*/
_output_shapes
:���������
 
�
=gradients/model/conv1/BiasAdd_grad/tuple/control_dependency_1Identity.gradients/model/conv1/BiasAdd_grad/BiasAddGrad4^gradients/model/conv1/BiasAdd_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients/model/conv1/BiasAdd_grad/BiasAddGrad*
_output_shapes
: 
�
0gradients/model_2/conv1/BiasAdd_grad/BiasAddGradBiasAddGrad0gradients/model_2/conv1/conv1/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes
: 
�
5gradients/model_2/conv1/BiasAdd_grad/tuple/group_depsNoOp1^gradients/model_2/conv1/BiasAdd_grad/BiasAddGrad1^gradients/model_2/conv1/conv1/Relu_grad/ReluGrad
�
=gradients/model_2/conv1/BiasAdd_grad/tuple/control_dependencyIdentity0gradients/model_2/conv1/conv1/Relu_grad/ReluGrad6^gradients/model_2/conv1/BiasAdd_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/model_2/conv1/conv1/Relu_grad/ReluGrad*/
_output_shapes
:���������
 
�
?gradients/model_2/conv1/BiasAdd_grad/tuple/control_dependency_1Identity0gradients/model_2/conv1/BiasAdd_grad/BiasAddGrad6^gradients/model_2/conv1/BiasAdd_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/model_2/conv1/BiasAdd_grad/BiasAddGrad*
_output_shapes
: 
�
*gradients/model_1/conv1/Conv2D_grad/ShapeNShapeNpositive_inputconv1/weights/read*
T0*
out_type0*
N* 
_output_shapes
::
�
7gradients/model_1/conv1/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput*gradients/model_1/conv1/Conv2D_grad/ShapeNconv1/weights/read=gradients/model_1/conv1/BiasAdd_grad/tuple/control_dependency*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*/
_output_shapes
:���������
*
	dilations
*
T0
�
8gradients/model_1/conv1/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterpositive_input,gradients/model_1/conv1/Conv2D_grad/ShapeN:1=gradients/model_1/conv1/BiasAdd_grad/tuple/control_dependency*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*&
_output_shapes
: 
�
4gradients/model_1/conv1/Conv2D_grad/tuple/group_depsNoOp9^gradients/model_1/conv1/Conv2D_grad/Conv2DBackpropFilter8^gradients/model_1/conv1/Conv2D_grad/Conv2DBackpropInput
�
<gradients/model_1/conv1/Conv2D_grad/tuple/control_dependencyIdentity7gradients/model_1/conv1/Conv2D_grad/Conv2DBackpropInput5^gradients/model_1/conv1/Conv2D_grad/tuple/group_deps*
T0*J
_class@
><loc:@gradients/model_1/conv1/Conv2D_grad/Conv2DBackpropInput*/
_output_shapes
:���������

�
>gradients/model_1/conv1/Conv2D_grad/tuple/control_dependency_1Identity8gradients/model_1/conv1/Conv2D_grad/Conv2DBackpropFilter5^gradients/model_1/conv1/Conv2D_grad/tuple/group_deps*&
_output_shapes
: *
T0*K
_classA
?=loc:@gradients/model_1/conv1/Conv2D_grad/Conv2DBackpropFilter
�
(gradients/model/conv1/Conv2D_grad/ShapeNShapeNanchor_inputconv1/weights/read*
N* 
_output_shapes
::*
T0*
out_type0
�
5gradients/model/conv1/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput(gradients/model/conv1/Conv2D_grad/ShapeNconv1/weights/read;gradients/model/conv1/BiasAdd_grad/tuple/control_dependency*/
_output_shapes
:���������
*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME
�
6gradients/model/conv1/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilteranchor_input*gradients/model/conv1/Conv2D_grad/ShapeN:1;gradients/model/conv1/BiasAdd_grad/tuple/control_dependency*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*&
_output_shapes
: 
�
2gradients/model/conv1/Conv2D_grad/tuple/group_depsNoOp7^gradients/model/conv1/Conv2D_grad/Conv2DBackpropFilter6^gradients/model/conv1/Conv2D_grad/Conv2DBackpropInput
�
:gradients/model/conv1/Conv2D_grad/tuple/control_dependencyIdentity5gradients/model/conv1/Conv2D_grad/Conv2DBackpropInput3^gradients/model/conv1/Conv2D_grad/tuple/group_deps*
T0*H
_class>
<:loc:@gradients/model/conv1/Conv2D_grad/Conv2DBackpropInput*/
_output_shapes
:���������

�
<gradients/model/conv1/Conv2D_grad/tuple/control_dependency_1Identity6gradients/model/conv1/Conv2D_grad/Conv2DBackpropFilter3^gradients/model/conv1/Conv2D_grad/tuple/group_deps*&
_output_shapes
: *
T0*I
_class?
=;loc:@gradients/model/conv1/Conv2D_grad/Conv2DBackpropFilter
�
*gradients/model_2/conv1/Conv2D_grad/ShapeNShapeNnegative_inputconv1/weights/read*
T0*
out_type0*
N* 
_output_shapes
::
�
7gradients/model_2/conv1/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput*gradients/model_2/conv1/Conv2D_grad/ShapeNconv1/weights/read=gradients/model_2/conv1/BiasAdd_grad/tuple/control_dependency*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*/
_output_shapes
:���������
*
	dilations
*
T0
�
8gradients/model_2/conv1/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilternegative_input,gradients/model_2/conv1/Conv2D_grad/ShapeN:1=gradients/model_2/conv1/BiasAdd_grad/tuple/control_dependency*
paddingSAME*&
_output_shapes
: *
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
�
4gradients/model_2/conv1/Conv2D_grad/tuple/group_depsNoOp9^gradients/model_2/conv1/Conv2D_grad/Conv2DBackpropFilter8^gradients/model_2/conv1/Conv2D_grad/Conv2DBackpropInput
�
<gradients/model_2/conv1/Conv2D_grad/tuple/control_dependencyIdentity7gradients/model_2/conv1/Conv2D_grad/Conv2DBackpropInput5^gradients/model_2/conv1/Conv2D_grad/tuple/group_deps*
T0*J
_class@
><loc:@gradients/model_2/conv1/Conv2D_grad/Conv2DBackpropInput*/
_output_shapes
:���������

�
>gradients/model_2/conv1/Conv2D_grad/tuple/control_dependency_1Identity8gradients/model_2/conv1/Conv2D_grad/Conv2DBackpropFilter5^gradients/model_2/conv1/Conv2D_grad/tuple/group_deps*
T0*K
_classA
?=loc:@gradients/model_2/conv1/Conv2D_grad/Conv2DBackpropFilter*&
_output_shapes
: 
�
gradients/AddN_9AddN?gradients/model_1/conv1/BiasAdd_grad/tuple/control_dependency_1=gradients/model/conv1/BiasAdd_grad/tuple/control_dependency_1?gradients/model_2/conv1/BiasAdd_grad/tuple/control_dependency_1*
N*
_output_shapes
: *
T0*C
_class9
75loc:@gradients/model_1/conv1/BiasAdd_grad/BiasAddGrad
�
gradients/AddN_10AddN>gradients/model_1/conv1/Conv2D_grad/tuple/control_dependency_1<gradients/model/conv1/Conv2D_grad/tuple/control_dependency_1>gradients/model_2/conv1/Conv2D_grad/tuple/control_dependency_1*
N*&
_output_shapes
: *
T0*K
_classA
?=loc:@gradients/model_1/conv1/Conv2D_grad/Conv2DBackpropFilter
�
8conv1/weights/Momentum/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*%
valueB"             * 
_class
loc:@conv1/weights
�
.conv1/weights/Momentum/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    * 
_class
loc:@conv1/weights
�
(conv1/weights/Momentum/Initializer/zerosFill8conv1/weights/Momentum/Initializer/zeros/shape_as_tensor.conv1/weights/Momentum/Initializer/zeros/Const*
T0*

index_type0* 
_class
loc:@conv1/weights*&
_output_shapes
: 
�
conv1/weights/Momentum
VariableV2*
shared_name * 
_class
loc:@conv1/weights*
	container *
shape: *
dtype0*&
_output_shapes
: 
�
conv1/weights/Momentum/AssignAssignconv1/weights/Momentum(conv1/weights/Momentum/Initializer/zeros*
use_locking(*
T0* 
_class
loc:@conv1/weights*
validate_shape(*&
_output_shapes
: 
�
conv1/weights/Momentum/readIdentityconv1/weights/Momentum*
T0* 
_class
loc:@conv1/weights*&
_output_shapes
: 
�
'conv1/biases/Momentum/Initializer/zerosConst*
valueB *    *
_class
loc:@conv1/biases*
dtype0*
_output_shapes
: 
�
conv1/biases/Momentum
VariableV2*
	container *
shape: *
dtype0*
_output_shapes
: *
shared_name *
_class
loc:@conv1/biases
�
conv1/biases/Momentum/AssignAssignconv1/biases/Momentum'conv1/biases/Momentum/Initializer/zeros*
use_locking(*
T0*
_class
loc:@conv1/biases*
validate_shape(*
_output_shapes
: 
�
conv1/biases/Momentum/readIdentityconv1/biases/Momentum*
T0*
_class
loc:@conv1/biases*
_output_shapes
: 
�
8conv2/weights/Momentum/Initializer/zeros/shape_as_tensorConst*%
valueB"          @   * 
_class
loc:@conv2/weights*
dtype0*
_output_shapes
:
�
.conv2/weights/Momentum/Initializer/zeros/ConstConst*
valueB
 *    * 
_class
loc:@conv2/weights*
dtype0*
_output_shapes
: 
�
(conv2/weights/Momentum/Initializer/zerosFill8conv2/weights/Momentum/Initializer/zeros/shape_as_tensor.conv2/weights/Momentum/Initializer/zeros/Const*
T0*

index_type0* 
_class
loc:@conv2/weights*&
_output_shapes
: @
�
conv2/weights/Momentum
VariableV2*
dtype0*&
_output_shapes
: @*
shared_name * 
_class
loc:@conv2/weights*
	container *
shape: @
�
conv2/weights/Momentum/AssignAssignconv2/weights/Momentum(conv2/weights/Momentum/Initializer/zeros*
use_locking(*
T0* 
_class
loc:@conv2/weights*
validate_shape(*&
_output_shapes
: @
�
conv2/weights/Momentum/readIdentityconv2/weights/Momentum*
T0* 
_class
loc:@conv2/weights*&
_output_shapes
: @
�
'conv2/biases/Momentum/Initializer/zerosConst*
valueB@*    *
_class
loc:@conv2/biases*
dtype0*
_output_shapes
:@
�
conv2/biases/Momentum
VariableV2*
dtype0*
_output_shapes
:@*
shared_name *
_class
loc:@conv2/biases*
	container *
shape:@
�
conv2/biases/Momentum/AssignAssignconv2/biases/Momentum'conv2/biases/Momentum/Initializer/zeros*
use_locking(*
T0*
_class
loc:@conv2/biases*
validate_shape(*
_output_shapes
:@
�
conv2/biases/Momentum/readIdentityconv2/biases/Momentum*
T0*
_class
loc:@conv2/biases*
_output_shapes
:@
�
8conv3/weights/Momentum/Initializer/zeros/shape_as_tensorConst*%
valueB"      @   �   * 
_class
loc:@conv3/weights*
dtype0*
_output_shapes
:
�
.conv3/weights/Momentum/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    * 
_class
loc:@conv3/weights
�
(conv3/weights/Momentum/Initializer/zerosFill8conv3/weights/Momentum/Initializer/zeros/shape_as_tensor.conv3/weights/Momentum/Initializer/zeros/Const*
T0*

index_type0* 
_class
loc:@conv3/weights*'
_output_shapes
:@�
�
conv3/weights/Momentum
VariableV2*
dtype0*'
_output_shapes
:@�*
shared_name * 
_class
loc:@conv3/weights*
	container *
shape:@�
�
conv3/weights/Momentum/AssignAssignconv3/weights/Momentum(conv3/weights/Momentum/Initializer/zeros*
use_locking(*
T0* 
_class
loc:@conv3/weights*
validate_shape(*'
_output_shapes
:@�
�
conv3/weights/Momentum/readIdentityconv3/weights/Momentum*'
_output_shapes
:@�*
T0* 
_class
loc:@conv3/weights
�
'conv3/biases/Momentum/Initializer/zerosConst*
dtype0*
_output_shapes	
:�*
valueB�*    *
_class
loc:@conv3/biases
�
conv3/biases/Momentum
VariableV2*
dtype0*
_output_shapes	
:�*
shared_name *
_class
loc:@conv3/biases*
	container *
shape:�
�
conv3/biases/Momentum/AssignAssignconv3/biases/Momentum'conv3/biases/Momentum/Initializer/zeros*
T0*
_class
loc:@conv3/biases*
validate_shape(*
_output_shapes	
:�*
use_locking(
�
conv3/biases/Momentum/readIdentityconv3/biases/Momentum*
_output_shapes	
:�*
T0*
_class
loc:@conv3/biases
�
8conv4/weights/Momentum/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*%
valueB"      �      * 
_class
loc:@conv4/weights
�
.conv4/weights/Momentum/Initializer/zeros/ConstConst*
valueB
 *    * 
_class
loc:@conv4/weights*
dtype0*
_output_shapes
: 
�
(conv4/weights/Momentum/Initializer/zerosFill8conv4/weights/Momentum/Initializer/zeros/shape_as_tensor.conv4/weights/Momentum/Initializer/zeros/Const*(
_output_shapes
:��*
T0*

index_type0* 
_class
loc:@conv4/weights
�
conv4/weights/Momentum
VariableV2* 
_class
loc:@conv4/weights*
	container *
shape:��*
dtype0*(
_output_shapes
:��*
shared_name 
�
conv4/weights/Momentum/AssignAssignconv4/weights/Momentum(conv4/weights/Momentum/Initializer/zeros*
validate_shape(*(
_output_shapes
:��*
use_locking(*
T0* 
_class
loc:@conv4/weights
�
conv4/weights/Momentum/readIdentityconv4/weights/Momentum*(
_output_shapes
:��*
T0* 
_class
loc:@conv4/weights
�
'conv4/biases/Momentum/Initializer/zerosConst*
valueB�*    *
_class
loc:@conv4/biases*
dtype0*
_output_shapes	
:�
�
conv4/biases/Momentum
VariableV2*
dtype0*
_output_shapes	
:�*
shared_name *
_class
loc:@conv4/biases*
	container *
shape:�
�
conv4/biases/Momentum/AssignAssignconv4/biases/Momentum'conv4/biases/Momentum/Initializer/zeros*
T0*
_class
loc:@conv4/biases*
validate_shape(*
_output_shapes	
:�*
use_locking(
�
conv4/biases/Momentum/readIdentityconv4/biases/Momentum*
_output_shapes	
:�*
T0*
_class
loc:@conv4/biases
�
8conv5/weights/Momentum/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*%
valueB"            * 
_class
loc:@conv5/weights
�
.conv5/weights/Momentum/Initializer/zeros/ConstConst*
valueB
 *    * 
_class
loc:@conv5/weights*
dtype0*
_output_shapes
: 
�
(conv5/weights/Momentum/Initializer/zerosFill8conv5/weights/Momentum/Initializer/zeros/shape_as_tensor.conv5/weights/Momentum/Initializer/zeros/Const*'
_output_shapes
:�*
T0*

index_type0* 
_class
loc:@conv5/weights
�
conv5/weights/Momentum
VariableV2*
	container *
shape:�*
dtype0*'
_output_shapes
:�*
shared_name * 
_class
loc:@conv5/weights
�
conv5/weights/Momentum/AssignAssignconv5/weights/Momentum(conv5/weights/Momentum/Initializer/zeros*
validate_shape(*'
_output_shapes
:�*
use_locking(*
T0* 
_class
loc:@conv5/weights
�
conv5/weights/Momentum/readIdentityconv5/weights/Momentum*'
_output_shapes
:�*
T0* 
_class
loc:@conv5/weights
�
'conv5/biases/Momentum/Initializer/zerosConst*
valueB*    *
_class
loc:@conv5/biases*
dtype0*
_output_shapes
:
�
conv5/biases/Momentum
VariableV2*
shared_name *
_class
loc:@conv5/biases*
	container *
shape:*
dtype0*
_output_shapes
:
�
conv5/biases/Momentum/AssignAssignconv5/biases/Momentum'conv5/biases/Momentum/Initializer/zeros*
T0*
_class
loc:@conv5/biases*
validate_shape(*
_output_shapes
:*
use_locking(
�
conv5/biases/Momentum/readIdentityconv5/biases/Momentum*
_output_shapes
:*
T0*
_class
loc:@conv5/biases
[
Momentum/learning_rateConst*
valueB
 *
�#<*
dtype0*
_output_shapes
: 
V
Momentum/momentumConst*
valueB
 *�p}?*
dtype0*
_output_shapes
: 
�
+Momentum/update_conv1/weights/ApplyMomentumApplyMomentumconv1/weightsconv1/weights/MomentumMomentum/learning_rategradients/AddN_10Momentum/momentum*
T0* 
_class
loc:@conv1/weights*
use_nesterov(*&
_output_shapes
: *
use_locking( 
�
*Momentum/update_conv1/biases/ApplyMomentumApplyMomentumconv1/biasesconv1/biases/MomentumMomentum/learning_rategradients/AddN_9Momentum/momentum*
T0*
_class
loc:@conv1/biases*
use_nesterov(*
_output_shapes
: *
use_locking( 
�
+Momentum/update_conv2/weights/ApplyMomentumApplyMomentumconv2/weightsconv2/weights/MomentumMomentum/learning_rategradients/AddN_8Momentum/momentum*
use_locking( *
T0* 
_class
loc:@conv2/weights*
use_nesterov(*&
_output_shapes
: @
�
*Momentum/update_conv2/biases/ApplyMomentumApplyMomentumconv2/biasesconv2/biases/MomentumMomentum/learning_rategradients/AddN_7Momentum/momentum*
use_nesterov(*
_output_shapes
:@*
use_locking( *
T0*
_class
loc:@conv2/biases
�
+Momentum/update_conv3/weights/ApplyMomentumApplyMomentumconv3/weightsconv3/weights/MomentumMomentum/learning_rategradients/AddN_6Momentum/momentum*
use_locking( *
T0* 
_class
loc:@conv3/weights*
use_nesterov(*'
_output_shapes
:@�
�
*Momentum/update_conv3/biases/ApplyMomentumApplyMomentumconv3/biasesconv3/biases/MomentumMomentum/learning_rategradients/AddN_5Momentum/momentum*
use_nesterov(*
_output_shapes	
:�*
use_locking( *
T0*
_class
loc:@conv3/biases
�
+Momentum/update_conv4/weights/ApplyMomentumApplyMomentumconv4/weightsconv4/weights/MomentumMomentum/learning_rategradients/AddN_4Momentum/momentum*
use_nesterov(*(
_output_shapes
:��*
use_locking( *
T0* 
_class
loc:@conv4/weights
�
*Momentum/update_conv4/biases/ApplyMomentumApplyMomentumconv4/biasesconv4/biases/MomentumMomentum/learning_rategradients/AddN_3Momentum/momentum*
use_nesterov(*
_output_shapes	
:�*
use_locking( *
T0*
_class
loc:@conv4/biases
�
+Momentum/update_conv5/weights/ApplyMomentumApplyMomentumconv5/weightsconv5/weights/MomentumMomentum/learning_rategradients/AddN_2Momentum/momentum*
use_locking( *
T0* 
_class
loc:@conv5/weights*
use_nesterov(*'
_output_shapes
:�
�
*Momentum/update_conv5/biases/ApplyMomentumApplyMomentumconv5/biasesconv5/biases/MomentumMomentum/learning_rategradients/AddN_1Momentum/momentum*
use_locking( *
T0*
_class
loc:@conv5/biases*
use_nesterov(*
_output_shapes
:
�
Momentum/updateNoOp+^Momentum/update_conv1/biases/ApplyMomentum,^Momentum/update_conv1/weights/ApplyMomentum+^Momentum/update_conv2/biases/ApplyMomentum,^Momentum/update_conv2/weights/ApplyMomentum+^Momentum/update_conv3/biases/ApplyMomentum,^Momentum/update_conv3/weights/ApplyMomentum+^Momentum/update_conv4/biases/ApplyMomentum,^Momentum/update_conv4/weights/ApplyMomentum+^Momentum/update_conv5/biases/ApplyMomentum,^Momentum/update_conv5/weights/ApplyMomentum

Momentum/valueConst^Momentum/update*
_class
loc:@Variable*
value	B :*
dtype0*
_output_shapes
: 
�
Momentum	AssignAddVariableMomentum/value*
use_locking( *
T0*
_class
loc:@Variable*
_output_shapes
: 
Y
save/filename/inputConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
n
save/filenamePlaceholderWithDefaultsave/filename/input*
dtype0*
_output_shapes
: *
shape: 
e

save/ConstPlaceholderWithDefaultsave/filename*
dtype0*
_output_shapes
: *
shape: 
�
save/SaveV2/tensor_namesConst*�
value�B�BVariableBconv1/biasesBconv1/biases/MomentumBconv1/weightsBconv1/weights/MomentumBconv2/biasesBconv2/biases/MomentumBconv2/weightsBconv2/weights/MomentumBconv3/biasesBconv3/biases/MomentumBconv3/weightsBconv3/weights/MomentumBconv4/biasesBconv4/biases/MomentumBconv4/weightsBconv4/weights/MomentumBconv5/biasesBconv5/biases/MomentumBconv5/weightsBconv5/weights/Momentum*
dtype0*
_output_shapes
:
�
save/SaveV2/shape_and_slicesConst*=
value4B2B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:
�
save/SaveV2SaveV2
save/Constsave/SaveV2/tensor_namessave/SaveV2/shape_and_slicesVariableconv1/biasesconv1/biases/Momentumconv1/weightsconv1/weights/Momentumconv2/biasesconv2/biases/Momentumconv2/weightsconv2/weights/Momentumconv3/biasesconv3/biases/Momentumconv3/weightsconv3/weights/Momentumconv4/biasesconv4/biases/Momentumconv4/weightsconv4/weights/Momentumconv5/biasesconv5/biases/Momentumconv5/weightsconv5/weights/Momentum*#
dtypes
2
}
save/control_dependencyIdentity
save/Const^save/SaveV2*
_output_shapes
: *
T0*
_class
loc:@save/Const
�
save/RestoreV2/tensor_namesConst"/device:CPU:0*�
value�B�BVariableBconv1/biasesBconv1/biases/MomentumBconv1/weightsBconv1/weights/MomentumBconv2/biasesBconv2/biases/MomentumBconv2/weightsBconv2/weights/MomentumBconv3/biasesBconv3/biases/MomentumBconv3/weightsBconv3/weights/MomentumBconv4/biasesBconv4/biases/MomentumBconv4/weightsBconv4/weights/MomentumBconv5/biasesBconv5/biases/MomentumBconv5/weightsBconv5/weights/Momentum*
dtype0*
_output_shapes
:
�
save/RestoreV2/shape_and_slicesConst"/device:CPU:0*=
value4B2B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:
�
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices"/device:CPU:0*h
_output_shapesV
T:::::::::::::::::::::*#
dtypes
2
�
save/AssignAssignVariablesave/RestoreV2*
T0*
_class
loc:@Variable*
validate_shape(*
_output_shapes
: *
use_locking(
�
save/Assign_1Assignconv1/biasessave/RestoreV2:1*
use_locking(*
T0*
_class
loc:@conv1/biases*
validate_shape(*
_output_shapes
: 
�
save/Assign_2Assignconv1/biases/Momentumsave/RestoreV2:2*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*
_class
loc:@conv1/biases
�
save/Assign_3Assignconv1/weightssave/RestoreV2:3*
use_locking(*
T0* 
_class
loc:@conv1/weights*
validate_shape(*&
_output_shapes
: 
�
save/Assign_4Assignconv1/weights/Momentumsave/RestoreV2:4*
use_locking(*
T0* 
_class
loc:@conv1/weights*
validate_shape(*&
_output_shapes
: 
�
save/Assign_5Assignconv2/biasessave/RestoreV2:5*
use_locking(*
T0*
_class
loc:@conv2/biases*
validate_shape(*
_output_shapes
:@
�
save/Assign_6Assignconv2/biases/Momentumsave/RestoreV2:6*
T0*
_class
loc:@conv2/biases*
validate_shape(*
_output_shapes
:@*
use_locking(
�
save/Assign_7Assignconv2/weightssave/RestoreV2:7*
use_locking(*
T0* 
_class
loc:@conv2/weights*
validate_shape(*&
_output_shapes
: @
�
save/Assign_8Assignconv2/weights/Momentumsave/RestoreV2:8*
use_locking(*
T0* 
_class
loc:@conv2/weights*
validate_shape(*&
_output_shapes
: @
�
save/Assign_9Assignconv3/biasessave/RestoreV2:9*
use_locking(*
T0*
_class
loc:@conv3/biases*
validate_shape(*
_output_shapes	
:�
�
save/Assign_10Assignconv3/biases/Momentumsave/RestoreV2:10*
use_locking(*
T0*
_class
loc:@conv3/biases*
validate_shape(*
_output_shapes	
:�
�
save/Assign_11Assignconv3/weightssave/RestoreV2:11*
use_locking(*
T0* 
_class
loc:@conv3/weights*
validate_shape(*'
_output_shapes
:@�
�
save/Assign_12Assignconv3/weights/Momentumsave/RestoreV2:12*
use_locking(*
T0* 
_class
loc:@conv3/weights*
validate_shape(*'
_output_shapes
:@�
�
save/Assign_13Assignconv4/biasessave/RestoreV2:13*
T0*
_class
loc:@conv4/biases*
validate_shape(*
_output_shapes	
:�*
use_locking(
�
save/Assign_14Assignconv4/biases/Momentumsave/RestoreV2:14*
use_locking(*
T0*
_class
loc:@conv4/biases*
validate_shape(*
_output_shapes	
:�
�
save/Assign_15Assignconv4/weightssave/RestoreV2:15*
use_locking(*
T0* 
_class
loc:@conv4/weights*
validate_shape(*(
_output_shapes
:��
�
save/Assign_16Assignconv4/weights/Momentumsave/RestoreV2:16*
use_locking(*
T0* 
_class
loc:@conv4/weights*
validate_shape(*(
_output_shapes
:��
�
save/Assign_17Assignconv5/biasessave/RestoreV2:17*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*
_class
loc:@conv5/biases
�
save/Assign_18Assignconv5/biases/Momentumsave/RestoreV2:18*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*
_class
loc:@conv5/biases
�
save/Assign_19Assignconv5/weightssave/RestoreV2:19*
T0* 
_class
loc:@conv5/weights*
validate_shape(*'
_output_shapes
:�*
use_locking(
�
save/Assign_20Assignconv5/weights/Momentumsave/RestoreV2:20*
validate_shape(*'
_output_shapes
:�*
use_locking(*
T0* 
_class
loc:@conv5/weights
�
save/restore_allNoOp^save/Assign^save/Assign_1^save/Assign_10^save/Assign_11^save/Assign_12^save/Assign_13^save/Assign_14^save/Assign_15^save/Assign_16^save/Assign_17^save/Assign_18^save/Assign_19^save/Assign_2^save/Assign_20^save/Assign_3^save/Assign_4^save/Assign_5^save/Assign_6^save/Assign_7^save/Assign_8^save/Assign_9
�
initNoOp^Variable/Assign^conv1/biases/Assign^conv1/biases/Momentum/Assign^conv1/weights/Assign^conv1/weights/Momentum/Assign^conv2/biases/Assign^conv2/biases/Momentum/Assign^conv2/weights/Assign^conv2/weights/Momentum/Assign^conv3/biases/Assign^conv3/biases/Momentum/Assign^conv3/weights/Assign^conv3/weights/Momentum/Assign^conv4/biases/Assign^conv4/biases/Momentum/Assign^conv4/weights/Assign^conv4/weights/Momentum/Assign^conv5/biases/Assign^conv5/biases/Momentum/Assign^conv5/weights/Assign^conv5/weights/Momentum/Assign
N
	step/tagsConst*
valueB
 Bstep*
dtype0*
_output_shapes
: 
P
stepScalarSummary	step/tagsVariable/read*
T0*
_output_shapes
: 
N
	loss/tagsConst*
dtype0*
_output_shapes
: *
valueB
 Bloss
G
lossScalarSummary	loss/tagsMean*
T0*
_output_shapes
: 
c
conv1/weights_1/tagConst* 
valueB Bconv1/weights_1*
dtype0*
_output_shapes
: 
m
conv1/weights_1HistogramSummaryconv1/weights_1/tagconv1/weights/read*
T0*
_output_shapes
: 
a
conv1/biases_1/tagConst*
valueB Bconv1/biases_1*
dtype0*
_output_shapes
: 
j
conv1/biases_1HistogramSummaryconv1/biases_1/tagconv1/biases/read*
T0*
_output_shapes
: 
c
conv2/weights_1/tagConst* 
valueB Bconv2/weights_1*
dtype0*
_output_shapes
: 
m
conv2/weights_1HistogramSummaryconv2/weights_1/tagconv2/weights/read*
T0*
_output_shapes
: 
a
conv2/biases_1/tagConst*
valueB Bconv2/biases_1*
dtype0*
_output_shapes
: 
j
conv2/biases_1HistogramSummaryconv2/biases_1/tagconv2/biases/read*
T0*
_output_shapes
: 
c
conv3/weights_1/tagConst*
dtype0*
_output_shapes
: * 
valueB Bconv3/weights_1
m
conv3/weights_1HistogramSummaryconv3/weights_1/tagconv3/weights/read*
T0*
_output_shapes
: 
a
conv3/biases_1/tagConst*
valueB Bconv3/biases_1*
dtype0*
_output_shapes
: 
j
conv3/biases_1HistogramSummaryconv3/biases_1/tagconv3/biases/read*
T0*
_output_shapes
: 
c
conv4/weights_1/tagConst* 
valueB Bconv4/weights_1*
dtype0*
_output_shapes
: 
m
conv4/weights_1HistogramSummaryconv4/weights_1/tagconv4/weights/read*
T0*
_output_shapes
: 
a
conv4/biases_1/tagConst*
valueB Bconv4/biases_1*
dtype0*
_output_shapes
: 
j
conv4/biases_1HistogramSummaryconv4/biases_1/tagconv4/biases/read*
T0*
_output_shapes
: 
c
conv5/weights_1/tagConst* 
valueB Bconv5/weights_1*
dtype0*
_output_shapes
: 
m
conv5/weights_1HistogramSummaryconv5/weights_1/tagconv5/weights/read*
T0*
_output_shapes
: 
a
conv5/biases_1/tagConst*
valueB Bconv5/biases_1*
dtype0*
_output_shapes
: 
j
conv5/biases_1HistogramSummaryconv5/biases_1/tagconv5/biases/read*
T0*
_output_shapes
: 
�
Merge/MergeSummaryMergeSummarysteplossconv1/weights_1conv1/biases_1conv2/weights_1conv2/biases_1conv3/weights_1conv3/biases_1conv4/weights_1conv4/biases_1conv5/weights_1conv5/biases_1*
N*
_output_shapes
: ""�
model_variables��
k
conv1/weights:0conv1/weights/Assignconv1/weights/read:02*conv1/weights/Initializer/random_uniform:08
^
conv1/biases:0conv1/biases/Assignconv1/biases/read:02 conv1/biases/Initializer/zeros:08
k
conv2/weights:0conv2/weights/Assignconv2/weights/read:02*conv2/weights/Initializer/random_uniform:08
^
conv2/biases:0conv2/biases/Assignconv2/biases/read:02 conv2/biases/Initializer/zeros:08
k
conv3/weights:0conv3/weights/Assignconv3/weights/read:02*conv3/weights/Initializer/random_uniform:08
^
conv3/biases:0conv3/biases/Assignconv3/biases/read:02 conv3/biases/Initializer/zeros:08
k
conv4/weights:0conv4/weights/Assignconv4/weights/read:02*conv4/weights/Initializer/random_uniform:08
^
conv4/biases:0conv4/biases/Assignconv4/biases/read:02 conv4/biases/Initializer/zeros:08
k
conv5/weights:0conv5/weights/Assignconv5/weights/read:02*conv5/weights/Initializer/random_uniform:08
^
conv5/biases:0conv5/biases/Assignconv5/biases/read:02 conv5/biases/Initializer/zeros:08"�
	summaries�
�
step:0
loss:0
conv1/weights_1:0
conv1/biases_1:0
conv2/weights_1:0
conv2/biases_1:0
conv3/weights_1:0
conv3/biases_1:0
conv4/weights_1:0
conv4/biases_1:0
conv5/weights_1:0
conv5/biases_1:0"�
trainable_variables��
k
conv1/weights:0conv1/weights/Assignconv1/weights/read:02*conv1/weights/Initializer/random_uniform:08
^
conv1/biases:0conv1/biases/Assignconv1/biases/read:02 conv1/biases/Initializer/zeros:08
k
conv2/weights:0conv2/weights/Assignconv2/weights/read:02*conv2/weights/Initializer/random_uniform:08
^
conv2/biases:0conv2/biases/Assignconv2/biases/read:02 conv2/biases/Initializer/zeros:08
k
conv3/weights:0conv3/weights/Assignconv3/weights/read:02*conv3/weights/Initializer/random_uniform:08
^
conv3/biases:0conv3/biases/Assignconv3/biases/read:02 conv3/biases/Initializer/zeros:08
k
conv4/weights:0conv4/weights/Assignconv4/weights/read:02*conv4/weights/Initializer/random_uniform:08
^
conv4/biases:0conv4/biases/Assignconv4/biases/read:02 conv4/biases/Initializer/zeros:08
k
conv5/weights:0conv5/weights/Assignconv5/weights/read:02*conv5/weights/Initializer/random_uniform:08
^
conv5/biases:0conv5/biases/Assignconv5/biases/read:02 conv5/biases/Initializer/zeros:08"
train_op


Momentum"�
	variables��
k
conv1/weights:0conv1/weights/Assignconv1/weights/read:02*conv1/weights/Initializer/random_uniform:08
^
conv1/biases:0conv1/biases/Assignconv1/biases/read:02 conv1/biases/Initializer/zeros:08
k
conv2/weights:0conv2/weights/Assignconv2/weights/read:02*conv2/weights/Initializer/random_uniform:08
^
conv2/biases:0conv2/biases/Assignconv2/biases/read:02 conv2/biases/Initializer/zeros:08
k
conv3/weights:0conv3/weights/Assignconv3/weights/read:02*conv3/weights/Initializer/random_uniform:08
^
conv3/biases:0conv3/biases/Assignconv3/biases/read:02 conv3/biases/Initializer/zeros:08
k
conv4/weights:0conv4/weights/Assignconv4/weights/read:02*conv4/weights/Initializer/random_uniform:08
^
conv4/biases:0conv4/biases/Assignconv4/biases/read:02 conv4/biases/Initializer/zeros:08
k
conv5/weights:0conv5/weights/Assignconv5/weights/read:02*conv5/weights/Initializer/random_uniform:08
^
conv5/biases:0conv5/biases/Assignconv5/biases/read:02 conv5/biases/Initializer/zeros:08
H

Variable:0Variable/AssignVariable/read:02Variable/initial_value:0
�
conv1/weights/Momentum:0conv1/weights/Momentum/Assignconv1/weights/Momentum/read:02*conv1/weights/Momentum/Initializer/zeros:0
�
conv1/biases/Momentum:0conv1/biases/Momentum/Assignconv1/biases/Momentum/read:02)conv1/biases/Momentum/Initializer/zeros:0
�
conv2/weights/Momentum:0conv2/weights/Momentum/Assignconv2/weights/Momentum/read:02*conv2/weights/Momentum/Initializer/zeros:0
�
conv2/biases/Momentum:0conv2/biases/Momentum/Assignconv2/biases/Momentum/read:02)conv2/biases/Momentum/Initializer/zeros:0
�
conv3/weights/Momentum:0conv3/weights/Momentum/Assignconv3/weights/Momentum/read:02*conv3/weights/Momentum/Initializer/zeros:0
�
conv3/biases/Momentum:0conv3/biases/Momentum/Assignconv3/biases/Momentum/read:02)conv3/biases/Momentum/Initializer/zeros:0
�
conv4/weights/Momentum:0conv4/weights/Momentum/Assignconv4/weights/Momentum/read:02*conv4/weights/Momentum/Initializer/zeros:0
�
conv4/biases/Momentum:0conv4/biases/Momentum/Assignconv4/biases/Momentum/read:02)conv4/biases/Momentum/Initializer/zeros:0
�
conv5/weights/Momentum:0conv5/weights/Momentum/Assignconv5/weights/Momentum/read:02*conv5/weights/Momentum/Initializer/zeros:0
�
conv5/biases/Momentum:0conv5/biases/Momentum/Assignconv5/biases/Momentum/read:02)conv5/biases/Momentum/Initializer/zeros:0��Xǽ:      �3�	���/���A*�u

step    

loss��>
�
conv1/weights_1*�	    �G��    �B�?     `�@!  �r����)��
��@2�����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ�1��a˲���[���FF�G ���~]�[�>��>M|K�>��d�r?�5�i}1?�.�?ji6�9�?�S�F !?U�4@@�$?+A�F�&?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?�������:�              Q@     �j@     `f@     �d@     �c@      e@      a@      ^@     �[@     �S@     �W@     �S@      O@     @Q@     �I@      K@      P@     �I@      @@      D@      6@      A@     �C@      <@      A@      3@      9@      .@      ,@      (@      (@      ,@      $@      "@      @      $@      $@      @      *@      @      @      @      @      @      @       @              �?      @      �?      @       @      �?      @              �?      �?               @              �?       @      �?              �?              �?              �?      �?              �?               @              �?      �?              �?              �?       @       @       @               @      �?      �?              @       @      @      @       @      �?      @      @      @      @      @      *@      @      @      .@      *@      ,@      1@       @      .@      *@      (@      :@      .@      7@      B@      =@      ;@      E@     �A@     �A@      C@      J@      L@     �M@     �L@     �O@     �S@     @U@     @Y@     �Y@      ^@     �`@      a@     �c@     @g@     �h@     `h@      F@        
O
conv1/biases_1*=      @@2        �-���q=�������:              @@        
�
conv2/weights_1*�	   �)���   �o��?      �@!  `	��!@)ل�L�UE@2����g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(龢f�����uE���⾮��%ᾙѩ�-߾�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿ�*��ڽ�G&�$����n�����豪}0ڰ���n����>�u`P+d�>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?�������:�             ��@     ��@     ��@     ��@     T�@     �@     �@     T�@     ��@     h�@     H�@     X�@     @�@     ��@     H�@      �@     @�@     �}@     �z@      z@     �u@     �u@     Ps@     @p@     �n@     �i@     @l@     �h@      f@     �c@      `@     �Z@     �^@      W@      ^@     @T@     �P@     �S@     �U@     @P@     �L@     �P@      H@     �H@     �C@      6@     �F@      B@      =@      2@      9@      6@      1@      6@      1@       @      2@      "@      @      @       @      @      @      @      @       @      @      @       @      @      @      @      @       @      @      �?      @       @               @      �?      �?              �?      �?              �?              �?      @      @              �?              �?              �?              �?              �?              �?              �?              �?      �?              �?       @       @      �?      �?       @      �?              �?       @      �?      �?              @      @      @              �?      "@      @       @      @      @      @      @      @      &@      (@       @      .@      "@      *@      ,@      4@      .@      1@      ,@      :@      <@     �@@      :@     �D@      F@     �C@      C@     �I@     �L@     �P@     �U@     @S@     �V@     �W@      Y@     @]@     @a@     �`@     @b@     �f@     �g@     `j@      k@      p@     q@      s@     �q@     `v@     �y@     @y@     `~@     ��@     @�@     �@     ��@     X�@     P�@     ��@     �@     4�@      �@     �@     l�@     Ț@     d�@     ��@     F�@     �@        
O
conv2/biases_1*=      P@2        �-���q=�������:              P@        
�
conv3/weights_1*�	   �|+��   `<+�?      �@!  ��é�)rW�s,KU@2�����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%�E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;;�"�qʾ
�/eq
Ⱦ�[�=�k���*��ڽ���n�����豪}0ڰ�f^��`{>�����~>���?�ګ>����>�_�T�l�>�iD*L��>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?�������:�             ��@     |�@     ��@     (�@     V�@     ҡ@     $�@     l�@     �@     ��@     $�@     t�@     ��@     h�@      �@     ��@     ؇@     ��@     �@     8�@      �@     P}@     �{@     �x@     �v@     pu@     �u@     �p@      p@      k@     �i@      g@     �f@     @f@      c@     �^@     @b@     @Y@     @W@     @Y@     @U@     �T@      N@      I@      M@     �L@      C@      B@      ;@     �F@      :@      @@      =@      7@      1@      1@      (@      ,@      &@      ,@      @      $@      4@      @      @      ,@       @      @      @      "@      @      @      @      @       @       @              �?      �?      �?              �?              �?      �?      �?       @              �?              �?              @       @              �?              �?              �?              �?      �?      �?              �?              �?              �?              �?               @              �?              �?              �?      �?       @              �?      �?      �?      @               @              �?      @      @      @      @      @      @      @      @       @      @      @      @       @      @      (@      "@      $@      $@      &@      3@      ,@      ,@      7@      8@      :@      ;@      <@      5@      E@     �B@     �G@      C@     �K@     �P@     @P@     @Q@     @T@     �S@      S@     �Y@      X@     �_@     �\@     @d@     `c@     �g@     �l@      k@     �k@     �m@     Pq@     0r@     �s@     u@      x@     �y@     @     Ȁ@     H�@     �@     ��@     ��@      �@     0�@     @�@     �@     ܒ@     �@     0�@     0�@     D�@     ��@      �@     r�@     �@     N�@     ��@     ȃ@        
O
conv3/biases_1*=      `@2        �-���q=�������:              `@        
�
conv4/weights_1*�	    P���   ����?      �@!   @���)d�*Q��e@2�Ӗ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9����ڋ��vV�R9��T7����5�i}1���d�r�>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s�����(��澢f�����_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ��(���>a�Ϭ(�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?1��a˲?6�]��?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�������:�             @]@      �@     ��@     ��@     `�@     0�@     h�@     ��@     `�@      �@     ��@     x�@     `}@     pz@      y@     �x@     `u@     �v@     `q@     �q@     �l@     `k@     `f@      f@     �e@      c@      ^@     �]@     �W@     @W@      T@     @V@      O@     �N@      P@     �I@      G@     �D@     �B@      K@     �B@      ?@      9@      <@      ;@      8@      1@      5@      3@      ,@      *@      0@      &@      $@      @       @      @       @      @      @      @      @       @      @      @      �?      @       @      @      @       @      �?      �?      �?              �?              �?      �?               @       @      �?       @      �?              �?              �?              �?      �?              �?              �?              �?               @      �?              @              �?      �?      @               @      �?               @      @      @      �?      @      �?      @      @       @      �?      @       @      @      @      $@      @      @       @       @      @      @      *@      (@      1@      6@      3@      1@      .@      :@      4@      ;@      @@      B@      F@     �F@     �D@      H@     �L@      Q@     @Q@     �O@      U@     �W@      X@     �[@      _@     �`@      a@     �`@     �e@      g@     �h@     �j@     �p@     `q@     �r@     pu@      w@     Px@      x@     �|@      @     ��@     Ђ@      �@     0�@     ��@     ��@     ��@     <�@     ܒ@     Е@     �@     �b@        
O
conv4/biases_1*=      p@2        �-���q=�������:              p@        
�
conv5/weights_1*�	   �И¿   `���?      �@!   }	�)�m��KQI@2�	�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��7Kaa+�I�I�)�(��S�F !�ji6�9���T7����5�i}1���d�r�x?�x��>h�'��K+�E���>jqs&\��>x?�x�?��d�r?�5�i}1?�T7��?ji6�9�?�S�F !?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?��%�V6?uܬ�@8?��%>��:?d�\D�X=?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?�������:�	             `k@     �t@     �p@     �n@     �n@     @l@      f@     �e@     �d@      `@     �]@      \@      Z@     @Z@      W@     �R@     @R@     �P@      I@     �L@     �J@     �B@      E@      B@      >@      4@      :@      :@      .@      <@      <@      (@      3@      *@      3@      "@       @      @      &@       @       @      @      @      @      @      @       @      @      @      @      @      �?      @      @      @      �?              @      @       @              �?              �?              �?              �?              �?      �?              �?              �?              �?      �?      �?               @              �?              �?               @      �?               @      �?       @              �?      �?       @       @       @               @      @      @      @      @      @      @      @      @      @      @       @      @      ,@      $@      ,@      &@      @      1@      (@      =@      5@      5@      *@      2@      <@      =@      <@      F@      >@     �C@     �C@     �N@     �H@      S@     �U@      Q@      O@     �X@      V@     @[@     �Z@     @^@      d@      b@      e@     �f@     �k@      l@     �p@     �r@     �q@     �j@        
O
conv5/biases_1*=      <@2        �-���q=�������:              <@        �����U      ��	3�/���A*��

step  �?

loss[j�>
�
conv1/weights_1*�	    �L��   `�C�?     `�@! @Z�8��)���h@2�����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.�I�I�)�(�+A�F�&�U�4@@�$��S�F !�ji6�9��6�]���1��a˲�>�?�s���O�ʗ���>h�'�?x?�x�?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?�������:�             �P@     �j@     �f@     �d@     �c@     @e@     �`@      ^@     �[@     �S@     �V@     @T@      O@     @Q@     �I@      K@     �O@     �H@     �B@      C@      6@     �A@      C@      >@      >@      6@      8@      .@      (@      .@      &@      ,@      $@      "@      @      &@      @      @      &@      @      @      @      @      @      @      �?      @              @      @      �?      �?      �?      �?              @       @      �?              �?      @              �?              �?              �?              �?              �?               @              �?               @      �?      �?              @      @      �?      �?       @      @      @      @      �?      @      �?      @      $@      @      @      @      (@       @      @      .@      0@      &@      .@      &@      ,@      .@      &@      9@      1@      8@     �@@      >@      ;@      D@      B@     �A@      C@      K@     �J@     �M@      M@     �O@     �S@      U@     �Y@      Z@     �]@      a@      a@     �c@     `g@     �h@     `h@      F@        
�
conv1/biases_1*�	   ���    �W(?      @@!   f!�%?)2Ѳ�"
�>2�ji6�9���.����ڋ��vV�R9��T7����5�i}1�x?�x��>h�'�������6�]���I��P=��pz�w�7���ߊ4F��h���`�a�Ϭ(���(��澮��%ᾙѩ�-߾��~]�[Ӿjqs&\�ѾX$�z�>.��fc��>�*��ڽ>�[�=�k�>E��a�W�>�ѩ�-�>�f����>��(���>8K�ߝ�>�h���`�>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>��[�?1��a˲?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?+A�F�&?I�I�)�(?�������:�              �?      �?      �?      �?      �?              �?              @              �?              �?              �?               @              �?              �?              �?              �?              �?               @              �?              �?      �?              �?              �?      �?              �?              �?       @              �?        
�
conv2/weights_1*�	   `����   @띩?      �@!�D_�"@)^���VE@2����g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پK+�E��Ͼ['�?�;;�"�qʾ��~��¾�[�=�k��G&�$��5�"�g�����z!�?�>��ӤP��>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>E��a�W�>�ѩ�-�>���%�>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?�������:�             ��@     ��@     ��@     �@     h�@     �@     �@     \�@     ��@     X�@     H�@     X�@     P�@     ��@     0�@     8�@     P�@     �}@     �z@     �y@      v@     �u@     @s@     �p@     �n@     `i@     �l@     �h@      f@      d@     �^@      \@     �\@     �X@     �]@     �T@      Q@     @R@      W@      P@      M@     �N@      I@      J@      B@      >@      D@     �A@      >@      2@      9@      2@      5@      2@      0@      *@      ,@      $@      "@      @      $@      "@      �?       @      @      @       @       @      �?      @      @              @      @              �?      @      �?              @      �?      �?              �?      �?              �?      �?               @              �?      �?       @      �?              �?       @              �?              �?              �?              �?      �?               @      �?              �?      �?               @      �?      �?      �?              �?      �?              �?      �?      �?      �?      �?       @      �?      @      @      @      @      �?      �?      @      @      &@       @      @      @      @      @      *@      ,@      @      ,@      @      ,@      1@      2@      2@      ,@      *@      :@     �@@      A@      8@      C@      F@      C@      D@     �J@      L@     @P@     �U@      U@     �T@     @X@     @Y@     @]@      a@      a@     `b@      f@     �h@      j@      k@     0p@      q@     @s@     �q@     �v@     �y@     Py@     p~@     ��@      �@     ��@     ��@     p�@     (�@      �@     �@     ,�@     ,�@     ��@     p�@     К@     X�@     ��@     @�@     ��@        
�	
conv2/biases_1*�		   @�0-�    �' ?      P@!  x!`D?)�E�k�>2���VlQ.��7Kaa+��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7���ߊ4F��h���`�8K�ߝ�a�Ϭ(龙ѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾jqs&\�ѾK+�E��Ͼ['�?�;;�"�qʾ��~��¾�[�=�k��39W$:��>R%�����>���]���>�5�L�>G&�$�>�*��ڽ>�[�=�k�>��~���>;�"�q�>['�?��>��~]�[�>��>M|K�>�ѩ�-�>���%�>�f����>��(���>a�Ϭ(�>8K�ߝ�>�ߊ4F��>})�l a�>O�ʗ��>>�?�s��>�FF�G ?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�������:�              �?              �?      �?              �?              �?      �?       @       @              �?              �?      �?      �?              �?      �?              �?      �?      �?              �?              �?              �?       @      �?              �?              �?              �?              �?              �?              �?              �?              �?              @              �?               @              �?      �?              @      �?       @      �?               @      �?      �?      @      �?      @       @       @        
�
conv3/weights_1*�	    .��    �,�?      �@!෌�>&�)_r,�6KU@2�����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[��>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾�iD*L�پ�_�T�l׾�����0c>cR�k�e>�����>
�/eq
�>��~]�[�>��>M|K�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?�������:�             X�@     ��@     ��@      �@     Z�@     ԡ@     *�@     X�@     �@     ��@     $�@     ��@     Б@     x�@     �@     ��@     ؇@     ��@      �@     H�@      �@     `}@     �{@     �x@     w@     �u@     `u@     �p@     p@      k@      j@     �f@      g@     �e@     �c@     �^@     @b@      Y@     @W@      Z@     �T@     �T@      O@      I@     �L@     �M@      B@      @@      >@     �E@      >@      =@      ?@      6@      1@      3@      (@      (@      &@      ,@      "@      &@      2@       @      @      (@      $@       @      @      @      @       @      @       @       @       @              @       @              �?               @      �?      �?       @               @      �?      �?       @      �?      �?               @              �?              �?              �?               @       @      �?              �?              �?      �?      �?      �?               @      �?      �?      �?       @      �?      @               @      @      @      @      @      @      @      @      @       @      @       @      @       @      $@      "@      (@      @      0@      ,@      $@      4@      5@      9@      ;@      6@      ?@      6@      E@     �A@      I@      B@     �J@      Q@      P@     @Q@      U@      S@     @S@     �X@     @Y@      ^@     @^@      d@     �c@      g@     �l@     �k@     �j@      n@      q@     Pr@     �s@     �t@     0x@     �y@     P@     Ȁ@     H�@     �@     ��@     x�@     (�@     @�@     D�@     �@     �@     �@     ,�@     ,�@     T�@     ��@      �@     z�@     �@     H�@     ��@     Ѓ@        
�
conv3/biases_1*�	   ��g#�   �X�%?      `@! �+��NP?)�DM�`̛>2�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a�a�Ϭ(���(����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;
�/eq
Ⱦ����žG&�$��5�"�g���
�}�����4[_>����H5�8�t�BvŐ�r�        �-���q=���m!#�>�4[_>��>
�}���>39W$:��>R%�����>���?�ګ>����>5�"�g��>G&�$�>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?�.�?ji6�9�?�S�F !?U�4@@�$?+A�F�&?�������:�               @      �?              �?      �?      �?      @      �?       @      �?      @              @      @       @      @              �?      @      �?      @      �?      �?              �?               @              �?              �?              �?              �?              �?              �?              �?              �?              @              �?      �?              �?              �?              �?              �?      �?               @              �?               @              �?       @      @      �?      @      @      �?      �?       @       @               @      �?       @      �?      @      @      @      @      @      @      @      @              @      @              �?        
�
conv4/weights_1*�	   `O���   ����?      �@! p�����)�B\��e@2�Ӗ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �8K�ߝ�a�Ϭ(龢f�����uE������>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ�h���`�>�ߊ4F��>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�������:�             @]@      �@     ��@     ��@     `�@     �@     h�@     ��@     `�@     �@     ��@     x�@     �}@     @z@     @y@     �x@     `u@     �v@     pq@     �q@     �l@     @k@     `f@     @f@     �e@     @c@     �]@     �]@      X@      W@      T@     @V@      O@     �N@      P@     �I@      G@     �D@      B@     �K@     �B@      ?@      :@      ;@      ;@      6@      3@      4@      4@      .@      (@      1@      $@      $@      @       @      @       @      @      @      @      @       @      @      @       @       @      @      @      @       @               @      �?              �?              �?              �?              �?      �?      @      �?       @      �?              �?              �?              �?              �?              �?              �?       @      �?              �?      �?              �?               @      �?      �?      �?       @      �?               @      @      @      �?      @      �?      @      @      @      �?      @       @      @      @      $@      @      @       @       @      @      @      *@      (@      2@      5@      4@      0@      ,@      :@      5@      :@      @@     �B@      F@     �F@     �D@      H@     �L@      Q@      Q@     �O@     �U@     @W@     @X@     �[@     @_@     ``@      a@     �`@     �e@      g@     �h@     �j@     �p@     @q@      s@     pu@     �v@     `x@      x@     �|@     @      �@     Ђ@      �@     0�@     ��@     Ȍ@     x�@     8�@     ��@     Е@     �@     �b@        
�
conv4/biases_1*�	   @A��   �3?      p@!�F���_?)hL�;.}�>2���ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(����uE���⾮��%ᾙѩ�-߾�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�Ѿ['�?�;;�"�qʾ
�/eq
Ⱦ����ž�XQ�þ��~��¾5�"�g���0�6�/n����������?�ګ����m!#���
�%W��'j��p���1���=��]����/�4��ݟ��uy�z�����i@4[���Qu�R"�PæҭUݽH����ڽ���X>ؽ��
"
ֽ�|86	Խ(�+y�6ҽ;3���н��.4Nν�!p/�^˽�d7���Ƚ��؜�ƽ�b1�ĽK?�\��½�
6������Bb�!澽5%����G�L������6���Į#�������/���EDPq���8�4L���<QGEԬ�y�訥�V���Ұ����@�桽�>�i�E�������嚽��s�����:������z5����x�����1�ͥ��̴�L�����/k��ڂ�        �-���q=����z5�=���:�=��s�=������=_�H�}��=�>�i�E�=V���Ұ�=y�訥=��M�eӧ=|_�@V5�=<QGEԬ=�8�4L��=����/�=�Į#��=���6�=G�L��=5%���=�Bb�!�=�
6����=K?�\���=�d7����=�!p/�^�=��.4N�=;3����=(�+y�6�=�|86	�=��
"
�=���X>�=H�����=PæҭU�=�Qu�R"�=i@4[��=z�����=ݟ��uy�=�/�4��==��]���=��1���='j��p�=��-��J�=�K���=����%�=f;H�\Q�=R%�����>�u��gr�>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�XQ��>�����>['�?��>K+�E���>�_�T�l�>�iD*L��>E��a�W�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?I�I�)�(?�7Kaa+?��82?�u�w74?�������:�               @      �?              �?       @      @              @      @              �?      �?      @      �?              �?              @      �?      �?       @       @      �?              �?      �?              �?      �?      �?               @      �?      �?              �?              �?              �?              �?              �?              �?      @       @      �?              @      @      @       @       @              @      @               @       @               @      �?               @               @              �?      �?               @               @              �?              �?              @              �?               @             �@@              �?              �?               @               @      �?       @              @              �?              @               @       @      �?              @              �?      @               @      @      �?       @       @      �?       @      �?       @      �?      �?              �?      �?              �?              �?              �?      �?              �?      �?              �?              �?               @      @              �?              �?       @       @      @      �?      �?       @      @       @       @       @       @      �?       @      �?               @      @       @      �?      @      @       @       @       @      �?      �?      �?       @              �?              �?        
�
conv5/weights_1*�	   �٘¿   �;��?      �@! @z�	�)k���_QI@2�	�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��7Kaa+�I�I�)�(��S�F !�ji6�9���T7����5�i}1������6�]������%�>�uE����>x?�x�?��d�r?�5�i}1?�T7��?ji6�9�?�S�F !?U�4@@�$?+A�F�&?I�I�)�(?��VlQ.?��bȬ�0?��82?��%�V6?uܬ�@8?��%>��:?d�\D�X=?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?�������:�	             `k@     �t@     �p@     �n@     �n@     @l@      f@     �e@     �d@      `@     �]@      \@      Z@     @Z@      W@     �R@     @R@     �P@      I@     �L@     �J@      B@     �E@      B@      >@      4@      :@      :@      .@      <@      <@      &@      4@      *@      3@      "@       @      @      &@       @       @      @      @      @      @      @       @      @      @      @      @              @      @      @      �?              @      @       @              �?              �?              �?              �?               @              �?              �?              �?      �?      �?               @              �?      �?               @      �?               @      �?       @              �?      �?       @       @       @               @      @      @      @      @      @      @      @      @      @       @       @      @      ,@      $@      ,@      $@      @      1@      (@      =@      5@      5@      *@      2@      <@      =@      <@      F@      ?@      C@     �C@     �N@     �H@      S@     �U@      Q@      O@     �X@      V@     @[@     �Z@     @^@      d@      b@      e@     �f@     �k@      l@     �p@     �r@     �q@     �j@        
�
conv5/biases_1*�	   ����   �'�=      <@!  ��v��=)EGf��<2��/�4��ݟ��uy�i@4[���Qu�R"�PæҭUݽH����ڽ���X>ؽ��
"
ֽ�|86	Խ�!p/�^˽�d7���Ƚ�b1�ĽK?�\��½y�訥�V���Ұ��        �-���q=�/k��ڂ=̴�L���=����z5�=���:�=5%���=�Bb�!�=K?�\���=�b1��=��؜��=�d7����=�!p/�^�=��.4N�=;3����=(�+y�6�=�|86	�=��
"
�=���X>�=�Qu�R"�=i@4[��=z�����=�K���=�9�e��=�������:�               @              �?      �?      �?      �?              �?              �?              �?              �?              @              �?              �?              �?               @              �?      �?      �?      �?      �?              �?      �?              �?      �?              �?        �1�XU      ��b	���/���A*ɪ

step   @

loss�[�>
�
conv1/weights_1*�	    5V��   ��V�?     `�@! �5�=��)����q@2�����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���5�i}1���d�r�6�]���1��a˲��uE���⾮��%�>h�'�?x?�x�?�T7��?�vV�R9?��ڋ?�.�?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��82?�u�w74?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?�������:�              Q@     �j@     @f@      e@     �c@     @e@     �`@     �]@     @\@     �S@     �U@     �U@     �N@      Q@     �J@      K@     �O@      G@      D@      B@      9@     �A@      B@      <@      >@      6@      <@      &@      ,@      ,@      *@      *@      (@      @      &@      "@      $@      @      @       @       @      @      @      �?      @              �?      �?       @       @      @       @      �?       @      @      �?       @              �?              �?      �?      �?      �?              �?              �?              �?              �?              �?              �?              �?              �?              �?              @      �?      @      @       @       @      @      �?      @       @       @      @      @      @      @      @      &@      "@      @      .@      1@      @      1@      $@      0@      .@      $@      9@      1@      9@     �@@      <@      =@     �C@     �@@     �C@     �C@      K@      I@      N@     �M@      P@      S@     �T@     @Z@     �Y@      ^@     �`@     @a@     @c@     �g@      i@     �g@      G@        
�
conv1/biases_1*�	    
�/�   ���;?      @@!  �!�=?)^e�D�[�>2���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��.����ڋ��vV�R9��T7����5�i}1�f�ʜ�7
������6�]�����(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ��>M|Kվ��~]�[Ӿjqs&\�Ѿ���]���>�5�L�>��~]�[�>��>M|K�>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>O�ʗ��>>�?�s��>6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?ji6�9�?�S�F !?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?��%>��:?d�\D�X=?�������:�              @              �?              �?              �?      �?              �?              �?      �?              �?              �?              �?              �?      �?              �?              �?              �?              �?      �?              �?              �?      �?      �?              �?              �?              �?      �?               @      �?              �?        
�
conv2/weights_1*�	   ����   �С�?      �@! �dJ�I"@)s�2VE@2����g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澮��%ᾙѩ�-߾E��a�Wܾ['�?�;;�"�qʾ豪}0ڰ>��n����>��~���>�XQ��>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>E��a�W�>�ѩ�-�>���%�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?�������:�             ��@     �@     ��@     �@     l�@     �@     $�@     \�@     �@     X�@     h�@     @�@     `�@     ��@      �@     H�@     H�@     ~@     Pz@     �y@     �u@      v@     Ps@     �p@     �n@     �i@     �k@     �h@     �f@     �c@      `@     �[@     �\@     @X@     �^@     @S@     @P@     �S@      V@     �Q@     �K@     �N@      I@     �J@      D@      <@      E@      <@      =@      4@      8@      4@      1@      5@      0@      1@       @      $@      (@      @      "@      @      @      @      @      @      @              @      @      @       @      @       @       @               @      �?      �?      @      �?      �?      �?      �?      @      @      �?              �?              �?      �?              �?              �?              �?              �?              �?               @              �?      @              �?      �?      �?      �?       @              �?       @      �?      @       @      �?      �?      @      @      �?      @       @      @      @      @      @      @      @      @      @       @      "@      .@      @      *@      @      "@      0@      6@      2@      .@      $@      <@      >@     �A@      :@      D@      D@     �A@      F@     �J@     �K@     @P@     �U@      U@     @U@     �X@     �X@     �]@     �`@     �`@     @b@     `f@     �h@      j@     @k@     �o@     �p@     s@     �q@     �v@      z@     0y@     �~@     ��@     (�@     Ȅ@     Ȅ@     ��@      �@     ؍@     �@     ,�@      �@     �@     |�@     ̚@     T�@     ��@     D�@     ��@        
�	
conv2/biases_1*�		   �?>�    �7?      P@!  @f� [?)0�ta̹>2����#@�d�\D�X=���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[��>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`���(��澢f���侄iD*L�پ�_�T�l׾�u��gr��R%�������5�L�>;9��R�>K+�E���>jqs&\��>��>M|K�>�_�T�l�>a�Ϭ(�>8K�ߝ�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?����?f�ʜ�7
?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?�������:�              �?               @              �?       @              �?               @       @      �?      �?              �?              �?      �?              �?              �?      �?              �?              �?      @              �?              �?              �?              �?              �?              �?              �?              @      �?              �?      �?              �?              �?               @      �?      �?       @              �?       @       @              @      �?      @               @       @              �?        
�
conv3/weights_1*�	    �/��   �/�?      �@!��$l��)�K�FKU@2�����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F����(��澢f�����uE��������ž�XQ�þ.��fc���X$�z���XQ��>�����>;�"�q�>['�?��>��>M|K�>�_�T�l�>�iD*L��>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?�������:�             `�@     z�@     ��@      �@     ^�@     Ρ@     2�@     T�@     �@     ��@     �@     ��@     đ@     x�@     �@     ��@     ��@     ��@     ��@     p�@     ؀@     �}@     �{@     @y@     �v@     �u@      u@     �p@      p@      k@     `i@     �g@     `g@     �e@     �b@     �_@      b@     @X@     @X@      Y@     �U@      T@     �L@     �M@     �K@     �L@     �B@      @@      ;@     �F@      <@      =@      :@      ;@      4@      0@      .@      &@      ,@      &@      $@      (@      0@      @       @      "@      "@      @      @      @      @      @      @      @      @              �?      �?      @       @      �?      �?       @      �?       @      �?       @              �?       @              �?              �?              �?              �?               @      �?               @      �?               @      �?       @               @       @      �?      @      �?      @      �?      �?      �?       @      @      @      @      @      @      �?      @      "@      @      @       @      @      (@      @      *@      &@      (@      *@      .@      *@      7@      5@      >@      9@      =@      9@      C@     �@@     �J@      C@     �I@     �Q@     �O@     �Q@     @U@      S@      S@      Y@      X@     �^@     @^@     �c@     `d@     �f@     �l@     �k@     �j@     �m@     0q@     0r@      t@     �t@     @x@     �y@     P@     ��@     H�@     �@     X�@     ��@     �@     0�@     @�@     $�@     ��@     ��@     0�@     4�@     D�@     ��@     �@     x�@      �@     :�@     ��@     ��@        
�
conv3/biases_1*�	   �X�8�   @�6?      `@!  �a�h[?)*���h��>2���%>��:�uܬ�@8��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(龢f�����uE���⾮��%���n�����豪}0ڰ���������?�ګ�        �-���q=豪}0ڰ>��n����>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>�iD*L��>E��a�W�>�f����>��(���>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?��%�V6?uܬ�@8?�������:�              �?              �?      �?      �?       @       @      �?      @       @      �?       @       @      @      �?       @       @      �?      @      �?      @      �?      �?       @      �?              �?              �?      �?      @              �?               @      �?              �?      �?              @              �?              @              �?              �?              �?              �?               @              �?              �?      �?      �?       @              @       @              �?      @       @      �?      @       @              @      �?      �?      @       @      @      @      @              �?      @              @      @              �?        
�
conv4/weights_1*�	   @����   ����?      �@! �!$��)鈊U��e@2�Ӗ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �O�ʗ�����Zr[v��I��P=���h���`�8K�ߝ뾙ѩ�-߾E��a�Wܾ��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ�uE����>�f����>��(���>a�Ϭ(�>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>��[�?1��a˲?6�]��?>h�'�?x?�x�?��d�r?�T7��?�vV�R9?��ڋ?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�������:�             @]@      �@     �@     ��@     d�@     �@     h�@     ��@     `�@     �@     p�@     x�@     �}@     Pz@     0y@     �x@     pu@     �v@     `q@     �q@     �l@     @k@     @f@     `f@     �e@     `c@     �]@      ^@      X@      W@      T@     @V@     �O@      N@      P@      J@      F@      E@     �B@     �J@     �B@      @@      :@      ;@      :@      8@      2@      3@      5@      ,@      *@      1@      $@      $@      @      @      @       @      @       @      @      @       @      @      @       @      @      �?      @       @       @              @      �?              �?              �?              �?      �?      �?      �?      �?      �?      �?      �?              �?      �?              �?              �?              �?              �?              �?              �?              �?              �?      �?              �?      �?              @       @              @      �?               @       @      @      �?      @              @      @       @      �?      @      �?      @      @      "@      @      @      @      "@      @      $@      $@      &@      2@      6@      3@      0@      .@      9@      5@      ;@      ?@     �B@     �F@     �E@     �E@      H@      L@     �Q@     �P@      O@     �U@     �W@     �W@     @\@     �_@      `@      a@     �`@     �e@      g@      i@     �j@     �p@     0q@      s@     �u@     �v@     Px@      x@     �|@     @      �@     Ȃ@     �@     0�@     ��@     ��@     ��@     8�@     �@     Е@     �@     �b@        
�
conv4/biases_1*�	   ��n,�    �aE?      p@!cY"�0Lr?)	���y�>2���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ
�/eq
Ⱦ����ž�f׽r����tO��������%���9�e����K��󽉊-��J�'j��p���1���=��]����/�4��ݟ��uy�z�����i@4[���Qu�R"�PæҭUݽH����ڽ���X>ؽ��
"
ֽ�|86	Խ(�+y�6ҽ;3���н��.4Nν�!p/�^˽�d7���Ƚ��؜�ƽ�b1�ĽK?�\��½�
6������Bb�!澽5%����G�L������6���Į#���|_�@V5����M�eӧ�y�訥�V���Ұ���>�i�E��_�H�}������:������z5��!���)_���1�ͥ��G-ֺ�І�̴�L����:[D[Iu�z����Ys�        �-���q=e���]�=���_���=����z5�=���:�=_�H�}��=�>�i�E�=��@��=��M�eӧ=|_�@V5�=<QGEԬ=�8�4L��=�EDPq�=G�L��=5%���=�Bb�!�=�b1��=��؜��=��.4N�=;3����=(�+y�6�=�|86	�=��
"
�=���X>�=H�����=PæҭU�=�Qu�R"�=i@4[��=z�����=ݟ��uy�=�/�4��==��]���=��1���='j��p�=��-��J�=�K���=�9�e��=����%�=f;H�\Q�=�tO���=�f׽r��=nx6�X� >��|�~�>���]���>5�"�g��>G&�$�>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>K+�E���>jqs&\��>��~]�[�>��>M|K�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?�T���C?a�$��{E?�������:�              @      �?              �?      �?      @       @       @      @      �?              �?      @              �?              �?      �?       @      �?       @      �?       @              �?              �?       @              �?              �?              �?               @              @      �?              @      �?      �?      �?      @      @      �?       @       @      @              �?      @       @      �?      �?      @      �?               @       @      �?      �?      �?      �?              �?       @       @              �?              �?      �?              �?      �?              �?              >@              �?              �?              �?       @              @      �?               @              �?      �?              �?              �?      �?       @               @      @               @      �?       @       @      �?      @      �?      �?      �?      �?      �?      �?               @       @      �?              �?              �?              �?      �?      �?              �?              @      �?       @              �?              �?              �?       @               @              �?               @              �?               @      �?       @      @       @      �?              @      �?      @      @      @      @              �?      �?      @       @      �?       @      @              @      �?      �?               @              �?        
�
conv5/weights_1*�	   ��¿   ����?      �@! �ig���)��Y�QI@2�	�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6�I�I�)�(�+A�F�&��S�F !�ji6�9���T7����5�i}1��ѩ�-߾E��a�Wܾa�Ϭ(�>8K�ߝ�>��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?��VlQ.?��bȬ�0?��82?��%�V6?uܬ�@8?��%>��:?d�\D�X=?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?�������:�	             `k@     �t@     �p@     �n@     �n@     @l@      f@     �e@     �d@      `@     �]@     @\@      Z@     @Z@      W@     �R@     @R@     �P@      I@     �L@     �J@     �A@      F@      B@      >@      4@      ;@      9@      .@      <@      <@      &@      4@      *@      3@      "@       @      @      (@       @      @       @      @      @      @      @       @      @      @      @       @              @      @      @      �?              @      @       @              �?              �?              �?              �?               @              �?              �?              �?      �?              �?              �?              �?              �?      �?               @      �?               @      �?       @              �?      �?       @       @       @               @      @      @      @      @      @      @       @      @      @       @       @      @      ,@      "@      .@      $@      @      1@      (@      =@      5@      5@      *@      2@      =@      <@      <@      F@      ?@     �B@      D@     �N@     �H@      S@     �U@     �P@     �O@     �X@     @V@      [@     �Z@     @^@      d@      b@      e@     �f@     �k@      l@     �p@     �r@     �q@     �j@        
�
conv5/biases_1*�	   ��M�   `��>      <@!   D.� �)b
�><2�y�+pm��mm7&c�����%���9�e�����-��J�'j��p���1���=��]����/�4��ݟ��uy�z�����i@4[���Qu�R"�PæҭUݽ�|86	Խ(�+y�6ҽ�!p/�^˽�d7���Ƚ        �-���q=��s�=������=�EDPq�=����/�=���6�=G�L��=�!p/�^�=��.4N�=;3����=PæҭU�=�Qu�R"�=z�����=ݟ��uy�=�/�4��=�K���=�9�e��=���">Z�TA[�>�������:�              �?              �?               @              �?      �?              �?      �?               @               @              �?               @              �?              �?              �?              �?      @              �?              �?       @              �?              �?        ׉8U      ���?	;��/���A*��

step  @@

lossP��>
�
conv1/weights_1*�	   @�]��   ��r�?     `�@! �&J����)Վ�`�@2�����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(��[^:��"��S�F !�ji6�9���.��pz�w�7��})�l a�;�"�q�>['�?��>��Zr[v�>O�ʗ��>f�ʜ�7
?>h�'�?x?�x�?��d�r?�.�?ji6�9�?U�4@@�$?+A�F�&?I�I�)�(?��82?�u�w74?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?�������:�              Q@     �j@     �e@     @e@      d@     `e@     �`@     �]@     @\@     �S@     �U@      U@     �O@     �P@     �J@      L@      N@      I@      C@      A@      ;@     �B@     �A@      :@      =@      8@      :@      *@      (@      0@      *@      *@      $@      $@      "@      @      $@      @       @      @       @      @      @      @      @      �?      @              @      �?      @      @               @              �?      �?       @      �?               @              �?              �?              �?              �?              �?               @              �?              �?      �?               @              @      �?      �?       @      @      �?      @       @      @       @       @      @      @      @      @      @       @      @      (@      *@      1@      @      ,@      *@      ,@      0@      &@      9@      1@      <@      =@      >@      :@     �D@      @@      C@     �C@     �K@     �I@      N@     �K@     �P@     �R@      U@     �Z@      Y@      _@      a@      a@      c@     �g@      i@     @g@     �I@        
�
conv1/biases_1*�	   �і7�    f�A?      @@!  (Bh`K?)C��~�ּ>2�uܬ�@8���%�V6��u�w74���bȬ�0���VlQ.�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����d�r�x?�x��>h�'�������6�]����ѩ�-߾E��a�WܾX$�z�>.��fc��>����>豪}0ڰ>�*��ڽ>�[�=�k�>�ѩ�-�>���%�>�ߊ4F��>})�l a�>��[�?1��a˲?x?�x�?��d�r?�5�i}1?�T7��?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?�������:�              @      �?              �?              �?              �?      �?              �?              �?      �?              �?              �?              �?              �?              �?              �?              �?              �?              �?      �?      �?              �?      �?      �?      �?               @      �?              �?      �?              �?        
�
conv2/weights_1*�	   @󟩿   @ன?      �@! Y<�Ϝ"@)��m3bVE@2����g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澮��%ᾙѩ�-߾K+�E��Ͼ['�?�;��~��¾�[�=�k��G&�$��5�"�g���5�"�g��>G&�$�>
�/eq
�>;�"�q�>K+�E���>jqs&\��>��~]�[�>�_�T�l�>�iD*L��>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>pz�w�7�>I��P=�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?�������:�             ��@     �@     ��@     ��@     T�@     �@     0�@     P�@     ܑ@     ��@     @�@     8�@     p�@     ��@     (�@     0�@     P�@     P~@     `z@      y@     v@     �u@     `s@     �p@     �n@     �i@     @k@     `i@     �f@     �b@     ``@     @[@      _@     �X@      ]@     �R@     @Q@      R@     @U@     @S@      J@     �P@     �E@      M@     �A@      B@      B@      @@      ;@      2@      9@      5@      4@      3@      2@      $@      $@      ,@      "@      @      @      "@      @      �?      @       @      @       @      @      @      @      @      @       @       @               @       @      �?      @      �?       @      �?       @       @      �?               @       @              �?              �?              �?              �?              �?              �?              �?      �?              �?              �?      @               @              �?      �?       @               @              �?      �?              @              �?       @       @      @      @      @      @      "@      @       @      @      @      @       @      @      &@      *@      @      0@      @      &@      0@      6@      0@      ,@      *@      ?@      <@     �A@      8@      C@      F@      @@     �D@     �K@      J@      Q@      V@     �U@     @T@     �X@     �Y@     �^@     �`@     �_@     �b@     @f@     �h@     `j@     `k@     `o@     �p@     s@     r@     Pv@     @z@     �x@     @~@     ��@     0�@     �@     ��@     x�@     H�@     ȍ@     �@     $�@     �@     ԕ@     ��@     ܚ@     L�@     ��@     R�@     ��@        
�
conv2/biases_1*�	   �ռJ�    qLF?      P@!  @^�j?)9����>2�IcD���L��qU���I����#@�d�\D�X=���%>��:�uܬ�@8���%�V6���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(��[^:��"��S�F !�ji6�9���.���vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[��a�Ϭ(���(����uE����>�f����>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?6�]��?����?>h�'�?x?�x�?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?a�$��{E?
����G?�������:�              �?              �?               @       @              �?       @              �?               @      �?      �?               @      �?      �?      �?      �?      �?              �?              �?               @              �?              �?      �?       @              �?              �?              @               @       @              @      �?              @      �?      �?       @      �?      �?               @      @       @      �?       @      �?              �?        
�
conv3/weights_1*�	   ��7��   @�:�?      �@!�#}h�)]���^KU@2�����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[��>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ;9��R���5�L��G&�$�>�*��ڽ>['�?��>K+�E���>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?�������:�             P�@     ~�@     ��@     �@     l�@     ȡ@     $�@     x�@     �@     ��@     �@     ��@     ȑ@     ��@      �@     ��@     ȇ@     ��@     ؄@     x�@     ��@     �}@     �{@     @y@     �v@      v@     �t@     Pq@     p@      k@     �i@     �f@     �g@     �e@     �b@     �_@      b@     �W@     @Z@     �W@     �U@     �R@     �M@      L@      P@      K@     �C@      ?@      <@     �F@      8@      =@      ;@      6@      9@      &@      5@      &@      &@      1@      &@      &@      ,@      @      "@      "@      @      @      "@      @      @      @      @      @      @      �?       @       @       @              �?      �?      �?      �?              �?      �?              �?      �?              �?              �?              �?               @              �?              �?              �?              �?      �?              �?       @               @      �?      �?      @      �?      �?      �?              �?      �?      �?      �?       @              @      �?      @      @      @       @      @               @      @      @      @      @      "@      @      &@      (@      (@      $@      "@      2@      &@      *@      4@      5@     �B@      5@      =@      <@     �C@      >@      J@     �D@     �I@      R@     �O@     �Q@      T@     @S@     �S@      W@      Y@      ^@     �`@     �b@     @d@     �f@      m@     @k@     �j@     �m@     Pq@     @r@     �s@     Pu@      x@     �y@      @     ��@      �@     0�@     p�@     ��@     ��@     �@     H�@      �@      �@     �@     $�@     8�@     <�@     ��@     �@     r�@     &�@     8�@     |�@     �@        
�
conv3/biases_1*�	   ���D�   @J�B?      `@!  x"%g?)�[SU���>2�a�$��{E��T���C�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
�6�]���1��a˲���[��>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f���侙ѩ�-߾E��a�WܾG&�$��5�"�g���        �-���q=�H5�8�t>�i����v>0�6�/n�>5�"�g��>['�?��>K+�E���>jqs&\��>�ѩ�-�>���%�>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>��Zr[v�>O�ʗ��>�FF�G ?��[�?1��a˲?6�]��?����?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?�������:�              �?               @      @      �?      �?              @      @      @      �?       @       @       @      @              @      @      @      �?      @              �?              �?              �?      �?               @      �?              �?              �?      �?      �?              �?      �?              �?              �?              @              �?              �?              �?      �?              �?              �?               @              �?              �?              �?      �?              @      �?               @      �?              @       @      �?      @       @       @      @       @       @      @      @              @      �?       @       @       @       @      �?        
�
conv4/weights_1*�	     ��    h �?      �@! �[v�>��)aV��Ɛe@2�Ӗ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9����ڋ��vV�R9��T7����5�i}1�>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G ���>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;��|�~�>���]���>a�Ϭ(�>8K�ߝ�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>�FF�G ?��[�?1��a˲?6�]��?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�������:�             @]@     ��@     ��@     ��@     h�@     �@     p�@     ��@     X�@     �@     `�@     ��@     �}@     `z@     0y@     �x@     �u@     pv@     `q@     �q@     @l@     `k@      f@     `f@     `e@     �c@     �]@     �]@     @X@      W@      T@     �V@      O@      N@     �O@      I@     �G@     �E@     �A@      K@      C@      >@      :@      <@      ;@      6@      4@      2@      3@      0@      ,@      .@      (@      "@      @      @      @      "@      @       @      @      @       @      @      @       @       @       @      @              @               @      �?               @              �?              �?      @               @       @      �?              �?      �?       @      �?              �?              �?               @              �?              �?               @              �?       @              �?               @      �?      �?       @               @      @      �?      @      �?      @       @      @      �?      @              @      @      "@       @      @      @      "@      @      $@      $@      *@      ,@      7@      3@      2@      ,@      :@      4@      8@     �@@      B@     �F@      F@     �F@      H@      K@     �Q@     �P@      P@     �U@      W@      X@      \@     @_@     @`@     @a@     �`@     �e@      g@      i@     �j@     `p@     Pq@      s@     pu@     �v@     `x@     �w@     �|@     @     ��@     Ђ@      �@     H�@     ��@     ��@     ��@     <�@     ��@     ̕@     �@     �b@        
�
conv4/biases_1*�	   ��\;�   ���P?      p@!�����?)@�A���>2�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7���f�ʜ�7
������6�]���1��a˲�I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;
�/eq
Ⱦ����ž�XQ�þ��~��¾y�+pm��mm7&c��f׽r����tO����f;H�\Q������%���9�e����K��󽉊-��J�'j��p���1���=��]����/�4��ݟ��uy�z�����i@4[���Qu�R"�PæҭUݽH����ڽ���X>ؽ��
"
ֽ�|86	Խ(�+y�6ҽ;3���н��.4Nν�!p/�^˽�d7���Ƚ��؜�ƽ�b1�ĽK?�\��½�
6�����G�L������6������/���EDPq��|_�@V5����M�eӧ�����z5��!���)_�����_����e���]��        �-���q=�/k��ڂ=̴�L���=_�H�}��=�>�i�E�=V���Ұ�=y�訥=|_�@V5�=<QGEԬ=�8�4L��=�EDPq�=����/�=�Į#��=���6�=�
6����=K?�\���=��؜��=�d7����=�!p/�^�=��.4N�=;3����=(�+y�6�=PæҭU�=�Qu�R"�=i@4[��=z�����=ݟ��uy�=�/�4��==��]���=��1���='j��p�=��-��J�=�K���=�9�e��=����%�=f;H�\Q�=�tO���=�f׽r��=nx6�X� >�`��>�mm7&c>y�+pm>��|�~�>���]���>�[�=�k�>��~���>;�"�q�>['�?��>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?k�1^�sO?nK���LQ?�������:�              �?              �?       @               @      @      �?               @      @       @       @      �?      �?       @      @      �?               @              �?              �?              �?              �?      �?              �?      �?               @              �?      �?              �?      �?      �?              �?              �?              �?              �?      �?      �?      @      @       @      @      @       @       @      @      @       @      �?      @      @       @      �?       @              �?       @              �?      �?      �?              �?              �?               @              �?              �?              �?              >@              �?              �?              �?              �?      �?      �?      �?              �?               @              �?              �?      �?      �?              @       @              @      �?       @      �?      �?              @      �?              @       @       @              @      �?      �?              �?              �?              �?              �?      �?      �?              �?      �?      �?      �?       @              �?       @               @              �?      �?      �?      �?      �?      �?      @      @              �?      @      �?      @      �?       @      �?      �?       @       @      @      �?      �?      @              @       @      �?      �?      �?              �?              �?        
�
conv5/weights_1*�	   ���¿   �N��?      �@!  ?y��)��άQI@2�	�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6�+A�F�&�U�4@@�$��S�F !�ji6�9���T7����5�i}1���d�r�pz�w�7�>I��P=�>6�]��?����?��d�r?�5�i}1?�T7��?��ڋ?�.�?ji6�9�?U�4@@�$?+A�F�&?I�I�)�(?��VlQ.?��bȬ�0?��82?��%�V6?uܬ�@8?��%>��:?d�\D�X=?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?�������:�	             `k@     �t@     �p@     �n@     �n@     `l@     �e@     �e@     �d@      `@     �]@     @\@      Z@     �Z@      W@     �R@     @R@     �P@      I@     �L@     �J@     �A@      F@      B@      >@      4@      :@      :@      0@      ;@      <@      &@      4@      *@      2@      $@       @      @      (@       @      @       @      @      @      @      @      "@      @      @      @       @      �?      @      @      @      �?              @      @       @              �?              �?              �?              �?              �?      �?              �?              �?              �?      �?              �?      �?               @      �?               @      �?               @      �?       @              �?      �?       @       @       @      �?      @      @       @      @      @      @      @      @      @      @      @      "@      @      *@      $@      .@      $@      @      0@      *@      =@      5@      5@      *@      2@      =@      <@      <@      F@      ?@     �B@      D@     �N@     �H@      S@     �U@     �P@     �O@      X@     �V@      [@     �Z@     @^@      d@      b@      e@     �f@     �k@      l@     �p@     �r@     �q@     �j@        
�
conv5/biases_1*�	   ��6�   ��'>      <@!   @��)u�w>>AR<2��#���j�Z�TA[���tO����f;H�\Q������%���9�e����K��󽉊-��J�'j��p�ݟ��uy�z�����i@4[���Qu�R"�PæҭUݽH����ڽ��؜�ƽ�b1�ĽK?�\��½5%���=�Bb�!�=K?�\���=�b1��=(�+y�6�=�|86	�=���X>�=H�����==��]���=��1���='j��p�=��-��J�=�K���=�tO���=�f׽r��=���">Z�TA[�>�J>2!K�R�>�������:�              �?               @              @              �?      �?              �?              �?              @              �?      �?              �?              �?              �?               @              �?              �?      �?              �?              �?              �?        4Ӳ�U      ����	�_"0���A*��

step  �@

lossv�>
�
conv1/weights_1*�	   `[n��   �ܓ�?     `�@! �I\�n��)�W�@2�����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��>�?�s���O�ʗ���8K�ߝ�a�Ϭ(���(���>a�Ϭ(�>>�?�s��>�FF�G ?�5�i}1?�T7��?��ڋ?�.�?ji6�9�?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?�������:�              P@      k@      f@     �d@      e@     �d@     �`@      ^@      [@     @U@     @U@     @V@     �M@     �P@      K@     �L@     �L@      H@      C@      D@      6@      D@      A@      9@      ?@      7@      7@      1@      "@      1@      .@      .@       @      &@      "@      $@      @      @      "@      @      @      @      �?      @      @      @      �?      �?      �?      @      @       @      �?      �?       @              �?               @              �?              �?              �?      �?              �?              �?              �?              �?              �?               @      �?               @      �?      �?      �?       @              �?       @      �?      �?              @      @       @      @      @      @      @      @      @      @       @      @      &@      "@      0@      2@      @      &@      1@      *@      ,@      1@      6@      0@      =@      ;@      @@      9@     �D@      >@      E@      E@      J@      H@     �N@      M@      O@     @S@     �T@      [@     @X@     �_@     �`@      a@     `c@     �g@     �i@      g@      J@        
�
conv1/biases_1*�	    �BA�    �^G?      @@!  8�hT?)�V�C�>2��!�A����#@�d�\D�X=���%>��:�uܬ�@8��u�w74���82���VlQ.��7Kaa+�I�I�)�(�U�4@@�$��[^:��"�ji6�9���.���vV�R9��T7�����[���FF�G �0�6�/n�>5�"�g��>8K�ߝ�>�h���`�>})�l a�>pz�w�7�>f�ʜ�7
?>h�'�?��d�r?�5�i}1?�S�F !?�[^:��"?U�4@@�$?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?��%>��:?d�\D�X=?���#@?�T���C?a�$��{E?
����G?�������:�              �?       @      �?      �?              �?              �?      �?               @              �?               @              �?              �?              �?              �?              �?              �?               @      �?               @              �?      �?              �?       @              �?       @        
�
conv2/weights_1*�	   `K���   ��é?      �@!�#T,;�"@)[�N��VE@2����g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾�iD*L�پ�_�T�l׾jqs&\�ѾK+�E��Ͼ;�"�qʾ
�/eq
Ⱦ0�6�/n���u`P+d����n����������>
�/eq
�>jqs&\��>��~]�[�>�_�T�l�>�iD*L��>�f����>��(���>a�Ϭ(�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?�������:�             ��@     �@     ��@     ؜@     l�@     ԗ@     P�@     8�@     �@     ��@     H�@     (�@     x�@     ��@     X�@     ؁@     ��@     �}@     �z@     @y@     �u@      v@     Ps@     �p@     �n@      j@     `k@     @i@     �f@      c@     ``@     @[@     @]@     @Z@      \@     �S@     �P@      R@     �U@     �Q@      M@      Q@      G@      J@      E@      C@      ?@      ;@      ;@      9@      8@      7@      0@      (@      4@      (@      *@      @      (@      ,@       @      @      @      @      @       @      @      �?      @      @       @      @      @       @      @       @       @      �?              @      �?       @               @      �?              �?              �?      �?              @              �?              �?              �?              �?      �?              �?              �?              �?              �?      �?               @               @              �?       @      �?       @      @               @      @      @              @      @      @      @      @      @      @      @      @      @      ,@      $@      .@      "@      $@      $@      ,@      8@      0@      ,@      *@      =@      <@      A@      <@     �A@      E@     �B@      C@      O@      J@      O@     �W@     �T@     �R@     �X@      Z@     �^@     ``@     �_@     `c@     �e@      h@      k@     �j@     `p@     �p@     �r@     �r@     0v@     �y@     0y@     0~@     ؀@     @�@     �@     h�@     H�@     x�@     ؍@     �@     <�@     ��@     ��@     ��@     Ԛ@     `�@     ��@     H�@     �@        
�
conv2/biases_1*�	   ���T�    f�P?      P@!  �.�s?)J�\D���>2�<DKc��T��lDZrS��qU���I�
����G��T���C��!�A����#@�d�\D�X=���%>��:���82���bȬ�0��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7�����[���FF�G ���Zr[v��I��P=���uE���⾮��%ᾕXQ�þ��~��¾jqs&\��>��~]�[�>})�l a�>pz�w�7�>����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?k�1^�sO?nK���LQ?�������:�              �?              �?              �?       @       @      �?               @              �?       @      �?      �?              @              �?              �?              �?              �?              �?              �?              �?              �?              �?       @              �?      �?              �?      �?      @      �?              �?       @      �?              �?       @      �?      @       @               @      @      �?       @       @               @              �?        
�
conv3/weights_1*�	   ��>��    �H�?      �@! ��<�O �)�%�~KU@2�����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��a�Ϭ(���(��澢f�����uE���⾮��%ᾄiD*L�پ�_�T�l׾;�"�qʾ
�/eq
Ⱦ豪}0ڰ��������|�~���MZ��K����n����>�u`P+d�>;�"�q�>['�?��>��~]�[�>��>M|K�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?�������:�             P�@     ~�@     ��@     �@     n�@     ȡ@      �@     x�@     �@     ��@     �@     ��@     �@     ��@     ��@     ��@     ��@     Ѕ@     ��@     ��@     Ȁ@     P}@     �{@     py@     Pv@     @v@      u@     0q@     p@     �k@     `i@     �f@     �g@      e@      c@     �_@     @b@     @V@     �Z@      Y@      T@     �S@     �M@      O@      N@      K@     �D@      ;@      @@     �E@      6@      <@      8@      7@      6@      .@      3@      2@      *@      ,@      "@      0@      1@      @      @      @      @       @      @      @      @       @      @      �?      @      �?      @      @      �?      �?      �?      �?      �?      �?               @       @              �?              �?      �?              �?              �?              �?              �?              �?              �?              �?              �?      �?      �?       @      �?      �?      �?               @              �?      �?      �?       @      �?      @      �?      @      @      @      @       @      @      @      @      @      @      @      @      @      &@      ,@      $@      $@      0@      *@       @      0@      3@      9@      ?@      5@      ?@      >@     �B@      ?@     �H@     �G@      H@     �R@      M@     @R@     �T@     �S@     �R@     @X@      X@     �]@     �`@     `c@     `d@     �f@     �k@     �m@     `i@     �m@     0q@     0r@     �s@     �u@     �w@     �y@     @     ��@     X�@      �@     P�@     ��@     �@     ��@     P�@     �@     �@      �@     �@     @�@     4�@     ��@     �@     p�@     "�@     ,�@     ��@     �@        
�
conv3/biases_1*�	   `WN�    �.N?      `@!  ����p?)o��7K�>2�k�1^�sO�IcD���L�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.���vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������1��a˲���[���FF�G �>�?�s���O�ʗ���pz�w�7��})�l a�8K�ߝ�a�Ϭ(���~��¾�[�=�k��5�"�g���0�6�/n����������?�ګ�.��fc���X$�z��        �-���q=�ߊ4F��>})�l a�>1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?�������:�              �?              �?      @              �?      @       @      @      �?       @       @       @       @      @      @      �?      �?       @      @       @      �?              �?      �?               @              �?       @              �?      �?      �?      �?              �?              �?              �?              �?              �?              �?              @              �?              �?              �?      �?      �?      @               @      �?       @      �?              @      �?              @       @      �?      @      @      @      �?       @      @      @              @      �?      @      �?       @      �?      �?        
�
conv4/weights_1*�	     ��   �'�?      �@! H�����)�?�Րe@2�Ӗ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9����ڋ��vV�R9��T7����5�i}1���d�r�>h�'��f�ʜ�7
������1��a˲���[���FF�G �>�?�s�����Zr[v��I��P=��})�l a��ߊ4F��f�����uE������>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��ϾK+�E���>jqs&\��>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�������:�             @]@     ��@     ��@     ��@     h�@     �@     x�@     ��@     `�@     �@     p�@     p�@     p}@     �z@     0y@     �x@     �u@     `v@     �q@     �q@     `l@     `k@     �e@     �f@      e@     �c@     �]@      ^@     @X@     @V@     �T@     @V@      O@     �N@     �N@     �I@     �H@      E@      B@      J@     �B@      >@      <@      :@      =@      5@      3@      5@      .@      4@      &@      1@      &@      $@      @      @      @       @      @       @      @      @      @       @      @      @      �?       @      @      �?      @              �?      @              �?      �?              �?              @      �?               @              �?              �?              �?              �?              �?              �?              �?              �?              �?              @              �?               @               @               @              �?      �?       @              �?       @       @       @      @       @      �?      @       @       @      @      @      �?       @      @      "@      @      @      @       @      "@      "@      $@      *@      ,@      5@      3@      3@      ,@      9@      5@      ;@      ?@      B@      F@      F@     �F@      I@      I@     �Q@     @Q@      P@     @U@      W@     @X@      \@     �_@      `@     @a@     �`@     `e@      g@     @i@     �j@     `p@     `q@     �r@     �u@     �v@     �x@      x@     �|@     �~@      �@     ؂@     �@     8�@     ��@     ��@     ��@     <�@     ��@     ̕@     �@     �b@        
�
conv4/biases_1*�	   ��'G�   �	�X?      p@!���2
�?)Ax�t�0�>2�
����G�a�$��{E��!�A����#@���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9���d�r�x?�x��>h�'��O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F����(��澢f�����uE���⾄iD*L�پ�_�T�l׾��>M|KվK+�E��Ͼ['�?�;
�/eq
Ⱦ����žRT��+��y�+pm��mm7&c��`���nx6�X� ��f׽r����tO����f;H�\Q������%���9�e����K��󽉊-��J�'j��p���1���=��]����/�4��ݟ��uy�z�����i@4[���Qu�R"�PæҭUݽH����ڽ���X>ؽ��
"
ֽ�|86	Խ(�+y�6ҽ;3���н�!p/�^˽�d7���Ƚ��؜�ƽ�b1�Ľ���6���Į#�������/���EDPq���8�4L���<QGEԬ�e���]����x����        �-���q=!���)_�=����z5�=������=_�H�}��=�>�i�E�=��@��=V���Ұ�=�8�4L��=�EDPq�=���6�=G�L��=��؜��=�d7����=�!p/�^�=;3����=(�+y�6�=�|86	�=��
"
�=H�����=PæҭU�=�Qu�R"�=i@4[��=z�����=ݟ��uy�=�/�4��==��]���=��1���='j��p�=��-��J�=�K���=�9�e��=����%�=f;H�\Q�=�tO���=�f׽r��=nx6�X� >�`��>�mm7&c>y�+pm>RT��+�>���">Z�TA[�>�5�L�>;9��R�>���?�ګ>����>�u`P+d�>0�6�/n�>K+�E���>jqs&\��>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?f�ʜ�7
?>h�'�?x?�x�?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�������:�              �?               @              @               @       @      �?      @       @              �?      �?              �?      @      �?               @              �?      �?               @       @              �?      �?               @      �?              �?      �?              �?              �?              �?       @              �?      �?      �?      @      @      @      @      @      @       @       @      �?      �?      �?       @      �?      �?      �?       @      @      �?       @       @              �?       @      �?              �?              �?              �?              �?              <@              �?              �?      �?              �?              �?              �?              �?      �?              @              �?              �?       @              @      �?      �?              @              @      �?       @       @      @      @      �?      �?              �?      �?      �?              @              �?              �?              �?              �?              �?              �?               @              �?      �?              �?      �?      �?              �?      �?      @      �?               @              �?      �?              �?       @       @      �?      @      @      �?      @       @      @      @       @              �?      �?      �?      �?      @       @      @               @      @       @      @       @              �?              �?        
�
conv5/weights_1*�	   ��¿   ����?      �@!  �H��)�)Ɨ�QI@2�	�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6�U�4@@�$��[^:��"�ji6�9���.���vV�R9��T7�����d�r�x?�x���FF�G ?��[�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?��VlQ.?��bȬ�0?��82?��%�V6?uܬ�@8?��%>��:?d�\D�X=?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?�������:�	             `k@     �t@     �p@     �n@     �n@     `l@     �e@     �e@     �d@      `@     �]@     �\@     �Y@     �Z@     @W@     �R@     @R@     �P@      I@     �L@     �J@     �A@      F@      B@      >@      4@      :@      :@      0@      ;@      <@      &@      4@      (@      3@      $@      @      @      (@       @      @      @      @      @      @      @      $@      @      @      @       @      �?      @      @      @      �?              @       @      @              �?              �?              �?              �?              �?              �?              �?              �?              �?      �?              �?      �?               @      �?               @      �?               @      �?       @              �?      �?      �?      @       @      �?      @      @       @      @      @      @      @      @      @      @      @      &@      @      (@      "@      .@      &@      @      0@      *@      <@      5@      5@      .@      1@      <@      =@      <@      F@      >@      C@      D@     �N@     �H@      S@     �U@     �P@      P@     @X@     �V@      [@     �Z@     @^@      d@      b@      e@     �f@     �k@      l@     �p@     �r@     �q@     �j@        
�
conv5/biases_1*�	   �
g�    m6>      <@!   ���)���_<2�2!K�R���J����"�RT��+���`���nx6�X� ��f׽r����tO�����9�e����K��󽉊-��J�=��]����/�4���Qu�R"�PæҭUݽ���X>ؽ��
"
ֽ�|86	Խ�!p/�^˽�d7���Ƚ��؜�ƽ�b1�Ľ�
6����=K?�\���=H�����=PæҭU�=i@4[��=z�����==��]���=��1���='j��p�=�9�e��=����%�=f;H�\Q�=RT��+�>���">Z�TA[�>�#���j>��R���>Łt�=	>�������:�              �?              �?               @       @      �?               @       @              �?              �?              �?      �?              �?      �?      �?              �?              �?              �?              �?      �?              �?      �?              �?              �?              �?        ��nZ�T      �e��	6xE0���A*��

step  �@

loss���>
�
conv1/weights_1*�	   �g{��   ����?     `�@! ��2�_��)?��ӷ@2�����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0�U�4@@�$��[^:��"��S�F !��.����ڋ��vV�R9��T7����u`P+d�>0�6�/n�>��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?�������:�             �O@     @k@     �e@     �d@      e@      e@     @`@      ^@      Z@      V@     @U@     @V@      M@     �O@     �L@      L@      L@      I@      D@      B@      :@      C@      @@      :@      @@      5@      6@      1@      ,@      *@      1@      ,@      $@      (@      @      (@      @      @      @       @              @       @      @      @       @      @      @       @       @      @      �?              �?      �?              �?       @              �?      �?              �?              �?              �?              �?      �?              �?              �?              @              @       @              �?      @      �?               @      @              �?      �?       @      �?      @      @      @      @       @      @      @      @      @      &@      $@      .@      .@      $@      @      1@      *@      ,@      1@      7@      1@      <@      ;@      @@      <@     �B@      @@     �E@     �E@     �I@     �F@     �N@     �N@      O@     @S@     @T@     �Z@      Y@     �^@     �`@      a@     �c@     @g@     `i@     @g@     �J@        
�
conv1/biases_1*�	   ��-F�   �+�P?      @@!  0�E]?)m	+����>2�
����G�a�$��{E��T���C��!�A����#@���%>��:�uܬ�@8��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"�ji6�9���.����ڋ��[�=�k�>��~���>})�l a�>pz�w�7�>��[�?1��a˲?�vV�R9?��ڋ?�S�F !?�[^:��"?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?a�$��{E?
����G?�qU���I?k�1^�sO?nK���LQ?�������:�              �?               @       @              �?              �?      �?      �?              �?              �?      �?              �?      �?              �?              �?              �?              �?              �?              �?              �?       @              �?              �?       @              @      �?              �?        
�
conv2/weights_1*�	    ����   ��ש?      �@!��#/Uo#@)�N4�VE@2����g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ���(��澢f���侮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پK+�E��Ͼ['�?�;;�"�qʾ�XQ�þ��~��¾豪}0ڰ���������]������|�~��
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>})�l a�>pz�w�7�>I��P=�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?�������:�             ��@     �@     ��@     ��@     d�@     ̗@     d�@     $�@     �@     ؏@     0�@     �@     ��@     X�@     ��@     ��@     ��@     �}@      {@     Py@     �u@     �u@     �s@     Pp@     �n@     �j@     �j@     �i@     �f@     �b@      `@     �\@     �[@     @[@     �Z@     �V@     �M@     @Q@     @W@     �P@      N@     �M@     �G@      L@      C@      C@      C@      =@      9@      5@      <@      .@      2@      1@      7@       @      *@      @      .@      "@       @      @      @      "@      @       @       @      @      @      @      @      @       @       @       @      �?       @               @      @      �?               @              �?               @              �?              �?      �?      �?              �?      �?              �?              �?              �?              �?              �?              �?      �?              �?      �?      �?      �?      �?              �?      �?              @      �?               @      @      @      @      �?       @       @      @      @      @      @      @      @      @      @      @      (@      $@      &@      $@      $@       @      1@      .@      2@      2@      3@      &@      <@      9@      B@      ;@     �@@     �D@      D@     �D@      N@      J@     �P@     @V@     �U@      R@     @Y@     @[@     �\@      a@     �_@     �b@     �e@     @h@     �j@     `j@     Pp@     �p@     �r@      r@     w@     �y@     �y@     �}@     �@     ��@     p�@     @�@     h�@     h�@     ��@     ,�@     (�@     �@     ��@     ��@     К@     L�@     ��@     L�@     (�@        
�	
conv2/biases_1*�		    ��[�    �TW?      P@!   �{?)΀��x��>2�E��{��^��m9�H�[�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���T7����5�i}1���d�r�x?�x��f�ʜ�7
������6�]���1��a˲��FF�G �>�?�s���
�/eq
Ⱦ����žx?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?ܗ�SsW?��bB�SY?�������:�              �?              �?              �?              �?      @              �?      �?              �?       @      �?      �?      �?      �?              �?      �?              �?              �?              �?              �?              �?              �?              �?      �?      �?              �?      �?              �?       @               @      �?       @       @      �?              @       @      �?      �?       @              �?      �?      @      �?      @      �?              �?      �?              �?        
�
conv3/weights_1*�	    �F��   `HX�?      �@! �>����)$/���KU@2�����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��pz�w�7��})�l a��ߊ4F��h���`f�����uE���⾙ѩ�-߾E��a�Wܾ�_�T�l׾��>M|Kվ�*��ڽ�G&�$�����]���>�5�L�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?�������:�             H�@     ��@     ��@     &�@     h�@     ġ@     �@     ��@     �@     ȗ@     ��@     t�@     �@     ��@     ��@     ��@     ��@     ��@     ��@     ��@     p�@     `}@     0|@     Py@     @v@     `v@     �t@     Pq@      p@     @l@     @h@     @g@      h@     `d@     �b@     ``@      b@     �V@     �Y@     @Z@      T@     �Q@      P@     �P@      K@     �J@      D@     �B@      9@      E@      ;@      :@      9@      5@      4@      6@      .@      .@      &@      (@      .@      ,@      3@       @      @      "@      @      @      @      @      @      @       @       @      @       @       @      @      �?      �?       @       @      �?              �?       @      @              �?              �?              �?              �?              �?              �?      @               @      �?              �?              �?               @       @      �?       @              @       @      @      @      @      �?       @      �?      �?      @      @      @      @       @      @      "@      0@      &@      0@      &@      (@      @      3@      2@      6@      >@      9@      A@      5@     �C@      @@      J@     �I@     �E@     �R@      N@     �Q@     �T@      R@     �U@      V@      Z@      \@     `a@     �b@     �d@     �f@     �k@     @n@      i@     �m@     �p@     �r@     @s@     �u@     0x@     @y@     �~@     @�@     �@     8�@     p�@     ��@     �@     Ќ@     H�@     �@     �@     ��@     $�@     $�@     \�@     x�@     ,�@     h�@      �@     4�@     ��@     �@        
�
conv3/biases_1*�	   �H�R�   �ĨU?      `@! ��ו;y?)�P�Pl��>2��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1�1��a˲���[��O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F����(��澢f���侄iD*L�پ�_�T�l׾jqs&\�ѾK+�E��Ͼf^��`{�E'�/��x�        �-���q=a�Ϭ(�>8K�ߝ�>�h���`�>O�ʗ��>>�?�s��>�FF�G ?1��a˲?6�]��?����?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?�������:�              �?               @               @              @      @      @      @              @       @       @      @       @      @              �?       @      @      �?      �?       @              �?      �?      �?       @              �?              �?              �?      �?      �?              �?              �?              �?              �?              @              �?      �?              �?      �?               @      �?              �?              �?              @       @      �?              �?      �?       @      @      @      �?               @      �?      @      @      @      @      @      @       @              @      �?      @      �?      �?      �?      �?        
�
conv4/weights_1*�	    � ��    ��?      �@! WI����)�G9��e@2�Ӗ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.���T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������1��a˲���[��>�?�s���O�ʗ���8K�ߝ�a�Ϭ(���>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ�u`P+d�>0�6�/n�>��>M|K�>�_�T�l�>���%�>�uE����>��(���>a�Ϭ(�>�ߊ4F��>})�l a�>��Zr[v�>O�ʗ��>�FF�G ?��[�?1��a˲?6�]��?>h�'�?x?�x�?��d�r?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�������:�              ]@     ��@     ��@     ��@     h�@      �@     x�@     ��@     P�@      �@     h�@     x�@     �}@     pz@     0y@     �x@     �u@     Pv@     �q@     �q@     `l@      k@      f@     �f@     @e@     �c@     �]@     �]@     @X@     �U@     �U@     �U@     �N@      O@     �N@      I@      H@      F@      B@     �I@      C@      ?@      ;@      :@      =@      4@      4@      5@      0@      4@      $@      1@      &@      $@      @       @      @      @      @       @      @      @       @      @      @      @      �?       @      @      @      @      �?      �?      �?      �?               @      �?              �?      �?      �?               @               @              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?      �?       @               @      �?               @      �?      @              �?      @       @      @       @      �?      @       @      �?      @      @      �?      @      @      @      "@      @      @      @      $@      "@      *@      $@      ,@      4@      3@      4@      *@      ;@      4@      9@      @@     �A@      G@      E@     �G@      I@      H@      R@     �P@     �P@     @U@     @W@     @X@     �[@      `@     �_@     `a@     �`@     @e@      g@     `i@     �j@     Pp@     �q@     �r@     `u@     �v@     Px@     `x@     �|@     �~@     �@     ؂@     �@     0�@     ��@     ��@     ��@     <�@     �@     ̕@      �@     `b@        
�
conv4/biases_1*�	   @\�O�   `[�_?      p@!?4-'[�?)k�G�k�>2�nK���LQ�k�1^�sO��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !��.����ڋ��vV�R9��5�i}1���d�r�1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��pz�w�7��})�l a��ߊ4F��h���`ѩ�-߾E��a�Wܾ��>M|Kվ��~]�[Ӿ�*��ڽ�G&�$���#���j�Z�TA[�����"�RT��+��y�+pm��mm7&c��`���nx6�X� ��f׽r����tO����f;H�\Q������%���9�e����K��󽉊-��J�'j��p���1���=��]����/�4��ݟ��uy�z�����i@4[���Qu�R"�PæҭUݽH����ڽ���X>ؽ��
"
ֽ;3���н��.4Nν�!p/�^˽�d7���ȽK?�\��½�
6�����5%����G�L����Į#�������/��<QGEԬ�|_�@V5����M�eӧ�y�訥�        �-���q=��x���=e���]�=����z5�=���:�=���6�=G�L��=�
6����=K?�\���=�b1��=��؜��=�|86	�=��
"
�=���X>�=i@4[��=z�����=ݟ��uy�=�/�4��=��1���='j��p�=��-��J�=�K���=�9�e��=����%�=f;H�\Q�=�tO���=�f׽r��=nx6�X� >�`��>�mm7&c>RT��+�>���">�#���j>�J>2!K�R�>f^��`{>�����~>�u`P+d�>0�6�/n�>G&�$�>�*��ڽ>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?��d�r?�5�i}1?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?��bB�SY?�m9�H�[?E��{��^?�l�P�`?�������:�              �?              �?      �?               @       @      �?       @      @              @       @              �?      �?      �?               @      �?              �?       @              @               @               @              �?              �?               @              �?              �?              �?              �?              �?       @      �?       @              @      @      @      @       @              @      @      @               @       @      @       @      @              �?      �?      �?              �?               @              �?              �?              �?              �?              �?              <@              �?              �?              �?              �?              �?              �?      �?              @              @              �?      @      @       @      �?      @      @      @       @               @               @              �?       @              �?              �?              �?               @      �?              �?              @              �?              �?      �?       @              �?      �?       @      �?      �?      �?      �?               @               @       @       @      �?      @              �?      �?      @      @       @      �?      @      @       @              �?               @       @       @       @      @       @      @              @      �?              �?              �?        
�
conv5/weights_1*�	    ��¿   `Ҝ�?      �@! ���O��)��@�7RI@2�	�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9�>h�'��f�ʜ�7
�����?f�ʜ�7
?��d�r?�5�i}1?�vV�R9?��ڋ?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?��%�V6?uܬ�@8?��%>��:?d�\D�X=?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?�������:�	             @k@     �t@     �p@     �n@     �n@     `l@     �e@      f@     �d@      `@     �]@     �\@     �Y@     �Z@     @W@     �R@      R@      Q@      I@     �L@     �J@     �A@      F@      B@      =@      5@      :@      :@      0@      ;@      <@      &@      5@      $@      4@      &@      @      @      (@       @      @      @      @      @       @      "@      "@      @      @      @       @      �?      @      @       @      �?              @       @      @              �?              �?              �?              �?              �?              �?              �?               @              �?               @      �?               @       @      �?               @      �?       @              �?      �?      �?      @       @       @      @      @       @      @      @      @      @      @      @      @      @      &@      @      *@      "@      .@      "@      "@      0@      *@      ;@      7@      4@      .@      0@      >@      <@      ;@      F@      ?@      C@     �D@      N@     �H@     �R@     �U@     @P@     @P@      X@     �V@      [@     �Z@      ^@      d@     @b@      e@     �f@     �k@      l@     �p@     �r@     �q@     �j@        
�
conv5/biases_1*�	    y5�    �w>      <@!  ��*�)z�-�'�i<2���f��p�Łt�=	�2!K�R���J�RT��+��y�+pm��mm7&c��`����tO����f;H�\Q������%���9�e����K���=��]����/�4���Qu�R"�PæҭUݽH����ڽ���X>ؽ��.4Nν�!p/�^˽��M�eӧ�y�訥����6�=G�L��=;3����=(�+y�6�='j��p�=��-��J�=�K���=�tO���=�f׽r��=nx6�X� >�`��>���">Z�TA[�>�J>2!K�R�>��f��p>�i
�k>�������:�              �?              �?              @       @      �?               @              �?      �?              �?              �?              �?               @              �?              �?               @              �?      �?              �?              �?              �?              �?              �?        ��ǱT      >�	DQj0���A*��

step  �@

loss�3�>
�
conv1/weights_1*�	    T���   ���?     `�@!  ��m���)�5�U@2�����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��uE���⾮��%���~]�[�>��>M|K�>>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?�.�?ji6�9�?�S�F !?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?�������:�             �O@      k@      f@     @e@     �d@     �d@      `@     @_@     �Y@     @U@     @T@     �W@      N@     �M@      N@     �L@     �L@     �H@      C@      D@      7@     �C@      =@      :@      @@      8@      0@      4@      ,@      ,@      3@      (@      ,@      "@      @      (@      @      @      @      @       @      @       @       @      @      @      @      �?       @      @      @       @              �?      �?              �?      �?               @      �?      �?              �?              �?              �?      �?       @              �?              �?      �?              �?      �?      �?      @      �?      �?      @       @       @      �?      @       @      �?      @      @      @      @      @      @      @      $@       @      &@      ,@      $@      (@      $@      1@      (@      .@      *@      5@      7@      :@      =@      =@     �@@      A@      @@     �G@     �@@      K@     �G@      O@     �N@     �O@     �R@     @T@      Z@     @Z@      _@     @`@      a@     @d@     @g@     �h@     �g@     �K@        
�
conv1/biases_1*�	   ��;J�   `��U?      @@!  `V�c?)��<��5�>2�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�uܬ�@8���%�V6��u�w74���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.�������>
�/eq
�>pz�w�7�>I��P=�>x?�x�?��d�r?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?<DKc��T?ܗ�SsW?�������:�              �?      �?               @      �?      �?              �?      �?               @              �?              �?              �?              �?              �?              �?              �?              �?               @      �?              �?      �?               @              �?      �?              �?              @              �?        
�
conv2/weights_1*�	    �©�   ��?      �@!�	�y$@),�pqWE@2����g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(龢f�����uE���⾮��%ᾙѩ�-߾�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;�z��6��so쩾4�豪}0ڰ>��n����>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>���%�>�uE����>�f����>��(���>a�Ϭ(�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?�������:�             |�@     ��@     ��@     ܜ@     d�@     ė@     L�@     (�@     �@     �@     P�@     �@     ��@     8�@     X�@     Ё@     Ȁ@     @}@     @{@     �y@     pu@      v@     Ps@     �p@      n@      k@     @j@     �i@     `f@     �c@      _@     �\@     @\@     �Y@      ]@     @T@     �Q@     @P@     �S@     �T@      L@     �K@     �J@      L@      E@      A@     �A@      >@      7@      =@      3@      8@      ,@      3@      2@      (@      @       @      .@       @      $@      @      @      @      @       @      @       @      @       @       @      @      @      @      @               @      �?      �?      @       @              �?              �?              �?      @              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?      �?      �?      �?               @       @       @       @      @               @              �?      @       @      @      @      @      @      @      @      @      @      �?      @      (@      @      $@      (@      @      &@      "@      &@      1@      4@      4@      4@      *@      ;@      ;@      ;@      =@     �B@     �D@     �B@      F@     �P@      I@      P@      V@      U@     �Q@     �Y@      ]@     @\@     @a@     @_@      b@      f@     �h@      j@      k@     �p@      p@     �r@     @r@     w@     �y@     �y@     0}@     P�@     ��@     `�@     H�@     X�@     ��@     P�@     L�@      �@     �@     ��@     ��@     ؚ@     8�@     ��@     R�@      �@        
�
conv2/biases_1*�	    6�a�    X�\?      P@!  �3�$�?)՗����>2����%��b��l�P�`�ܗ�SsW�<DKc��T�nK���LQ�k�1^�sO�IcD���L�
����G�a�$��{E�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !���ڋ��vV�R9���[���FF�G ��vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?�m9�H�[?E��{��^?�������:�              �?              �?              �?       @              @               @      �?      @       @      �?              �?              �?              �?              �?              �?              �?              @      �?              �?              �?              @      �?               @      @      @      @              �?       @      �?      �?       @      @              @      �?               @              �?        
�
conv3/weights_1*�	   ��N��   �4i�?      �@! �m��t��)s�d9�KU@2�����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��8K�ߝ�a�Ϭ(���(��澢f�����uE����E��a�Wܾ�iD*L�پ�_�T�l׾��~]�[Ӿjqs&\�ѾK+�E��Ͼ
�/eq
Ⱦ����žG&�$��5�"�g���豪}0ڰ>��n����>�[�=�k�>��~���>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?�������:�             @�@     x�@     ��@     &�@     d�@     ̡@     &�@     T�@      �@     ��@     ԕ@     x�@     �@     ��@     ��@      �@     ��@     ��@     ؄@     ��@     p�@      }@     @|@     �y@     0v@     �u@     �t@     pq@     �p@      l@     �h@     `g@     @g@     �d@     �b@     �`@     �a@     �W@     �Y@     @Y@     �T@     @P@     �Q@      P@     �L@     �I@      C@      B@      8@      F@      9@      >@      8@      >@      6@      ,@      3@      $@      "@      "@      .@      2@      1@       @       @      &@      @      @      "@      @       @      @      @               @              �?       @       @      @              @      �?      �?       @              �?              �?      �?               @              �?       @              �?      �?              �?              �?              �?              �?              �?               @              �?       @               @              �?       @       @      �?       @       @               @      @      @      @               @      @      @       @      @      @      @      @      $@       @      (@      (@      (@      ,@      .@      (@      0@      .@      5@      A@      2@      A@      8@      E@      =@      I@     �H@      H@     @P@     �Q@     �Q@     �R@      T@      T@     �V@      Z@     �\@      a@     `c@      e@     @f@     �k@     �m@     �h@     �m@     `q@     Pr@     Ps@     �u@     �w@     �y@     �~@     �@     �@     @�@     ��@     �@      �@     ��@     8�@     0�@     ��@     �@     �@     (�@     ��@     X�@      �@     v�@     $�@     &�@     ��@     (�@        
�
conv3/biases_1*�
	    �#V�   ��H\?      `@!  3o�6�?)�p�?2�ܗ�SsW�<DKc��T��lDZrS�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�>h�'��f�ʜ�7
�>�?�s���O�ʗ�����(��澢f�����uE������>M|Kվ��~]�[Ӿ��n�����豪}0ڰ�        �-���q=�iD*L��>E��a�W�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?x?�x�?��d�r?�vV�R9?��ڋ?�.�?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�������:�              �?       @              @       @       @      @      @      @              @      @      �?      @      �?      �?       @       @       @       @      �?              �?              �?      �?      �?               @      �?              �?              �?              �?      �?              �?              �?              @              �?              �?              �?      �?              �?              �?      @               @      �?      @      �?      �?       @      �?      @      �?               @       @       @      @      @      @      @      @       @       @      @              �?       @              �?        
�
conv4/weights_1*�	   `���   @��?      �@! ��p#�)�G��e@2�Ӗ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������1��a˲���[���FF�G �>�?�s���O�ʗ����ߊ4F��h���`���>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼa�Ϭ(�>8K�ߝ�>})�l a�>pz�w�7�>��Zr[v�>O�ʗ��>>�?�s��>1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�������:�              \@     �@     �@     ��@     h�@     �@     ��@     ��@     P�@      �@     p�@     x�@     p}@     pz@      y@     �x@     �u@     �v@     Pq@     �q@     `l@      k@     �e@     �f@     �e@     �c@     @]@      ^@     �X@     �U@     �U@     �U@     �N@     �N@      O@      I@      H@      F@     �B@      J@      C@      ?@      9@      <@      ;@      4@      3@      6@      4@      .@      &@      1@      *@      "@      @       @      @      @      @      @      @      @       @      @      @      @      @              @      �?      @       @      �?      �?               @              �?      �?       @              �?      �?              �?       @              �?              �?               @              �?              �?              �?               @      �?              �?               @      �?      �?              �?       @       @      �?              @      @      @      @      �?      @      @      @      @      @       @       @      @      @      @      @      @       @      $@      (@      $@      $@      ,@      4@      3@      3@      *@      ;@      5@      9@      ?@     �A@      G@     �E@     �G@      H@     �H@      R@      Q@     @P@     �U@     �V@     �X@     �[@     �`@      _@     �a@     �`@     `e@     �f@     �i@     �j@     @p@     �q@     �r@     pu@     �v@     @x@     px@     �|@     �~@     �@     �@     ��@     0�@     ��@     Ȍ@     p�@     D�@     �@     ȕ@     $�@     `b@        
�
conv4/biases_1*�	   ���V�   ��Sc?      p@!'&���?)6戚��	?2�ܗ�SsW�<DKc��T�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"���ڋ��vV�R9��T7�����d�r�x?�x��>h�'��f�ʜ�7
�1��a˲���[���FF�G �O�ʗ�����Zr[v��I��P=��pz�w�7���h���`�8K�ߝ�a�Ϭ(��uE���⾮��%��_�T�l׾��>M|Kվ0�6�/n���u`P+d��2!K�R���J��#���j�Z�TA[�����"�RT��+��y�+pm��mm7&c��`���nx6�X� ��f׽r����tO����f;H�\Q���9�e����K��󽉊-��J�'j��p���1���=��]����/�4��ݟ��uy�z�����i@4[���Qu�R"�PæҭUݽH����ڽ��
"
ֽ�|86	Խ(�+y�6ҽ��؜�ƽ�b1�ĽK?�\��½�
6������Bb�!澽5%����e���]����x�����1�ͥ��G-ֺ�І�        �-���q=G�L��=5%���=K?�\���=�b1��=��؜��=�d7����=�!p/�^�=��.4N�=;3����=�Qu�R"�=i@4[��=ݟ��uy�=�/�4��==��]���=��1���='j��p�=��-��J�=�K���=�9�e��=����%�=f;H�\Q�=�tO���=�f׽r��=nx6�X� >�`��>�mm7&c>y�+pm>Z�TA[�>�#���j>2!K�R�>��R���>Łt�=	>��f��p>G&�$�>�*��ڽ>�XQ��>�����>;�"�q�>['�?��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>�f����>��(���>a�Ϭ(�>8K�ߝ�>})�l a�>pz�w�7�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?�l�P�`?���%��b?5Ucv0ed?�������:�              �?              �?      �?              �?      �?       @      @      �?       @      @      �?              �?              �?              �?      @              �?       @              �?      @              �?              �?              �?      �?              �?               @              �?      �?              �?              �?              �?              �?              �?       @       @      �?       @      @      @              @      @              @      @      @              �?      @      @       @      �?      �?      �?      �?              �?      @              �?              �?              �?              �?              �?              <@               @              �?      �?              �?              �?               @               @      �?      @      �?      �?      �?      �?      @      �?       @      @      @      @       @      �?               @              �?      �?      �?              �?              �?              �?              �?              �?      �?      �?               @              �?              �?              �?      �?      �?      �?       @      �?      �?              @       @               @              �?       @              @      �?       @              �?      �?      @      @       @       @      @              @       @              �?       @      �?       @      @      �?      @      @      @      �?      �?              �?      �?        
�
conv5/weights_1*�	    ��¿   `���?      �@! `x�1Q�)`�?��RI@2�	�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��S�F !�ji6�9���.����ڋ������6�]�������?f�ʜ�7
?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?�������:�	             @k@     �t@     �p@     �n@     �n@     @l@     �e@     �e@     �d@     �_@     �]@     @\@      Z@     �Z@      W@     �R@      R@     @Q@     �H@      M@      K@     �@@     �F@     �A@      >@      5@      :@      :@      0@      ;@      ;@      (@      5@      "@      5@      &@      @      @      *@      @      @      @      @      @       @      "@      &@      @      @      @       @      �?      @      @       @      �?               @      @      @              �?              �?              �?      �?      �?              �?              �?              �?      �?              �?              �?      �?              �?               @       @      �?      �?      �?      �?       @              �?               @      �?       @       @      @      @      @       @      @      @      @      @      @      @      @      @      $@      @      *@      "@      ,@      $@      "@      0@      *@      ;@      7@      4@      .@      .@      ?@      <@      ;@      F@      @@     �B@     �D@     �N@      H@      S@     �U@      P@     �P@     �W@     �V@      [@     �Z@      ^@      d@      b@     @e@     �f@     �k@      l@     �p@     �r@     �q@     �j@        
�
conv5/biases_1*�	   ����   �t2 >      <@!   Z�2�)uT�b�t<2��i
�k���f��p�Łt�=	���R����2!K�R���J��#���j�Z�TA[�����"�RT��+���mm7&c��`���nx6�X� ��tO����f;H�\Q������%���9�e�����-��J�'j��p�=��]����/�4���Qu�R"�PæҭUݽH����ڽ;3����=(�+y�6�=�|86	�=PæҭU�=�Qu�R"�=i@4[��=�`��>�mm7&c>y�+pm>RT��+�>���">Z�TA[�>�#���j>��R���>Łt�=	>%���>��-�z�!>�������:�              �?              �?              �?              �?       @       @              �?      �?              �?              �?               @              �?              �?      �?              �?      �?              �?       @              �?              �?      �?              �?              �?              �?        �c��hU      � �	^��0���A*٪

step  �@

lossY5�>
�
conv1/weights_1*�	   @ş��   `2�?     `�@! �u"��)��Y	� @2�����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A���%>��:�uܬ�@8��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��vV�R9��T7����5�i}1���d�r�>h�'��f�ʜ�7
���Zr[v��I��P=��5�"�g��>G&�$�>K+�E���>jqs&\��>f�ʜ�7
?>h�'�?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?�������:�              P@      k@     �e@     @e@     �d@     �d@      `@      `@      Y@     �U@     �S@     �X@     �M@     �N@     �M@     �L@      K@      H@     �C@     �D@      5@      D@      >@      =@      ?@      8@      .@      2@      *@      0@      3@      *@      *@      ,@      @      *@       @      @      @      @      @      @              @       @      @      @       @      @      @              �?              �?      @      �?       @      �?              �?              �?      �?      �?              �?              �?              �?              �?              �?              �?              �?      �?      �?              �?      �?               @              �?      �?       @              �?       @              @      @      @      @      @      @      @      @      @      @      @      $@      .@      &@      &@      @      $@      2@      (@      .@      .@      2@      =@      8@      ;@      <@     �@@     �@@     �@@     �E@      C@      K@     �G@     �M@     �P@      N@     @R@     �T@     �Y@      [@      _@     �_@     @a@      d@     `g@     �h@     �f@      P@        
�
conv1/biases_1*�	   ��N�    ��[?      @@!  P��j?)=8��'��>2�k�1^�sO�IcD���L�
����G�a�$��{E��T���C��!�A����#@���%>��:�uܬ�@8���bȬ�0���VlQ.�+A�F�&�U�4@@�$�ji6�9���.�������6�]���;�"�q�>['�?��>��Zr[v�>O�ʗ��>��ڋ?�.�?�7Kaa+?��VlQ.?��bȬ�0?�u�w74?��%�V6?uܬ�@8?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?��bB�SY?�m9�H�[?�������:�               @               @      �?               @              �?              @              �?              �?              �?              �?              �?              �?              �?      �?              �?      @              �?              �?      �?      �?      �?      �?      �?      �?              �?        
�
conv2/weights_1*�	   ��ҩ�   `.
�?      �@! iB�C�$@)�F	�XE@2����g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澮��%ᾙѩ�-߾jqs&\�ѾK+�E��ϾG&�$�>�*��ڽ>jqs&\��>��~]�[�>�_�T�l�>�iD*L��>�ѩ�-�>���%�>�f����>��(���>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?�������:�             p�@      �@     ��@     ؜@     \�@     ��@     8�@     8�@     ԑ@     ȏ@     h�@     (�@     ��@     @�@     H�@     ��@     ��@     �}@     �z@      y@     `u@     pv@     `s@     �p@     �m@      l@     �i@     `i@     �e@     @d@     �_@      ^@     @[@     �Z@     @Y@     @V@     @R@      Q@     �S@     �Q@     �N@     �L@     �M@     �K@      ?@      >@      A@     �C@      <@      >@      3@      6@      4@      0@      ,@      *@      (@      @       @      &@       @       @      "@      @       @      @      @       @       @       @      �?      @      @      @      �?       @       @      @               @      �?               @      �?       @               @      �?      �?               @              �?              �?               @              �?              �?              �?               @       @      @              �?      �?              �?              �?      @      @      �?       @      �?      @      �?       @              @      "@      @      "@       @       @      $@      "@      @      &@      @      1@      "@      .@      ,@      1@      3@      ,@      4@      ;@      8@      9@      ?@     �B@      C@      G@      D@     @Q@      E@     �Q@     �U@      V@     @P@     @[@     @Y@      `@      a@     �^@     `b@      f@     `h@     �j@     �j@     0p@     �p@      r@     �r@     �v@     py@     �y@     P}@     @�@     ؂@     H�@     x�@     x�@     x�@     h�@     D�@     �@     (�@     p�@     Ԙ@     ��@      �@     ğ@     F�@     0�@        
�
conv2/biases_1*�	   ��oe�   �fa?      P@!  `�ֆ?)J HQ#'?2�Tw��Nof�5Ucv0ed��m9�H�[���bB�SY�<DKc��T��lDZrS�nK���LQ��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+��.����ڋ��vV�R9��T7���1��a˲���[��f�ʜ�7
?>h�'�?�T7��?�vV�R9?ji6�9�?�S�F !?�[^:��"?I�I�)�(?�7Kaa+?��VlQ.?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?�l�P�`?���%��b?�������:�              �?              �?              �?       @               @      �?              �?              �?       @      @      �?      �?              �?              �?              �?              �?              �?              �?              �?              �?      �?              �?       @              @      �?       @      �?      @       @      @       @       @               @       @       @              @               @      �?              �?        
�
conv3/weights_1*�	   �;V��   `�z�?      �@! l�g����)N�V�&LU@2�����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ���I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE����E��a�Wܾ�iD*L�پK+�E��Ͼ['�?�;�[�=�k���*��ڽ�5�"�g���0�6�/n���u`P+d���MZ��K���u��gr���MZ��K�>��|�~�>�u`P+d�>0�6�/n�>G&�$�>�*��ڽ>
�/eq
�>;�"�q�>K+�E���>jqs&\��>�iD*L��>E��a�W�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?�������:�              �@     ��@     ��@     �@     r�@     ��@     �@     t�@     ��@     ��@     �@     T�@      �@     ��@     ؍@     �@     ��@     ��@     ��@     ��@     ��@     �|@     `|@     y@     �v@     pu@     �t@     �q@     �p@      k@     �i@     @f@      g@     `d@      d@     �`@     �a@     �W@     @W@     �[@     @S@     �Q@     @Q@      M@      O@     �I@     �A@     �B@      9@      E@      ?@      :@      =@      8@      5@      2@      .@      "@      $@      0@      ,@      &@      2@      &@      @      "@      @      @      @      @      @      @      @      @       @      �?              @       @              �?      @              �?               @               @              �?               @              �?              �?              �?              �?      �?              �?              �?              �?              �?              �?              �?              �?              �?      �?              �?       @      �?              �?              �?               @      @      @       @       @      @      �?      @       @      �?       @      @       @      @       @       @      @       @      @      "@       @      2@      @      &@      2@      (@      9@      1@      0@     �A@      0@      ?@      ?@      A@      B@     �G@     �H@     �E@     �R@     �N@     �R@     �Q@     @S@      T@      X@     �Y@     @\@      a@      d@     @d@     �f@     @l@     @m@     `i@     �l@     �q@     0r@     `s@      v@     �w@     �y@     �~@     (�@     �@     @�@     ��@     ��@      �@     ��@     (�@     4�@     �@     �@     ��@     <�@     ��@     d�@     0�@     f�@     2�@     �@     ��@     0�@        
�
conv3/biases_1*�	    '�Z�   ��b?      `@!  �A�b�?)w9H'h?2��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��T7����5�i}1������6�]�����Zr[v��I��P=��8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%�G&�$��5�"�g���        �-���q=��>M|K�>�_�T�l�>���%�>�uE����>�h���`�>�ߊ4F��>})�l a�>6�]��?����?��d�r?�5�i}1?��ڋ?�.�?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?�������:�              �?       @      �?               @       @      @      @      @              @      @      @      �?       @              �?      �?      @      �?      @       @              �?      �?              �?              �?      �?              �?              �?              �?              �?              �?              �?              �?              @              �?              �?              �?      �?              �?              �?              �?              �?       @      �?       @      �?       @              @              @      @      @      �?      �?       @              @      @      @      @      @      �?      @      �?               @      �?      �?        
�
conv4/weights_1*�	   ���   ��	�?      �@!  W��L�)��>1#�e@2�Ӗ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9����ڋ��vV�R9��5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��})�l a��ߊ4F����>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�������:�              \@     �@     �@     ��@     l�@     ��@     ��@     ��@     X�@      �@     p�@     ��@     @}@     �z@     y@     �x@     �u@     �v@     Pq@     �q@     @l@     �j@     @f@     @f@     �e@     �c@     �]@      ^@     �X@     �U@     @U@     �U@      O@     �O@      N@     �I@     �G@     �E@     �C@     �I@      D@      =@      :@      >@      9@      3@      2@      8@      4@      ,@      $@      2@      0@       @      @      "@      @      @      @       @      @      @      @      @      @       @      �?      @      @      @       @      �?       @       @              �?               @              �?      �?      �?              �?      �?      �?              �?              �?              �?              �?              �?              �?      �?              �?      �?              �?              �?       @      �?              �?       @      �?      �?               @              @      @       @      @      @      @       @      @      �?      @      �?      @      @      @      @      @      @      "@      (@      &@       @      $@      0@      3@      4@      1@      .@      9@      7@      9@      =@     �C@     �D@     �G@     �F@     �H@     �H@     �Q@     �P@     @P@      V@     @V@     @X@      \@     ``@     @_@     �a@     �`@     @e@     �f@      j@     �j@     @p@     �q@     �r@     �u@     �v@     @x@     �x@     p|@     �~@     �@     ��@     �@     (�@     ��@     ،@     ��@     @�@     ܒ@     ȕ@     0�@     @b@        
�
conv4/biases_1*�	   �of]�   �\�f?      p@!�ދ��?)��X`��?2�E��{��^��m9�H�[�<DKc��T��lDZrS�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�U�4@@�$��[^:��"���ڋ��vV�R9��T7�����d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲��FF�G �>�?�s���O�ʗ�����Zr[v��pz�w�7��})�l a��ߊ4F��h���`�a�Ϭ(���(��澢f�����uE���⾙ѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿ�[�=�k���*��ڽ�Łt�=	���R����2!K�R���J��#���j�Z�TA[�����"�RT��+��y�+pm��mm7&c��`���nx6�X� ��f׽r���f;H�\Q������%���9�e����K��󽉊-��J�'j��p���1���=��]����/�4��ݟ��uy�z�����i@4[���Qu�R"���
"
ֽ�|86	Խ(�+y�6ҽ;3���н��.4Nν�!p/�^˽K?�\��½�
6������Bb�!澽5%�����Į#�������/���8ŜU|�x�_��y�        �-���q=5%���=�Bb�!�=��
"
�=���X>�=H�����=PæҭU�=�Qu�R"�=i@4[��=z�����=ݟ��uy�=�/�4��==��]���=��1���='j��p�=��-��J�=�K���=�9�e��=����%�=f;H�\Q�=�tO���=�f׽r��=nx6�X� >�`��>�mm7&c>y�+pm>RT��+�>���">�#���j>�J>2!K�R�>Łt�=	>��f��p>�i
�k>%���>��-�z�!>��n����>�u`P+d�>�[�=�k�>��~���>;�"�q�>['�?��>E��a�W�>�ѩ�-�>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>I��P=�>��Zr[v�>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?�������:�              �?               @               @               @      @      @      �?      @              �?      �?      �?      �?      �?       @      �?              �?              �?      �?              �?               @              �?      �?              �?               @              �?      �?      �?              �?              �?              �?              �?              �?              �?              �?              �?              �?      @      @      @      @       @      @      @              @      @       @       @      @      �?      �?       @              �?       @      @              �?              �?              �?              �?              �?              �?              �?              ;@              �?              �?              �?      �?              �?              �?      �?      @       @              �?       @              @      @      @       @       @      @      �?       @      �?              �?      �?              �?      �?              �?              �?              �?              �?               @               @              �?              �?              �?       @      �?       @              �?              �?      �?      �?       @       @      �?               @      �?      �?      �?      @       @              @              �?      @      @       @      �?       @      @      �?      �?       @       @      @      �?       @      @       @       @      @      @              �?      �?              �?        
�
conv5/weights_1*�	   `��¿   �Þ�?      �@!  (��&�)��SI@2�	�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6�U�4@@�$��[^:��"��.����ڋ�>�?�s���O�ʗ���})�l a�>pz�w�7�>��d�r?�5�i}1?�T7��?�vV�R9?ji6�9�?�S�F !?+A�F�&?I�I�)�(?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?�������:�	              k@     �t@     �p@     �n@     �n@     `l@     �e@     �e@     �d@     �_@     �]@      \@      Z@     �Z@     �V@     �R@     @R@      Q@      I@      L@     �K@     �@@     �F@     �A@      ?@      4@      :@      9@      1@      ;@      ;@      (@      3@      $@      5@      (@      @      @      (@      @       @      @      @       @       @      &@      "@      @      @      @      @      �?      @      @       @              �?       @      @       @      �?      �?              �?              �?               @              �?              �?              �?      �?      �?              �?              �?              @      �?               @       @      �?       @              �?      �?      �?      �?       @      @      �?      @      @       @      @      @      @       @      @      @      @      @      &@      @      *@       @      .@      $@      $@      *@      .@      8@      :@      4@      .@      0@      >@      <@      ;@      F@      @@      C@      D@     �N@      H@     �R@     �U@      P@     �P@     �W@     �V@      [@     �Z@     �]@      d@      b@     `e@     �f@      l@     �k@     �p@     �r@     �q@     �j@        
�
conv5/biases_1*�	   `� �    m�!>      <@!   �947�)���C}<2���-�z�!�%������f��p�Łt�=	�2!K�R���J��#���j�Z�TA[�����"�RT��+��y�+pm��`���nx6�X� �����%���9�e�����-��J�'j��p�ݟ��uy�z�����i@4[���Qu�R"�PæҭUݽH����ڽ5%����G�L����Qu�R"�=i@4[��=ݟ��uy�=�/�4��=��1���='j��p�=f;H�\Q�=�tO���=�mm7&c>y�+pm>Z�TA[�>�#���j>�J>2!K�R�>��f��p>�i
�k>%���>��-�z�!>�������:�              �?               @              �?       @       @      �?               @              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?               @              �?              �?              �?        ��RHT      b-	�N�0���A*��

step   A

loss��>
�
conv1/weights_1*�	    ����    "<�?     `�@! ��l�)	���;&@2�����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��vV�R9��T7����5�i}1���d�r�x?�x��1��a˲���[���_�T�l׾��>M|Kվ>�?�s��>�FF�G ?>h�'�?x?�x�?�T7��?�vV�R9?��ڋ?ji6�9�?�S�F !?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?uܬ�@8?��%>��:?d�\D�X=?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?�������:�             �O@     �j@     �e@     @e@     �d@     `d@      `@     �`@     @X@     @V@     @T@     �X@      L@     �M@      L@      P@     �J@     �H@     �B@      E@      7@      ?@     �A@      ?@      >@      6@      0@      .@      1@      0@      0@      3@      *@      $@       @      "@      @      @      @      @       @      @      @      �?      @      @      @      �?       @      @      �?               @              �?              �?       @              �?      �?              �?              �?      �?              �?               @              �?              �?              �?              �?       @              �?              �?      �?               @      �?              �?      @              @      �?      �?      @      @       @      @      @      &@      �?      "@      @       @      $@      "@      $@      (@      $@      &@      *@      (@      2@      1@      0@      ;@      >@      8@      <@      @@      A@      @@     �D@      F@     �I@      G@      N@     �O@      N@      S@     �T@     @Y@     �[@     �_@      _@      a@      d@     `g@      i@     �f@     @P@        
�
conv1/biases_1*�	   @(&S�   ��_`?      @@!  ���q?)�I�!��>2�<DKc��T��lDZrS�nK���LQ�k�1^�sO��qU���I�
����G�a�$��{E��T���C��!�A���%>��:�uܬ�@8��u�w74���82���bȬ�0���VlQ.�I�I�)�(�+A�F�&��vV�R9��T7���jqs&\��>��~]�[�>})�l a�>pz�w�7�>>�?�s��>�FF�G ?�[^:��"?U�4@@�$?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?�u�w74?��%�V6?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?E��{��^?�l�P�`?�������:�              �?              �?               @              �?      �?               @              �?      �?      �?              �?              �?              �?              �?              �?              �?              �?              �?               @              �?               @       @               @       @              �?              �?        
�
conv2/weights_1*�	    R穿    a,�?      �@! �dL�_%@)N���XE@2����g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ���I��P=��pz�w�7��})�l a��ߊ4F��8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվjqs&\�ѾK+�E��Ͼ0�6�/n���u`P+d���[�=�k�>��~���>�XQ��>�����>;�"�q�>['�?��>jqs&\��>��~]�[�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?�������:�             |�@     ��@     |�@     �@     \�@     �@     D�@     �@     ��@     ��@     ��@      �@     H�@     (�@     @�@     8�@     Ѐ@     }@      {@     �x@      v@     �v@     ps@     �p@      n@      l@     �j@     �g@     `e@     @d@     �_@     @_@     @Z@      Z@     @Y@     �W@     �R@     �P@     �Q@     �R@     �L@     �P@      N@     �G@      B@      =@      ;@     �E@      :@      >@      2@      <@      *@      3@      0@      *@      ,@      $@       @      @      (@      @      @      @      @      @      @      @      @      �?       @       @      @       @       @               @              �?      @              @      �?       @              �?              �?       @              �?      �?      �?               @              �?              �?              �?              �?              �?              �?              �?              �?      �?              �?              �?      @      @       @      �?      @      @      @       @               @       @      @       @              �?       @      @      @      @      @      "@       @      "@       @      (@      @      0@      0@      .@      2@      .@      1@      &@      4@      ;@      6@      7@      ?@     �E@     �E@      D@      D@      Q@      J@      R@     �T@     �T@     @Q@     @\@     �X@     �^@     @a@     �_@     ``@      g@     �h@     �i@     @k@     �o@     `q@     r@     Ps@     �u@     �y@      z@     p}@      �@     8�@      �@     `�@     ��@     h�@     h�@     �@      �@     H�@     p�@     Ș@     ܚ@     0�@     ��@     `�@     ,�@        
�
conv2/biases_1*�	   � �h�   `Y�d?      P@!  ����?)�	h�?2�P}���h�Tw��Nof��l�P�`�E��{��^���bB�SY�ܗ�SsW�<DKc��T��lDZrS�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C����#@�d�\D�X=���%>��:��u�w74���82���bȬ�0���VlQ.�U�4@@�$��[^:��"������6�]����iD*L��>E��a�W�>��ڋ?�.�?�S�F !?�[^:��"?��bȬ�0?��82?�u�w74?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?5Ucv0ed?Tw��Nof?�������:�              �?              �?              �?      �?      �?              �?               @       @      �?              @      @              �?              �?              �?               @              �?              �?              @              �?       @              @      �?      �?      @       @      �?      @      �?       @       @       @              �?               @       @      �?      �?              �?        
�
conv3/weights_1*�	   `�]��    t��?      �@! `C nW�)*A��zLU@2�����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ뾢f�����uE������~]�[Ӿjqs&\�Ѿ
�/eq
Ⱦ����ž��>M|K�>�_�T�l�>�iD*L��>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?�������:�             ��@     ��@     ��@     �@     r�@     ġ@     �@     ��@     ؚ@     �@     �@     D�@      �@     ��@     ȍ@     ��@     ��@     ؅@     H�@     ��@     Ѐ@     p|@     �|@     �x@     �v@     Pu@      u@     pq@     pq@     @j@      j@     `e@     `g@      d@     �d@      _@     �a@     @[@      U@     �Z@     �T@     @R@      Q@     �K@     �K@     �H@     �E@      B@      @@      D@      =@      9@      >@      ;@      2@      (@      .@      .@      .@      &@      &@      $@      2@       @      *@      (@      @      @      @      @      @      @      @              @      �?      �?      @      @      �?      �?      �?      �?      �?      �?              �?              �?              �?              �?               @              �?      �?              �?      �?       @      �?              �?      �?      �?              �?      @      �?       @      �?       @      �?       @      @      @      @      @      @      �?      @       @      @      "@      @      $@      "@      @       @      @      ,@       @      *@      2@      (@      8@      .@      7@      =@      6@      :@      ?@      >@     �B@     �F@     �J@      K@      M@     �Q@     �Q@     �Q@     �S@     @V@     �V@     �Z@     �Z@     ``@      e@      d@     �f@     �l@     @m@     `i@      l@     �q@     Pr@     @s@     Pv@     Pw@     �y@     @@     �@     �@     h�@     8�@     P�@     ��@     ��@     �@     D�@     �@     �@     ؘ@     ,�@     ��@     `�@     *�@     j�@     .�@     $�@     ��@     P�@        
�
conv3/biases_1*�	    1�`�   ��f?      `@!  D���?)���-?2����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��.����ڋ�x?�x��>h�'��f�ʜ�7
������I��P=��pz�w�7����(��澢f����;�"�qʾ
�/eq
Ⱦ��~��¾�[�=�k��        �-���q=8K�ߝ�>�h���`�>O�ʗ��>>�?�s��>�FF�G ?��[�?��d�r?�5�i}1?�T7��?�vV�R9?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?�������:�              �?               @               @      �?      �?      @       @      @      @       @       @      @       @      �?       @      �?              @      �?               @              �?              �?              �?              �?               @              �?              �?              �?              �?              �?              @              �?              �?              �?              �?      �?      �?              �?              �?      �?              �?      @               @              @      @      �?       @      @       @      �?      @               @      @      @      @      @      @       @      @               @      �?      �?        
�
conv4/weights_1*�	   ���   �t�?      �@! P	! ��)�,�H�e@2�Ӗ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.���vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���pz�w�7��})�l a���>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ�iD*L��>E��a�W�>I��P=�>��Zr[v�>��[�?1��a˲?6�]��?����?>h�'�?x?�x�?��d�r?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�������:�             �[@     �@     ��@     ��@     \�@     ��@     ��@     `�@     `�@     (�@     p�@     ��@     P}@     �z@     0y@     @x@     �u@     `v@     `q@     �q@     @l@     @k@     �e@      f@      f@     @c@      ^@     @^@     �X@      V@     �T@     �U@      O@      O@      N@     �H@      H@     �E@      D@      I@     �D@      ?@      9@      :@      9@      4@      3@      7@      7@      (@      &@      ,@      1@      $@      @      @      @      @      @      @      @      @      @      @      @      @      �?       @      @      @       @      �?      �?      @      �?              �?      �?      �?      �?               @      �?       @              �?               @              �?              �?              �?              �?               @              �?      �?      �?              �?       @               @               @       @      �?      @       @      @       @      @      @      @      @      @      @      �?      @      �?      @      $@      @      @      $@      *@      @      $@      &@      .@      3@      1@      2@      1@      7@      9@      :@      :@      D@      F@     �F@      F@      H@      I@     �Q@     �P@      Q@     �U@     �U@     @X@     �\@     ``@     @_@     @a@     �`@     �d@      g@     �i@     �j@     @p@     pq@     s@     �u@     �v@     0x@     �x@     p|@     @      �@      �@     Ȅ@     @�@     ��@     Ќ@     `�@     L�@     ��@     ĕ@     $�@     �b@        
�
conv4/biases_1*�	   `�$b�   ��sj?      p@!V�qCD�?)r�a4f?2����%��b��l�P�`���bB�SY�ܗ�SsW��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��T7����5�i}1�x?�x��>h�'��f�ʜ�7
������1��a˲���[���FF�G �>�?�s����h���`�8K�ߝ뾙ѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾
�/eq
Ⱦ����ž�i
�k���f��p���R����2!K�R���J��#���j�Z�TA[�����"�RT��+��y�+pm��mm7&c��`���nx6�X� ��f׽r����tO����f;H�\Q������%���9�e����K��󽉊-��J�'j��p���1����/�4��ݟ��uy�z�����i@4[���Qu�R"���
"
ֽ�|86	Խ(�+y�6ҽ;3���н�d7���Ƚ��؜�ƽ�b1�ĽK?�\��½�
6������Bb�!澽        �-���q=V���Ұ�=y�訥=K?�\���=�b1��=��؜��=�d7����=��.4N�=;3����=(�+y�6�=�|86	�=��
"
�=���X>�=z�����=ݟ��uy�=�/�4��==��]���=��1���='j��p�=��-��J�=�K���=�9�e��=����%�=f;H�\Q�=�f׽r��=nx6�X� >�`��>�mm7&c>y�+pm>RT��+�>���">�J>2!K�R�>��R���>Łt�=	>�i
�k>%���>��-�z�!>4�e|�Z#>�XQ��>�����>K+�E���>jqs&\��>E��a�W�>�ѩ�-�>���%�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�������:�              �?               @               @              �?      @      �?       @       @      �?              �?      �?      �?              �?       @       @              �?              �?              �?      �?               @              �?       @      �?              �?       @       @              �?              �?               @              �?              �?              �?               @      @      @       @      @      @       @      @      �?      �?      @       @       @      �?       @      @       @              �?      �?       @      �?              �?              �?              �?              �?              �?              ;@              �?              �?               @              �?      �?      �?              �?              �?              �?      �?              @              �?      @      �?              @      @               @      @       @              �?               @               @              �?              �?              �?              �?      �?              �?              �?      �?              �?               @              �?      �?      �?      �?              �?       @      �?      �?       @      �?               @              �?      �?      �?      @       @      @      �?       @      �?      �?      �?      @      �?      @       @       @      �?              @       @      �?      @      @       @       @              @      @       @              �?      �?        
�
conv5/weights_1*�	   �Ģ¿   ` ��?      �@! U���)[���SI@2�	�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8�I�I�)�(�+A�F�&��.����ڋ��vV�R9��T7����uE���⾮��%ᾙѩ�-߾E��a�Wܾx?�x�?��d�r?�5�i}1?��ڋ?�.�?ji6�9�?+A�F�&?I�I�)�(?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?�������:�	              k@      u@     �p@     �n@     �n@     `l@     �e@     �e@     �d@     �_@     �]@      \@      Z@     �Z@     �V@     �R@      R@     �P@      J@      L@     �K@     �@@     �F@     �A@      ?@      3@      ;@      9@      1@      ;@      <@      &@      4@      "@      5@      *@      @      @      $@      @      @      @       @       @       @      "@      &@      @      @      @      @      @      @      @       @               @      �?      @       @      �?      �?      �?              �?              �?              �?              �?              �?              �?      �?              �?      �?              �?               @       @      �?              �?      @      �?              �?      �?       @              @      @      �?      @      @      @       @      @       @      @      @      @      @      @      (@      @      *@      "@      ,@      $@      $@      *@      .@      8@      :@      4@      .@      0@      >@      <@      ;@      F@      @@      C@      D@     �M@      I@     �R@      V@     �O@     �P@     �W@     �V@     �Z@     �[@     �]@     @d@     �a@     @e@     �f@      l@     �k@     �p@     �r@     �q@     �j@        
�
conv5/biases_1*�	   ��� �   `�%#>      <@!   �+�8�)�q�7b�<2���-�z�!�%�����i
�k���f��p�Łt�=	���R����2!K�R���J�Z�TA[�����"�y�+pm��mm7&c��tO����f;H�\Q������%���K��󽉊-��J�'j��p��/�4��ݟ��uy����X>ؽ��
"
ֽ�����嚽��s���Qu�R"�=i@4[��=��-��J�=�K���=�9�e��=����%�=�mm7&c>y�+pm>RT��+�>���">Z�TA[�>�#���j>�J>��R���>Łt�=	>�i
�k>%���>��-�z�!>4�e|�Z#>�������:�              �?      �?      �?              �?      @      �?               @              �?              �?      �?              �?      �?              �?              �?              �?              �?              �?              �?              �?              �?              �?      �?              �?              �?              �?        ����8T      ��bg	���0���A	*��

step  A

lossB��>
�
conv1/weights_1*�	   �5߮�   �~u�?     `�@!  U�:�)��9��,@2�����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(��[^:��"��S�F !��.����ڋ������>
�/eq
�>�f����>��(���>�T7��?�vV�R9?��ڋ?�.�?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?�������:�              O@      k@     �e@     @e@     �d@     �c@     �`@     �`@     �W@     �W@     �T@     @X@      L@     �M@     �K@     �O@     �K@     �F@      D@      E@      ;@      =@      <@      C@      =@      5@      0@      .@      0@      0@      2@      4@       @      *@      "@      @      @      @       @       @      �?      @      @      @      �?      @      @               @      @      �?       @      �?      �?      �?       @              �?      �?      �?      �?              �?              �?              �?              �?              �?              �?               @      @               @               @       @      �?       @      �?      �?       @       @      �?      @      �?      @      @      @      @      @      $@      $@      @      "@      @      (@      $@      ,@      $@      "@      0@      .@      4@      2@      7@      =@      :@      =@      =@      B@     �B@     �A@      G@     �J@      G@     �L@      P@      N@     �S@     �S@     �Y@      [@     �_@     @_@     �`@     �d@     @g@     @h@     �g@     �P@        
�
conv1/biases_1*�	   `�V�    kc?      @@!  b �w?)���8 �>2�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0�I�I�)�(�+A�F�&��u`P+d����n������_�T�l�>�iD*L��>�FF�G ?��[�?x?�x�?��d�r?U�4@@�$?+A�F�&?�7Kaa+?��VlQ.?��82?�u�w74?��%�V6?uܬ�@8?�!�A?�T���C?a�$��{E?
����G?�qU���I?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?���%��b?5Ucv0ed?�������:�              �?              �?               @              �?              �?               @              �?               @              �?              �?              �?              �?              �?              �?              �?              �?               @              �?      �?      �?       @              �?      �?       @      �?              �?        
�
conv2/weights_1*�	   �����    Q�?      �@! ��r�^&@)ê��YE@2����g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ���I��P=��pz�w�7��})�l a��ߊ4F��a�Ϭ(���(��澢f���侮��%ᾙѩ�-߾E��a�Wܾ��>M|Kվ��~]�[Ӿjqs&\�Ѿ��n�����豪}0ڰ�.��fc���X$�z�����?�ګ>����>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>���%�>�uE����>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?�������:�             x�@     �@     ��@     �@     t�@     З@     8�@     �@     �@     ��@     p�@     `�@     0�@     8�@     0�@     8�@     Ѐ@     P}@      {@     px@     pv@     �u@     Ps@     �p@      o@      l@     �j@     `g@     �e@     �d@     �^@      ]@     �[@     �Y@     @[@     �V@      R@     �P@     �R@     �Q@      M@     �Q@     �I@      L@     �C@      9@      <@     �A@      8@     �@@      3@      9@      &@      4@      1@      *@      *@       @      $@       @      $@      &@       @      @      @      �?      @      @      @      @      @      @              @              @      �?               @       @              @       @      �?              �?      �?               @      �?              �?      �?              �?              �?              �?              �?              �?              �?              �?              @      �?               @       @      �?       @              @              �?      @       @      @      @       @      @      @      @      @      @      @      @       @      @      &@      "@       @      ,@      (@      4@      1@      4@      2@      *@      0@      ;@      8@      6@     �A@      D@      C@     �F@      E@     �N@      N@     �S@     @S@     �R@     @S@     �\@     �W@     @]@     �a@      `@     �`@     �g@      h@     @i@     `k@     �o@     @q@     @r@     �r@     �u@     0z@     �y@     p}@     ��@     X�@      �@     h�@     �@     h�@     0�@     4�@     �@     0�@     ��@     ��@     ��@     4�@     ��@     b�@     <�@        
�
conv2/biases_1*�	   �d�k�   �9�g?      P@!   9��?)|��%�?2��N�W�m�ߤ�(g%k����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8��u�w74���82���bȬ�0���VlQ.�+A�F�&�U�4@@�$���d�r�x?�x����(��澢f���侢FF�G ?��[�?��d�r?�5�i}1?�T7��?U�4@@�$?+A�F�&?I�I�)�(?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?�������:�              �?              �?              �?      �?              �?      �?      �?               @              �?      �?       @      �?      �?       @      �?              �?              �?              �?              �?              �?              �?              �?      �?              �?      �?              �?      �?               @       @       @       @      @       @      @       @       @      �?      @      �?              @      �?      �?      �?      �?              �?        
�
conv3/weights_1*�	   �Ce��    ���?      �@! p�"�=��)U����LU@2�����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(龢f�����uE����E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ['�?�;;�"�qʾ�u`P+d�>0�6�/n�>��>M|K�>�_�T�l�>E��a�W�>�ѩ�-�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?�������:�              �@     l�@     ȧ@     �@     ��@     ��@     �@     ��@     ؚ@     ��@     �@     H�@      �@     x�@     ��@     ��@     ��@     ȅ@     ��@     Ђ@     �@     �{@      }@      y@      v@     �u@     �t@     �q@      q@     @k@     �i@     �e@     �f@     `d@     �d@     �_@     �a@     �X@     �W@     �W@     �U@     @R@     �P@      L@      K@     �G@     �D@      F@      B@      B@     �@@      :@      :@      8@      0@      $@      6@      *@      *@      0@       @      $@      2@      "@      (@       @      @      @      @      @      @       @      @      @      �?      @              @               @      �?      @      �?      �?              �?      �?      �?              �?              �?              �?              �?              �?              �?              �?              �?              �?      �?              �?              �?              �?      �?       @       @              @              @      @       @      $@      @       @      @      @      @       @      @      @      @      @       @      (@      $@      $@      .@      (@      ,@      @      6@      4@      5@      =@      9@      :@      9@     �A@      C@      J@     �F@     �I@     �M@     �P@     �Q@     �P@     �T@      V@     �W@      [@     �Z@      `@      e@     �d@     �f@     �l@      m@     �i@      l@      q@      s@     Pr@     w@     �v@     z@     �~@     h�@     ��@     ��@     ��@     h�@     `�@     �@     �@     D�@     (�@     �@     �@     ,�@     ��@     \�@     ,�@     v�@     &�@     �@     ��@     ��@        
�
conv3/biases_1*�
	   �r�d�   @�cj?      `@!  dӖ?)�B���$?2�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��S�F !�ji6�9����ڋ��vV�R9������6�]���8K�ߝ�a�Ϭ(�
�/eq
Ⱦ����ž        �-���q=0�6�/n�>5�"�g��>>�?�s��>�FF�G ?f�ʜ�7
?>h�'�?x?�x�?��d�r?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�������:�              �?              �?      �?      �?       @      �?       @      @      @       @      @      @      �?      @      �?              �?       @      @      �?      �?              �?      �?              �?       @      �?      �?              �?              �?              �?              �?              �?              @              �?              �?              �?              �?              �?      �?      �?      �?              �?       @      �?       @      �?       @       @      @       @       @      @       @              @       @       @      @      @      @      @       @      @              @              �?        
�
conv4/weights_1*�	   �o��   ���?      �@! �G��)�K��u�e@2�Ӗ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7�����d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`��_�T�l׾��>M|Kվjqs&\�ѾK+�E��Ͼ['�?�;��Zr[v�>O�ʗ��>�FF�G ?��[�?1��a˲?6�]��?����?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�������:�             �[@     �@     ��@     ��@     X�@     �@     ��@     `�@     `�@     0�@     p�@     p�@     P}@     �z@     y@     `x@     �u@     pv@     `q@     �q@     `l@     @k@     �e@     @f@      f@      c@     @^@      ^@     �X@     @V@     @T@     @V@     �N@      P@     �K@      K@      G@     �E@      D@     �I@     �E@      =@      9@      ;@      7@      4@      3@      8@      7@      &@      &@      1@      ,@       @      "@      @      @       @      @      @      @      @      @       @      @       @       @      �?      @      @      @              �?      �?       @      �?              �?              �?               @       @              �?      �?      �?      �?              �?              �?              �?              �?              �?      �?              �?              �?      �?      �?       @              �?              �?               @      �?      �?      �?      @      @       @       @      @       @       @       @      @      @      @       @      @      @      @       @      @      @      @      *@      "@       @      (@      0@      3@      2@      2@      1@      1@      A@      7@      9@     �B@      G@     �G@     �E@     �G@      H@      R@      Q@      Q@     �T@     @V@     �W@     �\@     �`@      _@     �a@     �`@      e@     @g@     �i@     �j@     @p@     pq@      s@     �u@     �v@     Px@     px@     `|@     @     ��@      �@     ��@     0�@     ��@     Ќ@     X�@     L�@     �@     ��@     ,�@     �b@        
�
conv4/biases_1*�	   �}Ue�    ��m?      p@!��L%��?))M�[�?$?2�Tw��Nof�5Ucv0ed�E��{��^��m9�H�[�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A�d�\D�X=���%>��:�uܬ�@8���%�V6���bȬ�0���VlQ.��7Kaa+��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1�>h�'��f�ʜ�7
�6�]���1��a˲���[���uE���⾮��%ᾄiD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿ0�6�/n���u`P+d����-�z�!�%�����i
�k���R����2!K�R���J��#���j�Z�TA[�����"�RT��+��y�+pm��mm7&c��`���nx6�X� ��f׽r����tO����f;H�\Q������%���9�e����K��󽉊-��J���1���=��]���ݟ��uy�z�����PæҭUݽH����ڽ�!p/�^˽�d7���Ƚ�b1�ĽK?�\��½�8�4L���<QGEԬ�̴�L�����/k��ڂ�        �-���q=��؜��=�d7����=�!p/�^�=��.4N�=;3����=(�+y�6�=��
"
�=���X>�=PæҭU�=�Qu�R"�=i@4[��=z�����=ݟ��uy�==��]���=��1���='j��p�=��-��J�=�K���=�9�e��=����%�=f;H�\Q�=�tO���=�f׽r��=nx6�X� >�`��>�mm7&c>y�+pm>RT��+�>���">Z�TA[�>�#���j>�J>2!K�R�>��R���>Łt�=	>��f��p>�i
�k>%���>��-�z�!>��o�kJ%>4��evk'>
�/eq
�>;�"�q�>jqs&\��>��~]�[�>��>M|K�>�ѩ�-�>���%�>�uE����>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��bȬ�0?��82?�u�w74?��%�V6?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?�������:�              �?               @               @              �?      @      @      �?      @              �?      �?              �?      �?       @               @      @              �?      �?               @       @      �?              �?               @       @              �?              �?      �?      �?              �?              �?      �?               @       @      @      @       @      @      @      @      �?       @       @      @       @      �?      @      �?      @              @              �?               @              �?              �?               @              �?              :@              �?              �?              �?              �?              �?      �?      �?      �?               @      �?      �?      @              �?      �?       @              @      �?       @       @      @       @      @      �?      �?              �?              �?      �?               @              �?              �?              �?      �?              �?      �?              �?      �?      �?      �?              �?              �?              �?      �?      �?              �?      �?       @               @       @      �?      �?      �?      �?      �?      �?              @              @       @      @              �?       @      @      �?      @      @               @      �?              @      @              @      @      @              @      �?      @      �?       @      �?        
�
conv5/weights_1*�	   ��¿    ���?      �@! �F�G��)[��_TI@2�	�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���VlQ.��7Kaa+���ڋ��vV�R9�x?�x��>h�'�������6�]���I��P=�>��Zr[v�>��[�?1��a˲?��d�r?�5�i}1?�vV�R9?��ڋ?�S�F !?�[^:��"?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?�������:�	              k@     �t@     �p@     �n@      o@     �l@     `e@     �e@     �d@     �_@     �]@     �[@     @Z@      [@     �V@     �R@     @R@     �P@     �J@     �K@     �K@      A@     �F@      @@     �@@      3@      ;@      :@      0@      :@      =@      (@      2@      $@      6@      (@      @      @      &@      @      @      @       @      @       @      @      $@      @      @      @      @      �?      @      @      �?      �?       @              @       @               @      �?              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?              @      �?      �?              �?       @       @      �?      �?      �?       @      �?      @      �?              @      @      @      @      @       @      @      @      @      @      @      (@      @      &@      $@      *@      &@      $@      *@      .@      8@      :@      3@      1@      .@      =@      =@      =@     �D@     �@@      C@     �C@     �M@      I@     �R@     �U@      P@     �P@     �W@     �V@     �Z@     �[@      ^@      d@     �a@     `e@     �f@      l@     �k@     �p@     �r@     �q@      k@        
�
conv5/biases_1*�	   @�#�   ��%>      <@!   z�d?�)MT3�<2���o�kJ%�4�e|�Z#���-�z�!�%�����i
�k���f��p�Łt�=	���R����2!K�R���J��#���j�Z�TA[�����"��`���nx6�X� ��tO����f;H�\Q������%���9�e���'j��p���1���=��]����/�4��ݟ��uy�H�����=PæҭU�=�K���=�9�e��=f;H�\Q�=�tO���=�f׽r��=�mm7&c>y�+pm>Z�TA[�>�#���j>�J>�i
�k>%���>��-�z�!>4�e|�Z#>��o�kJ%>�������:�              �?              �?      �?      @      �?              �?              �?      �?      �?              �?              �?              �?              �?      �?              �?              �?              �?              �?      �?              �?               @      �?              �?      �?              �?        ���T      >�	��0���A
*��

step   A

loss*8�>
�
conv1/weights_1*�	    $��   �&��?     `�@!  ��=pٿ)�,�55@2�����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�uܬ�@8���%�V6��u�w74���82���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��.����ڋ��vV�R9��_�T�l׾��>M|Kվ�ߊ4F��>})�l a�>����?f�ʜ�7
?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?�������:�              O@     �j@     �e@     �e@     �d@     �c@     �`@      `@     �X@      W@     �U@     @V@     �N@     �N@      J@      N@     �L@     �G@     �E@      B@     �@@      ;@      :@      @@      A@      2@      :@      $@      (@      5@      ,@      0@      ,@      (@      "@      @      @      @      @      @      @       @      �?      @      @      @       @              �?       @      �?              @      �?      �?              �?       @              �?              �?       @              �?              �?              �?               @              �?              �?      �?      �?      �?      �?              �?      @      �?      �?       @      @      @       @       @      @       @       @      @       @      @      @      @      @       @       @      $@      .@      &@      &@      $@      &@      .@      7@      4@      7@      <@      7@      ?@      =@     �B@     �B@      B@     �F@      H@     �E@     @P@      O@      O@     �S@     �S@     �W@     �[@     @`@     �^@     �a@     �c@     `g@     �h@     �g@     �Q@        
�
conv1/biases_1*�	   ���Y�   ���f?      @@!  T�{f�?)X��'���>2��m9�H�[���bB�SY��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G����#@�d�\D�X=���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(��S�F !�ji6�9��E��a�W�>�ѩ�-�>��[�?1��a˲?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?��82?�u�w74?��%>��:?d�\D�X=?���#@?�!�A?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?Tw��Nof?P}���h?�������:�              �?              �?              �?      �?      �?               @              �?              �?      �?              �?              �?              �?              �?              �?              �?      �?               @              �?      �?      �?              @      �?               @              �?      �?      �?              �?        
�
conv2/weights_1*�	    +��   @�x�?      �@! �1B��'@)�����ZE@2����g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾�iD*L�پ�_�T�l׾;9��R�>���?�ګ>jqs&\��>��~]�[�>�ѩ�-�>���%�>�uE����>�f����>��(���>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?�������:�             P�@     ��@     ��@     �@     T�@     ��@     H�@     ؓ@     p�@     X�@     X�@     ��@     @�@     (�@     H�@      �@     Ȁ@     P}@     �z@     Px@     �v@     �u@     �s@     0p@     �n@     @m@     �j@     �f@      e@     �e@      _@      ^@      [@     �X@     @[@     @W@     �Q@      S@      Q@     �P@     @P@      N@      M@     �I@      G@      3@      ;@     �@@      B@      7@      6@      5@      .@      2@      3@      &@      "@      &@      &@      @       @      (@      @      (@      @      @      @       @       @      @      @      �?      �?      �?      @       @      @              �?      @      �?              �?      �?      @              �?      �?      �?      �?      �?              �?              �?              �?               @              �?      �?              �?               @      �?       @              �?              �?       @       @       @      @      �?       @      �?      @      @               @      @      @      @      @      "@      @       @      @       @       @      *@      @      1@      "@      *@      3@      .@      3@      7@      0@      ;@      ;@      4@      C@      =@      G@     �G@     �B@      O@     �N@     @T@     �R@     @U@     �R@      [@     �X@      \@     �`@     `a@     @_@     �g@      h@     �i@     �j@     `o@     pq@     r@     0s@     �u@     z@     0z@     �}@     ��@     X�@      �@     ��@     ��@     ��@     @�@     �@     ,�@     ,�@     ��@     ��@     К@     (�@     ̟@     V�@     `�@        
�
conv2/biases_1*�	   @_Ko�    �8j?      P@!  �R~'�?)�����?2�;8�clp��N�W�m�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6�I�I�)�(�+A�F�&�U�4@@�$�ji6�9���.��8K�ߝ�a�Ϭ(�K+�E���>jqs&\��>��[�?1��a˲?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�������:�              �?              �?              �?      �?               @       @              �?               @      �?              @              �?      �?       @              �?      �?              �?              �?              �?              �?              �?              �?              �?      �?              �?               @      �?              �?       @      @       @      @      �?      @      �?       @      @      �?              �?       @       @      �?      �?              �?        
�
conv3/weights_1*�	    �m��   �弮?      �@! �{�1��)L�y`MU@2�����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[��>�?�s���O�ʗ�����Zr[v��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%�E��a�Wܾ�iD*L�پG&�$��5�"�g���R%�����>�u��gr�>����>豪}0ڰ>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?�������:�              �@     d�@     ʧ@     ��@     ��@     ��@     �@     ��@     К@      �@     �@     <�@     H�@     X�@     ��@     ��@     ȇ@     ��@     Ȅ@     X�@     ��@     �{@     �|@     �x@     v@     0v@     �t@     �q@     �p@      l@     @i@     �e@     `f@     �d@     �d@      `@     �a@     �Y@      U@     �X@     @V@     @R@     �P@     �O@     �E@     �G@     �A@      L@      =@     �D@      B@      :@      :@      3@      5@      (@      0@      ,@      *@      0@      ,@       @      .@      @      &@      *@      @       @      @      "@      @       @       @      @      @      �?      �?      �?      �?               @       @              �?      �?      �?              �?               @      �?      �?               @              �?              �?              �?              �?       @      �?      �?      �?      �?              �?              @      �?      �?      @      @       @              @       @      @              @       @      @       @      @      @      @      @      @      @       @      &@      (@      @      .@      .@      "@      *@      .@      ,@      5@      7@     �@@      5@      <@      9@      A@      C@     �E@     �H@     �H@     @P@     �N@     @Q@     �Q@      S@     �Y@     �V@     @Z@      [@     ``@      e@      d@     �f@     �m@     `k@     �j@      l@     �p@     �s@     `r@     @w@     �v@     0z@      ~@     h�@     �@     ��@      �@     ��@      �@     �@     (�@     \�@     �@     ԕ@     ܘ@     D�@     ��@     D�@     4�@     v�@     �@     .�@     ~�@     ��@        
�

conv3/biases_1*�
	   @��h�    qo?      `@!  jo͛?)��x7�,?2�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"�O�ʗ�����Zr[v���ߊ4F��h���`�;�"�qʾ
�/eq
Ⱦ        �-���q=�f����>��(���>6�]��?����?�T7��?�vV�R9?��ڋ?�.�?�[^:��"?U�4@@�$?+A�F�&?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?P}���h?ߤ�(g%k?�N�W�m?;8�clp?�������:�              �?      �?               @      �?              @       @      @      �?      @      @      �?      @      @      �?       @       @      �?       @      �?      �?      �?              �?              @      �?               @              �?              �?              �?              @              �?              �?              �?              �?              �?       @              �?              �?      �?      �?      @       @               @       @      @      @       @       @      �?      @      �?      @      @      @      @      @      �?      @              @              �?        
�
conv4/weights_1*�	    �
��   @�?      �@! �D����)R����e@2�Ӗ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[��>�?�s���O�ʗ���})�l a��ߊ4F��h���`�8K�ߝ뾄iD*L�پ�_�T�l׾jqs&\�ѾK+�E��Ͼ39W$:���.��fc�����Zr[v�>O�ʗ��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�������:�             @[@     �@     �@     ��@     P�@      �@     ��@     p�@     X�@     0�@     h�@     x�@      }@      {@     �x@     �x@     �u@     `v@     pq@     �q@     @l@     @k@      f@     �e@      f@     @c@     �]@     @^@     @Y@      V@     @T@      V@      O@     �O@     �K@      K@     �F@      G@     �C@      J@     �D@      >@      7@      <@      6@      4@      1@      9@      :@      $@      @      9@      $@      &@      @       @       @      @      @      @      @      @       @       @      @       @       @      @      @      @      @      �?      �?      @              �?              �?               @      �?      �?              �?      �?               @              �?              �?              �?              �?              �?              �?               @               @              �?               @              �?       @       @               @       @       @      @      �?      @       @      @      �?      @      @       @      @      �?      @       @       @      @      @      "@      &@      @      *@       @      0@      2@      1@      5@      0@      3@      ?@      ;@      6@      B@      G@      H@      F@      F@     �I@     �Q@     �Q@     �P@      T@      W@      X@     @\@     �`@     @_@     @a@      a@      e@      g@     �i@     �j@     Pp@     @q@      s@     �u@     �v@     x@     px@     0|@     @@      �@      �@     Ȅ@     8�@     ��@     Ȍ@     h�@     <�@     �@     ��@     8�@     �b@        
�
conv4/biases_1*�	   �H�h�    f�p?      p@!�����Ы?)|n|�9�*?2�P}���h�Tw��Nof��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6���82���bȬ�0���VlQ.��7Kaa+�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �pz�w�7��})�l a��ߊ4F��8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾄiD*L�پ�_�T�l׾��>M|Kվ4�e|�Z#���-�z�!�%������f��p�Łt�=	���R����2!K�R���J��#���j�Z�TA[�����"�RT��+��y�+pm��mm7&c��`���nx6�X� ��f׽r����tO����f;H�\Q������%���9�e����K��󽉊-��J�'j��p��/�4��ݟ��uy�z�����i@4[��(�+y�6ҽ;3���н��.4Nν�!p/�^˽�d7���Ƚ��؜�ƽK?�\��½�
6�����        �-���q=�!p/�^�=��.4N�=�|86	�=��
"
�=�Qu�R"�=i@4[��=z�����=ݟ��uy�=�/�4��==��]���=��1���='j��p�=��-��J�=�K���=�9�e��=����%�=f;H�\Q�=�tO���=�f׽r��=nx6�X� >�`��>�mm7&c>y�+pm>RT��+�>���">Z�TA[�>�#���j>�J>2!K�R�>��R���>Łt�=	>��f��p>�i
�k>%���>��-�z�!>4�e|�Z#>4��evk'>���<�)>豪}0ڰ>��n����>;�"�q�>['�?��>�_�T�l�>�iD*L��>�ѩ�-�>���%�>�uE����>�f����>})�l a�>pz�w�7�>I��P=�>��Zr[v�>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?�������:�              �?               @              �?      �?               @      @      @      �?      �?      �?       @              �?      �?       @              �?              �?      �?      �?              �?      �?      �?      �?              �?              �?              �?      �?      �?       @      �?              �?              �?      �?              �?              �?              �?               @      �?              �?      �?              @      �?              @      @      @      @      @       @       @      @      �?       @              @       @       @       @               @              �?      �?      �?              �?      �?              �?      �?              �?              9@              �?              �?              �?      �?      �?               @              �?              @      �?      �?      @              �?      �?       @      �?       @      @       @              @       @      �?              �?              �?      �?      �?      �?              �?              �?              �?              �?              �?      �?      �?              @               @              �?              �?      �?              �?              �?      �?       @               @      �?       @               @              �?      @              @       @      @       @              �?      �?              @      @       @       @      �?       @              �?      @      �?       @      @      @       @      �?       @      @       @      �?      �?      �?        
�
conv5/weights_1*�	   ���¿   `k��?      �@! @�^���)�Ǔ�:UI@2�	�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���82���bȬ�0���ڋ��vV�R9��T7�����[���FF�G �8K�ߝ�>�h���`�>f�ʜ�7
?>h�'�?��d�r?�5�i}1?�T7��?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?�������:�	              k@      u@     �p@     �n@     �n@     �l@     `e@     �e@     �d@     �_@     �]@     �[@     @Z@     @[@     @V@     �R@     @R@     �P@      K@      K@      L@      @@      G@     �@@     �@@      2@      9@      <@      2@      8@      <@      ,@      2@      $@      4@      &@      @      @      &@      @      @      @       @       @      @      @      "@      $@       @       @      @      �?       @      @      �?       @      �?      �?      @       @      �?      �?      �?              �?              �?      �?              �?              �?              �?              �?      �?              �?              �?              �?       @      �?              �?       @       @              �?       @      @      �?       @       @               @      @      @      @      @      @       @      @      �?      @      @      @      (@      @      &@      "@      *@      &@       @      .@      .@      7@      8@      6@      1@      0@      ;@      >@      <@     �E@     �@@     �B@      C@      N@     �I@     �R@     �U@     @P@     �P@     @W@     @W@     �Z@     �[@     �]@      d@      b@     @e@     �f@     @l@      l@     �p@     �r@     �q@      k@        
�
conv5/biases_1*�	   `�Y'�   `7�&>      <@!  ���B�)sQ���O�<2�4��evk'���o�kJ%�4�e|�Z#���-�z�!�%�����i
�k���f��p�Łt�=	��J��#���j�Z�TA[��RT��+��y�+pm�f;H�\Q������%���K��󽉊-��J�'j��p���1���i@4[���Qu�R"���1���='j��p�=��-��J�=�f׽r��=nx6�X� >�`��>�mm7&c>RT��+�>���">�#���j>�J>2!K�R�>��-�z�!>4�e|�Z#>��o�kJ%>4��evk'>�������:�              �?              �?      @              �?      �?               @      �?              �?              �?               @              �?              �?              �?      �?              �?              �?              �?               @      �?               @              �?        [��\�R      ��Q�	v1���A*�

step  0A

loss�Y�>
�
conv1/weights_1*�	   @�o��    ���?     `�@!  ��Ҝ?)U
+0?@2�����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A�d�\D�X=���%>��:�uܬ�@8���%�V6���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !��vV�R9��T7�����d�r�x?�x�������6�]����ߊ4F��>})�l a�>>h�'�?x?�x�?�S�F !?�[^:��"?U�4@@�$?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?�������:�              P@     `j@     �d@      f@     �d@     `c@     �`@     �`@     �W@     @W@     �U@     �V@     �O@      N@      I@      N@      L@     �G@     �E@     �A@      A@      =@      8@     �@@      @@      8@      4@      (@      3@      ,@      ,@      .@      0@      $@       @      @      "@      @      @      @      @      @       @      @      �?       @      @       @       @      �?              �?      �?      @               @       @               @      �?      �?              �?              �?              �?              �?              �?               @      �?              �?      �?       @      �?              �?       @      �?       @      @      @      @       @      �?       @      @      @      @      @      @      @       @      "@      *@      &@      (@      "@      *@      &@      *@      8@      5@      8@      8@      6@     �@@      A@      A@     �B@     �B@      D@     �H@     �H@     �M@      O@     �O@     �S@     �R@     �X@     @[@     ``@     �^@     @a@      d@      h@     �g@     �g@      S@        
�
conv1/biases_1*�	   ���[�   �j#j?      @@!   {��?)�8�7�&?2�E��{��^��m9�H�[��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��!�A����#@�uܬ�@8���%�V6��u�w74���82���VlQ.��7Kaa+�ji6�9���.��['�?��>K+�E���>�uE����>�f����>})�l a�>pz�w�7�>6�]��?����?I�I�)�(?�7Kaa+?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?�T���C?a�$��{E?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?��bB�SY?�m9�H�[?E��{��^?P}���h?ߤ�(g%k?�������:�              �?              �?              �?      �?              �?              �?              �?      �?      �?              �?              �?              �?              �?              �?              �?              �?              �?      �?               @               @              @      �?       @              �?       @              �?        
�
conv2/weights_1*�	   �:A��   �!��?      �@! k!�\�(@)��iX�[E@2����g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ���I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(龮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�����>
�/eq
�>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>���%�>�uE����>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?�������:�             �@     �@     ��@     �@     L�@     ��@     P�@     ��@     \�@     x�@     @�@     P�@     ��@     �@     X�@     ��@     ��@     �}@     �z@     0x@     �v@      v@     �s@      o@     �o@     �n@     �j@     `f@     �d@     @e@     �^@     �^@     @\@      X@     �Z@      W@     �R@     �R@      S@      S@      H@      O@      J@     �I@     �C@      >@      ?@      ?@      :@      ;@      4@      :@      4@      1@      *@      *@       @      @      .@      "@      $@      @      @      @      @      @      @      @      @      @      @       @      �?      �?      �?      @      @       @      �?       @              �?       @      �?      �?      �?      �?               @              �?              �?               @              �?      �?              �?               @      �?       @      �?       @      @      @       @      �?      �?       @      �?      �?      @      @       @      @      �?       @      @      @       @      @      @      @      @      @      "@      *@      *@      &@      0@      *@      .@      7@      3@      3@      6@      ;@      ;@     �B@      B@      G@     �D@     �D@      K@     �Q@     @T@     �S@     �U@     @R@     �X@     �X@     �^@      `@     �`@      a@     `f@     �g@      l@     @i@      p@     q@     r@      s@     0v@     �y@     �z@     `}@     ��@     h�@     ��@     �@     P�@     ��@     P�@     �@     $�@     D�@     |�@     ��@     К@     4�@     ��@     f�@     t�@        
�
conv2/biases_1*�	   ���p�    ��l?      P@!  �����?)�_č2� ?2�uWy��r�;8�clp�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���VlQ.��7Kaa+�U�4@@�$��[^:��"�ji6�9���.��f�ʜ�7
������1��a˲?6�]��?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?�������:�              �?              �?              �?              �?               @      �?              �?       @              �?       @               @              �?      �?      �?               @              �?              �?              �?               @              �?              �?              �?              �?               @      �?      �?               @      @      @      @       @       @       @              @       @      �?      �?      �?       @       @      �?              �?        
�
conv3/weights_1*�	   ��y��   �<ٮ?      �@!  �CV̿)�6.�MU@2�����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾K+�E��Ͼ['�?�;X$�z��
�}�������%�>�uE����>�f����>�h���`�>�ߊ4F��>})�l a�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?�������:�              �@     l�@     ҧ@     �@     ��@     ��@     �@     ��@     �@     ė@     �@     0�@     <�@     P�@     ��@     ��@     ȇ@     ��@     ��@     p�@     ؁@     0|@     0|@      y@     @v@     �u@     �t@     �q@     �p@      m@     `h@     �f@     �e@     �d@     @d@      a@     �a@      X@     �W@      W@     @V@     �R@     �N@     �P@     �E@      H@      D@     �E@     �D@     �D@      >@      <@      4@      ;@      2@      0@      0@      1@      0@      ,@      $@      "@      0@      @      @      "@      "@      @      @      @       @      @      @       @      �?      @      @       @       @      �?       @       @              �?      �?      �?      �?              @      �?      �?              �?      �?      �?      �?              �?              �?              �?              �?       @              �?      �?              @      �?              �?       @       @      @      (@      @      �?      @      @      @      @      @       @      �?      @      @      "@      (@      &@      (@      $@      &@      1@      9@      .@      8@      9@      ;@      <@     �A@      <@      A@      G@     �B@      K@     �O@      O@      Q@     @Q@      V@     �V@     �V@     �\@     �Z@     �`@     @c@     �e@     �f@     �l@     �l@     @j@     �k@     �p@     ps@     �r@     �v@     �w@     0y@     �~@     `�@     ��@     ��@     X�@     `�@     X�@     �@     �@     0�@     <�@     ԕ@     ��@     h�@     ��@     0�@     B�@     f�@     �@     4�@     |�@     ��@        
�
conv3/biases_1*�
	   �*�k�    ݻq?      `@!  �0�?)��A���2?2��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%�V6��u�w74���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&��[^:��"��S�F !��T7����5�i}1�1��a˲���[��})�l a��ߊ4F��K+�E��Ͼ['�?�;        �-���q=�h���`�>�ߊ4F��>�FF�G ?��[�?�5�i}1?�T7��?��ڋ?�.�?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��bȬ�0?��82?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?�������:�              �?      �?              �?      �?      �?              @      @      @      �?      @      @       @      @      �?      @      �?       @      @      �?      �?              �?              �?      �?      �?       @              �?              �?              �?              �?              �?              @              �?              �?              �?              �?              �?      �?      �?              �?              @      �?               @       @      �?              @      @      @       @      �?      @      @      @      @      @      @      @      @       @      �?              @              �?        
�
conv4/weights_1*�	   ����   �J�?      �@!  eVu��)Y�:�e@2�Ӗ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[��O�ʗ�����Zr[v���h���`�8K�ߝ뾙ѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾jqs&\�ѾK+�E��ϾE'�/��x>f^��`{>G&�$�>�*��ڽ>O�ʗ��>>�?�s��>1��a˲?6�]��?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�������:�              [@     �@     �@     ��@     T�@     ��@     ��@     ��@     P�@      �@     ��@     X�@      }@     0{@     �x@     �x@     �u@     pv@     `q@     �q@     �l@      k@     �e@      f@     �e@     �c@     @\@     @_@     �Y@      U@     �S@     �V@     �N@      O@     �M@     �K@     �E@     �E@     �D@     �J@      D@      >@      8@      =@      6@      0@      2@      =@      6@      $@       @      5@      &@      (@      @      "@      @      "@      @      @      @      @       @      @       @      @       @       @      @       @      @       @      �?      �?      �?              �?              �?      �?      �?      �?      �?              �?               @               @              �?              �?              �?              �?              �?               @               @              �?       @              �?              �?       @      �?              @      �?      @      @      @      �?      @      @      �?      @      @      @      @      @       @       @      @       @      @      "@      "@      *@      &@      (@      4@      0@      3@      2@      5@      >@      :@      6@     �@@     �H@      G@     �G@      F@     �I@     �Q@     �Q@     �Q@     �R@     @W@     @X@     �\@     @`@     @_@     �a@     �`@     `e@      g@     �i@     �j@     0p@     Pq@     0s@     �u@     �v@     �w@     Px@     �|@      @     ��@     ��@     ��@     P�@     ��@     Ќ@     x�@     <�@     ��@     ��@     0�@     �b@        
�
conv4/biases_1*�	    �k�   `�r?      p@!D�=��ί?)ng�2@1?2��N�W�m�ߤ�(g%k����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�uܬ�@8���%�V6��u�w74���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !���d�r�x?�x��>h�'��f�ʜ�7
������6�]���pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(龙ѩ�-߾E��a�Wܾ�iD*L�پ4��evk'���o�kJ%�4�e|�Z#���-�z�!�%�����i
�k���f��p�Łt�=	���R����2!K�R���J��#���j�Z�TA[�����"�RT��+��y�+pm��mm7&c��`���nx6�X� ��f׽r����tO����f;H�\Q������%���9�e����K��󽉊-��J�'j��p���1���=��]����/�4��ݟ��uy�z�����PæҭUݽH����ڽ(�+y�6ҽ;3���н�!p/�^˽�d7���Ƚ��؜�ƽ�b1�Ľ        �-���q=�Į#��=���6�=�b1��=��؜��=�Qu�R"�=i@4[��=�/�4��==��]���=��1���='j��p�=��-��J�=�K���=�9�e��=����%�=f;H�\Q�=�tO���=�f׽r��=nx6�X� >�`��>�mm7&c>y�+pm>RT��+�>���">Z�TA[�>�#���j>�J>2!K�R�>��R���>Łt�=	>��f��p>��-�z�!>4�e|�Z#>��o�kJ%>4��evk'>���<�)>�����>
�/eq
�>['�?��>K+�E���>�_�T�l�>�iD*L��>E��a�W�>���%�>�uE����>�f����>��(���>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>�FF�G ?��[�?����?f�ʜ�7
?>h�'�?x?�x�?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?�������:�              �?               @              �?      �?              @       @       @       @      �?      �?       @              �?       @      �?              �?      �?               @       @               @      �?               @              �?      �?      �?      �?       @               @              �?              �?              �?      �?              �?              �?              �?      @               @      �?      @      @      @      @      @      �?      �?      @               @       @      �?       @      �?      �?              @      �?      �?      �?              �?              �?              �?              �?              �?              9@              �?              �?               @              �?      �?              �?       @              �?      �?       @       @      �?      �?      @       @              @      �?       @       @              @               @               @      �?              �?              �?              �?              �?      �?               @      �?      @              �?      �?               @              �?              �?              �?              �?      �?              �?      �?       @              �?      �?              @               @      @      @      �?      @       @      �?      �?              �?      @       @      @              @      �?      �?      @      �?       @      @      @      �?       @      �?       @      @      �?       @      �?        
�
conv5/weights_1*�	   ��¿   @���?      �@! P�nE�)�wL';VI@2�	�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74��[^:��"��S�F !���ڋ��vV�R9�a�Ϭ(���(��澤����~>[#=�؏�>>h�'�?x?�x�?��d�r?�5�i}1?�vV�R9?��ڋ?�.�?ji6�9�?+A�F�&?I�I�)�(?��VlQ.?��bȬ�0?��82?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?�������:�	             �j@      u@     �p@      o@     �n@     �l@     �e@     �e@     �d@      `@     @]@     @\@     �Y@     �[@     @V@     �R@     �R@     �P@      J@      K@     �K@     �@@     �F@     �A@      @@      2@      9@      <@      3@      7@      =@      *@      2@      $@      3@      (@      @      @      $@      @      @      @      "@       @      @      @       @      @      @      @      @      @      �?      @      �?      �?      �?      �?      @       @      �?              �?               @              �?              �?              �?              �?              �?              �?              �?              �?              �?              @      �?              @       @              �?      @      �?       @       @      �?      �?              @      @      @      �?      @      @      @       @      @      @      @      *@      @      (@       @      *@      (@      @      0@      .@      8@      6@      6@      2@      0@      9@      @@      <@      F@      ?@     �C@      C@      M@     �I@     �R@     �U@      P@     �P@     @W@     �W@     �Z@     �[@     @]@     @d@      b@     @e@     �f@     @l@      l@     �p@     �r@     �q@      k@        
�
conv5/biases_1*�	    .3*�   ���)>      <@!  ��@C�)f�/b��<2��'v�V,����<�)���o�kJ%�4�e|�Z#���-�z�!�%�����i
�k�2!K�R���J�Z�TA[�����"�nx6�X� ��f׽r�����-��J�'j��p���1���=��]����|86	Խ(�+y�6ҽ�9�e��=����%�=�f׽r��=nx6�X� >���">Z�TA[�>�J>2!K�R�>��R���>Łt�=	>��f��p>4�e|�Z#>��o�kJ%>4��evk'>���<�)>�'v�V,>�������:�              �?               @      @               @              @              �?              �?              �?       @      �?              �?              �?              �?              �?              �?      �?               @              �?      �?              �?        b�࿸S      �E�+	yZE1���A*��

step  @A

loss�t�>
�
conv1/weights_1*�	   @ʯ�   ��)�?     `�@! ��e�S�?)It)�oK@2�����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���82���bȬ�0���VlQ.��[^:��"��S�F !�ji6�9���.����ڋ���d�r�x?�x��>h�'��f�ʜ�7
���(��澢f����O�ʗ��>>�?�s��>�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?�������:�             �P@     `i@     �d@      f@     `e@      c@     �_@     `a@      W@     �V@     �X@      U@     �P@      K@      L@     �K@     �J@      N@      C@     �@@     �@@      ;@      <@      @@      =@      7@      8@      "@      9@      ,@      &@      *@      .@      $@      *@      @      @      @      @      @      @      @      @       @      @       @       @       @       @      �?      @      �?               @               @       @               @              �?       @              �?              �?              �?              �?              �?      �?              �?      �?       @       @               @      �?      �?       @              �?      @              @       @      @       @      @       @      @      @      @       @      @      *@      @      @      1@      (@      @      *@      (@      ,@      3@      <@      7@      4@      9@      @@     �A@      ;@      G@     �A@      D@      J@      G@      M@     �N@     @P@      T@      R@     @X@     @[@     ``@     �^@      a@     �d@     �g@     �g@     �g@     �T@        
�
conv1/biases_1*�	   ��]�   @un?      @@!  ����?)�RW�R?2�E��{��^��m9�H�[�<DKc��T��lDZrS�IcD���L��qU���I�a�$��{E��T���C��!�A����#@��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(��vV�R9��T7����f����>��(���>>�?�s��>�FF�G ?f�ʜ�7
?>h�'�?�[^:��"?U�4@@�$?I�I�)�(?�7Kaa+?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?
����G?�qU���I?IcD���L?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?E��{��^?�l�P�`?���%��b?�N�W�m?;8�clp?�������:�              �?              �?               @              �?              �?              �?      �?              �?      �?              �?              �?              �?              �?              �?              �?              �?      �?              �?      �?              �?      �?              �?      @      �?      �?               @      �?              �?        
�
conv2/weights_1*�	   ��j��   @�Ϊ?      �@! ��'�3*@)s!��]]E@2����g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'�������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ���I��P=��pz�w�7��})�l a��ߊ4F��8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ��>M|Kվ��~]�[Ӿjqs&\�Ѿ['�?�;;�"�qʾG&�$��5�"�g�����ӤP�����z!�?�����?�ګ>����>�[�=�k�>��~���>jqs&\��>��~]�[�>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?�������:�             ��@     
�@     ��@     �@     H�@     x�@     T�@      �@     P�@     ��@     �@     �@     ��@     (�@     H�@     ��@     Ѐ@     �}@     Pz@     �x@      v@      v@      t@      o@     �n@     @o@      j@     �g@     �c@     �c@     �`@     �]@      [@     �[@     �W@     @U@     �S@     @S@      S@      R@      N@     �L@     �D@     �G@     �H@      >@      C@      9@      ;@      5@      7@      7@      5@      .@       @      2@      $@      *@      .@      @      &@      @      &@       @       @      @      @      �?      @      @       @      @               @      �?      @       @      �?      @              �?       @       @              �?              �?               @      �?      �?              �?       @               @              �?              �?              �?              �?               @              �?               @      �?      @      �?      �?       @      @      �?      @      @      @      �?      @       @      &@      @      @      @      @      @      @       @      @      "@      &@      0@      &@      4@      &@      .@      ,@      0@      >@      5@      8@      ?@      E@      C@     �G@     �C@     �G@     �I@      R@     �S@      S@      W@      R@     @V@     @X@     ``@      _@     �`@      b@     �e@     `g@     @k@      k@     �n@     �p@     �r@     �r@     �v@     @y@     �z@     �}@     ��@     ��@     ��@     �@     ��@     ��@      �@     D�@     ؑ@     |�@     ��@     ��@     К@     P�@     ��@     b�@     ��@        
�
conv2/biases_1*�	   @jr�   @��o?      P@!  p|��?)|�=M`{$?2�uWy��r�;8�clp�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW��lDZrS�nK���LQ�IcD���L��qU���I�
����G�a�$��{E��T���C�d�\D�X=���%>��:�uܬ�@8���%�V6���82���bȬ�0�I�I�)�(�+A�F�&�U�4@@�$��S�F !�ji6�9���.����ڋ�pz�w�7�>I��P=�>����?f�ʜ�7
?�T7��?�vV�R9?I�I�)�(?�7Kaa+?��82?�u�w74?��%�V6?uܬ�@8?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?�������:�              �?              �?      �?              �?       @              �?              �?              �?       @      �?      �?              �?       @      �?              �?              �?      �?              �?              �?              �?              �?              �?               @               @              �?               @              �?              �?      @              @              @      �?      �?      �?       @      @      �?      �?       @      �?      @      �?              �?        
�
conv3/weights_1*�	   @L���   `���?      �@! �{�ys�?)�A��NU@2�����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ���I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(龢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�_�T�l׾��>M|Kվ��~��¾�[�=�k���u`P+d����n�����4�j�6Z�Fixі�W��iD*L��>E��a�W�>���%�>�uE����>�f����>��(���>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?�������:�             (�@     N�@     �@     �@     ~�@     ҡ@      �@     ��@     �@     ��@     (�@     T�@     �@     ��@     Ѝ@     ��@     ؇@     Ѕ@     @�@     h�@     �@     �{@     �{@     �x@     �v@     �u@     �t@     �q@     �p@     �m@     �h@     �f@      e@     `d@     �d@      b@      ^@     @\@     �T@     �Y@     �S@      T@     �L@      O@     �J@      K@      B@     �H@      C@     �@@     �B@      ;@      7@      3@      4@      7@      1@      2@      &@      &@      ,@      @      .@      @      $@      "@       @      @      @      @      @      @       @      @      @      �?      �?      @               @               @              �?      �?       @       @      �?      �?              �?      �?              �?              �?              �?              �?              �?              �?              �?      @       @              �?      �?              @               @      �?              �?      @      @       @      @       @      �?      @      @      @      @      @      $@       @      @      @      @      (@      &@      0@      (@      .@      "@      .@      .@      9@      =@      @@      <@      A@      ;@      C@     �D@     �C@      L@     �J@     �L@      S@      R@      T@     �W@     �V@     �]@     @[@     �a@      b@     @f@     �f@      l@     �l@     `j@      l@     �p@     �s@     s@     0v@     x@     `y@      @     (�@     p�@     ��@     0�@     h�@     h�@     ��@     �@     �@     \�@     ܕ@     ̘@     T�@     ��@     (�@     B�@     j�@     �@     >�@     ��@     ��@        
�

conv3/biases_1*�
	   `@�o�   ���s?      `@!  �wB�?)��`�>�7?2�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A�d�\D�X=���%>��:�uܬ�@8���%�V6�I�I�)�(�+A�F�&�U�4@@�$��[^:��"�ji6�9���.���FF�G �>�?�s���O�ʗ���pz�w�7��})�l a�jqs&\�ѾK+�E��Ͼ        �-���q=pz�w�7�>I��P=�>>�?�s��>�FF�G ?�vV�R9?��ڋ?�.�?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?�������:�              �?      �?              �?      �?      �?       @      @      @       @       @      @      @       @      �?      @       @      @               @       @              �?      �?       @               @      �?      �?              �?              �?      �?              �?              �?              @              �?              �?              �?      �?              �?       @              �?      �?      @               @      @      �?      �?      @      @       @              @      @      @      @      @      @      @       @       @       @               @      �?      �?        
�
conv4/weights_1*�	   �@��    m$�?      �@! ��@�׿)�p>�e@2�Ӗ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.���vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[����Zr[v��I��P=���ѩ�-߾E��a�Wܾ�iD*L�پjqs&\�ѾK+�E��Ͼ��Zr[v�>O�ʗ��>�FF�G ?��[�?1��a˲?6�]��?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�������:�             �Z@     �@     �@     ��@     L�@     �@     ؍@     P�@     X�@     �@     ��@     p�@     }@     P{@     �x@     �x@     �u@     Pv@     pq@     pq@      m@      k@     �e@     �e@     @f@     �c@     @\@     �]@     �Z@     �U@     �R@     @W@      N@      N@     �N@      M@     �D@      D@     �D@      M@      C@      >@      8@      <@      7@      0@      2@      9@      8@      "@      ,@      1@       @      (@      @      "@      @      @      @      @      @      @       @      @      @      @      @      @       @      �?      @      �?      �?       @      �?               @      �?      �?               @      �?      �?              �?       @              �?              �?      �?              �?              �?              �?              �?              @      �?      �?       @               @       @               @       @      @      @      @      �?      �?      �?      @      @      @      �?      @      @      @      "@      @      $@      @      $@       @      $@      &@      (@      9@      .@      1@      4@      6@      :@      <@      5@     �@@     �H@      G@      F@     �E@     �L@     @Q@     �Q@     @Q@     @R@     �W@     @X@     �]@     �_@     @_@     �a@     �`@     `e@      g@     �i@      j@     �p@     q@     `s@     �u@     �v@     @x@      x@     `|@      @      �@     ��@     ��@     X�@     ��@     ��@     X�@     T�@     ��@     ��@     ,�@     �b@        
�
conv4/biases_1*�	   @�n�   @*ct?      p@!5��~�?)I�>c�5?2�;8�clp��N�W�m�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�+A�F�&�U�4@@�$��[^:��"�ji6�9���.���vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������O�ʗ�����Zr[v��I��P=��pz�w�7���ߊ4F��h���`f�����uE����E��a�Wܾ�iD*L�پ���<�)�4��evk'���o�kJ%�4�e|�Z#���-�z�!�%�����i
�k���f��p�Łt�=	���R����2!K�R���J��#���j�Z�TA[�����"�RT��+��y�+pm��mm7&c��`���nx6�X� ��f׽r�������%���9�e����K��󽉊-��J�'j��p���1���=��]����/�4��ݟ��uy��Qu�R"�PæҭUݽ�|86	Խ(�+y�6ҽ��s�����:��        �-���q=|_�@V5�=<QGEԬ=5%���=�Bb�!�=ݟ��uy�=�/�4��==��]���='j��p�=��-��J�=�K���=�9�e��=����%�=f;H�\Q�=�tO���=�f׽r��=nx6�X� >�`��>�mm7&c>y�+pm>RT��+�>���">Z�TA[�>�#���j>�J>2!K�R�>��R���>Łt�=	>��f��p>�i
�k>4�e|�Z#>��o�kJ%>4��evk'>���<�)>�'v�V,>K+�E���>jqs&\��>E��a�W�>�ѩ�-�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?�������:�              �?               @      �?              �?      @               @      @               @       @               @      �?       @              �?      �?              �?               @              �?      @              �?       @              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?      �?       @      �?              @      @      @      @       @      @       @       @      @       @      �?              @               @       @      �?      �?      �?       @      �?       @               @              �?              �?              9@              �?              �?              �?      �?              �?       @      �?      @      �?      �?              �?      �?      @      �?       @       @      �?      @              @      �?       @       @       @               @      �?              �?              �?              �?              �?               @              �?              �?      �?              �?              @              �?              �?      �?              �?      �?       @      �?      �?              �?               @      �?              �?              �?      @       @      @      @              �?       @              @       @      @       @      �?       @      �?      �?      @      �?      @       @      @       @      �?       @      �?      @      �?       @      �?        
�
conv5/weights_1*�	   ��¿   �5��?      �@! @����)o}g�pWI@2�	�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0��7Kaa+�I�I�)�(��.����ڋ�1��a˲���[��>�?�s��>�FF�G ?��[�?1��a˲?6�]��?��d�r?�5�i}1?�S�F !?�[^:��"?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?�������:�	             �j@      u@     �p@      o@     `n@      m@     �e@     �e@     �d@     �_@     �]@     �[@     @Y@      \@     �V@     �Q@     �R@     @Q@      J@     �J@     �K@     �A@      F@      B@      >@      3@      :@      =@      1@      7@      =@      ,@      0@      $@      4@      &@      "@      @       @      @      @      @      $@      �?      @      @      "@      @      @       @       @      @              @       @      �?      �?      �?      @      �?      �?      �?      �?              �?              �?              �?              �?              �?              �?              �?      �?              �?              �?              �?       @       @              �?       @      @               @      @      �?      �?              @               @      &@       @              @      @      �?      @      @      @      @      *@      @      *@      @      &@      .@      @      .@      1@      7@      5@      8@      2@      .@      :@      @@      ;@     �E@      >@     �D@      C@     �L@      I@      S@      U@     �P@      Q@      W@     @W@     �Z@     @[@     �]@     @d@     �a@     @e@      g@      l@      l@     �p@     pr@     �q@     �j@        
�
conv5/biases_1*�	   ��--�    6�+>      <@!  ���hD�)PО�<2�7'_��+/��'v�V,����<�)�4��evk'���o�kJ%�4�e|�Z#���-�z�!�%�����i
�k���R����2!K�R���J��#���j�Z�TA[��nx6�X� ��f׽r����9�e����K��󽉊-��J���:������z5��<QGEԬ=�8�4L��=�tO���=�f׽r��=�mm7&c>y�+pm>�#���j>�J>2!K�R�>��R���>Łt�=	>��f��p>�i
�k>%���>��-�z�!>��o�kJ%>4��evk'>���<�)>�'v�V,>�������:�              �?              �?       @       @              �?      �?              @      �?              �?              �?              �?      �?              �?              �?              �?              �?              �?              �?              �?              �?      �?              �?      �?      �?        �ږ�XT      �9<	�{k1���A*ɨ

step  PA

loss^��>
�
conv1/weights_1*�	    A��   ��j�?     `�@!  �E��?)�|�I9Z@2�����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0�I�I�)�(�+A�F�&�U�4@@�$���ڋ��vV�R9��5�i}1���d�r�6�]���1��a˲��iD*L�پ�_�T�l׾��[�?1��a˲?6�]��?����?��ڋ?�.�?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�������:�             @R@      h@     �d@     �f@     �c@     `d@      `@     �`@      X@     @U@     �Y@     @T@     �Q@      L@     �L@      I@     �L@     �L@     �D@     �A@      =@      8@     �@@      @@      8@      7@      :@      $@      4@      8@      "@       @      &@      1@      &@      "@      @      @      @       @      @      @       @      �?      @      @      �?       @       @       @      �?      �?               @      @               @      �?              �?      �?              �?              �?              �?              �?              �?              �?              �?              @              �?      �?              �?               @              �?      �?       @              �?      @       @      @      @       @      @      �?      @      @      @      @      @      (@      &@       @      "@      ,@      $@      &@      .@      (@      4@      6@      ;@      5@      9@      A@      @@      >@      D@     �C@     �B@     �H@      K@      L@      M@     �Q@     �R@     @R@     �X@      [@     @`@      _@     ``@     `d@     `h@      h@     `h@      T@      @        
�
conv1/biases_1*�	   @��_�   �,�p?      @@!  �&��?),�;%?2��l�P�`�E��{��^�<DKc��T��lDZrS��qU���I�
����G�a�$��{E���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�ji6�9���.����ڋ�a�Ϭ(�>8K�ߝ�>>h�'�?x?�x�?U�4@@�$?+A�F�&?��VlQ.?��bȬ�0?��82?��%>��:?d�\D�X=?���#@?�!�A?�T���C?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?;8�clp?uWy��r?�������:�              �?              �?              �?       @              �?              �?      �?      �?              �?      �?              �?              �?              �?              �?      �?              �?              �?      �?               @              �?      �?       @       @              �?              �?       @              �?        
�
conv2/weights_1*�	   �T���    s�?      �@!  -;��+@)�?+�1_E@2����g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ���(��澢f�����uE���⾮��%ᾙѩ�-߾�_�T�l׾��>M|Kվ��~��¾�[�=�k���u`P+d�>0�6�/n�>�*��ڽ>�[�=�k�>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�uE����>�f����>�h���`�>�ߊ4F��>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?�������:�             ؐ@     �@     ��@     �@     L�@     ��@     �@     ,�@     L�@     ��@     ȋ@     0�@     ��@     0�@     8�@     ȁ@      �@     p}@     �y@     py@     �u@     �v@     0s@     �o@      o@     �n@     �j@      f@     @e@     �b@     �`@     �_@     �[@      [@     �V@     �T@     �V@     @T@     @Q@     @Q@     �H@     @P@      B@     �I@     �I@      ?@      =@     �A@      <@      5@      4@      6@      4@      4@       @      *@      .@      "@      $@      $@      "@      "@      @      @      @       @      @      @      @      @       @      �?       @      @      �?      @      @              �?      @      �?      �?              �?       @              �?              �?      �?              �?              �?              �?              �?              �?              �?              �?              �?               @              @              �?      �?      �?      @      �?       @      @       @      @       @      "@      @      @       @       @      @       @      @       @      @      @      ,@      @      0@      @      3@      *@      3@      0@      3@      5@      7@      5@      =@      A@     �G@     �I@      E@     �G@      K@     �P@     �U@     �Q@      W@     �Q@     @Y@     �V@     �`@     @]@     �`@     �c@     `d@     �h@     �j@      j@     �m@     `q@     �r@     �r@     �v@     �y@     �z@     �}@     ��@     8�@     ��@     0�@      �@     Љ@     ��@     P�@     ܑ@     l�@     ��@     ��@     ��@     �@     ȟ@     b�@     ��@      @        
�
conv2/biases_1*�	   �y!s�   `mYq?      P@!  p��?)���}�(?2�hyO�s�uWy��r�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��qU���I�
����G�a�$��{E����#@�d�\D�X=���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+���d�r�x?�x��>�?�s���O�ʗ���pz�w�7��})�l a򾂼d�r?�5�i}1?�T7��?�[^:��"?U�4@@�$?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�T���C?a�$��{E?
����G?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?�������:�              �?              �?              �?      �?       @              �?              �?               @       @               @              �?              �?       @      �?              �?              �?              �?              �?      �?              �?              �?              �?      �?              �?              �?      �?      �?              �?       @              �?       @      @       @       @      �?      @      �?      @               @       @      �?       @       @              �?        
�
conv3/weights_1*�	   @ܔ��   `;�?      �@! @yLw��?)��OU@2�����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�Ѿ�u`P+d�>0�6�/n�>�*��ڽ>�[�=�k�>�uE����>�f����>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?�������:�             �@     d�@     ֧@     �@     ~�@     ȡ@     ��@     Ԝ@     ̚@     ��@      �@     H�@     T�@     ��@     ؍@     h�@     ؇@     �@     �@     ��@     �@     �{@     �{@      y@      w@     �u@     �s@     `r@     @p@      n@     �h@      f@      e@     @e@      d@     @a@      ^@      \@      X@     �X@      V@      N@      N@     �O@     �J@      P@     �A@      J@     �A@     �A@      =@      ;@      :@      >@      (@      5@      4@      0@      *@      *@      @      &@      3@      @      "@      "@      @      @      @      @      @      @      @      @       @      @      @      @       @      �?       @       @       @              �?              �?              �?              �?       @              �?              �?              �?              �?              �?              �?               @              �?               @               @      @      �?      �?      �?      �?       @               @       @      �?      @       @      @      @      @      $@      @      @       @      "@      1@      *@      $@      $@      $@      0@      1@      1@      9@      7@      >@      =@      8@      A@      A@      G@     �H@      I@     �E@     @P@     �P@     �S@     @T@     �U@     �Y@     @[@     @]@     �`@     �a@     `f@     @h@     @j@     �l@     �j@     @l@      p@     �s@     s@     `v@      x@     `y@     `@     @�@     `�@     Є@     ��@     8�@     @�@     ��@     �@     ԑ@     ��@     ��@     ��@      �@     ��@     �@     \�@     j�@     ��@     @�@     ��@     �@        
�
conv3/biases_1*�	   ���q�   �"�u?      `@!  ����?)'U�E�=?2�uWy��r�;8�clp��N�W�m�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E����#@�d�\D�X=���%>��:�uܬ�@8���%�V6���82���bȬ�0��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��S�F !�ji6�9����ڋ��vV�R9������6�]���O�ʗ�����Zr[v��I��P=��pz�w�7����~]�[Ӿjqs&\�Ѿ        �-���q=�h���`�>�ߊ4F��>>h�'�?x?�x�?ji6�9�?�S�F !?U�4@@�$?+A�F�&?��VlQ.?��bȬ�0?��82?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?�������:�              �?      �?               @      �?      @      @      @      @      �?      @      �?      @              @      �?      @      �?       @              @      �?              �?              �?              �?              �?               @              �?              �?              �?              �?              �?              @              �?              �?              �?              �?               @      �?               @      �?              �?      �?              �?       @      �?               @      @       @      @      �?       @      @      @      @      @      @      @       @      �?      @              �?       @      �?        
�
conv4/weights_1*�	    W��   �9-�?      �@!  I�[ſ)G�TF��e@2�Ӗ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F�𾮙�%ᾙѩ�-߾jqs&\�ѾK+�E��Ͼ�[�=�k�>��~���>['�?��>K+�E���>�h���`�>�ߊ4F��>�FF�G ?��[�?1��a˲?6�]��?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�������:�             @[@      �@     �@     ��@     T�@     �@     �@     0�@     x�@     �@     ��@     h�@     @}@     0{@     �x@     �x@     �u@     0v@     �q@      q@     `m@     @k@     �e@      f@      f@     �c@     �\@      ^@      Z@     �U@     �R@      W@      N@      O@      N@     �M@      B@      I@     �B@      M@     �C@      :@      :@      ;@      6@      4@      0@      4@      ;@      &@      1@      *@      @      $@      $@      &@      @      @      @      @      @      @      @      @      @      @       @      @       @      �?      @      �?      @       @              �?               @      @              �?               @      �?              �?      �?              �?      �?              �?              �?              �?              �?              �?              �?              �?              �?               @              @      �?       @      �?              �?      @       @      @      �?      @      �?      @      �?      �?       @      @       @      @      @       @      "@      "@      (@      @      "@      @      "@      (@      *@      9@      ,@      .@      6@      6@      9@      ;@      7@      B@     �D@     �G@      G@      G@     �L@      Q@     @Q@     �Q@     �R@     @W@      Y@     �\@      `@     @_@     `a@     �`@     �e@     �f@     �i@     @j@     �p@     q@     ps@     @u@     �v@      x@     0x@     �|@      @     8�@     ؂@     ��@     h�@     ��@     ��@     X�@     D�@     �@     ��@     0�@     �b@        
�
conv4/biases_1*�	   �~�p�    ��v?      p@!��8��X�?)�zY�y�:?2�uWy��r�;8�clp�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
�})�l a��ߊ4F��a�Ϭ(���(���E��a�Wܾ�iD*L�پ���?�ګ�;9��R�����<�)�4��evk'���o�kJ%�4�e|�Z#���-�z�!�%�����i
�k���f��p�Łt�=	���R����2!K�R���J��#���j�Z�TA[�����"�RT��+��nx6�X� ��f׽r����tO����f;H�\Q������%���9�e����K��󽉊-��J�=��]����/�4��ݟ��uy�z�����i@4[���Qu�R"�(�+y�6ҽ;3���н�!p/�^˽�d7���Ƚ�
6������Bb�!澽�Į#�������/��        �-���q=����z5�=���:�=�|86	�=��
"
�=�Qu�R"�=i@4[��=z�����=ݟ��uy�=��1���='j��p�=�9�e��=����%�=f;H�\Q�=�tO���=�`��>�mm7&c>y�+pm>RT��+�>���">Z�TA[�>�#���j>�J>2!K�R�>��R���>Łt�=	>��f��p>�i
�k>%���>��o�kJ%>4��evk'>���<�)>�'v�V,>7'_��+/>jqs&\��>��~]�[�>�ѩ�-�>���%�>�f����>��(���>a�Ϭ(�>8K�ߝ�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�vV�R9?��ڋ?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?�������:�              �?               @      �?              @      �?              �?      @      �?      �?      @      �?       @      �?      �?       @              �?      �?              �?               @       @              �?      �?       @              �?              �?              �?              �?              �?              �?              �?              �?               @      �?       @              �?      �?      @      @      @      @      @      @      �?       @       @              �?      �?      �?      �?       @              @               @      �?      �?      �?      �?              �?              �?              �?              �?              9@              �?              �?              �?              �?              @               @      �?       @              @      @              �?      @      @      �?       @               @       @      @      �?               @      �?              �?              �?               @              �?              �?              �?               @       @               @              �?      �?              �?      �?      �?       @               @              �?       @       @              �?              �?       @      �?      @      @      �?      �?       @       @      �?       @       @      @      @              �?      @              @      �?      @       @      @       @      �?      �?       @      @              @      �?        
�
conv5/weights_1*�	   �c�¿   �g��?      �@! �!T��)�:���XI@2�	�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=�uܬ�@8���%�V6���82���bȬ�0���VlQ.��7Kaa+�ji6�9���.���5�i}1���d�r�x?�x��>h�'��8K�ߝ�a�Ϭ(��ߊ4F��>})�l a�>��d�r?�5�i}1?�T7��?�vV�R9?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?�������:�	              k@     �t@     �p@     @o@     `n@     �l@     �e@     �e@     �d@      `@     �]@     �[@     @Y@      \@     �V@     �Q@     �R@     @Q@     �I@      K@      K@      B@     �F@     �A@      ?@      2@      :@      =@      2@      6@      >@      (@      1@       @      4@      (@      $@      @      @      @       @      @       @       @      @      @      @      @      @      @      @      @       @      �?      @       @      �?       @      @       @       @              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?              @      �?       @      �?      �?      @       @      �?      �?       @              �?      �?               @       @      "@      @              @      @              @      @      @       @      $@       @      (@      @      "@      0@      @      .@      1@      7@      5@      8@      0@      2@      8@      A@      ;@     �D@      ?@     �D@      C@      L@     �H@     @S@     @U@     �P@     @Q@     @W@      W@      [@      [@     �]@     @d@     �a@      e@      g@     `l@     �k@     pp@     �r@     �q@     �j@        
�
conv5/biases_1*�	   ���0�   @�N.>      <@!  @��C�)��RQZ�<2�_"s�$1�7'_��+/��'v�V,����<�)�4��evk'���o�kJ%�4�e|�Z#�%�����i
�k���f��p���R����2!K�R���#���j�Z�TA[���mm7&c��`����K��󽉊-��J�PæҭUݽH����ڽ|_�@V5����M�eӧ�;3����=(�+y�6�=�`��>�mm7&c>���">Z�TA[�>�J>2!K�R�>��R���>��f��p>�i
�k>��-�z�!>4�e|�Z#>4��evk'>���<�)>�'v�V,>7'_��+/>�������:�              �?               @      �?      �?      �?              �?       @              @              �?              �?              �?              �?              �?              �?              �?              �?              �?      �?              �?               @               @              �?        s��(S      �&Y	�~�1���A*��

step  `A

lossK��>
�
conv1/weights_1*�	   ��W��   `��?     `�@! ���v��?)U��)l@2���]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&��.����ڋ��vV�R9��T7����5�i}1�f�ʜ�7
������pz�w�7�>I��P=�>��Zr[v�>��ڋ?�.�?�S�F !?�[^:��"?U�4@@�$?�u�w74?��%�V6?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�������:�              �?      R@     @h@     �c@     �f@      d@     �c@      `@      `@     @Z@     @S@     �Z@     �U@      O@      O@     �I@     �M@     �K@      L@      B@      B@      >@      ?@      6@      E@      8@      6@      .@      6@      0@      4@      &@      $@      (@      &@      &@      $@      @      @      @      @      @      @      �?      @      @      @      �?      @       @      @       @              �?      �?               @      �?      �?              �?              �?              �?      �?              �?              �?              �?      �?              �?               @       @               @              �?      �?              �?              @      @       @       @              @       @       @      @       @      $@      "@      $@      @      "@      3@      @      (@      .@      4@      .@      0@      <@      ;@      9@      ?@      @@      >@      C@      C@     �G@      D@      N@     �I@      N@     �O@     @S@     �S@     @W@     �Z@     �`@     �^@     ``@      e@      h@      h@     `h@     �U@      @        
�
conv1/biases_1*�	    ,�`�   ���r?      @@!  8�1��?)1�ts]K?2��l�P�`�E��{��^�<DKc��T��lDZrS�
����G�a�$��{E��T���C�uܬ�@8���%�V6���VlQ.��7Kaa+��S�F !�ji6�9���ߊ4F��h���`h���`�>�ߊ4F��>��d�r?�5�i}1?�T7��?��82?�u�w74?��%�V6?��%>��:?d�\D�X=?���#@?a�$��{E?
����G?IcD���L?k�1^�sO?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?uWy��r?hyO�s?�������:�              �?              �?              �?      �?              �?              @              �?              �?              �?              �?      �?              �?      �?              �?      �?               @              �?              �?      �?       @       @      �?      �?              �?       @              �?        
�
conv2/weights_1*�	   ��ת�   ��c�?      �@! E��
~-@)*�daE@2����g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[��>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پK+�E��Ͼ['�?�;
�/eq
Ⱦ����žjqs&\��>��~]�[�>�_�T�l�>�iD*L��>�ѩ�-�>���%�>�uE����>�f����>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?�������:�             А@     �@     Ȟ@     ��@     ,�@     ė@     ,�@     �@     @�@     �@     ��@     �@      �@     X�@     P�@     ؁@     Ȁ@     �|@      {@     �x@     @v@     Pv@      s@     pp@     �n@     `n@     @j@     �f@     �d@     `c@      `@     �_@     �\@     �[@     @X@     �T@     �T@     �T@     �O@     �P@      L@     �L@      G@      E@     �G@     �C@      <@     �@@      7@      ;@      2@      8@      2@      7@      (@      (@      1@      @      (@      @      "@       @      @      @      @      @       @      �?       @      @      �?      @      @               @      �?       @              @       @               @      @      @              �?              �?              �?      �?      �?              �?              �?              �?              �?              �?      �?      �?              �?              �?      �?       @              �?      �?      @       @      @      @      @      @      @      @      @      "@      @      @       @      @      .@      @      @      (@      (@      (@      @      0@      (@      1@      ,@      .@      2@      :@      5@      <@      G@      E@     �H@      C@      F@      R@      P@     @T@     �R@     @W@     �R@      Y@     �W@     �\@     �]@     �_@     `e@     �c@     �g@     �l@      j@     �l@     �q@     �r@      r@     �v@     `z@     �y@      ~@     �@     Ђ@     �@      �@      �@     �@     ��@     d�@     ̑@     t�@     t�@     ��@      �@     D�@     ��@     T�@     В@      @        
�	
conv2/biases_1*�		   `0t�   ���r?      P@!  p#[�?)~���-?2�&b՞
�u�hyO�s�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�IcD���L��qU���I�
����G��T���C��!�A����#@�d�\D�X=���%�V6��u�w74�I�I�)�(�+A�F�&�ji6�9���.�������6�]���>�?�s���O�ʗ����ߊ4F��h���`�I��P=�>��Zr[v�>�5�i}1?�T7��?ji6�9�?�S�F !?�[^:��"?U�4@@�$?I�I�)�(?�7Kaa+?��VlQ.?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?�������:�              �?              �?              @      �?              �?              �?              �?       @              �?      �?      �?              �?              �?              �?              �?              �?              �?              �?              �?              �?      �?      �?              �?      �?              �?      �?              �?      �?       @              �?              �?      �?               @      @       @       @      �?      @      �?      @               @       @       @       @      �?              �?        
�
conv3/weights_1*�	    ����    �>�?      �@! @֥��?)<O.�PU@2�����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE��������ž�XQ�þjqs&\��>��~]�[�>�_�T�l�>�iD*L��>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?�������:�             (�@     Z�@     ҧ@     �@     x�@     ġ@     �@     ؜@     К@     ��@     ܕ@     t�@     D�@     ��@     ��@     ��@     Ї@     Ѕ@     8�@     p�@     ��@     �{@     �{@     Py@     �v@     �u@     ps@     �r@     pp@     �m@      i@     �e@      f@     �d@      c@     @a@      `@     �Z@      X@     �Y@      T@      Q@      N@      O@     �M@      J@      F@      G@     �B@      C@      ?@      ;@      6@      5@      9@      9@      (@      .@      .@      ,@       @      $@      4@       @      @      &@      �?      @      @      @      @      @      @      @       @      �?      �?      @      @               @       @      �?               @       @      �?               @              �?               @              �?              �?              �?              �?      �?               @      @               @              �?               @              �?              �?      �?      �?      @      @      �?      @      @      @      @      "@      @      @      @      "@       @      @      *@      $@      &@      ,@      ,@      1@      1@      9@      8@      <@      <@      ;@     �A@      B@     �E@      B@      O@     �I@      I@     @R@     @R@     @U@     @U@     �Y@     @\@     �^@      _@      b@     �e@     �i@      j@     �k@     `l@     �k@      p@     ps@     �r@     �u@     �x@     `y@     �@     �@     X�@     ��@     Ȇ@     ��@     h�@      �@     ȏ@     Б@     ��@     Е@     Ș@     0�@     ��@     �@     >�@     ��@     �@     F�@     ��@     @�@        
�
conv3/biases_1*�	   @��s�   `�Lw?      `@!  ��M�?)��Oق@B?2�hyO�s�uWy��r�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@���%>��:�uܬ�@8���%�V6��u�w74���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&��S�F !�ji6�9���.����ڋ�>h�'��f�ʜ�7
���Zr[v��I��P=���_�T�l׾��>M|Kվ        �-���q=�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?U�4@@�$?+A�F�&?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?�������:�               @               @      @      �?      @      @      @      �?      @      �?       @       @       @       @       @       @              �?              �?       @              �?      �?      �?              �?              �?      �?              �?      �?      �?              �?              �?              �?              @              �?       @      �?              �?              �?               @              �?              �?               @              �?       @      �?      �?      @       @      @       @      �?       @      @      @      @       @      @      @       @       @      @              �?       @      �?        
�
conv4/weights_1*�	   ��&��   ��7�?      �@!  ,�z3�?)�<�#�e@2�Ӗ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������1��a˲���[��pz�w�7��})�l a��uE���⾮��%�jqs&\�ѾK+�E��Ͼ���?�ګ>����>���%�>�uE����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>1��a˲?6�]��?����?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�������:�             @[@     ��@     �@     ��@     @�@     ��@     ��@     0�@     X�@     �@     ��@     x�@     @}@     @{@     px@      y@     �u@     v@     �q@     q@     �m@     �j@     �d@     �f@     �f@     �b@     �]@     �]@     �Y@     �V@     �Q@     �W@     �M@     �N@      O@      L@     �C@      G@     �C@     �O@     �@@      <@      =@      8@      5@      5@      3@      4@      6@      *@      2@      &@      "@       @      @      (@       @      @       @      @      @      @      @      @      @      @       @       @      @              @      �?       @       @      �?      �?       @      �?       @              @              �?      �?              �?               @              �?              �?              �?              �?              �?              �?              �?              �?      �?              �?      �?              �?       @              �?      @      @      �?      @      �?       @       @      @       @      �?      @      @       @       @      @      @      (@      @      &@      @      $@      @      $@      *@      "@      :@      .@      ,@      7@      6@      ;@      =@      8@      =@     �F@      E@      G@     �H@      M@     @Q@     �P@      R@     �R@     �V@     �Y@     �\@      `@     �_@     �`@     �`@     �e@     �f@     �i@     �j@     `p@     q@     ps@     Pu@      w@     �w@     Px@     �|@      @      �@      �@     ��@     x�@     ��@     ��@     p�@     @�@      �@     ��@     $�@      c@        
�
conv4/biases_1*�	     Tr�   ���x?      p@!@�#dn��?)@�S�"@?2�hyO�s�uWy��r�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���VlQ.��7Kaa+�I�I�)�(�U�4@@�$��[^:��"��T7����5�i}1���d�r�x?�x��>h�'��pz�w�7��})�l a�h���`�8K�ߝ���(��澢f����E��a�Wܾ�iD*L�پ7'_��+/��'v�V,����<�)�4��evk'���o�kJ%�4�e|�Z#���-�z�!�%�����i
�k���f��p�Łt�=	���R����2!K�R���J��#���j�Z�TA[�����"�RT��+���`���nx6�X� ��f׽r����tO����f;H�\Q������%���9�e�����1���=��]����/�4���Qu�R"�PæҭUݽH����ڽ���X>ؽ�|86	Խ(�+y�6ҽ        �-���q=���6�=G�L��=�Bb�!�=�
6����=K?�\���=�b1��=��؜��=�!p/�^�=��.4N�=H�����=PæҭU�=�/�4��==��]���=��1���='j��p�=����%�=f;H�\Q�=�tO���=�f׽r��=�mm7&c>y�+pm>RT��+�>���">Z�TA[�>�#���j>�J>2!K�R�>��R���>Łt�=	>��f��p>�i
�k>%���>��-�z�!>��o�kJ%>4��evk'>���<�)>�'v�V,>7'_��+/>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>���%�>�uE����>�f����>��(���>8K�ߝ�>�h���`�>��Zr[v�>O�ʗ��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?��ڋ?�.�?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?�������:�              �?              �?       @              @              @      @               @       @      @              �?      �?              �?      �?      �?       @              �?              �?      @              �?               @      �?              �?              �?              �?              �?              �?              �?               @       @              �?       @      �?      @      @       @      @              @      @      @      @              �?              �?      @       @      �?               @      �?              �?              �?              �?              9@              �?              �?              �?       @              �?              @              �?      �?       @              �?       @       @              @      �?      �?      �?      @       @       @      �?      �?               @      @      @              �?      �?      �?      �?              �?              �?              �?              �?              �?               @               @      @      �?              �?      �?      �?       @              �?               @               @       @      �?      �?      �?              �?              �?      �?      @       @               @      �?       @       @      �?      @      @      @      @       @      �?      @       @      @      �?      @       @      �?      �?       @      @              @      �?        
�
conv5/weights_1*�	   ���¿   @;��?      �@! ��)�)>��"�ZI@2�	�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��T7����5�i}1��iD*L�پ�_�T�l׾��d�r?�5�i}1?�S�F !?�[^:��"?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?�������:�	              k@     �t@     �p@      o@     �n@     �l@     �e@     �e@     �d@     @`@     @]@      \@     @Y@     �[@     �V@     �Q@     �R@     �Q@      J@      J@     �K@      B@      F@      A@     �@@      2@      :@      =@      1@      6@      @@      &@      1@       @      4@      ,@      @      "@       @      @      @      @       @      @      @      @      @      @      @      @      �?      @      @      @      @       @              �?      @      �?               @               @              �?               @              �?              �?              �?              �?              �?               @      @      @       @      @      @               @      �?      �?      �?      �?      �?              @      @      @       @       @       @              @      @      @      $@       @      @      .@      @      $@      .@      "@      *@      1@      8@      5@      :@      *@      3@      7@     �@@      =@     �C@     �@@      D@     �C@     �L@      H@      S@     �U@     �P@      Q@     @W@      W@      [@      [@     �]@      d@      b@      e@      g@     @l@      l@     pp@     pr@      r@     �j@        
�
conv5/biases_1*�	    �m3�   �YW/>      <@!  ���jE�)hr��8�<2��so쩾4�6NK��2�7'_��+/��'v�V,����<�)�4��evk'���o�kJ%�4�e|�Z#��i
�k���f��p�Łt�=	���R����2!K�R���J�Z�TA[�����"�'j��p���1���PæҭUݽH����ڽ�!p/�^�=��.4N�=��1���='j��p�=�mm7&c>y�+pm>�#���j>�J>2!K�R�>��R���>Łt�=	>��f��p>��-�z�!>4�e|�Z#>��o�kJ%>4��evk'>���<�)>�'v�V,>7'_��+/>_"s�$1>�������:�              �?               @      �?               @      �?               @               @               @              �?              �?              �?              �?              �?              �?              �?              �?              �?               @              �?               @              �?        ��'�HT      b-	���1���A*��

step  pA

lossU�>
�
conv1/weights_1*�	    2���   �y��?     `�@! ���X@)�̒h�@2���]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&��[^:��"��S�F !�ji6�9���.����ڋ�1��a˲���[��;�"�q�>['�?��>���%�>�uE����>�T7��?�vV�R9?�.�?ji6�9�?�S�F !?�[^:��"?+A�F�&?I�I�)�(?�7Kaa+?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�������:�               @     �R@      g@     �c@     �f@     �c@     �b@     �`@     �^@     �[@      U@     �W@     @W@      N@      M@     �M@     �L@      K@     �K@      B@     �C@      =@     �@@      8@     �C@      8@      2@      :@      "@      2@      2@      (@      (@      0@      @      "@      @      &@      @      $@      @      @      @      @      @      @      �?       @       @      @       @       @      �?      �?      �?              �?               @      �?      �?       @              �?       @              �?              �?              �?              �?              �?              �?              �?              �?      �?              �?       @      �?      �?      �?      �?               @       @       @      @       @      �?      �?      �?      @      @      @      @      @       @      *@      &@      &@      ,@      @      1@      1@      *@      2@      5@      9@      2@      ?@      >@     �@@     �A@      @@     �E@      E@     �E@     �M@     �G@      P@      N@     �T@     �S@     �U@     �Z@     �`@      `@      `@     �e@     �g@     @g@      i@     @V@       @        
�
conv1/biases_1*�	    �ma�   @�t?      @@!  (c�\�?)�Vг�D?2����%��b��l�P�`��lDZrS�nK���LQ�a�$��{E��T���C�d�\D�X=���%>��:��7Kaa+�I�I�)�(��[^:��"��S�F !�ji6�9���.����ڋ�})�l a�>pz�w�7�>�5�i}1?�T7��?I�I�)�(?�7Kaa+?�u�w74?��%�V6?uܬ�@8?��%>��:?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?hyO�s?&b՞
�u?�������:�              �?              �?              �?              �?              �?              �?       @              �?              �?              �?              �?               @               @               @      �?              �?              �?              �?       @      @              �?              @              �?        
�
conv2/weights_1*�	   `��   ���?      �@! K3s/@)x�zydE@2�I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
�6�]���1��a˲���[��>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پX$�z�>.��fc��>���]���>�5�L�>�XQ��>�����>
�/eq
�>;�"�q�>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>E��a�W�>�ѩ�-�>���%�>�uE����>�h���`�>�ߊ4F��>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?�������:�              �?     Ȑ@     ��@     ��@      �@     T�@     ��@     (�@     0�@     $�@     ��@     8�@     ��@     h�@     �@     �@     ��@     ��@     �|@      {@     px@     �v@     0v@     `s@     �o@     �n@     �n@     �h@     @h@     �c@     �b@     `a@     �`@     �\@     @X@      [@     �T@      T@      U@      O@     �K@     �K@      M@     �G@     �G@     �E@      E@      ;@      ?@      ;@      A@      6@      .@      &@      2@      .@      0@      .@      &@      $@      &@      (@      @      @       @      @      @       @      @      �?      @       @      @      �?              �?      �?               @      �?      @               @       @               @      �?      @              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?               @              @      �?       @      �?      �?      @      @               @       @      @      @       @      �?      @      @      @      @      @      @      @      "@       @      (@      @      $@      &@      "@      1@      0@      0@      0@      3@      7@      <@      <@      E@      F@     �F@     �A@     �N@     @P@     @P@     �R@     �Q@      U@     �V@     �V@     �Z@      _@      [@     �^@      e@     �d@     �h@     �j@     �j@     �k@     �q@     �r@      s@     �u@     0z@     �y@     �}@     �@     h�@     @�@     Є@     ��@     ؉@     X�@     l�@     ��@     p�@     ��@     ��@     ��@     8�@     ��@     d�@      �@      @        
�
conv2/biases_1*�	   `u�   @K't?      P@!  0�f��?)-D���1?2�&b՞
�u�hyO�s�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�k�1^�sO�IcD���L��qU���I�
����G��T���C��!�A���%�V6��u�w74���82��7Kaa+�I�I�)�(�6�]���1��a˲���[��>h�'�?x?�x�?��d�r?�5�i}1?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?��bȬ�0?��82?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?�������:�              �?              �?      @              �?      �?              �?              �?       @      �?              �?              �?      �?              �?              �?      �?              �?              �?              �?              �?               @              �?              �?      �?      �?       @              �?              �?      �?              �?       @      @              �?       @      @               @      @       @              @      �?      �?      @              �?      �?        
�
conv3/weights_1*�	    O���    �h�?      �@! ������?)�S��QU@2�����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(龢f�����uE���⾮��%ᾙѩ�-߾.��fc��>39W$:��>jqs&\��>��~]�[�>��>M|K�>E��a�W�>�ѩ�-�>�uE����>�f����>��(���>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?�������:�             �@     R�@     ֧@     �@     z�@     ʡ@     ��@     ��@     ��@     ��@     ��@     ��@     (�@     ��@     (�@     Њ@     �@     ��@     ��@     0�@     ؁@     �{@     `{@     �y@     `w@     `u@     �s@      r@     �p@     �n@     �g@     �g@     @e@     �d@     �b@     ``@     ``@     �Z@      Z@     �Y@     �S@     �P@     @P@      O@      I@     �H@      K@     �D@      G@      @@      >@      >@      5@      ;@      6@      0@      3@      .@      2@      0@      @      (@      0@      @      @      $@      @       @      @      @      @      @      @      @              @      @      �?       @      @      @      �?               @      �?      @      �?      @      �?      �?               @              �?              �?               @      �?               @              �?      �?              @       @              �?               @      �?       @       @      @       @      @      @      @      �?      @      @      @      @      &@      @      @      @       @      @      $@      (@       @       @      0@      ,@      ,@      2@      3@      7@      :@      :@     �B@     �A@      >@     �I@     �A@      N@     �G@     �J@     �R@     �T@      P@     �Z@     @X@     @Z@      ^@     ``@     @b@      g@     �g@     @j@     �k@      l@      l@     Pp@     �s@     �r@     v@     `w@     z@     �@     ��@     ��@     ��@     Ȇ@     ��@     `�@     Ѝ@     ��@     ��@     ܓ@     ܕ@     Ș@     �@     ܜ@     ܞ@     6�@     ��@     ��@     2�@     ��@     ��@        
�

conv3/biases_1*�
	    ��u�   ��y?      `@!  �����?)r��/�E?2�&b՞
�u�hyO�s�uWy��r��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+��[^:��"��S�F !�ji6�9���T7����5�i}1���d�r���[���FF�G �O�ʗ�����Zr[v���_�T�l׾��>M|Kվ        �-���q=��d�r?�5�i}1?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?�������:�              �?      �?               @       @       @      @      @      @       @      @      �?      �?      �?      @              @      @               @              �?               @              �?      �?      �?      �?              �?      �?              �?      �?              �?              �?               @              @              �?              �?       @      �?              �?              �?      �?               @              �?      �?       @      @      @      �?      @      @      �?      @      @      @      @      �?      "@      @      �?       @      @              �?       @      �?        
�
conv4/weights_1*�	   @�1��   �<E�?      �@! �����?)�V��e@2�Ӗ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���})�l a��ߊ4F����(��澢f����jqs&\�ѾK+�E��Ͼ�u��gr��R%������E'�/��x��i����v��*��ڽ>�[�=�k�>pz�w�7�>I��P=�>1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�������:�             @[@     ��@     �@     ��@     <�@     ��@     �@     H�@     P�@      �@     x�@     ��@     @}@     `{@     @x@     �x@     �u@     0v@     �q@     �p@     `n@     @k@     `d@      f@      g@     �b@     �\@     �^@      Z@     @V@     @S@     �U@     �O@      O@     �L@     �L@      E@      F@     �D@      M@      A@      >@      :@      7@      8@      5@      2@      7@      2@      *@      3@      ,@      @      @      "@      "@      @      ,@              @      @      @      @      @      @      @      �?       @       @      �?      @      �?       @      �?      �?      @      �?      �?      @      �?              �?       @       @              �?      �?              �?              �?              �?              �?              �?              �?              �?              �?              �?       @      �?              �?              �?              �?              �?       @       @       @       @      @       @      �?       @      @      @       @      @       @       @      @      "@      @       @      $@      &@      @      $@      &@      "@      &@      6@      1@      *@      8@      7@      <@      7@      =@      9@      H@      E@      F@     �H@     �L@     @R@      P@     @S@     �R@     �U@     �Y@     �\@      `@     �_@     �`@     �`@     @e@     @g@     �i@     @j@     @p@     q@     ps@     �u@     �v@     �w@     �x@     �|@     �~@     8�@     �@     `�@     ��@     Ȋ@     ��@     `�@     <�@      �@     ��@     �@      d@        
�
conv4/biases_1*�	   `�s�    $,{?      p@!t즀�ɹ?)�Q���FC?2�hyO�s�uWy��r�P}���h�Tw��Nof�5Ucv0ed����%��b��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(��[^:��"��S�F !�ji6�9���.���vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
�I��P=��pz�w�7��})�l a��ߊ4F��E��a�Wܾ�iD*L�پ_"s�$1�7'_��+/��'v�V,����<�)�4��evk'���o�kJ%�4�e|�Z#���-�z�!�%�����i
�k���f��p�Łt�=	���R����2!K�R���J��#���j�Z�TA[�����"�RT��+��nx6�X� ��f׽r����tO����f;H�\Q������%���9�e����K��󽉊-��J�'j��p���1���=��]����/�4��ݟ��uy�i@4[���Qu�R"����X>ؽ��
"
ֽ�|86	Խ(�+y�6ҽ;3���н�!p/�^˽�d7���Ƚ        �-���q=K?�\���=�b1��=(�+y�6�=�|86	�=��
"
�=���X>�=H�����=z�����=ݟ��uy�=�/�4��=�9�e��=����%�=�tO���=�f׽r��=nx6�X� >�`��>�mm7&c>y�+pm>RT��+�>���">Z�TA[�>�#���j>�J>2!K�R�>��R���>Łt�=	>��f��p>�i
�k>%���>��-�z�!>4�e|�Z#>4��evk'>���<�)>�'v�V,>7'_��+/>��>M|K�>�_�T�l�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?�������:�              �?               @      @       @              @              @       @      @      �?              �?               @      �?       @      �?              �?      �?              �?      �?              �?              �?              �?               @      �?              �?              �?              �?              �?              �?              �?              @      �?      �?               @      �?      @      @       @       @       @      @      �?       @      @       @              �?              @      �?              �?              �?      �?              �?       @              �?              �?              �?      �?              �?              9@              �?              �?      �?              �?              �?       @              �?              @       @      �?              �?      �?      �?       @      �?      @      @              �?       @              @      �?      @               @               @              �?              �?              �?              �?      �?               @      �?       @      �?              �?               @       @      �?              �?              �?       @               @      �?      �?       @              �?      �?       @              �?       @      �?      �?      �?      �?      �?              @              @      �?      @      @      @      �?      @      @       @      @      �?      @       @      �?              @      �?      �?       @      �?        
�
conv5/weights_1*�	   ���¿   ����?      �@!  Ǚ���)�H�\I@2�	�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82�I�I�)�(�+A�F�&��vV�R9��T7�����d�r�x?�x����[���FF�G �x?�x�?��d�r?�5�i}1?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?�������:�	              k@     �t@     �p@     �n@     �n@     �l@      f@      f@      d@     �`@     @]@      \@     �Y@     �Z@     �V@     @R@     @R@      R@     �I@      J@     �J@     �C@      F@     �@@      @@      4@      9@      =@      2@      6@      ?@      &@      1@       @      3@      .@      @      "@       @      @      @      @       @      @      @      @      @      @      @      @       @      @      �?      �?      @      @      @      @       @      @               @      �?              �?      �?              �?              �?              �?              �?              �?      �?              �?      �?               @              @      �?      �?      �?       @      @      @               @      �?              �?              �?      �?      @      @      @       @       @      @      @      @      @      @      "@      "@      @      ,@       @      @      .@      (@      (@      .@      9@      4@      ;@      ,@      3@      :@      ?@      ;@      D@     �@@     �D@      C@      L@      I@     �R@     �U@     �P@     �P@      W@      W@     �[@     �Z@     �]@      d@     @b@     �d@     `g@     @l@      l@     �p@     Pr@     r@     �j@        
�
conv5/biases_1*�	   ��6�   ��L0>      <@!   ��H�)ؙ� 0��<2��z��6��so쩾4�6NK��2�_"s�$1�7'_��+/��'v�V,����<�)�4��evk'���o�kJ%���-�z�!�%������f��p�Łt�=	���R����2!K�R���#���j�Z�TA[��'j��p���1���z�����i@4[���Qu�R"�����%�=f;H�\Q�=y�+pm>RT��+�>2!K�R�>��R���>Łt�=	>�i
�k>%���>��-�z�!>4�e|�Z#>��o�kJ%>4��evk'>���<�)>�'v�V,>7'_��+/>_"s�$1>�������:�              �?              �?      �?      �?      �?               @              �?               @      �?       @              �?              �?              �?      �?              �?              �?              �?      �?              �?              �?              �?              �?       @      �?        @%�3HT      b-	�H�1���A*��

step  �A

loss���>
�
conv1/weights_1*�	   �H�    �F�?     `�@!  �a�
@)�Z���@2���]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+��.����ڋ��vV�R9���d�r�x?�x��pz�w�7��})�l a��5�i}1?�T7��?�vV�R9?��ڋ?�.�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?��bȬ�0?��82?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�������:�               @     @R@     `g@     �c@     �e@     `d@     @b@     `a@     �]@      \@     �W@     �T@      Y@      O@      K@     �N@     �L@      I@      J@      F@     �A@      >@      B@      :@      @@      ?@      4@      0@      *@      .@      3@       @       @      2@      *@      (@      @      @      "@      @      "@      @      @      @      @      @      @              �?       @      �?               @              �?      @       @               @       @              �?      �?              �?              �?              �?              �?      �?              �?      �?              �?              �?              �?      �?      �?      �?              �?       @       @      @      @      @      �?      @      @      @      @      �?      @      @      @      "@      (@      ,@       @      *@      8@      1@      *@      9@      4@      7@      =@      <@      C@      @@      >@     �F@      E@     �D@      M@      K@      O@     �M@     �T@     �S@     @T@     @Z@     �a@     �^@      `@     �e@     �g@     `g@     �h@     �V@      1@        
�
conv1/biases_1*�	    ��a�    :�w?      @@!  `*��?)�xՓ?2����%��b��l�P�`�k�1^�sO�IcD���L��T���C��!�A��7Kaa+�I�I�)�(�U�4@@�$��[^:��"��T7����5�i}1������6�]���K+�E��Ͼ['�?�;pz�w�7�>I��P=�>�vV�R9?��ڋ?I�I�)�(?�7Kaa+?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?&b՞
�u?*QH�x?�������:�              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?      �?      �?      �?              �?               @      �?               @               @       @      �?      �?      �?      @              �?        
�
conv2/weights_1*�	   ��*��   ��?      �@!���
��0@)ٰ��:gE@2�I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ뾮��%ᾙѩ�-߾E��a�Wܾjqs&\�ѾK+�E��Ͼ��|�~���MZ��K����n����>�u`P+d�>�*��ڽ>�[�=�k�>
�/eq
�>;�"�q�>K+�E���>jqs&\��>��~]�[�>�uE����>�f����>��(���>a�Ϭ(�>�ߊ4F��>})�l a�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?�������:�              @     ��@     ��@     ��@     М@     ��@     ��@     $�@     �@     ,�@     ��@     p�@     ��@     @�@      �@     �@     �@     ��@      }@     P{@      x@     v@     �u@     0t@      o@      o@      n@      i@     @h@     �d@     �a@     �a@     �`@     �_@     �X@     �X@     �U@     �R@     �S@     @Q@     �P@      G@     �M@      E@      F@      C@      E@     �@@     �B@      A@      6@      4@      2@      (@      1@      1@      &@      &@      (@      &@       @      *@      @      @       @      @      @       @      �?      @      @       @      �?      @              @      @      �?       @               @      �?      �?      �?       @      @              �?              �?       @              �?              �?              �?              �?              �?              �?      �?               @              �?              �?              @      �?      �?      �?              @       @      �?       @       @       @      �?      @      @      @      �?      @      @       @       @      *@      @      ,@      &@      "@      $@       @      &@      1@      (@      1@      8@      2@      6@      :@      A@      B@     �B@     �G@      E@     �O@      Q@      N@      S@      N@     �S@     �X@     �W@     @Z@     �\@     �^@     @]@     �d@     �d@     �i@      k@      i@     �l@     �q@     �r@     `s@     �u@     @z@     �y@     0}@     �@     ��@     Є@      �@     ��@      �@     �@     X�@     T�@     X�@     t�@     h�@     �@     �@     ��@     p�@     ,�@      &@        
�
conv2/biases_1*�	    tv�    ��u?      P@!   ��J�?)\��"�4?2�*QH�x�&b՞
�u�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[�nK���LQ�k�1^�sO�IcD���L��qU���I�a�$��{E��T���C�uܬ�@8���%�V6��u�w74�I�I�)�(�+A�F�&�x?�x��>h�'��6�]���1��a˲�pz�w�7�>I��P=�>U�4@@�$?+A�F�&?I�I�)�(?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?d�\D�X=?���#@?�T���C?a�$��{E?
����G?�qU���I?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?�������:�              �?              �?      �?      �?       @      �?              �?              �?      �?       @              �?              �?      �?              �?              �?              �?              �?              �?       @              �?      �?      �?      �?              �?              �?      �?       @               @      @      �?              �?       @      �?      �?      @      @      @              @       @      �?       @               @        
�
conv3/weights_1*�	   ��ᮿ   �Y��?      �@! ��A=@)O�BOSU@2�����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾��~]�[Ӿjqs&\�Ѿ
�/eq
Ⱦ����žG&�$��5�"�g����MZ��K�>��|�~�>G&�$�>�*��ڽ>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?�������:�             �@     L�@     �@     ڤ@     ��@     ��@     ��@     ,�@     l�@     �@     ��@     ��@     �@     x�@     ��@     ��@     ؇@     �@     `�@     p�@     X�@     `{@     0|@     �y@     �w@     �t@      t@     �q@     �p@     �m@     �h@     @f@     �g@     �b@      d@      ]@     @a@     �Y@     @[@      Y@     @R@     �N@     @R@     @Q@     �F@     �J@     �H@     �C@     �F@      ?@      ;@      ?@      :@      9@      1@      ;@      .@      6@      3@      &@      $@      2@      *@      "@      (@      (@      @      @       @       @      @      �?      @      �?      @       @      �?      @      @      �?      �?      �?               @      �?      �?       @      �?              �?      �?              �?      �?      �?              �?              �?              �?              �?              �?              �?              �?               @      �?      �?      �?              �?      �?              �?      �?      @      @              �?      @      �?      @      @      @      �?      �?      @      @      @      @      @      $@      @       @      @      (@      "@       @      0@      2@      0@      .@      2@      8@      <@      :@      B@      9@      ?@     �H@     �E@      N@      F@      K@     @R@     @S@     �S@     @V@     @Z@     �Y@     �]@     �`@     �b@     �f@     �g@     `i@     �l@     `k@     �l@      p@     �s@     �s@     @v@     �v@     Pz@      @     ��@     ��@     ��@     ��@     ��@     ��@     P�@     ��@     ��@     ��@     �@     ��@     ,�@     ��@     �@     (�@     ��@     �@     R�@     ^�@     H�@        
�
conv3/biases_1*�	   ���w�   �.�z?      `@! ����ϰ?)ۍ~h�I?2�*QH�x�&b՞
�u�hyO�s�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0�I�I�)�(�+A�F�&��vV�R9��T7����5�i}1���d�r������6�]���1��a˲���[��>�?�s���O�ʗ����iD*L�پ�_�T�l׾        �-���q=�f����>��(���>��d�r?�5�i}1?U�4@@�$?+A�F�&?�7Kaa+?��VlQ.?��bȬ�0?��82?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?�������:�              �?      �?               @       @       @       @      @      @      @       @       @       @       @               @       @       @      @      �?       @              �?              �?              �?              �?      �?              �?              �?              �?               @              �?              �?              �?              @              �?              �?              �?              �?      �?      �?              �?      �?       @      �?              �?              �?      @      @       @       @      @       @      �?       @      @      @      @      @      @      �?      �?       @      �?       @      �?      �?        
�
conv4/weights_1*�	   ��;��    �T�?      �@! ����K�?)%���2�e@2�Ӗ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7�����d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�a�Ϭ(���(��澢f����jqs&\�ѾK+�E��ϾI��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?>h�'�?x?�x�?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�������:�             �[@     ��@     �@     ��@     H�@     ��@     ؍@     `�@     X�@     ��@     ��@     ��@      }@     �{@     Px@     �x@     �u@     v@     �q@     �p@     �n@     `k@     @d@     �e@      g@     `b@      _@     �]@     �Z@      V@     �S@      U@      O@     �O@     �M@      K@      F@     �E@      D@     �M@     �B@      ;@      :@      8@      8@      6@      0@      8@      2@      $@      2@      .@       @      "@      @      @      @      $@      @       @      @      @      @      @      @      �?       @      @       @       @       @      �?       @      �?               @       @      �?               @              �?      �?               @      �?      �?               @              �?       @              �?              �?      �?              �?              �?              �?      �?              �?      �?      �?              �?              �?       @      �?       @      �?      �?      @      �?      �?       @      �?       @      @      @      @      @      @      @      @      @      @      @      0@       @      @      &@      @      $@      .@      6@      .@      *@      9@      3@      ?@      :@      ;@      8@      H@     �D@     �D@      K@      L@     �Q@     �P@      S@     �R@     �V@     �Y@     �[@     �`@      _@     `a@     �`@     @d@     @h@     �i@     �i@     pp@      q@     �s@     �u@     �v@     �w@     px@     p|@     �~@     @�@      �@     X�@     ��@     �@     p�@     ��@     8�@      �@     ��@     �@     @d@        
�
conv4/biases_1*�	   �Y�t�    B]}?      p@!2<���?)��\��F?2�&b՞
�u�hyO�s�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&��[^:��"��S�F !�ji6�9����ڋ��vV�R9��T7����5�i}1���d�r�pz�w�7��})�l a�ѩ�-߾E��a�Wܾ_"s�$1�7'_��+/��'v�V,����<�)�4��evk'���o�kJ%�4�e|�Z#���-�z�!�%�����i
�k���f��p�Łt�=	���R����2!K�R���J��#���j�Z�TA[�����"�RT��+���`���nx6�X� ��f׽r����tO����f;H�\Q���9�e����K��󽉊-��J�'j��p���1���=��]����/�4��ݟ��uy�z������b1�ĽK?�\��½�Į#�������/��        �-���q=�EDPq�=����/�=�d7����=�!p/�^�=��.4N�=�/�4��==��]���='j��p�=��-��J�=�K���=�9�e��=����%�=f;H�\Q�=nx6�X� >�`��>�mm7&c>y�+pm>RT��+�>���">Z�TA[�>�#���j>�J>2!K�R�>��R���>Łt�=	>��f��p>�i
�k>%���>��-�z�!>4�e|�Z#>��o�kJ%>���<�)>�'v�V,>7'_��+/>_"s�$1>�_�T�l�>�iD*L��>�f����>��(���>a�Ϭ(�>�h���`�>�ߊ4F��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?�������:�              �?               @       @      @              �?              @              @       @      �?      @               @              �?      �?      �?              �?               @              �?              �?              �?              �?              �?      �?               @      �?      �?      �?               @              �?              �?      @      �?      �?              �?      @      @      @      @      �?      �?      @               @      @      @      �?              �?      �?              �?              @               @              �?      @      �?      �?              �?              �?              9@              �?              �?      �?              �?              �?              �?      �?       @               @      �?       @      �?       @      �?      �?       @      @       @      �?               @      �?       @      �?      @              @              �?              �?              �?      �?              �?              �?      �?       @      �?      �?       @              �?      �?               @      �?      �?              �?              �?      �?      �?       @      �?       @              �?      �?      �?              �?      @      �?      �?      �?              �?      �?      �?      @       @      �?      @       @       @      @      @      @      @       @      @       @       @      �?              @      �?       @      �?      �?        
�
conv5/weights_1*�	   ���¿   ���?      �@! �5Ad�)��Kz_I@2�	�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���VlQ.��7Kaa+�U�4@@�$��[^:��"��.����ڋ�x?�x��>h�'�������>
�/eq
�>��(���>a�Ϭ(�>I��P=�>��Zr[v�>��d�r?�5�i}1?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?a�$��{E?
����G?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?�������:�	              k@     �t@     �p@      o@     �n@     @l@     �f@     �e@      d@     �`@     �\@      \@      Z@      Z@     �V@     �R@      R@     �Q@      J@      K@     �I@      C@      G@      @@     �@@      3@      :@      =@      2@      6@      @@      &@      0@       @      3@      .@      @      $@      @      "@      @      @      @       @      @      @      @      @      @      @       @       @      �?      �?      @       @       @       @      @      @              @       @      �?      �?      �?              �?              �?               @              �?              �?              �?              �?               @              �?       @      @               @      @      �?      @      �?      �?              �?               @      @      @      @      @       @       @      @      @      @       @      @      "@      @      &@      $@       @      ,@      &@      *@      .@      8@      6@      :@      .@      2@      :@      @@      ;@     �C@      A@     �C@      C@      L@      I@     �R@     �U@     �P@     �P@     @W@     �W@     @Z@     �[@     �]@     �c@     �b@     �d@      g@     @l@      l@     �p@     `r@     �q@     �j@        
�
conv5/biases_1*�	   ��8�   ���1>      <@!   xW�J�)F��ڭ<2�u 5�9��z��6��so쩾4�6NK��2�_"s�$1�7'_��+/��'v�V,����<�)�4��evk'���o�kJ%�4�e|�Z#�%�����i
�k���f��p�Łt�=	���R����2!K�R���J��#���j��9�e����K�����1���=��]����|86	�=��
"
�=�f׽r��=nx6�X� >Z�TA[�>�#���j>��R���>Łt�=	>��f��p>��-�z�!>4�e|�Z#>��o�kJ%>4��evk'>�'v�V,>7'_��+/>_"s�$1>6NK��2>�������:�              �?              �?      �?      �?      �?               @              �?              �?               @              �?      �?      �?              �?              �?              �?              �?              �?              �?      �?               @              �?               @      �?      �?        ɩH.HT      b-	�2���A*��

step  �A

loss�>
�
conv1/weights_1*�	   ��E��   ��?     `�@!  uA��@)���q�@2���]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�U�4@@�$��[^:��"�f�ʜ�7
������;�"�q�>['�?��>��Zr[v�>O�ʗ��>1��a˲?6�]��?>h�'�?x?�x�?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�������:�              @      S@     @f@     �c@     �e@     @d@     �b@     @b@     �[@      ]@      W@      U@     @W@      Q@      K@     �L@      O@     �H@      H@     �G@      B@      >@     �A@      ?@      ?@      =@      1@      .@      0@      ,@      ,@      &@       @      2@      *@      &@      $@      @      "@      @      @      @      @      @       @      @      @              @      @       @      �?      @              �?      �?      @              �?       @              �?              �?              �?              �?              �?              �?              �?              �?      �?              �?              �?              �?      @      �?              @       @              �?      @      @       @      @      @      �?      �?      @       @      @       @      @      &@      @       @      $@      *@      ,@      ,@      *@      2@      2@      5@      >@      6@      7@     �A@      ?@      B@      ?@     �A@     �E@     �H@     �K@      K@      O@     �O@      R@      V@     �S@      \@     �`@     �\@      a@     �d@     �g@      h@     �h@     �V@      ;@        
�
conv1/biases_1*�	    ~�a�   `�Mz?      @@!  ���?)�07+�"?2����%��b��l�P�`�IcD���L��qU���I��T���C��!�A��[^:��"��S�F !��5�i}1���d�r��FF�G �>�?�s���I��P=�>��Zr[v�>�FF�G ?��[�?��ڋ?�.�?ji6�9�?�S�F !?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?*QH�x?o��5sz?�������:�              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?               @               @              �?               @      �?      �?       @               @       @               @      �?      @              �?        
�
conv2/weights_1*�	   �Wv��   �v�?      �@! ?%
n�1@)8���jE@2�I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�a�Ϭ(���(��澮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�Ѿ;�"�qʾ
�/eq
Ⱦ�5�L�����]����BvŐ�r�ہkVl�p��u��gr�>�MZ��K�>�[�=�k�>��~���>jqs&\��>��~]�[�>E��a�W�>�ѩ�-�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?�������:�              @     ��@     ��@     ��@     Ԝ@     T�@     ��@     D�@     $�@      �@     ��@     @�@     ȉ@     P�@     `�@     ��@     ��@     ؀@     `}@     �{@     �x@     0u@      u@     t@     �o@      p@     `n@     �h@     @g@     �f@     @b@     �^@     @a@     @_@      [@     �T@     @Y@      R@      T@     �P@     @P@     �N@      I@     �E@      E@      E@      B@      :@      A@      C@      8@      1@      ,@      (@      0@      4@      .@      &@      @      &@      &@      &@      @      $@      @      @      @      @      @      @       @      @      @      �?      �?      @      @       @              �?      @      @      �?       @              �?      �?              �?              �?      �?              �?      �?               @              �?              �?              �?              �?              �?              �?              �?              �?      �?      �?      �?              �?      �?      �?      @              �?       @      �?      �?      @      �?      @      @      �?      "@      @       @      @      �?      @      &@      @      @       @      0@      $@      1@      .@      (@      1@      4@      3@      .@      3@     �B@      9@     �D@      @@     �I@     �G@     �E@     �R@     @P@      S@      O@     @T@     @V@     �W@      W@      ^@     �`@     @_@      d@      e@     @h@     �l@     �h@      n@     0q@     �r@     �s@      v@     py@     �z@     �|@     0�@     ��@     Ȅ@     �@     X�@     ȉ@     ��@     ,�@     x�@     ,�@     |�@     ��@     �@     Ԝ@     ��@     z�@     X�@      1@        
�
conv2/biases_1*�	   @vLw�   `*w?      P@!  �S)��?)ڍ((z8?2�*QH�x�&b՞
�u�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L�
����G�a�$��{E����#@�d�\D�X=���%>��:�uܬ�@8���ڋ��vV�R9���[���FF�G �O�ʗ�����Zr[v���ѩ�-�>���%�>ji6�9�?�S�F !?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?d�\D�X=?���#@?�!�A?�T���C?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?�������:�              �?               @      �?              @              �?              �?              �?      �?               @              �?              �?              �?              �?              �?              �?              �?              �?               @      �?      �?              �?              �?              �?      �?      @      �?      @      @               @      �?              �?      @      @       @      �?      @      �?      �?       @               @        
�
conv3/weights_1*�	   `���   �(̯?      �@! �����@)�{�UU@2�����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[��>�?�s���O�ʗ���I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ뾢f�����uE���⾮��%ᾧ5�L�����]�����_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>�uE����>�f����>��(���>a�Ϭ(�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?�������:�             (�@     H�@     ԧ@     Ƥ@     ��@     ơ@     ğ@     L�@     T�@     �@     ԕ@     ��@     ԑ@     ȏ@     H�@     Њ@     �@     8�@      �@     0�@     ��@      |@     �{@     �x@     `w@     @u@     �s@     0r@     �p@     �m@     �j@     @e@     �e@      e@      c@     @\@      a@     @\@     @X@     �W@     @R@      R@     �Q@      Q@     �L@     �I@     �C@     �C@      B@     �B@     �C@      8@      ;@      ?@      0@      <@      1@      .@      *@      (@      .@      *@      5@      @      ,@      @      @      @      @       @       @      @      @      @      @      @       @       @      �?              @              @               @       @      �?               @      �?              �?              �?              �?              �?              �?               @              �?              @              @              @      @       @      @      @              �?      @      @      @      @      @      @      @      @      @      @      (@      $@      $@      *@      0@      .@      4@      7@      8@      <@      :@      :@     �A@      ;@     �F@      D@      H@      L@     �N@     @Q@     �Q@      W@      V@     �W@      Y@     �`@     ``@     �b@     �g@     �e@      j@     `n@     �i@     �l@     �o@     �s@     �s@     @v@     �v@     �z@     @@     ��@     �@     p�@     ��@     P�@     ��@     ��@     �@     ��@     @�@     ��@     ��@     4�@     ��@     <�@     �@     ��@     �@     D�@     r�@     ��@        
�
conv3/biases_1*�	   ���y�   �}�|?      `@!  ��M%�?)<�u�PN?2�o��5sz�*QH�x�&b՞
�u�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.�U�4@@�$��[^:��"������6�]����FF�G �>�?�s���E��a�Wܾ�iD*L�پ5�"�g���0�6�/n��        �-���q=5�"�g��>G&�$�>})�l a�>pz�w�7�>�.�?ji6�9�?�S�F !?�[^:��"?I�I�)�(?�7Kaa+?��VlQ.?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?�!�A?�T���C?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?�������:�              �?      �?              �?      @       @       @       @      @       @       @       @       @       @      �?      �?      �?       @      @       @              �?              �?              @              �?              �?              �?              �?              �?              �?              �?              �?               @              �?              �?              �?              �?              @      �?              �?      �?       @      �?               @              �?      �?              @      @       @      @      @       @               @      @      @      @      @       @      �?      �?       @      �?       @       @        
�
conv4/weights_1*�	   ��E��   `e�?      �@! PX&���?)�����e@2�Ӗ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9�>h�'��f�ʜ�7
������1��a˲���[���FF�G ���Zr[v��I��P=���ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澮��%ᾙѩ�-߾��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;;�"�qʾ�ߊ4F��>})�l a�>pz�w�7�>��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�������:�              \@     �@     �@     ��@     @�@     ��@     ��@     P�@     x�@     ��@     ��@     ��@      }@     �{@      x@     �x@     �u@     �u@     �q@     @q@     `n@     �k@     @d@      f@     `f@      c@      ^@     @^@     @Y@      W@     �S@      U@     �P@      M@     �M@      L@      E@      H@      A@     �M@     �B@      >@      ;@      5@      ;@      6@      2@      6@      1@      @      4@      @      *@      (@      @      @       @      "@       @      @      �?      @      @      @      @      @               @      @              @              @      �?      �?       @      �?              �?      �?               @      �?              �?              �?      �?              �?              �?              �?      �?              �?              �?       @              �?      �?      �?      �?      �?      �?              �?      �?      �?      �?       @      �?      @      �?      �?       @      �?       @      �?       @      @      @       @      @      @      @       @      @      (@      $@      "@      "@      @       @      (@      ;@      (@      3@      5@      2@      A@      ;@      ;@      5@      F@     �E@      D@     �K@     �I@     �R@     �Q@      T@     @Q@     @V@     �Z@     �Z@      a@     �]@      a@     �a@      d@     �g@      j@      j@     �p@     �p@     �s@     �u@     pv@     x@     Px@     P|@     �~@     8�@     (�@     @�@     x�@     ��@     p�@     ��@     8�@     ��@     ��@     �@     `d@        
�
conv4/biases_1*�	   �WSu�   ���?      p@!jBȊ6�?)ޕ�╹J?2�&b՞
�u�hyO�s��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !���ڋ��vV�R9��T7����5�i}1���d�r���[���FF�G �I��P=��pz�w�7��})�l a��ߊ4F��ѩ�-߾E��a�Wܾ['�?�;;�"�qʾ6NK��2�_"s�$1�7'_��+/��'v�V,����<�)�4��evk'���o�kJ%�4�e|�Z#���-�z�!�%�����i
�k���f��p�Łt�=	���R����2!K�R���J��#���j�Z�TA[�����"�RT��+��y�+pm��mm7&c��`���nx6�X� ��f׽r����tO����f;H�\Q���9�e����K��󽉊-��J�'j��p���1���=��]����/�4���Qu�R"�PæҭUݽ��
"
ֽ�|86	Խ(�+y�6ҽ��؜�ƽ�b1�Ľ        �-���q=G�L��=5%���=��.4N�=;3����=��-��J�=�K���=�9�e��=����%�=f;H�\Q�=�tO���=�f׽r��=nx6�X� >�`��>y�+pm>RT��+�>���">Z�TA[�>�#���j>�J>2!K�R�>��R���>Łt�=	>��f��p>%���>��-�z�!>4�e|�Z#>��o�kJ%>4��evk'>���<�)>�'v�V,>7'_��+/>_"s�$1>�_�T�l�>�iD*L��>��(���>a�Ϭ(�>�h���`�>�ߊ4F��>})�l a�>��[�?1��a˲?6�]��?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?�������:�              �?              @      �?       @      �?              �?      �?       @      �?       @       @       @      �?      �?              �?      �?      �?              �?              �?       @      �?               @      �?              �?              �?      �?              �?      �?              �?              �?              �?              �?              �?              �?              �?      @      �?      �?              �?      �?      @      @      @       @              @      �?      �?       @       @      @      @              �?              �?              �?      �?              �?       @      �?      �?       @      �?              �?              �?      �?              �?              8@              �?               @              �?       @              �?              �?      �?       @               @      �?       @      �?       @      @       @       @      �?              @      �?       @      @               @      �?      �?              �?               @              �?      �?              �?      @              �?               @      �?      �?       @      �?              �?       @      �?              @              �?              �?      �?      �?      �?              �?      �?              @      �?      �?       @      �?              �?      @      �?      @      @      �?      @      @      @      @      @      �?      @      @       @      �?       @      @      �?       @       @        
�
conv5/weights_1*�	   �^�¿   `���?      �@! �t�jl�)r�nM�aI@2�	�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1�x?�x��>h�'��f�ʜ�7
���Zr[v�>O�ʗ��>��d�r?�5�i}1?�T7��?ji6�9�?�S�F !?+A�F�&?I�I�)�(?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?�������:�	              k@      u@     �p@     �n@     �n@      l@     �f@     �e@     �c@     �`@     �\@     �\@      Y@      [@     @V@     @S@     �Q@     �Q@      J@     �J@     �I@      C@     �G@     �@@      ?@      4@      9@      >@      2@      8@      >@      &@      0@      $@      .@      .@      @      "@      @       @       @      @      @      @      @       @      @      @      @      @      @              �?       @       @      �?      �?      @      @       @              �?      �?      �?      @       @      �?              �?      �?              �?              �?              �?              �?      �?              �?              �?      �?              �?               @              @      �?       @      �?      @       @              @              �?      �?               @      �?       @      @      �?      @       @      �?      @      @       @      @      @      $@      @      $@      $@      $@      $@      *@      .@      1@      4@      7@      8@      0@      3@      9@      =@      >@     �C@     �A@     �B@      C@      M@     �H@     �R@     �U@     �P@     �P@     �W@     �W@     �Z@      [@      ^@      d@     `b@     @d@     @g@     �l@      l@     pp@     pr@      r@     �j@        
�
conv5/biases_1*�	   �RA:�    j3>      <@!   ]�|L�)�\�!:��<2�p
T~�;�u 5�9��z��6��so쩾4�6NK��2�_"s�$1�7'_��+/��'v�V,����<�)�4��evk'���-�z�!�%�����i
�k���f��p���R����2!K�R���J��#���j�����%���9�e�����-��J�'j��pｉ�-��J�=�K���=���">Z�TA[�>�J>2!K�R�>Łt�=	>��f��p>�i
�k>4�e|�Z#>��o�kJ%>���<�)>�'v�V,>7'_��+/>_"s�$1>6NK��2>�so쩾4>�������:�              �?              �?      �?      �?      �?               @      �?              �?      �?       @              �?              �?              �?              �?              �?              �?              �?              �?      �?               @              �?               @      �?      �?        ���P�S      �E�+	��12���A*��

step  �A

loss���>
�
conv1/weights_1*�	    ���   �}��?     `�@! �ڛ:\@)�2�+��@2�	��]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���VlQ.��7Kaa+�I�I�)�(�+A�F�&��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1�f�ʜ�7
�������ߊ4F��h���`�x?�x�?��d�r?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?�������:�	              @     �S@     `e@     �c@      e@     �d@     �a@     �a@      \@     �\@     @X@     @V@     �S@      S@      O@     �J@      O@     �H@     �F@     �F@      C@      B@     �@@      ;@     �B@      =@      2@       @      .@      0@      0@      ,@      &@      (@      "@      (@      @      "@      @      @      $@       @      @      @      @      @      @      @      @       @       @              �?      @      �?       @      �?               @               @              �?              �?              �?      �?              �?               @              �?              �?      �?      �?      �?              �?              �?      �?      �?      �?       @       @      �?       @       @       @      @       @      @      �?              @      @      @      @      @      @      @      @      @      $@      (@      ,@      ,@      0@      .@      4@      2@      ?@      4@      ?@      9@     �D@     �@@      ?@     �B@     �A@      I@      K@     �N@      N@     �Q@     @P@     �S@     @V@      [@     @`@     �^@     �`@     �d@     �g@     @h@     �g@      Y@      =@      �?        
�
conv1/biases_1*�	   @=�a�   @&}?      @@!   ��E�?)H3�hR�&?2����%��b��l�P�`��qU���I�
����G��!�A����#@��vV�R9��T7����FF�G �>�?�s����f�����uE������Zr[v�>O�ʗ��>�.�?ji6�9�?�7Kaa+?��VlQ.?��bȬ�0?���#@?�!�A?�T���C?�qU���I?IcD���L?k�1^�sO?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?o��5sz?���T}?�������:�              �?              �?              �?              �?              �?              �?              �?              �?              �?      �?               @      �?              �?      �?              @              �?       @              �?      @               @      �?      �?       @              �?        
�
conv2/weights_1*�	   �ȫ�    ��?      �@! ���H3@)��Y�oE@2�I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ뾮��%ᾙѩ�-߾E��a�Wܾ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;;�"�qʾR%�����>�u��gr�>�XQ��>�����>
�/eq
�>K+�E���>jqs&\��>��~]�[�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>a�Ϭ(�>8K�ߝ�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?�������:�              @     ��@     �@     ��@     Ȝ@     (�@     ȗ@     @�@     ,�@      �@     ��@     P�@     ؉@     ȇ@     ��@      �@     p�@      �@     �}@     @{@     �x@     �u@     `t@     �s@     �o@     p@      o@     @h@     �f@     @g@     `c@     �^@     �`@      _@     @[@     �Y@     �S@     @R@     �R@     �R@      R@      O@      J@      A@     �H@      C@     �A@      ;@      D@      =@      4@      2@      ,@      &@      4@      .@      (@      @      "@      *@      &@      *@      @      @      @      @      @      @      @      @      @       @      �?      @              @      �?      @      @      @      @      �?       @       @              �?      �?      �?              �?      @              �?      �?              �?              �?              �?      �?              �?      �?              �?              �?      �?              @              �?      �?      �?      �?      �?      �?       @      @      @      @      @       @       @      @      �?      @      @      @       @      @      @      @      @      @      ,@       @      ,@      ,@      4@       @      6@      2@      2@      0@      8@      ;@     �A@     �C@      C@     �H@      E@     �E@      O@     �P@     @S@     �P@     �U@      U@     �T@     �[@      Y@     �b@     @`@     @b@     �f@     �f@      k@     �j@      o@      q@     �r@     ps@     @v@     �y@      z@     0|@     P�@     Ȃ@     x�@     (�@     ��@     ��@     ��@     �@     X�@     �@     ��@     ��@     �@     �@     ��@     x�@     `�@      :@        
�
conv2/biases_1*�	    &Gx�   `�x?      P@!  �#q�?)���j<?2�o��5sz�*QH�x��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^���bB�SY�ܗ�SsW�nK���LQ�k�1^�sO�
����G�a�$��{E��T���C��!�A���bȬ�0���VlQ.�U�4@@�$��[^:��"�>h�'�?x?�x�?�[^:��"?U�4@@�$?�7Kaa+?��VlQ.?��bȬ�0?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?�������:�              �?              �?      �?      �?       @              �?      �?              �?               @              �?      �?       @              �?              �?               @              �?              �?      �?               @              �?              �?              �?              �?      �?      �?      @       @       @       @      �?       @              �?       @      @      @      �?      @              �?       @               @        
�
conv3/weights_1*�	   @�I��   �& �?      �@! �����@)�a��VU@2�����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%�E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ5�"�g��>G&�$�>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?�������:�             x�@     8�@     ܧ@     ��@     ��@     �@     �@     �@     ��@     ��@     ��@     |�@     ��@      �@     �@     0�@     �@     �@     ȃ@     X�@     p�@     0|@     �{@     �x@     �w@     u@     �s@     �q@     q@     �m@     `j@     �e@     �d@     @f@     @b@     �^@     �_@     �\@     @[@     @W@     �Q@      Q@      R@      O@      L@     �J@     �I@      C@      =@     �C@      C@      :@      9@      ?@      9@      2@      6@      2@      *@       @      0@      "@      1@       @      *@       @      @       @       @       @      @      "@      @      �?       @      @      �?      @      �?      �?       @      �?              @      �?       @       @      �?              �?               @       @       @              �?              �?              �?              �?      �?       @       @               @               @               @      �?              @              �?       @      �?      @      @       @      @      @      �?      @      @      @      @      @      @      $@       @      $@       @      0@      *@      .@      1@      1@      <@      8@      <@      ;@      <@      <@      @@      D@      C@      J@     �E@      L@     @S@     @U@      U@     �T@     �Y@      X@     @`@     �`@     `d@     �e@      d@     �l@     @m@     `i@      m@     �o@     ps@     @t@     @v@     pv@     Pz@     �@     p�@     (�@     `�@     ��@     @�@     x�@     �@     T�@     ��@     �@     ܕ@     ��@     (�@     ��@     �@     "�@     ��@     �@     6�@     N�@     p�@        
�
conv3/biases_1*�	   ���{�   @��~?      `@! �y{A��?)"\|��Q?2����T}�o��5sz�*QH�x�&b՞
�u�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%�V6��u�w74���VlQ.��7Kaa+��FF�G �>�?�s����ѩ�-߾E��a�Wܾjqs&\�ѾK+�E��Ͼ        �-���q=f^��`{>�����~>jqs&\��>��~]�[�>��d�r?�5�i}1?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?k�1^�sO?nK���LQ?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?�������:�              �?              �?              @      �?      @       @      @      @       @       @       @      @              �?      @              @       @      �?      �?      �?               @              �?              �?              �?              �?              �?              �?               @              �?              �?              �?              �?              �?              �?              @              �?       @      �?              �?               @               @              @       @      @      @       @       @       @       @      @      @      @      @      "@               @      �?      �?       @       @        
�
conv4/weights_1*�	   `�O��   �Pw�?      �@! p�+��?):Kʈ��e@2�Ӗ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7���>h�'��f�ʜ�7
������1��a˲���[���FF�G �>�?�s���pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(���jqs&\�ѾK+�E��Ͼ})�l a�>pz�w�7�>I��P=�>1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�������:�             �\@     ԗ@     �@     ��@     \�@      �@     ��@     H�@     ��@     ��@     ��@     ��@     0}@     �{@      x@     �x@     �u@     �u@     �q@      q@     �n@      k@     �c@     �f@      f@     @c@     �]@     �^@     @Y@     �V@     �S@     @T@     @Q@      L@     �M@      O@     �C@     �E@     �A@     �M@      E@      :@      :@      8@      ;@      5@      3@      5@      4@      @      0@      $@      "@      (@      "@       @      @      &@       @      @      @      @      @       @      @       @      @      �?       @      @      �?       @       @      @      �?      �?       @      �?              �?       @              �?      �?      �?              �?              �?      �?              �?              �?              �?      �?              �?       @               @      �?      �?      �?              �?       @              �?      �?      �?      @       @      �?      @       @       @       @       @      @      @      @      @      &@      @      @      ,@       @      @      "@       @      @      .@      7@      1@      2@      3@      7@      >@      =@      ;@      4@      F@      D@      F@     �D@     �M@      S@     �R@     @S@     �Q@     @V@      Z@     �[@     ``@     @^@     `a@     �`@     �d@     �g@      j@     �i@     �p@     Pp@     �s@     �u@     �v@     �w@     px@     0|@     @     �@     8�@     8�@     x�@     ��@     x�@     ��@     @�@      �@     ��@     �@     �d@        
�
conv4/biases_1*�	   `�v�   ��ڀ?      p@!"͍@���?)�ZD�2O?2�*QH�x�&b՞
�u�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:��u�w74���82���bȬ�0�I�I�)�(�+A�F�&�U�4@@�$���ڋ��vV�R9��T7����5�i}1�6�]���1��a˲���Zr[v��I��P=���f�����uE���⾙ѩ�-߾E��a�Wܾ�iD*L�پ�so쩾4�6NK��2�_"s�$1�7'_��+/��'v�V,����<�)�4��evk'���o�kJ%�4�e|�Z#���-�z�!�%�����i
�k���f��p�Łt�=	���R����2!K�R���J��#���j�Z�TA[�����"�RT��+��y�+pm��mm7&c��`���nx6�X� ��f׽r����tO����f;H�\Q������%���9�e����K��󽉊-��J�'j��p�i@4[���Qu�R"���
"
ֽ�|86	Խ(�+y�6ҽ;3���н�
6������Bb�!澽_�H�}��������嚽        �-���q=G�L��=5%���=z�����=ݟ��uy�=�9�e��=����%�=f;H�\Q�=�tO���=�f׽r��=�`��>�mm7&c>y�+pm>RT��+�>���">Z�TA[�>�#���j>�J>2!K�R�>��R���>Łt�=	>��f��p>�i
�k>��-�z�!>4�e|�Z#>��o�kJ%>4��evk'>���<�)>�'v�V,>7'_��+/>_"s�$1>6NK��2>�XQ��>�����>�iD*L��>E��a�W�>��(���>a�Ϭ(�>8K�ߝ�>�ߊ4F��>})�l a�>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?�qU���I?IcD���L?k�1^�sO?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?�������:�              �?               @       @       @               @              �?      �?       @       @      �?       @       @      �?              �?      �?              �?              �?       @      �?               @      �?              �?      �?               @              �?              �?              �?              �?              �?      �?              �?      @      �?      �?              �?      �?      �?      @      @       @      �?       @      �?      �?              @      �?      �?       @      �?       @      @      �?              �?      �?               @      �?       @      �?              �?              �?      �?      �?              �?              �?              8@              �?              �?              �?      �?      �?       @              �?       @               @       @       @      �?       @      �?       @       @       @              @       @      @       @      �?       @              �?              �?              �?              �?      �?              �?              �?              �?       @      �?              �?      �?      �?      �?      �?      �?      �?      �?      �?      �?       @              @      �?      �?      �?              @              �?      �?      @      �?               @      �?               @       @       @       @      @      @      @      �?      @      @      @       @      @       @      @      �?       @       @      �?       @       @        
�
conv5/weights_1*�	   �y�¿   �&��?      �@! �o���)Z��dI@2�	�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+��S�F !�ji6�9���.��x?�x��>h�'����d�r?�5�i}1?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?�������:�	              k@     u@     �p@     �n@      o@     �k@      g@      f@     �c@     �`@     �\@     �\@      Y@     @[@     �U@      T@     @Q@     �Q@      J@      J@     �J@      B@      I@     �@@      ?@      3@      9@      =@      3@      8@      <@      *@      .@      &@      ,@      .@      @      $@      @       @      "@      @      @      @      @      �?       @       @      @      @      @       @       @      �?       @       @              @       @      @      �?              �?      �?      @      �?      �?       @              �?       @              �?              �?              �?              �?      �?      �?               @       @      �?       @              @       @       @       @      �?              �?               @      �?      �?      @      @      @      �?       @       @      @      @       @       @      @      &@      @      &@      $@       @      *@      *@      *@      1@      5@      7@      7@      2@      1@      9@      >@      ?@     �C@     �A@     �@@     �C@     �M@     �J@     �Q@     @V@     �O@      Q@     �W@     �W@     �Z@     �Z@     �]@      d@     �b@      d@     @g@     �l@     @l@     @p@     �r@      r@     �j@        
�
conv5/biases_1*�	   ���<�   �!4>      <@!   ٚ*N�)B� �^�<2�����W_>�p
T~�;�u 5�9��z��6��so쩾4�6NK��2�_"s�$1�7'_��+/��'v�V,����<�)���o�kJ%�4�e|�Z#���-�z�!�%�����i
�k���f��p�Łt�=	���R����Z�TA[�����"�nx6�X� ��f׽r���'j��p���1��콘f׽r��=nx6�X� >�J>2!K�R�>��R���>Łt�=	>��f��p>�i
�k>%���>4�e|�Z#>��o�kJ%>4��evk'>�'v�V,>7'_��+/>_"s�$1>6NK��2>�so쩾4>�������:�              �?              �?      �?      �?              �?       @      �?              �?              �?              �?               @              �?              �?              �?              �?              �?      �?              �?              �?              �?      �?              �?               @       @        ?�7�S      �E�+	@�]2���A*��

step  �A

loss��>
�
conv1/weights_1*�	   �����    B^�?     `�@!  #N�@)!TX�@2��{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74��7Kaa+�I�I�)�(�U�4@@�$��[^:��"��S�F !�ji6�9��x?�x��>h�'��f�ʜ�7
��iD*L��>E��a�W�>1��a˲?6�]��?����?x?�x�?��d�r?�5�i}1?ji6�9�?�S�F !?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?�������:�              �?      "@      T@     �c@     `c@     �d@     `e@     `b@     �a@     �\@     �Y@     @X@     �W@     �S@      S@     @P@     �K@      L@     �G@     �I@      D@      E@     �@@     �@@      B@      ?@      <@      4@       @      *@      4@      2@      $@      "@      *@      &@      &@      *@       @      @      @      @      @      @      �?      �?      @       @      @      @      @       @      @       @       @       @      �?      �?              �?              �?              �?              �?      �?              �?              �?      �?               @      �?              �?              �?      �?      �?               @      �?              �?      @      �?      �?       @      �?      @      @              @      �?      �?      �?      @      $@      @      $@      @      "@      @      "@      (@      $@      "@      0@      4@      0@      ;@      8@      6@      >@      =@     �B@      >@     �B@      D@     �A@     �D@      J@     �P@      N@     @R@     �Q@      Q@     @W@     �Z@      _@     @`@      `@      e@     �g@     `h@     @g@     �Z@     �A@      @        
�
conv1/biases_1*�	   �sJa�   �R�?      @@!  �f��?)�SZ�,?2����%��b��l�P�`��qU���I�
����G��!�A����#@�I��P=��pz�w�7���uE����>�f����>�ߊ4F��>})�l a�>��Zr[v�>O�ʗ��>ji6�9�?�S�F !?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?>	� �?����=��?�������:�              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?               @              �?              �?              �?      @              �?               @               @       @              �?      @               @              �?        
�
conv2/weights_1*�	   �F!��   ��g�?      �@! Y�U`4@)t��B�sE@2�I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`��%ᾙѩ�-߾�iD*L�پ�_�T�l׾5�"�g���0�6�/n������>豪}0ڰ>G&�$�>�*��ڽ>jqs&\��>��~]�[�>���%�>�uE����>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?�������:�              $@     ��@     �@     ��@     ��@     <�@     ȗ@     P�@     �@     ԑ@     �@     p�@     ��@     ��@     ��@     (�@     P�@     H�@     0}@      {@     y@     �u@      t@     t@      p@      n@      o@      i@     `g@     @f@      d@     �_@     �^@     ``@     �[@     �W@     �T@     �P@      S@     @Q@     �Q@     �P@     �I@     �E@     �D@     �D@      ?@      >@      B@      <@      :@      6@      3@      (@      6@      ,@      $@      &@      ,@      (@      "@      "@      @       @      "@      @      @      @      �?      @      @      �?       @       @      @       @      �?      @              �?       @              @       @               @       @               @              �?              �?              �?              �?              �?              �?               @              �?      �?      �?       @      @              �?       @      �?       @       @      @      @              @      �?      @      @      @      @      "@      @       @       @      @      @      (@      $@      ,@      2@      4@      $@      1@      3@      7@      2@      :@      @@     �B@      E@      <@      H@     �A@      J@      M@     �N@     �R@      R@     �R@     �V@     @V@     @Z@     �Z@     ``@     �`@     �d@     @f@     �f@     @i@     @j@     �p@      q@     �r@     Ps@     �u@     �z@      y@     P|@     ��@     ��@     ��@     X�@     ��@     �@     ��@     Џ@     L�@     0�@     ��@     ��@     �@     ܜ@     �@     P�@     ��@     �B@        
�
conv2/biases_1*�	    /.y�   �ptz?      P@!  ��s\�?)��ObP@?2�o��5sz�*QH�x��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY��lDZrS�nK���LQ�k�1^�sO��qU���I�
����G�a�$��{E����#@�d�\D�X=�uܬ�@8���%�V6���VlQ.��7Kaa+�x?�x�?��d�r?ji6�9�?�S�F !?�[^:��"?U�4@@�$?�7Kaa+?��VlQ.?��%�V6?uܬ�@8?��%>��:?d�\D�X=?�!�A?�T���C?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?�������:�              �?              �?       @      �?      �?              �?      �?              �?              �?      �?               @      �?              �?              �?              �?              �?              �?              �?              �?              �?               @               @              �?              �?      �?      @      �?               @      @              @              �?      @      @       @      @       @      �?               @      �?      �?        
�
conv3/weights_1*�	   �̓��   �_�?      �@! �E�a�@)�1�YU@2�����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾�iD*L�پ�_�T�l׾��~]�[Ӿjqs&\�ѾK+�E��Ͼ��~��¾�[�=�k�����]���>�5�L�>���?�ګ>����>�XQ��>�����>�uE����>�f����>��(���>a�Ϭ(�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?�������:�             ؅@     2�@     ��@     Ĥ@     l�@     �@     ȟ@     L�@     X�@     ��@     ԕ@     t�@     �@     p�@     X�@     H�@     ��@     (�@     ��@     X�@      �@     P}@     �z@     �y@     `w@     �t@     �s@     `r@     0q@     @l@     �j@      g@      d@      e@     �b@     @]@      _@     �\@      \@     @X@     �T@     �Q@     �M@      M@      L@      L@      H@     �D@      C@      A@     �A@      >@      >@      6@      4@      2@      6@      3@      5@      &@      ,@      *@      *@      @      "@      "@      @      @      @      @      @      @      @      @       @      �?      @      @              �?      �?      �?      �?              �?       @      �?      �?      �?      �?              �?              @              �?              �?      �?              �?              �?              �?              �?               @              �?              �?      @      @      @               @      @      @       @      @      @      @       @      @      @      @      @      @      @      @      @      "@      @      "@      (@      3@      1@      $@      (@      <@      7@      ?@      2@      :@      :@      ?@      @@      E@     �@@      F@      M@      E@     �Q@     �V@     �W@      T@     @W@     @Z@      `@     �`@     �c@      f@     �d@      l@     `m@     �h@     �m@     p@     Ps@     @t@      v@      w@     �y@     �@     @�@     p�@     (�@     ��@     ��@     ؊@     H�@     `�@     t�@     8�@     ��@     ܘ@      �@     ��@     @�@     �@     ��@     ޤ@     B�@     @�@     ��@        
�

conv3/biases_1*�
	    ��}�    <=�?      `@! �@|\�?)�m��-T?2�>	� �����T}�o��5sz�*QH�x�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�uܬ�@8���%�V6���[���FF�G ����%ᾙѩ�-߾E��a�Wܾ        �-���q=
�}���>X$�z�>�T7��?�vV�R9?ji6�9�?�S�F !?�[^:��"?U�4@@�$?��VlQ.?��bȬ�0?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?�������:�              �?              �?              @      �?      @      �?      @      �?      @      @      @      @      �?      �?               @              �?      @       @       @              �?              �?      �?              �?              �?              �?      �?               @              �?              �?               @              �?              �?              �?       @      �?      @      �?      �?      �?              �?      �?      �?              �?       @       @      �?      @      @      @      �?      @       @      @      @      @      @      @      �?      �?      �?       @      �?       @        
�
conv4/weights_1*�	   ��Y��   �ً�?      �@! `B�7k�?)'����e@2�Ӗ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s�����Zr[v��I��P=���ߊ4F��h���`f�����uE����jqs&\�ѾK+�E��Ͼ�_�T�l�>�iD*L��>1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�������:�              ]@     З@     �@     ��@     H�@     P�@     ��@     8�@     ��@     `�@     �@     ��@     0}@     p{@     px@     @x@     �u@     �u@     �q@      q@      n@     �k@      d@     �f@     �e@     �c@      ]@     @^@     �Z@     �U@     �S@     @S@     �Q@      L@      P@     �M@     �D@     �C@     �E@     �H@     �F@      <@      9@      :@      6@      9@      2@      6@      6@      @      (@      $@      &@       @      "@      "@      @      $@      @      @      @      @       @      @      @       @              @      �?       @      �?       @      @               @      �?      �?              �?      �?       @              @      �?      �?              �?              �?              �?              �?              �?              �?              �?              �?      �?              �?       @              �?      �?      �?       @               @      @               @              @      �?       @      @       @      @      "@       @      @      @      @      "@      @      *@      &@      @      "@      "@      @      *@      8@      4@      0@      8@      5@      ;@      >@      9@      8@      F@     �D@     �E@      D@      L@     �Q@     �T@      S@     �Q@     �V@      Z@     @\@     �_@      _@      a@     �`@     �d@     �g@     @i@      k@     �p@     0p@     �s@     �u@     �v@     �w@     �x@      |@     @@     �@     H�@     8�@     P�@     ��@     h�@     ��@     H�@     �@     ĕ@      �@     �e@        
�
conv4/biases_1*�	   @�v�   �s�?      p@!�(Lyya�?)9Fd��R?2�*QH�x�&b՞
�u�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO��qU���I�
����G�a�$��{E��T���C��!�A����#@���%�V6��u�w74���82���VlQ.��7Kaa+�+A�F�&�U�4@@�$��[^:��"��S�F !���ڋ��vV�R9��T7���O�ʗ�����Zr[v����(��澢f���侙ѩ�-߾E��a�Wܾ��>M|Kվ��~]�[Ӿ�z��6��so쩾4�6NK��2�_"s�$1�7'_��+/��'v�V,����<�)�4��evk'���o�kJ%�4�e|�Z#���-�z�!�%�����i
�k���f��p�Łt�=	���R����2!K�R���J��#���j�Z�TA[�����"�RT��+��y�+pm��mm7&c��`���nx6�X� ��f׽r����tO����f;H�\Q������%���9�e����K��󽉊-��J�'j��p���1���=��]���(�+y�6ҽ;3���н��.4Nν�!p/�^˽        �-���q=�!p/�^�=��.4N�=;3����=z�����=ݟ��uy�=�/�4��==��]���=����%�=f;H�\Q�=�tO���=�f׽r��=nx6�X� >�mm7&c>y�+pm>RT��+�>���">Z�TA[�>�#���j>�J>2!K�R�>��R���>Łt�=	>��f��p>�i
�k>��-�z�!>4�e|�Z#>��o�kJ%>4��evk'>���<�)>�'v�V,>7'_��+/>_"s�$1>6NK��2>�so쩾4>E��a�W�>�ѩ�-�>a�Ϭ(�>8K�ߝ�>})�l a�>pz�w�7�>6�]��?����?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?�qU���I?IcD���L?k�1^�sO?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?�������:�              �?               @       @       @      �?              �?      �?               @      @      �?      @      �?      �?      �?      �?              �?      �?      �?               @              �?      �?              �?              �?              �?               @      �?              �?              �?               @              �?              �?      �?      @              �?      �?              @      @      @      @      �?      @              �?              @      �?      �?      �?      @       @              �?      �?       @       @      @              �?              �?              �?      �?              �?              �?              8@              �?      �?               @              �?              �?               @       @              �?      @              @              �?       @      @      �?      �?      @               @       @      @      @       @      �?      �?              �?              �?               @              �?              @              �?      �?              �?      �?      �?      �?      �?              �?       @      �?       @      �?      �?      �?      �?              �?       @      �?      @      �?      @              �?       @               @       @      �?       @      @      @      @      �?      @      @      @      @      @      �?      @      �?      @      �?      �?      �?       @        
�
conv5/weights_1*�	   �T�¿   �S��?      �@!  ��?�)!2�]hI@2�	�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9��O�ʗ��>>�?�s��>6�]��?����?f�ʜ�7
?>h�'�?��d�r?�5�i}1?�vV�R9?��ڋ?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?�������:�	              k@     @u@     �p@      n@     @o@     �k@     @g@      f@     �c@     �`@      \@     �]@     �X@     �Z@      V@     �S@     �Q@     @Q@     �J@     �K@      I@      A@     �J@      >@      @@      4@      9@      =@      2@      8@      >@      (@      1@      "@      ,@      *@      "@      @      @       @      (@      @      @      @      @       @      @      @      @      @      @      @       @       @      �?       @       @       @      �?      @       @      @       @              �?              �?      �?              �?              �?       @              �?              �?              �?              �?              �?              �?      �?      �?               @      �?      �?       @      �?       @      �?               @       @               @      �?      �?      @      @      @      @      �?      @       @      @      @       @       @      @      $@      @      ,@       @      @      ,@      "@      3@      ,@      8@      7@      2@      3@      1@      :@      ?@      ?@     �C@      B@      >@      F@      L@     �J@     �Q@     �U@     @P@      Q@      W@      X@     �Z@     @Z@     �^@     �c@      c@      d@      g@     `l@     �l@     @p@     �r@     �q@      k@        
�
conv5/biases_1*�	   @��?�   ��;5>      <@!  ��d+N�)�a�W�߸<2�p��Dp�@�����W_>�p
T~�;�u 5�9��z��6��so쩾4�6NK��2�_"s�$1�7'_��+/��'v�V,����<�)���o�kJ%�4�e|�Z#��i
�k���f��p�2!K�R���J�Z�TA[�����"�y�+pm��mm7&c�=��]����/�4���mm7&c>y�+pm>2!K�R�>��R���>Łt�=	>��f��p>%���>��-�z�!>��o�kJ%>4��evk'>���<�)>�'v�V,>7'_��+/>_"s�$1>6NK��2>�so쩾4>�z��6>�������:�              �?              �?               @              �?       @              �?               @               @              �?              �?              �?              �?              �?              �?      �?      �?              �?              �?              �?              �?              @      �?        ����Q      ����	"��2���A*��

step  �A

lossNv�>
�
conv1/weights_1*�	   �	\��    �Ʋ?     `�@!  Y���@)wU�A5@2��{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$�ji6�9���.����ڋ��5�i}1���d�r��FF�G �>�?�s�����ڋ?�.�?ji6�9�?�[^:��"?U�4@@�$?��bȬ�0?��82?�u�w74?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?�������:�              �?      3@     �Q@     �c@     �c@     �d@     �d@     �b@     �`@     �\@     �Z@     �W@      X@     �T@      P@     �R@      K@     �L@     �F@     �G@      F@     �D@     �D@      ;@      ;@     �B@      9@      9@      ,@      *@      0@      (@      .@      *@      ,@      (@      "@      @      @      &@      �?              @      @       @      @      @      @      @      �?      @      @              �?      @      @       @              �?      �?      �?              �?      �?              �?              �?              �?      �?              �?              �?      �?              @      @      �?              �?              @      @      @      @      @      @      @      @      @      @      @      @      $@      @      "@      2@      ,@      $@      (@      8@      *@      4@      <@      7@      ;@      =@      C@     �A@      ?@      E@     �B@      C@     �J@      P@     �O@     @R@      S@     �O@     �V@     @Z@      ^@      a@     @`@     �d@     �f@     �h@     �f@      ]@     �D@      @        
�
conv1/biases_1*�	    ��`�   ����?      @@!  (��	�?)�FعO1?2��l�P�`�E��{��^�
����G�a�$��{E�d�\D�X=���%>��:�;�"�qʾ
�/eq
ȾI��P=�>��Zr[v�>O�ʗ��>�5�i}1?�T7��?�S�F !?�[^:��"?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?����=��?���J�\�?�������:�              �?              �?              �?              �?              �?      �?              �?              �?              �?              �?      �?      �?      �?              �?              �?       @      �?      �?               @      �?      �?       @              �?      @               @              �?        
�
conv2/weights_1*�	    0���   @H�?      �@! 	��6�5@)d�4yE@2�I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��})�l a��ߊ4F��8K�ߝ�a�Ϭ(���(��澢f���侮��%ᾙѩ�-߾G&�$��5�"�g���;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>�_�T�l�>�iD*L��>E��a�W�>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?�������:�              *@     ��@     �@     ��@     ��@     P�@     ܗ@     (�@     H�@     X�@     L�@     8�@     `�@     H�@     ��@     ��@     0�@     (�@     @~@     �z@     0y@     �t@     �t@      t@     �o@     �n@     �n@      i@     @g@     �g@      d@      _@     �_@     @]@     �^@     �X@      U@     �P@      R@     @Q@      L@     �P@     �L@      C@     �F@      B@      ?@     �E@      >@      6@      3@     �@@      .@      ,@      3@      0@      &@      ,@      (@      *@       @       @      "@      @      @      @      @      @      @      �?      @      @      @      �?      @       @      @      @      �?               @               @               @              �?              �?              �?              �?              �?       @              �?              �?      �?               @      �?               @              �?      @      �?       @      �?      @       @       @      @      @      @       @      @      @      @      @      @      @      @      @      *@      *@      &@      (@      8@      *@      7@      5@      :@      =@      7@     �A@      =@      :@     �@@     �C@     �H@     �E@      K@     @R@     �Q@     @Q@      T@     �U@     �W@      X@     �]@     @^@      `@     `d@     @g@     `g@     �i@     @j@     �o@     �q@     �q@     �s@     �u@      z@     �z@     P{@     ��@     �@     p�@     h�@     ��@     H�@     P�@     ��@     T�@     h�@     x�@     l�@      �@     ��@     �@     P�@     ��@     �H@      �?        
�
conv2/biases_1*�	   ���y�   �}|?      P@!  h�V�?)���u�B?2�o��5sz�*QH�x�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b�E��{��^��m9�H�[��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I��T���C��!�A���%�V6��u�w74�+A�F�&�U�4@@�$��5�i}1?�T7��?+A�F�&?I�I�)�(?��%>��:?d�\D�X=?���#@?�!�A?a�$��{E?
����G?�qU���I?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?�������:�              �?              �?      �?      �?       @              �?               @               @              �?       @              �?              �?              �?               @               @              �?               @               @      �?              @      �?      �?              @       @       @      �?      �?       @      @      @      @      @      �?      �?      �?      �?       @        
�
conv3/weights_1*�	   �8į�   ��6�?      �@! `{���@)�@�[U@2�����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��8K�ߝ�a�Ϭ(龢f�����uE���⾮��%ᾙѩ�-߾R%�����>�u��gr�>�����>
�/eq
�>��~]�[�>��>M|K�>�_�T�l�>�uE����>�f����>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?�������:�             @�@     �@     ��@     ��@     r�@     ֡@     ��@     H�@     x�@     З@     ĕ@     x�@     �@     x�@     x�@     p�@     ��@      �@     ؃@     ��@     ��@     �}@     pz@     �x@     0x@     �s@     pt@     �r@     0p@     �n@     �i@     @g@     `d@      e@     �a@      ]@     �_@     �Z@     �[@     �Z@      R@     �R@      P@     @R@     �G@     �M@      E@      @@      F@      E@      =@      <@     �C@      0@      4@      8@      1@      3@      *@      &@      $@      *@      4@      "@      &@      (@      @      @      @      @      @      @      @      @      �?      �?              @      �?      @      @      @              �?      �?      �?       @              �?              �?      �?      �?              �?              �?              �?      �?              �?              �?               @       @       @      @              �?               @      @      @      @      @      �?       @      @      @      @      @       @      @      @      @      "@      &@      ,@      *@      .@      2@      1@      .@      7@      5@      7@      8@      ?@      8@      >@      =@     �G@      @@     �E@      L@     �K@     @Q@      S@     �T@      V@     @Y@      [@     �^@     @`@     @d@     `e@     `f@     `j@     @m@     `j@     �m@     �o@     ps@      t@     Pu@      x@      y@      �@     X�@     �@     ��@     ��@     ȇ@      �@     ��@     �@     ��@     ,�@     \�@      �@     T�@     ��@     <�@     �@     ��@     �@     B�@     6�@     H�@        
�

conv3/biases_1*�
	   �@��   ��B�?      `@! `H��?)�d0m�V?2�>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E����#@�d�\D�X=�uܬ�@8���%�V6�1��a˲���[���f�����uE���⾮��%ᾙѩ�-߾        �-���q=G&�$�>�*��ڽ>pz�w�7�>I��P=�>U�4@@�$?+A�F�&?I�I�)�(?��VlQ.?��bȬ�0?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?�������:�              �?              �?              @              @      �?      @       @       @       @      @      @      �?      �?      �?       @               @       @      �?      �?       @       @       @              �?              �?              �?              �?              �?               @              �?              �?              �?      �?              �?              @      �?       @      �?      �?              �?      @              �?       @      �?      �?      @      @       @      @       @       @      @      @      @       @      @      @      @      �?      �?       @      �?      @        
�
conv4/weights_1*�	   ��c��   �n��?      �@! ;��&�?)w�y��e@2�Ӗ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7���x?�x��>h�'��f�ʜ�7
������1��a˲���[���FF�G �O�ʗ�����Zr[v��pz�w�7��})�l a�h���`�8K�ߝ�a�Ϭ(���(��澮��%ᾙѩ�-߾jqs&\�ѾK+�E��Ͼ�h���`�>�ߊ4F��>1��a˲?6�]��?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�������:�              \@     ܗ@     �@     ��@     @�@     @�@     ��@     �@     ��@     p�@     ��@     ��@     �|@     �{@     �x@     �w@     �u@      v@     �q@     q@     �m@     �k@     �d@     �f@     �d@      e@     @\@     �]@     @[@     @T@     @T@     �S@      R@     �K@      Q@     �K@      E@     �C@      F@      G@     �G@      :@      :@      :@      7@      5@      8@      5@      .@      *@      (@       @      *@      @      $@       @      @      @      @      @      @      @      @      @      @      @      �?      �?       @       @      @      @       @              @       @              �?              �?      @       @              �?      �?              �?              �?              �?      �?      �?              �?              �?              �?              �?              �?              @               @      �?      �?       @       @              �?      @       @      @      �?      @      @      @      @      @      @      @       @      @      @      $@       @      @      $@       @       @      &@      =@      .@      8@      4@      6@      ;@      <@      9@      ;@      E@     �D@     �D@      G@      K@      Q@      T@     �S@     @R@     �V@     @Y@     �Z@      a@     @_@     �`@     �`@     �d@     `h@     `h@     �k@     �p@     `p@     �s@     �u@     �v@     �w@     �x@     �{@     �@     ��@     P�@     �@     ��@     ��@     x�@     ��@     @�@     �@     ��@     ��@      f@        
�
conv4/biases_1*�	    ��w�    ���?      p@!Q�O���?)f��Ez�T?2�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO��qU���I�
����G��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74�+A�F�&�U�4@@�$�ji6�9���.����ڋ��vV�R9��T7���>�?�s���O�ʗ�����Zr[v���ѩ�-߾E��a�Wܾ�z��6��so쩾4�6NK��2�_"s�$1�7'_��+/��'v�V,����<�)�4��evk'���o�kJ%�4�e|�Z#���-�z�!�%�����i
�k�Łt�=	���R����2!K�R���J��#���j�Z�TA[�����"�RT��+��y�+pm��mm7&c��`���nx6�X� ��f׽r����tO����f;H�\Q������%���9�e����K��󽉊-��J�'j��p���1������X>ؽ��
"
ֽG-ֺ�І�̴�L����        �-���q=��
"
�=���X>�=PæҭU�=�Qu�R"�=��-��J�=�K���=�9�e��=����%�=f;H�\Q�=�tO���=�f׽r��=nx6�X� >�`��>�mm7&c>y�+pm>RT��+�>���">Z�TA[�>�#���j>�J>2!K�R�>��R���>Łt�=	>��f��p>�i
�k>%���>��-�z�!>4�e|�Z#>��o�kJ%>4��evk'>���<�)>�'v�V,>7'_��+/>_"s�$1>6NK��2>�so쩾4>���?�ګ>����>['�?��>K+�E���>E��a�W�>�ѩ�-�>8K�ߝ�>�h���`�>})�l a�>pz�w�7�>>�?�s��>�FF�G ?6�]��?����?f�ʜ�7
?�T7��?�vV�R9?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?�������:�              �?               @      �?       @       @               @               @              @       @      �?      �?       @      �?      �?               @              �?      �?              �?              �?      �?              �?              �?      �?              �?              �?      �?              �?               @      @              �?      �?              @      @      @      @      @       @              �?              @       @              @      �?       @               @       @      �?       @      �?              �?               @      �?      �?              �?              �?              8@              �?              �?              �?      �?      �?      �?              �?      �?              �?      �?       @       @      �?       @      �?               @      @      �?      @      �?      �?              �?      @      �?      @      @              �?      �?              �?              �?              �?              @              �?              �?              �?      @              �?              @              �?      �?              �?              �?      �?       @       @      �?      �?      �?      �?      �?      @      @      �?              �?       @      �?      �?              �?      �?       @      �?       @       @      @      @      @      @      @      @      @      @      �?       @      @       @      �?       @       @        
�
conv5/weights_1*�	   `��¿   ����?      �@! @��V�)�2lI@2�	�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(��S�F !�ji6�9���.����ڋ��vV�R9��T7����FF�G �>�?�s���I��P=��pz�w�7��O�ʗ��>>�?�s��>����?f�ʜ�7
?x?�x�?��d�r?�5�i}1?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?�������:�	              k@     u@     �p@     @n@      o@     �k@     @g@      f@     �c@     �`@     @\@     �]@      Y@      Z@      V@     �T@     �P@     �O@      M@     �K@      I@      @@     �J@      @@      ?@      5@      :@      <@      3@      :@      ;@      *@      0@      $@      *@      (@      "@      @      @      &@      "@       @      @      @      @      @      @      @      @      @      @       @       @      �?       @      �?       @       @       @       @      @       @              �?              @      �?              �?      �?              �?              �?              �?              �?              �?              �?              �?              �?      �?               @       @              �?      �?      �?      @      �?      �?       @      @              �?       @      @      @      @      @      �?       @      �?      @       @      @      @      &@      @      *@      &@      @      *@      &@      2@      ,@      8@      8@      1@      2@      2@      ;@      ?@      >@     �C@      B@      >@      F@     �L@     �I@      R@     �U@     �O@     �Q@     �V@     �X@      Z@      Z@     @_@     �c@     �b@      d@      g@     �l@     `l@     @p@     �r@     �q@      k@        
�
conv5/biases_1*�	   ���@�   �(6>      <@!   u^?O�)W�	p�q�<2�/�p`B�p��Dp�@�����W_>�p
T~�;�u 5�9��z��6�6NK��2�_"s�$1�7'_��+/��'v�V,����<�)�4��evk'���o�kJ%�%�����i
�k���R����2!K�R��RT��+��y�+pm��mm7&c���-��J�'j��p�Z�TA[�>�#���j>��R���>Łt�=	>��f��p>�i
�k>��-�z�!>4�e|�Z#>��o�kJ%>4��evk'>7'_��+/>_"s�$1>6NK��2>�so쩾4>�z��6>�������:�              �?              �?               @               @      �?              �?               @               @              �?              �?      �?              �?              �?              �?      �?      �?              �?              �?              �?      �?      �?      @        yҠ��R      ȘA	X!�2���A*��

step  �A

loss �o>
�
conv1/weights_1*�	    ����   ��+�?     `�@! �)��@)�W@�Lf@2��{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !��.����ڋ��5�i}1?�T7��?�[^:��"?U�4@@�$?��bȬ�0?��82?�u�w74?��%�V6?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?�������:�               @      6@     �R@     @b@      d@     `d@     �c@     @c@     �`@      ^@     �X@     �W@     �W@      W@      Q@      O@     �N@      K@      H@     �D@     �H@      A@     �C@     �B@      @@     �@@      4@      4@      3@      0@      5@      *@      0@      @      (@      $@      (@      @      "@      @      @       @      @      @       @              �?      �?      @      @       @       @              @      @              �?              �?               @               @              �?              �?               @              �?              �?               @       @       @              �?      �?      �?      @      �?      @      @      @              @      �?       @      @      @      @      @      "@      @       @      &@      0@      &@      $@      0@      1@      6@      3@      4@      <@      @@      <@      >@     �@@      B@      G@      A@      C@      N@      M@     �N@     @R@     �S@      Q@     @T@     @Z@     @_@     �`@     @`@      d@     �g@     �i@     �d@      `@     �D@      *@        
�
conv1/biases_1*�	   ���_�   `2@�?      @@!  vJ�B�?)�4W*�n4?2��l�P�`�E��{��^�
����G�a�$��{E�uܬ�@8���%�V6�f�ʜ�7
������O�ʗ��>>�?�s��>�5�i}1?�T7��?�[^:��"?U�4@@�$?+A�F�&?��%>��:?d�\D�X=?���#@?�!�A?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?����=��?���J�\�?�������:�              �?              �?              �?              �?              �?              �?              �?      �?              �?              �?              �?      �?              �?              �?      �?               @      �?      �?              @      �?       @              @               @      �?              �?        
�
conv2/weights_1*�	   �~߬�   �Y]�?      �@!�_A�7@)��!�~E@2�I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F�𾮙�%ᾙѩ�-߾�iD*L�پ�_�T�l׾��>M|KվK+�E��Ͼ['�?�;;�"�qʾK���7��[#=�؏��jqs&\��>��~]�[�>�iD*L��>E��a�W�>�uE����>�f����>a�Ϭ(�>8K�ߝ�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?�������:�              1@     ��@     �@     x�@     ��@     D�@     ̗@     4�@     H�@     l�@     0�@     0�@     ��@     �@     `�@      �@     8�@     Ȁ@     P~@     �z@     �x@     @u@     pt@     �t@     �n@      o@      n@      i@     `f@      h@     �c@      _@     �`@      \@      ^@     �Z@      R@     �T@     �N@     �R@      I@      P@      N@      F@     �C@     �A@     �@@     �B@     �B@      6@      .@      1@      1@      4@      9@      4@      @      (@      ,@      ,@      &@      ,@      @      @      @       @      @      @      @      @      @       @      @              @       @       @       @      @       @      @              @      �?               @              �?              �?      �?              �?      �?              �?              �?              �?              �?               @               @              �?              @      �?      @              �?      @              @       @              @      @      �?       @      @      @      @      @      "@      "@      @      &@       @      (@      &@      $@      0@      .@      5@      5@      7@      7@      ;@     �B@      6@      ?@      B@      F@      C@      J@      L@      Q@     @Q@     @R@      T@     �S@     @X@     �Y@     �\@     @`@     �`@     �a@     �f@     `g@     @i@      k@     @p@     `q@      r@     pt@     v@     �x@     @{@     �{@     Ȁ@     x�@     ��@      �@     ȇ@     P�@     ؋@     0�@     8�@     ��@     p�@     \�@     �@     Ĝ@     �@     `�@     ��@      N@      @        
�
conv2/biases_1*�	    �z�    ��}?      P@!  @��K�?)���
/E?2����T}�o��5sz�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`���bB�SY�ܗ�SsW��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E�d�\D�X=���%>��:���[���FF�G �jqs&\�ѾK+�E��Ͼ��ڋ?�.�?+A�F�&?I�I�)�(?��bȬ�0?��82?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?�������:�              �?              �?      �?               @      �?      �?              �?              �?              @      �?              �?              �?              �?              �?              �?              �?              �?              �?              �?       @      �?      �?      �?              �?              �?      @              @      �?      �?       @       @      �?      @              @      @      @              �?      �?       @      �?        
�
conv3/weights_1*�	    K��   `}S�?      �@! �+z��@)A��Y
^U@2�����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F����(��澢f�����uE���⾮��%ᾄiD*L�پ�_�T�l׾��>M|Kվjqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�������:�             ��@     Ԩ@     ֧@     ��@     `�@     ڡ@     �@     ,�@     ��@     ��@     ��@     ��@     �@     H�@     ��@     8�@     ��@     (�@     ��@     `�@     ��@     ~@     pz@     �x@     �w@     �t@      t@     @s@     �n@     �o@     @j@     �e@     `e@      e@     �b@     �Z@     �a@      V@     @^@     �W@     �T@     @R@     @S@     �H@      K@     �K@      F@     �H@     �B@     �D@      ;@     �A@     �@@      4@      .@      8@      3@      $@      *@      .@      @      (@      (@      1@       @      (@      @      $@      @      @      @      @      @      �?      �?      �?       @      �?      �?      �?               @              �?      �?      �?      �?              �?      �?      �?              �?       @              �?      �?      �?              �?              �?              �?              �?      �?               @               @       @       @      @      �?               @      �?      @      @      @       @      @      @      @      @      @      @      @      &@      $@      @       @      *@      .@       @      (@      ,@      7@      ,@      6@      1@      >@      ;@      9@      9@      C@      >@     �F@     �A@     �D@      L@      I@      S@     �R@     @T@     �U@     �X@     @X@     �_@     @`@     @f@     `d@     `f@     �i@     �k@     `l@      m@     �p@     �r@     �s@     �u@      x@     �x@      �@     x�@     ��@     Є@     ��@     ��@     �@     ��@     �@     ��@     l�@     �@     ,�@     P�@     ��@     $�@     ��@     ��@     �@     F�@     "�@     �@       @        
�

conv3/biases_1*�
	   ��ǀ�    �Z�?      `@! �&��˽?)0�л��Y?2�����=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G��T���C��!�A����#@�d�\D�X=���%>��:�1��a˲���[���ߊ4F��h���`�a�Ϭ(���(����uE���⾮��%�        �-���q=0�6�/n�>5�"�g��>�vV�R9?��ڋ?��VlQ.?��bȬ�0?�u�w74?��%�V6?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?�������:�              �?              �?              �?       @      �?      @      @       @       @      �?      @      @      @      �?      �?      �?       @      @              �?              @       @              �?              �?       @              �?              �?              �?              �?               @              �?              �?              �?              �?              �?      �?      �?       @      �?               @       @       @      �?      �?      @      @      �?      @      �?      @      �?      @      @      @      @      @      @      @       @       @      �?       @      �?       @        
�
conv4/weights_1*�	   @Cn��    ,��?      �@! t��]�@)��K"��e@2�Ӗ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������1��a˲���[��>�?�s���O�ʗ����ߊ4F��h���`f�����uE����E��a�Wܾ�iD*L�پjqs&\�ѾK+�E��Ͼ8K�ߝ�>�h���`�>�ߊ4F��>pz�w�7�>I��P=�>1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�������:�             �\@     ԗ@     �@     ��@     (�@     h�@     ��@     @�@     ��@     X�@     ��@     ��@     �|@     �{@      y@     `w@     @v@     �u@     �q@     q@     �m@     �k@     �d@     `f@     `d@      e@      ]@     @]@     �Z@      U@      S@     �T@     @R@     �K@     �Q@     �G@      I@     �C@      D@     �G@     �F@      =@      7@      7@      8@      ;@      7@      4@      ,@      $@      0@      "@      $@      (@      &@      @       @      @      @      @      @       @      @       @      @      @       @      @      @      @       @      �?       @       @      @              �?      �?              �?      �?              �?      �?              �?              �?               @              �?              �?              �?              �?      �?              �?              �?              �?      �?       @              �?      �?      @              @      �?      �?              �?      @      @       @      @      @       @      @      @       @       @      @       @      @      @      $@       @      @      @      @      @      3@      8@      *@      8@      7@      7@      ;@      9@      =@      9@     �E@      B@     �E@     �E@      K@     �Q@     �T@     @S@      R@     �W@     @X@     �Z@      a@     @`@     ``@     �`@     �d@      h@     �h@     `k@     �p@      p@     �s@     �u@     �v@      x@      y@     P{@     �@     ��@     P�@     0�@     h�@     ��@     `�@     ��@     L�@     �@     ��@     �@     @f@        
�
conv4/biases_1*�	   `��x�   `,�?      p@!��8���?)����W?2�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A�d�\D�X=���%>��:�uܬ�@8���%�V6�+A�F�&�U�4@@�$�ji6�9���.����ڋ��vV�R9�1��a˲���[���FF�G �>�?�s����ѩ�-߾E��a�Wܾ�5�L�����]����u 5�9��z��6��so쩾4�6NK��2�_"s�$1�7'_��+/��'v�V,����<�)�4��evk'���o�kJ%�4�e|�Z#���-�z�!�%�����i
�k���R����2!K�R���J��#���j�Z�TA[�����"�RT��+��y�+pm��mm7&c��`���nx6�X� ��f׽r���f;H�\Q������%���K��󽉊-��J�'j��p���1����/�4��ݟ��uy��b1�ĽK?�\��½!���)_�����_����        �-���q=i@4[��=z�����=��1���='j��p�=��-��J�=�K���=f;H�\Q�=�tO���=�f׽r��=�`��>�mm7&c>y�+pm>RT��+�>���">Z�TA[�>�#���j>�J>2!K�R�>��R���>Łt�=	>��f��p>�i
�k>%���>��-�z�!>4�e|�Z#>��o�kJ%>4��evk'>���<�)>�'v�V,>7'_��+/>_"s�$1>6NK��2>�so쩾4>�z��6>['�?��>K+�E���>�ѩ�-�>���%�>8K�ߝ�>�h���`�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>����?f�ʜ�7
?>h�'�?�T7��?�vV�R9?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?�������:�              �?              �?       @      �?      @              �?      �?       @              @      �?      �?       @      �?      @              �?              �?               @      �?              �?              �?              �?              �?      �?      �?              �?              �?              �?              �?               @      �?       @               @      �?       @      @      @      @      @      �?       @               @      �?      @      �?       @      @      �?      �?       @      �?      @              �?              �?      �?      �?              �?              �?              �?              7@              �?              �?              �?               @      �?               @       @       @      �?       @       @              �?      �?       @      �?      @       @      �?      �?      �?      �?      @       @      @      �?      �?      �?      �?              �?              �?               @              �?      �?              �?              @      �?              �?               @      �?      @              �?              �?               @       @       @      �?      �?      �?      �?      @      @       @              @               @      �?       @       @               @      @      @      @      @      @      @      @      @      �?       @      �?      @      �?      �?      �?       @        
�
conv5/weights_1*�	    �ÿ   `�?      �@! @/,��)R7��oI@2�	�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9����%ᾙѩ�-߾�XQ��>�����>x?�x�?��d�r?�5�i}1?�S�F !?�[^:��"?U�4@@�$?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?�������:�	             @k@      u@     �p@     @n@     �n@     �k@      g@     @f@     �c@     �`@      ]@     @]@      Y@     @Z@     �U@      U@      Q@     �O@     �L@      K@      J@     �@@      I@     �@@      >@      4@      >@      9@      5@      :@      ;@      ,@      ,@      $@      (@      (@       @      "@      @      (@       @      "@      @      @      @      �?      @      @      @       @      @              �?      �?      �?       @      @      �?      �?      �?       @      �?               @      �?       @      �?      �?               @      �?              �?      �?              �?              �?              �?              �?      �?              �?      �?              �?               @       @      �?               @              �?      �?      �?      @      �?      @      �?      �?      �?      @      @      @       @      �?       @      @      @       @      @      @      $@      @      *@      $@      @      (@      $@      4@      1@      8@      5@      1@      0@      5@      :@      >@     �@@      C@      A@      ?@      E@      M@      I@      R@     �U@     �O@      R@     �V@     �X@      Z@     �Y@     �_@     @d@     `b@      d@      g@     @l@     �l@     p@     �r@     �q@     �k@        
�
conv5/biases_1*�	   ��B�    }7>      <@!  ���N�)�ѿ�I�<2�/�p`B�p��Dp�@�����W_>�p
T~�;�u 5�9��z��6��so쩾4�6NK��2�_"s�$1��'v�V,����<�)�4��evk'���-�z�!�%������R����2!K�R��RT��+��y�+pm���1���=��]���2!K�R�>��R���>��f��p>�i
�k>%���>4�e|�Z#>��o�kJ%>4��evk'>���<�)>7'_��+/>_"s�$1>6NK��2>�z��6>u 5�9>�������:�              �?              �?               @      �?               @              �?       @               @              �?               @              �?              �?               @      �?              �?              �?              �?      �?              @        *�?�R      Ʉt�	�[�2���A*��

step  �A

lossZ|`>
�
conv1/weights_1*�	   ����   �Ñ�?     `�@! `�|8�!@)ue��@2��{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���VlQ.��7Kaa+�I�I�)�(�+A�F�&��[^:��"��S�F !��.����ڋ�f�ʜ�7
������.��fc��>39W$:��>�h���`�>�ߊ4F��>��[�?1��a˲?��ڋ?�.�?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?�������:�               @      9@     @S@      b@     `c@     �c@      c@     �c@      `@      ^@      Z@     �V@     �Y@     @U@      R@     �N@      M@      K@     �I@      ?@      I@      F@      >@     �E@      A@      >@      5@      2@      1@      2@      3@      1@      .@      @      $@      ,@      *@      @      @      @      @      @      @       @      @              �?      @       @              @       @              @      @               @      �?              �?              �?              �?              �?               @              �?              �?              �?              �?               @      �?      �?              �?      �?              �?              @      �?      @              �?       @      @      @      �?       @      �?       @      @      @      @      $@       @      @      @       @      *@      $@      *@      2@      2@      9@      ,@      6@      7@      C@      =@      9@     �B@      :@     �I@     �E@     �A@     �L@     �K@     �N@     @R@     @T@     �R@      Q@     �[@      _@      a@      `@     `c@      g@      j@      f@     �]@     �G@      5@        
�
conv1/biases_1*�	   ���]�   @Ƅ?      @@!  �G��?)�O���8?2�E��{��^��m9�H�[�a�$��{E��T���C���82���bȬ�0�})�l a��ߊ4F��O�ʗ��>>�?�s��>�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?��bȬ�0?��82?d�\D�X=?���#@?�!�A?�qU���I?IcD���L?k�1^�sO?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���J�\�?-Ա�L�?�������:�              �?              �?              �?              �?              �?              �?              �?              �?              �?      �?              �?      �?              �?               @      �?               @              �?              �?       @      �?       @              @               @      �?              �?        
�
conv2/weights_1*�	   ��B��    �خ?      �@! �B���8@)Ү�⭄E@2�I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ뾮��%ᾙѩ�-߾����ž�XQ�þ�[�=�k���*��ڽ�jqs&\��>��~]�[�>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?�������:�              5@     ��@     Р@     p�@     ��@      �@     �@     0�@     0�@     p�@     �@     Ȍ@     8�@     0�@     h�@     ؃@     ��@     ؀@     ~@     �z@     �w@     �v@     �s@     @t@     Pp@     `n@     �m@      h@      h@     �f@     �c@     �`@      a@     �]@     @[@     @Y@     �T@     �T@     �O@     @R@      K@      M@      P@     �B@     �B@      C@     �B@     �@@      B@      8@      4@      1@      1@      1@      7@      1@      &@      (@      &@      ,@       @      ,@      &@      @       @      @      @      @      @       @       @      @      @      �?      �?       @       @       @      @      �?      @      @      �?      �?              �?               @               @              �?               @               @              �?      �?              �?               @      �?      @              @       @       @      @       @       @       @       @      @      @      @      @      "@      @       @      @      @      "@      (@       @      0@       @      7@      ,@      5@      .@      5@      :@      =@     �@@      9@      ;@      E@     �F@     �@@     �F@     @R@     �L@     @S@     @Q@      T@      T@      X@     �V@      _@     @a@     �`@      b@     @e@      i@     �h@     �h@     �q@     �p@     @r@     �t@     �u@     �x@      {@     �{@     ��@     8�@     h�@     ��@     ��@     0�@     x�@     ��@     �@     ��@     @�@     p�@     ��@     ��@     �@     X�@     ē@     @S@      @        
�
conv2/biases_1*�	    2w{�   ��l?      P@!  ��"?�?)"�:[�G?2����T}�o��5sz�uWy��r�;8�clp�ߤ�(g%k�P}���h�Tw��Nof����%��b��l�P�`�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G��!�A����#@�U�4@@�$��[^:��"��T7��?�vV�R9?U�4@@�$?+A�F�&?I�I�)�(?��%>��:?d�\D�X=?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?�������:�              �?               @               @       @              �?              �?      �?      @              �?              �?              �?              �?              �?              �?      �?              �?              �?              �?      �?       @      �?              �?              �?      �?       @       @      �?      �?      �?       @       @      @      @              @       @      @      �?               @       @        
�
conv3/weights_1*�	   ��"��   �q�?      �@! n׊�0 @)ማ�`U@2�����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`f�����uE���⾙ѩ�-߾E��a�Wܾ��>M|Kվ��~]�[ӾK+�E��Ͼ['�?�;���?�ګ�;9��R��39W$:���.��fc����XQ��>�����>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>�uE����>�f����>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�������:�             ��@     ��@     ҧ@     ��@     J�@     �@     �@     <�@     ��@     ��@     ܕ@     t�@     (�@     p�@     ��@     ȋ@     ��@     p�@     ��@      �@     ��@     �}@     `z@     py@     `w@     0u@     �s@      s@     @o@     @o@      j@     �e@      e@     �f@      b@     �X@     �`@     �Z@     �[@     @Y@     @S@     �S@     �Q@     �K@      H@      J@      J@     �G@     �B@     �A@     �C@      =@      :@      6@      4@      3@      2@      *@      3@      *@      @      *@      ,@      &@      @      (@      @      @      @      @      @      @      @      �?      @      �?      �?      @      �?      �?      �?      �?      �?       @       @      @      �?      �?              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?               @               @       @               @      �?       @      @      �?      @      @      @       @      @      @      @      @      @      "@      @      @      "@      (@      0@      .@      0@      3@      3@      .@      4@      5@      >@      =@      :@      1@      A@     �D@      E@      :@      I@     �L@     �H@     �S@      V@      Q@     �T@     �W@     @X@     �^@     �b@     �c@     `d@     �f@      j@      m@     �i@     �n@      q@     �r@     s@      u@     y@     px@     �@     ��@     �@     Є@     P�@     ��@     Ȋ@     ��@     �@     ��@     ,�@     X�@     H�@     �@     ��@     <�@     �@     ��@     �@     <�@     "�@     ��@      @        
�

conv3/biases_1*�
	   @k���    �_�?      `@! ���B�?)u^q��5]?2����J�\������=������T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G����#@�d�\D�X=���%>��:���82���bȬ�0��5�i}1���d�r�6�]���1��a˲��h���`�8K�ߝ��uE���⾮��%�5�"�g���0�6�/n��        �-���q=�FF�G ?��[�?��%�V6?uܬ�@8?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?�������:�              �?              �?              @      �?       @      @      �?      @      �?      @      @              @       @      �?      �?      @      �?              �?      �?       @       @               @      �?              �?              �?              �?              �?              �?              �?               @              �?               @              �?      �?              �?      �?      �?      �?              @       @       @      �?      @      @              @       @      @       @      @       @      @      @      @      @      @      @      �?      �?       @       @      �?        
�
conv4/weights_1*�	   ��x��   �o��?      �@! �jP�@)�M�;��e@2�Ӗ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������1��a˲���[��>�?�s���O�ʗ����ߊ4F��h���`���~]�[Ӿjqs&\�ѾK+�E��Ͼ5�"�g���0�6�/n���f����>��(���>a�Ϭ(�>8K�ߝ�>��Zr[v�>O�ʗ��>>�?�s��>��[�?1��a˲?6�]��?>h�'�?x?�x�?��d�r?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�������:�             @\@     ��@     �@     ��@     �@     ��@     ��@     �@     ��@     x�@     ��@     Ѐ@     `|@     �{@      y@      w@     �v@     �u@      r@     q@     �m@     `k@      e@     �f@     @d@     @d@     @^@     @]@     @Z@     �U@      S@     �S@     �S@     �K@     �Q@      H@     �H@      B@      C@     �I@      G@      ;@      8@      6@      8@      <@      5@      5@      1@      @      0@      &@      "@      $@       @      &@      @      @      @      @      @      @      @      @      @      @       @      @       @      @      @      @      @               @      �?              �?      �?              �?      �?      �?               @              �?              �?              �?      �?              �?              �?              �?              �?      �?               @      @              �?       @              �?       @              @      @      �?      @      �?       @       @       @      �?      @      �?      @       @      @       @      @      @      "@       @      "@      @      @      @      &@      1@      3@      2@      7@      4@      9@      <@      <@      =@      8@     �B@     �C@     �E@     �G@     �I@     �Q@     �S@     @T@     �Q@     �W@     �W@     �Z@     �`@      `@     �`@     �`@     @d@     @h@      i@     �k@     �p@     �o@     �s@      u@     Pw@     Px@     �x@     @{@     @@     �@     P�@     �@     ��@     ��@     `�@     ��@     h�@     ��@     ��@     ��@     �f@        
�
conv4/biases_1*�	   ��~y�   `�a�?      p@!��N�hp�?)m��a��Z?2�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�IcD���L��qU���I�
����G�a�$��{E����#@�d�\D�X=���%>��:�uܬ�@8���%�V6�+A�F�&�U�4@@�$��S�F !�ji6�9���.����ڋ��vV�R9�f�ʜ�7
��������[���FF�G ����%ᾙѩ�-߾u 5�9��z��6��so쩾4�6NK��2�_"s�$1�7'_��+/��'v�V,����<�)�4��evk'���o�kJ%�4�e|�Z#���-�z�!�%����Łt�=	���R����2!K�R���J��#���j�Z�TA[�����"�RT��+��y�+pm��mm7&c��`���nx6�X� ��f׽r����K��󽉊-��J�'j��p���1���H����ڽ���X>ؽ�|86	Խ(�+y�6ҽ        �-���q=PæҭU�=�Qu�R"�=�/�4��==��]���='j��p�=��-��J�=�K���=�9�e��=nx6�X� >�`��>�mm7&c>y�+pm>RT��+�>���">Z�TA[�>�#���j>�J>2!K�R�>��R���>Łt�=	>��f��p>�i
�k>%���>��-�z�!>4�e|�Z#>��o�kJ%>4��evk'>���<�)>�'v�V,>7'_��+/>_"s�$1>6NK��2>�so쩾4>�z��6>u 5�9>�ѩ�-�>���%�>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>1��a˲?6�]��?����?f�ʜ�7
?>h�'�?�T7��?�vV�R9?ji6�9�?�S�F !?U�4@@�$?+A�F�&?I�I�)�(?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?�������:�              �?      �?       @              @      �?              �?               @      �?      @       @      �?              @               @      �?               @              �?              �?              �?      �?              �?              �?      �?              �?              �?              �?              �?               @       @      �?      �?       @       @      �?      @      @      @              @              �?      �?      @      �?       @              @              @      @       @      �?               @              �?              �?              �?              7@              �?              �?              �?      �?      �?               @              @       @              �?      @               @      �?      �?       @              @       @       @      �?      �?      �?       @      @       @               @              �?              �?               @              �?              �?              �?              �?        