       �K"	   xX��Abrain.Event:2�`Z�     [~"�	�Z,xX��A"ԣ
�
anchor_inputPlaceholder*
dtype0*0
_output_shapes
:���������2�*%
shape:���������2�
�
positive_inputPlaceholder*%
shape:���������2�*
dtype0*0
_output_shapes
:���������2�
�
negative_inputPlaceholder*%
shape:���������2�*
dtype0*0
_output_shapes
:���������2�
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
seed2 *
dtype0*&
_output_shapes
: *

seed *
T0* 
_class
loc:@conv1/weights
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
(conv1/weights/Initializer/random_uniformAdd,conv1/weights/Initializer/random_uniform/mul,conv1/weights/Initializer/random_uniform/min*&
_output_shapes
: *
T0* 
_class
loc:@conv1/weights
�
conv1/weights
VariableV2*
dtype0*&
_output_shapes
: *
shared_name * 
_class
loc:@conv1/weights*
	container *
shape: 
�
conv1/weights/AssignAssignconv1/weights(conv1/weights/Initializer/random_uniform*
T0* 
_class
loc:@conv1/weights*
validate_shape(*&
_output_shapes
: *
use_locking(
�
conv1/weights/readIdentityconv1/weights*&
_output_shapes
: *
T0* 
_class
loc:@conv1/weights
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
VariableV2*
shape: *
dtype0*
_output_shapes
: *
shared_name *
_class
loc:@conv1/biases*
	container 
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
model/conv1/Conv2DConv2Danchor_inputconv1/weights/read*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*0
_output_shapes
:���������2� 
�
model/conv1/BiasAddBiasAddmodel/conv1/Conv2Dconv1/biases/read*0
_output_shapes
:���������2� *
T0*
data_formatNHWC
n
model/conv1/conv1/ReluRelumodel/conv1/BiasAdd*
T0*0
_output_shapes
:���������2� 
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
:���������K 
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
,conv2/weights/Initializer/random_uniform/maxConst*
_output_shapes
: *
valueB
 *��L=* 
_class
loc:@conv2/weights*
dtype0
�
6conv2/weights/Initializer/random_uniform/RandomUniformRandomUniform.conv2/weights/Initializer/random_uniform/shape*
dtype0*&
_output_shapes
: @*

seed *
T0* 
_class
loc:@conv2/weights*
seed2 
�
,conv2/weights/Initializer/random_uniform/subSub,conv2/weights/Initializer/random_uniform/max,conv2/weights/Initializer/random_uniform/min*
T0* 
_class
loc:@conv2/weights*
_output_shapes
: 
�
,conv2/weights/Initializer/random_uniform/mulMul6conv2/weights/Initializer/random_uniform/RandomUniform,conv2/weights/Initializer/random_uniform/sub*
T0* 
_class
loc:@conv2/weights*&
_output_shapes
: @
�
(conv2/weights/Initializer/random_uniformAdd,conv2/weights/Initializer/random_uniform/mul,conv2/weights/Initializer/random_uniform/min*&
_output_shapes
: @*
T0* 
_class
loc:@conv2/weights
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
conv2/weights/AssignAssignconv2/weights(conv2/weights/Initializer/random_uniform*
use_locking(*
T0* 
_class
loc:@conv2/weights*
validate_shape(*&
_output_shapes
: @
�
conv2/weights/readIdentityconv2/weights*
T0* 
_class
loc:@conv2/weights*&
_output_shapes
: @
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
VariableV2*
_output_shapes
:@*
shared_name *
_class
loc:@conv2/biases*
	container *
shape:@*
dtype0
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
model/conv2/Conv2DConv2Dmodel/conv1/MaxPool2D/MaxPoolconv2/weights/read*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*/
_output_shapes
:���������K@
�
model/conv2/BiasAddBiasAddmodel/conv2/Conv2Dconv2/biases/read*
T0*
data_formatNHWC*/
_output_shapes
:���������K@
m
model/conv2/conv2/ReluRelumodel/conv2/BiasAdd*
T0*/
_output_shapes
:���������K@
�
model/conv2/MaxPool2D/MaxPoolMaxPoolmodel/conv2/conv2/Relu*
T0*
data_formatNHWC*
strides
*
ksize
*
paddingSAME*/
_output_shapes
:���������&@
�
.conv3/weights/Initializer/random_uniform/shapeConst*%
valueB"      @   �   * 
_class
loc:@conv3/weights*
dtype0*
_output_shapes
:
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
,conv3/weights/Initializer/random_uniform/maxConst*
valueB
 *�[q=* 
_class
loc:@conv3/weights*
dtype0*
_output_shapes
: 
�
6conv3/weights/Initializer/random_uniform/RandomUniformRandomUniform.conv3/weights/Initializer/random_uniform/shape*
dtype0*'
_output_shapes
:@�*

seed *
T0* 
_class
loc:@conv3/weights*
seed2 
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
dtype0*
_output_shapes
:*
valueB"      
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
:���������&�
�
model/conv3/BiasAddBiasAddmodel/conv3/Conv2Dconv3/biases/read*
data_formatNHWC*0
_output_shapes
:���������&�*
T0
n
model/conv3/conv3/ReluRelumodel/conv3/BiasAdd*
T0*0
_output_shapes
:���������&�
�
model/conv3/MaxPool2D/MaxPoolMaxPoolmodel/conv3/conv3/Relu*
ksize
*
paddingSAME*0
_output_shapes
:����������*
T0*
data_formatNHWC*
strides

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
6conv4/weights/Initializer/random_uniform/RandomUniformRandomUniform.conv4/weights/Initializer/random_uniform/shape*
T0* 
_class
loc:@conv4/weights*
seed2 *
dtype0*(
_output_shapes
:��*

seed 
�
,conv4/weights/Initializer/random_uniform/subSub,conv4/weights/Initializer/random_uniform/max,conv4/weights/Initializer/random_uniform/min* 
_class
loc:@conv4/weights*
_output_shapes
: *
T0
�
,conv4/weights/Initializer/random_uniform/mulMul6conv4/weights/Initializer/random_uniform/RandomUniform,conv4/weights/Initializer/random_uniform/sub*
T0* 
_class
loc:@conv4/weights*(
_output_shapes
:��
�
(conv4/weights/Initializer/random_uniformAdd,conv4/weights/Initializer/random_uniform/mul,conv4/weights/Initializer/random_uniform/min*(
_output_shapes
:��*
T0* 
_class
loc:@conv4/weights
�
conv4/weights
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
conv4/biases/Initializer/zerosConst*
_output_shapes	
:�*
valueB�*    *
_class
loc:@conv4/biases*
dtype0
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
conv4/biases/AssignAssignconv4/biasesconv4/biases/Initializer/zeros*
use_locking(*
T0*
_class
loc:@conv4/biases*
validate_shape(*
_output_shapes	
:�
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
model/conv4/Conv2DConv2Dmodel/conv3/MaxPool2D/MaxPoolconv4/weights/read*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*0
_output_shapes
:����������*
	dilations
*
T0
�
model/conv4/BiasAddBiasAddmodel/conv4/Conv2Dconv4/biases/read*
data_formatNHWC*0
_output_shapes
:����������*
T0
n
model/conv4/conv4/ReluRelumodel/conv4/BiasAdd*
T0*0
_output_shapes
:����������
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
:���������
�
�
.conv5/weights/Initializer/random_uniform/shapeConst*%
valueB"            * 
_class
loc:@conv5/weights*
dtype0*
_output_shapes
:
�
,conv5/weights/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *
valueB
 *���* 
_class
loc:@conv5/weights
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
,conv5/weights/Initializer/random_uniform/subSub,conv5/weights/Initializer/random_uniform/max,conv5/weights/Initializer/random_uniform/min* 
_class
loc:@conv5/weights*
_output_shapes
: *
T0
�
,conv5/weights/Initializer/random_uniform/mulMul6conv5/weights/Initializer/random_uniform/RandomUniform,conv5/weights/Initializer/random_uniform/sub*'
_output_shapes
:�*
T0* 
_class
loc:@conv5/weights
�
(conv5/weights/Initializer/random_uniformAdd,conv5/weights/Initializer/random_uniform/mul,conv5/weights/Initializer/random_uniform/min*'
_output_shapes
:�*
T0* 
_class
loc:@conv5/weights
�
conv5/weights
VariableV2*
shared_name * 
_class
loc:@conv5/weights*
	container *
shape:�*
dtype0*'
_output_shapes
:�
�
conv5/weights/AssignAssignconv5/weights(conv5/weights/Initializer/random_uniform*
use_locking(*
T0* 
_class
loc:@conv5/weights*
validate_shape(*'
_output_shapes
:�
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
shape:*
dtype0*
_output_shapes
:*
shared_name *
_class
loc:@conv5/biases*
	container 
�
conv5/biases/AssignAssignconv5/biasesconv5/biases/Initializer/zeros*
_output_shapes
:*
use_locking(*
T0*
_class
loc:@conv5/biases*
validate_shape(
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
model/conv5/Conv2DConv2Dmodel/conv4/MaxPool2D/MaxPoolconv5/weights/read*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*/
_output_shapes
:���������

�
model/conv5/BiasAddBiasAddmodel/conv5/Conv2Dconv5/biases/read*
T0*
data_formatNHWC*/
_output_shapes
:���������

�
model/conv5/MaxPool2D/MaxPoolMaxPoolmodel/conv5/BiasAdd*/
_output_shapes
:���������*
T0*
data_formatNHWC*
strides
*
ksize
*
paddingSAME
x
model/Flatten/flatten/ShapeShapemodel/conv5/MaxPool2D/MaxPool*
_output_shapes
:*
T0*
out_type0
s
)model/Flatten/flatten/strided_slice/stackConst*
_output_shapes
:*
valueB: *
dtype0
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
model/Flatten/flatten/ReshapeReshapemodel/conv5/MaxPool2D/MaxPool#model/Flatten/flatten/Reshape/shape*
Tshape0*(
_output_shapes
:����������*
T0
l
model_1/conv1/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
�
model_1/conv1/Conv2DConv2Dpositive_inputconv1/weights/read*
paddingSAME*0
_output_shapes
:���������2� *
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
�
model_1/conv1/BiasAddBiasAddmodel_1/conv1/Conv2Dconv1/biases/read*
T0*
data_formatNHWC*0
_output_shapes
:���������2� 
r
model_1/conv1/conv1/ReluRelumodel_1/conv1/BiasAdd*
T0*0
_output_shapes
:���������2� 
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
:���������K 
l
model_1/conv2/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
�
model_1/conv2/Conv2DConv2Dmodel_1/conv1/MaxPool2D/MaxPoolconv2/weights/read*/
_output_shapes
:���������K@*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME
�
model_1/conv2/BiasAddBiasAddmodel_1/conv2/Conv2Dconv2/biases/read*
T0*
data_formatNHWC*/
_output_shapes
:���������K@
q
model_1/conv2/conv2/ReluRelumodel_1/conv2/BiasAdd*
T0*/
_output_shapes
:���������K@
�
model_1/conv2/MaxPool2D/MaxPoolMaxPoolmodel_1/conv2/conv2/Relu*/
_output_shapes
:���������&@*
T0*
data_formatNHWC*
strides
*
ksize
*
paddingSAME
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
:���������&�*
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
:���������&�
r
model_1/conv3/conv3/ReluRelumodel_1/conv3/BiasAdd*
T0*0
_output_shapes
:���������&�
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
:����������
l
model_1/conv4/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
�
model_1/conv4/Conv2DConv2Dmodel_1/conv3/MaxPool2D/MaxPoolconv4/weights/read*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*0
_output_shapes
:����������
�
model_1/conv4/BiasAddBiasAddmodel_1/conv4/Conv2Dconv4/biases/read*
T0*
data_formatNHWC*0
_output_shapes
:����������
r
model_1/conv4/conv4/ReluRelumodel_1/conv4/BiasAdd*
T0*0
_output_shapes
:����������
�
model_1/conv4/MaxPool2D/MaxPoolMaxPoolmodel_1/conv4/conv4/Relu*
ksize
*
paddingSAME*0
_output_shapes
:���������
�*
T0*
data_formatNHWC*
strides

l
model_1/conv5/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
�
model_1/conv5/Conv2DConv2Dmodel_1/conv4/MaxPool2D/MaxPoolconv5/weights/read*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*/
_output_shapes
:���������
*
	dilations

�
model_1/conv5/BiasAddBiasAddmodel_1/conv5/Conv2Dconv5/biases/read*
T0*
data_formatNHWC*/
_output_shapes
:���������

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
:���������
|
model_1/Flatten/flatten/ShapeShapemodel_1/conv5/MaxPool2D/MaxPool*
out_type0*
_output_shapes
:*
T0
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
%model_1/Flatten/flatten/strided_sliceStridedSlicemodel_1/Flatten/flatten/Shape+model_1/Flatten/flatten/strided_slice/stack-model_1/Flatten/flatten/strided_slice/stack_1-model_1/Flatten/flatten/strided_slice/stack_2*
_output_shapes
: *
Index0*
T0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask 
r
'model_1/Flatten/flatten/Reshape/shape/1Const*
valueB :
���������*
dtype0*
_output_shapes
: 
�
%model_1/Flatten/flatten/Reshape/shapePack%model_1/Flatten/flatten/strided_slice'model_1/Flatten/flatten/Reshape/shape/1*
N*
_output_shapes
:*
T0*

axis 
�
model_1/Flatten/flatten/ReshapeReshapemodel_1/conv5/MaxPool2D/MaxPool%model_1/Flatten/flatten/Reshape/shape*
T0*
Tshape0*(
_output_shapes
:����������
l
model_2/conv1/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
�
model_2/conv1/Conv2DConv2Dnegative_inputconv1/weights/read*
paddingSAME*0
_output_shapes
:���������2� *
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
�
model_2/conv1/BiasAddBiasAddmodel_2/conv1/Conv2Dconv1/biases/read*0
_output_shapes
:���������2� *
T0*
data_formatNHWC
r
model_2/conv1/conv1/ReluRelumodel_2/conv1/BiasAdd*
T0*0
_output_shapes
:���������2� 
�
model_2/conv1/MaxPool2D/MaxPoolMaxPoolmodel_2/conv1/conv1/Relu*
ksize
*
paddingSAME*/
_output_shapes
:���������K *
T0*
data_formatNHWC*
strides

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
:���������K@*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
�
model_2/conv2/BiasAddBiasAddmodel_2/conv2/Conv2Dconv2/biases/read*
T0*
data_formatNHWC*/
_output_shapes
:���������K@
q
model_2/conv2/conv2/ReluRelumodel_2/conv2/BiasAdd*
T0*/
_output_shapes
:���������K@
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
:���������&@
l
model_2/conv3/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
�
model_2/conv3/Conv2DConv2Dmodel_2/conv2/MaxPool2D/MaxPoolconv3/weights/read*0
_output_shapes
:���������&�*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME
�
model_2/conv3/BiasAddBiasAddmodel_2/conv3/Conv2Dconv3/biases/read*
T0*
data_formatNHWC*0
_output_shapes
:���������&�
r
model_2/conv3/conv3/ReluRelumodel_2/conv3/BiasAdd*
T0*0
_output_shapes
:���������&�
�
model_2/conv3/MaxPool2D/MaxPoolMaxPoolmodel_2/conv3/conv3/Relu*
T0*
data_formatNHWC*
strides
*
ksize
*
paddingSAME*0
_output_shapes
:����������
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
:����������*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
�
model_2/conv4/BiasAddBiasAddmodel_2/conv4/Conv2Dconv4/biases/read*0
_output_shapes
:����������*
T0*
data_formatNHWC
r
model_2/conv4/conv4/ReluRelumodel_2/conv4/BiasAdd*
T0*0
_output_shapes
:����������
�
model_2/conv4/MaxPool2D/MaxPoolMaxPoolmodel_2/conv4/conv4/Relu*
data_formatNHWC*
strides
*
ksize
*
paddingSAME*0
_output_shapes
:���������
�*
T0
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
:���������
*
	dilations
*
T0*
strides
*
data_formatNHWC
�
model_2/conv5/BiasAddBiasAddmodel_2/conv5/Conv2Dconv5/biases/read*
T0*
data_formatNHWC*/
_output_shapes
:���������

�
model_2/conv5/MaxPool2D/MaxPoolMaxPoolmodel_2/conv5/BiasAdd*
T0*
data_formatNHWC*
strides
*
ksize
*
paddingSAME*/
_output_shapes
:���������
|
model_2/Flatten/flatten/ShapeShapemodel_2/conv5/MaxPool2D/MaxPool*
T0*
out_type0*
_output_shapes
:
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
%model_2/Flatten/flatten/strided_sliceStridedSlicemodel_2/Flatten/flatten/Shape+model_2/Flatten/flatten/strided_slice/stack-model_2/Flatten/flatten/strided_slice/stack_1-model_2/Flatten/flatten/strided_slice/stack_2*
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
'model_2/Flatten/flatten/Reshape/shape/1Const*
valueB :
���������*
dtype0*
_output_shapes
: 
�
%model_2/Flatten/flatten/Reshape/shapePack%model_2/Flatten/flatten/strided_slice'model_2/Flatten/flatten/Reshape/shape/1*
N*
_output_shapes
:*
T0*

axis 
�
model_2/Flatten/flatten/ReshapeReshapemodel_2/conv5/MaxPool2D/MaxPool%model_2/Flatten/flatten/Reshape/shape*
T0*
Tshape0*(
_output_shapes
:����������

mulMulmodel_1/Flatten/flatten/Reshapemodel_2/Flatten/flatten/Reshape*
T0*(
_output_shapes
:����������
W
Sum/reduction_indicesConst*
value	B :*
dtype0*
_output_shapes
: 
q
SumSummulSum/reduction_indices*#
_output_shapes
:���������*
	keep_dims( *

Tidx0*
T0
J
Pow/yConst*
valueB
 *   @*
dtype0*
_output_shapes
: 
e
PowPowmodel_1/Flatten/flatten/ReshapePow/y*
T0*(
_output_shapes
:����������
Y
Sum_1/reduction_indicesConst*
value	B :*
dtype0*
_output_shapes
: 
u
Sum_1SumPowSum_1/reduction_indices*#
_output_shapes
:���������*
	keep_dims( *

Tidx0*
T0
A
SqrtSqrtSum_1*#
_output_shapes
:���������*
T0
L
Pow_1/yConst*
valueB
 *   @*
dtype0*
_output_shapes
: 
i
Pow_1Powmodel_2/Flatten/flatten/ReshapePow_1/y*
T0*(
_output_shapes
:����������
Y
Sum_2/reduction_indicesConst*
value	B :*
dtype0*
_output_shapes
: 
w
Sum_2SumPow_1Sum_2/reduction_indices*
	keep_dims( *

Tidx0*
T0*#
_output_shapes
:���������
C
Sqrt_1SqrtSum_2*
T0*#
_output_shapes
:���������
H
mul_1MulSqrtSqrt_1*
T0*#
_output_shapes
:���������
L
truedivRealDivSummul_1*
T0*#
_output_shapes
:���������

mul_2Mulmodel_1/Flatten/flatten/Reshapemodel/Flatten/flatten/Reshape*
T0*(
_output_shapes
:����������
Y
Sum_3/reduction_indicesConst*
value	B :*
dtype0*
_output_shapes
: 
w
Sum_3Summul_2Sum_3/reduction_indices*
T0*#
_output_shapes
:���������*
	keep_dims( *

Tidx0
L
Pow_2/yConst*
dtype0*
_output_shapes
: *
valueB
 *   @
i
Pow_2Powmodel_1/Flatten/flatten/ReshapePow_2/y*
T0*(
_output_shapes
:����������
Y
Sum_4/reduction_indicesConst*
value	B :*
dtype0*
_output_shapes
: 
w
Sum_4SumPow_2Sum_4/reduction_indices*
T0*#
_output_shapes
:���������*
	keep_dims( *

Tidx0
C
Sqrt_2SqrtSum_4*#
_output_shapes
:���������*
T0
L
Pow_3/yConst*
valueB
 *   @*
dtype0*
_output_shapes
: 
g
Pow_3Powmodel/Flatten/flatten/ReshapePow_3/y*
T0*(
_output_shapes
:����������
Y
Sum_5/reduction_indicesConst*
value	B :*
dtype0*
_output_shapes
: 
w
Sum_5SumPow_3Sum_5/reduction_indices*
T0*#
_output_shapes
:���������*
	keep_dims( *

Tidx0
C
Sqrt_3SqrtSum_5*
T0*#
_output_shapes
:���������
J
mul_3MulSqrt_2Sqrt_3*#
_output_shapes
:���������*
T0
P
	truediv_1RealDivSum_3mul_3*
T0*#
_output_shapes
:���������

mul_4Mulmodel_2/Flatten/flatten/Reshapemodel/Flatten/flatten/Reshape*
T0*(
_output_shapes
:����������
Y
Sum_6/reduction_indicesConst*
dtype0*
_output_shapes
: *
value	B :
w
Sum_6Summul_4Sum_6/reduction_indices*
	keep_dims( *

Tidx0*
T0*#
_output_shapes
:���������
L
Pow_4/yConst*
valueB
 *   @*
dtype0*
_output_shapes
: 
i
Pow_4Powmodel_2/Flatten/flatten/ReshapePow_4/y*
T0*(
_output_shapes
:����������
Y
Sum_7/reduction_indicesConst*
dtype0*
_output_shapes
: *
value	B :
w
Sum_7SumPow_4Sum_7/reduction_indices*
T0*#
_output_shapes
:���������*
	keep_dims( *

Tidx0
C
Sqrt_4SqrtSum_7*
T0*#
_output_shapes
:���������
L
Pow_5/yConst*
dtype0*
_output_shapes
: *
valueB
 *   @
g
Pow_5Powmodel/Flatten/flatten/ReshapePow_5/y*
T0*(
_output_shapes
:����������
Y
Sum_8/reduction_indicesConst*
value	B :*
dtype0*
_output_shapes
: 
w
Sum_8SumPow_5Sum_8/reduction_indices*
T0*#
_output_shapes
:���������*
	keep_dims( *

Tidx0
C
Sqrt_5SqrtSum_8*
T0*#
_output_shapes
:���������
J
mul_5MulSqrt_4Sqrt_5*#
_output_shapes
:���������*
T0
P
	truediv_2RealDivSum_6mul_5*
T0*#
_output_shapes
:���������
}
subSubmodel/Flatten/flatten/Reshapemodel_1/Flatten/flatten/Reshape*
T0*(
_output_shapes
:����������
L
Pow_6/yConst*
valueB
 *   @*
dtype0*
_output_shapes
: 
M
Pow_6PowsubPow_6/y*
T0*(
_output_shapes
:����������
Y
Sum_9/reduction_indicesConst*
value	B :*
dtype0*
_output_shapes
: 
{
Sum_9SumPow_6Sum_9/reduction_indices*
	keep_dims(*

Tidx0*
T0*'
_output_shapes
:���������
G
Sqrt_6SqrtSum_9*
T0*'
_output_shapes
:���������

sub_1Submodel/Flatten/flatten/Reshapemodel_2/Flatten/flatten/Reshape*(
_output_shapes
:����������*
T0
L
Pow_7/yConst*
dtype0*
_output_shapes
: *
valueB
 *   @
O
Pow_7Powsub_1Pow_7/y*
T0*(
_output_shapes
:����������
Z
Sum_10/reduction_indicesConst*
value	B :*
dtype0*
_output_shapes
: 
}
Sum_10SumPow_7Sum_10/reduction_indices*
T0*'
_output_shapes
:���������*
	keep_dims(*

Tidx0
H
Sqrt_7SqrtSum_10*
T0*'
_output_shapes
:���������
N
sub_2SubSqrt_6Sqrt_7*
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
dtype0*
_output_shapes
:*
valueB"       
Z
MeanMeanMaximumConst*
	keep_dims( *

Tidx0*
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
gradients/grad_ys_0Const*
valueB
 *  �?*
dtype0*
_output_shapes
: 
o
gradients/FillFillgradients/Shapegradients/grad_ys_0*
T0*

index_type0*
_output_shapes
: 
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
gradients/Mean_grad/TileTilegradients/Mean_grad/Reshapegradients/Mean_grad/Shape*
T0*'
_output_shapes
:���������*

Tmultiples0
b
gradients/Mean_grad/Shape_1ShapeMaximum*
_output_shapes
:*
T0*
out_type0
^
gradients/Mean_grad/Shape_2Const*
dtype0*
_output_shapes
: *
valueB 
c
gradients/Mean_grad/ConstConst*
valueB: *
dtype0*
_output_shapes
:
�
gradients/Mean_grad/ProdProdgradients/Mean_grad/Shape_1gradients/Mean_grad/Const*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
e
gradients/Mean_grad/Const_1Const*
dtype0*
_output_shapes
:*
valueB: 
�
gradients/Mean_grad/Prod_1Prodgradients/Mean_grad/Shape_2gradients/Mean_grad/Const_1*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
_
gradients/Mean_grad/Maximum/yConst*
value	B :*
dtype0*
_output_shapes
: 
�
gradients/Mean_grad/MaximumMaximumgradients/Mean_grad/Prod_1gradients/Mean_grad/Maximum/y*
_output_shapes
: *
T0
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
gradients/Mean_grad/truedivRealDivgradients/Mean_grad/Tilegradients/Mean_grad/Cast*'
_output_shapes
:���������*
T0
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
gradients/Maximum_grad/Shape_2Shapegradients/Mean_grad/truediv*
_output_shapes
:*
T0*
out_type0
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
,gradients/Maximum_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/Maximum_grad/Shapegradients/Maximum_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
gradients/Maximum_grad/SelectSelect#gradients/Maximum_grad/GreaterEqualgradients/Mean_grad/truedivgradients/Maximum_grad/zeros*
T0*'
_output_shapes
:���������
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
 gradients/Maximum_grad/Reshape_1Reshapegradients/Maximum_grad/Sum_1gradients/Maximum_grad/Shape_1*
_output_shapes
: *
T0*
Tshape0
s
'gradients/Maximum_grad/tuple/group_depsNoOp^gradients/Maximum_grad/Reshape!^gradients/Maximum_grad/Reshape_1
�
/gradients/Maximum_grad/tuple/control_dependencyIdentitygradients/Maximum_grad/Reshape(^gradients/Maximum_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/Maximum_grad/Reshape*'
_output_shapes
:���������
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
gradients/add_grad/Sum_1Sum/gradients/Maximum_grad/tuple/control_dependency*gradients/add_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
gradients/add_grad/Reshape_1Reshapegradients/add_grad/Sum_1gradients/add_grad/Shape_1*
_output_shapes
: *
T0*
Tshape0
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
`
gradients/sub_2_grad/ShapeShapeSqrt_6*
T0*
out_type0*
_output_shapes
:
b
gradients/sub_2_grad/Shape_1ShapeSqrt_7*
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
/gradients/sub_2_grad/tuple/control_dependency_1Identitygradients/sub_2_grad/Reshape_1&^gradients/sub_2_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/sub_2_grad/Reshape_1*'
_output_shapes
:���������
�
gradients/Sqrt_6_grad/SqrtGradSqrtGradSqrt_6-gradients/sub_2_grad/tuple/control_dependency*
T0*'
_output_shapes
:���������
�
gradients/Sqrt_7_grad/SqrtGradSqrtGradSqrt_7/gradients/sub_2_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:���������
_
gradients/Sum_9_grad/ShapeShapePow_6*
T0*
out_type0*
_output_shapes
:
�
gradients/Sum_9_grad/SizeConst*
value	B :*-
_class#
!loc:@gradients/Sum_9_grad/Shape*
dtype0*
_output_shapes
: 
�
gradients/Sum_9_grad/addAddSum_9/reduction_indicesgradients/Sum_9_grad/Size*
T0*-
_class#
!loc:@gradients/Sum_9_grad/Shape*
_output_shapes
: 
�
gradients/Sum_9_grad/modFloorModgradients/Sum_9_grad/addgradients/Sum_9_grad/Size*
T0*-
_class#
!loc:@gradients/Sum_9_grad/Shape*
_output_shapes
: 
�
gradients/Sum_9_grad/Shape_1Const*
valueB *-
_class#
!loc:@gradients/Sum_9_grad/Shape*
dtype0*
_output_shapes
: 
�
 gradients/Sum_9_grad/range/startConst*
value	B : *-
_class#
!loc:@gradients/Sum_9_grad/Shape*
dtype0*
_output_shapes
: 
�
 gradients/Sum_9_grad/range/deltaConst*
value	B :*-
_class#
!loc:@gradients/Sum_9_grad/Shape*
dtype0*
_output_shapes
: 
�
gradients/Sum_9_grad/rangeRange gradients/Sum_9_grad/range/startgradients/Sum_9_grad/Size gradients/Sum_9_grad/range/delta*-
_class#
!loc:@gradients/Sum_9_grad/Shape*
_output_shapes
:*

Tidx0
�
gradients/Sum_9_grad/Fill/valueConst*
value	B :*-
_class#
!loc:@gradients/Sum_9_grad/Shape*
dtype0*
_output_shapes
: 
�
gradients/Sum_9_grad/FillFillgradients/Sum_9_grad/Shape_1gradients/Sum_9_grad/Fill/value*
T0*

index_type0*-
_class#
!loc:@gradients/Sum_9_grad/Shape*
_output_shapes
: 
�
"gradients/Sum_9_grad/DynamicStitchDynamicStitchgradients/Sum_9_grad/rangegradients/Sum_9_grad/modgradients/Sum_9_grad/Shapegradients/Sum_9_grad/Fill*
T0*-
_class#
!loc:@gradients/Sum_9_grad/Shape*
N*
_output_shapes
:
�
gradients/Sum_9_grad/Maximum/yConst*
value	B :*-
_class#
!loc:@gradients/Sum_9_grad/Shape*
dtype0*
_output_shapes
: 
�
gradients/Sum_9_grad/MaximumMaximum"gradients/Sum_9_grad/DynamicStitchgradients/Sum_9_grad/Maximum/y*
_output_shapes
:*
T0*-
_class#
!loc:@gradients/Sum_9_grad/Shape
�
gradients/Sum_9_grad/floordivFloorDivgradients/Sum_9_grad/Shapegradients/Sum_9_grad/Maximum*
T0*-
_class#
!loc:@gradients/Sum_9_grad/Shape*
_output_shapes
:
�
gradients/Sum_9_grad/ReshapeReshapegradients/Sqrt_6_grad/SqrtGrad"gradients/Sum_9_grad/DynamicStitch*
T0*
Tshape0*0
_output_shapes
:������������������
�
gradients/Sum_9_grad/TileTilegradients/Sum_9_grad/Reshapegradients/Sum_9_grad/floordiv*(
_output_shapes
:����������*

Tmultiples0*
T0
`
gradients/Sum_10_grad/ShapeShapePow_7*
T0*
out_type0*
_output_shapes
:
�
gradients/Sum_10_grad/SizeConst*
value	B :*.
_class$
" loc:@gradients/Sum_10_grad/Shape*
dtype0*
_output_shapes
: 
�
gradients/Sum_10_grad/addAddSum_10/reduction_indicesgradients/Sum_10_grad/Size*
T0*.
_class$
" loc:@gradients/Sum_10_grad/Shape*
_output_shapes
: 
�
gradients/Sum_10_grad/modFloorModgradients/Sum_10_grad/addgradients/Sum_10_grad/Size*
T0*.
_class$
" loc:@gradients/Sum_10_grad/Shape*
_output_shapes
: 
�
gradients/Sum_10_grad/Shape_1Const*
valueB *.
_class$
" loc:@gradients/Sum_10_grad/Shape*
dtype0*
_output_shapes
: 
�
!gradients/Sum_10_grad/range/startConst*
value	B : *.
_class$
" loc:@gradients/Sum_10_grad/Shape*
dtype0*
_output_shapes
: 
�
!gradients/Sum_10_grad/range/deltaConst*
dtype0*
_output_shapes
: *
value	B :*.
_class$
" loc:@gradients/Sum_10_grad/Shape
�
gradients/Sum_10_grad/rangeRange!gradients/Sum_10_grad/range/startgradients/Sum_10_grad/Size!gradients/Sum_10_grad/range/delta*.
_class$
" loc:@gradients/Sum_10_grad/Shape*
_output_shapes
:*

Tidx0
�
 gradients/Sum_10_grad/Fill/valueConst*
dtype0*
_output_shapes
: *
value	B :*.
_class$
" loc:@gradients/Sum_10_grad/Shape
�
gradients/Sum_10_grad/FillFillgradients/Sum_10_grad/Shape_1 gradients/Sum_10_grad/Fill/value*
T0*

index_type0*.
_class$
" loc:@gradients/Sum_10_grad/Shape*
_output_shapes
: 
�
#gradients/Sum_10_grad/DynamicStitchDynamicStitchgradients/Sum_10_grad/rangegradients/Sum_10_grad/modgradients/Sum_10_grad/Shapegradients/Sum_10_grad/Fill*
T0*.
_class$
" loc:@gradients/Sum_10_grad/Shape*
N*
_output_shapes
:
�
gradients/Sum_10_grad/Maximum/yConst*
value	B :*.
_class$
" loc:@gradients/Sum_10_grad/Shape*
dtype0*
_output_shapes
: 
�
gradients/Sum_10_grad/MaximumMaximum#gradients/Sum_10_grad/DynamicStitchgradients/Sum_10_grad/Maximum/y*
T0*.
_class$
" loc:@gradients/Sum_10_grad/Shape*
_output_shapes
:
�
gradients/Sum_10_grad/floordivFloorDivgradients/Sum_10_grad/Shapegradients/Sum_10_grad/Maximum*
_output_shapes
:*
T0*.
_class$
" loc:@gradients/Sum_10_grad/Shape
�
gradients/Sum_10_grad/ReshapeReshapegradients/Sqrt_7_grad/SqrtGrad#gradients/Sum_10_grad/DynamicStitch*0
_output_shapes
:������������������*
T0*
Tshape0
�
gradients/Sum_10_grad/TileTilegradients/Sum_10_grad/Reshapegradients/Sum_10_grad/floordiv*

Tmultiples0*
T0*(
_output_shapes
:����������
]
gradients/Pow_6_grad/ShapeShapesub*
T0*
out_type0*
_output_shapes
:
_
gradients/Pow_6_grad/Shape_1Const*
dtype0*
_output_shapes
: *
valueB 
�
*gradients/Pow_6_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/Pow_6_grad/Shapegradients/Pow_6_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
v
gradients/Pow_6_grad/mulMulgradients/Sum_9_grad/TilePow_6/y*
T0*(
_output_shapes
:����������
_
gradients/Pow_6_grad/sub/yConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
e
gradients/Pow_6_grad/subSubPow_6/ygradients/Pow_6_grad/sub/y*
_output_shapes
: *
T0
q
gradients/Pow_6_grad/PowPowsubgradients/Pow_6_grad/sub*
T0*(
_output_shapes
:����������
�
gradients/Pow_6_grad/mul_1Mulgradients/Pow_6_grad/mulgradients/Pow_6_grad/Pow*
T0*(
_output_shapes
:����������
�
gradients/Pow_6_grad/SumSumgradients/Pow_6_grad/mul_1*gradients/Pow_6_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
gradients/Pow_6_grad/ReshapeReshapegradients/Pow_6_grad/Sumgradients/Pow_6_grad/Shape*(
_output_shapes
:����������*
T0*
Tshape0
c
gradients/Pow_6_grad/Greater/yConst*
dtype0*
_output_shapes
: *
valueB
 *    

gradients/Pow_6_grad/GreaterGreatersubgradients/Pow_6_grad/Greater/y*(
_output_shapes
:����������*
T0
g
$gradients/Pow_6_grad/ones_like/ShapeShapesub*
T0*
out_type0*
_output_shapes
:
i
$gradients/Pow_6_grad/ones_like/ConstConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
gradients/Pow_6_grad/ones_likeFill$gradients/Pow_6_grad/ones_like/Shape$gradients/Pow_6_grad/ones_like/Const*
T0*

index_type0*(
_output_shapes
:����������
�
gradients/Pow_6_grad/SelectSelectgradients/Pow_6_grad/Greatersubgradients/Pow_6_grad/ones_like*
T0*(
_output_shapes
:����������
o
gradients/Pow_6_grad/LogLoggradients/Pow_6_grad/Select*(
_output_shapes
:����������*
T0
d
gradients/Pow_6_grad/zeros_like	ZerosLikesub*
T0*(
_output_shapes
:����������
�
gradients/Pow_6_grad/Select_1Selectgradients/Pow_6_grad/Greatergradients/Pow_6_grad/Loggradients/Pow_6_grad/zeros_like*
T0*(
_output_shapes
:����������
v
gradients/Pow_6_grad/mul_2Mulgradients/Sum_9_grad/TilePow_6*(
_output_shapes
:����������*
T0
�
gradients/Pow_6_grad/mul_3Mulgradients/Pow_6_grad/mul_2gradients/Pow_6_grad/Select_1*(
_output_shapes
:����������*
T0
�
gradients/Pow_6_grad/Sum_1Sumgradients/Pow_6_grad/mul_3,gradients/Pow_6_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
gradients/Pow_6_grad/Reshape_1Reshapegradients/Pow_6_grad/Sum_1gradients/Pow_6_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
m
%gradients/Pow_6_grad/tuple/group_depsNoOp^gradients/Pow_6_grad/Reshape^gradients/Pow_6_grad/Reshape_1
�
-gradients/Pow_6_grad/tuple/control_dependencyIdentitygradients/Pow_6_grad/Reshape&^gradients/Pow_6_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/Pow_6_grad/Reshape*(
_output_shapes
:����������
�
/gradients/Pow_6_grad/tuple/control_dependency_1Identitygradients/Pow_6_grad/Reshape_1&^gradients/Pow_6_grad/tuple/group_deps*
_output_shapes
: *
T0*1
_class'
%#loc:@gradients/Pow_6_grad/Reshape_1
_
gradients/Pow_7_grad/ShapeShapesub_1*
T0*
out_type0*
_output_shapes
:
_
gradients/Pow_7_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
�
*gradients/Pow_7_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/Pow_7_grad/Shapegradients/Pow_7_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
w
gradients/Pow_7_grad/mulMulgradients/Sum_10_grad/TilePow_7/y*(
_output_shapes
:����������*
T0
_
gradients/Pow_7_grad/sub/yConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
e
gradients/Pow_7_grad/subSubPow_7/ygradients/Pow_7_grad/sub/y*
T0*
_output_shapes
: 
s
gradients/Pow_7_grad/PowPowsub_1gradients/Pow_7_grad/sub*(
_output_shapes
:����������*
T0
�
gradients/Pow_7_grad/mul_1Mulgradients/Pow_7_grad/mulgradients/Pow_7_grad/Pow*
T0*(
_output_shapes
:����������
�
gradients/Pow_7_grad/SumSumgradients/Pow_7_grad/mul_1*gradients/Pow_7_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
gradients/Pow_7_grad/ReshapeReshapegradients/Pow_7_grad/Sumgradients/Pow_7_grad/Shape*
T0*
Tshape0*(
_output_shapes
:����������
c
gradients/Pow_7_grad/Greater/yConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
gradients/Pow_7_grad/GreaterGreatersub_1gradients/Pow_7_grad/Greater/y*
T0*(
_output_shapes
:����������
i
$gradients/Pow_7_grad/ones_like/ShapeShapesub_1*
T0*
out_type0*
_output_shapes
:
i
$gradients/Pow_7_grad/ones_like/ConstConst*
_output_shapes
: *
valueB
 *  �?*
dtype0
�
gradients/Pow_7_grad/ones_likeFill$gradients/Pow_7_grad/ones_like/Shape$gradients/Pow_7_grad/ones_like/Const*

index_type0*(
_output_shapes
:����������*
T0
�
gradients/Pow_7_grad/SelectSelectgradients/Pow_7_grad/Greatersub_1gradients/Pow_7_grad/ones_like*
T0*(
_output_shapes
:����������
o
gradients/Pow_7_grad/LogLoggradients/Pow_7_grad/Select*
T0*(
_output_shapes
:����������
f
gradients/Pow_7_grad/zeros_like	ZerosLikesub_1*(
_output_shapes
:����������*
T0
�
gradients/Pow_7_grad/Select_1Selectgradients/Pow_7_grad/Greatergradients/Pow_7_grad/Loggradients/Pow_7_grad/zeros_like*
T0*(
_output_shapes
:����������
w
gradients/Pow_7_grad/mul_2Mulgradients/Sum_10_grad/TilePow_7*(
_output_shapes
:����������*
T0
�
gradients/Pow_7_grad/mul_3Mulgradients/Pow_7_grad/mul_2gradients/Pow_7_grad/Select_1*
T0*(
_output_shapes
:����������
�
gradients/Pow_7_grad/Sum_1Sumgradients/Pow_7_grad/mul_3,gradients/Pow_7_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
gradients/Pow_7_grad/Reshape_1Reshapegradients/Pow_7_grad/Sum_1gradients/Pow_7_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
m
%gradients/Pow_7_grad/tuple/group_depsNoOp^gradients/Pow_7_grad/Reshape^gradients/Pow_7_grad/Reshape_1
�
-gradients/Pow_7_grad/tuple/control_dependencyIdentitygradients/Pow_7_grad/Reshape&^gradients/Pow_7_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/Pow_7_grad/Reshape*(
_output_shapes
:����������
�
/gradients/Pow_7_grad/tuple/control_dependency_1Identitygradients/Pow_7_grad/Reshape_1&^gradients/Pow_7_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/Pow_7_grad/Reshape_1*
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
gradients/sub_grad/SumSum-gradients/Pow_6_grad/tuple/control_dependency(gradients/sub_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
gradients/sub_grad/ReshapeReshapegradients/sub_grad/Sumgradients/sub_grad/Shape*
T0*
Tshape0*(
_output_shapes
:����������
�
gradients/sub_grad/Sum_1Sum-gradients/Pow_6_grad/tuple/control_dependency*gradients/sub_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
Z
gradients/sub_grad/NegNeggradients/sub_grad/Sum_1*
T0*
_output_shapes
:
�
gradients/sub_grad/Reshape_1Reshapegradients/sub_grad/Neggradients/sub_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:����������
g
#gradients/sub_grad/tuple/group_depsNoOp^gradients/sub_grad/Reshape^gradients/sub_grad/Reshape_1
�
+gradients/sub_grad/tuple/control_dependencyIdentitygradients/sub_grad/Reshape$^gradients/sub_grad/tuple/group_deps*
T0*-
_class#
!loc:@gradients/sub_grad/Reshape*(
_output_shapes
:����������
�
-gradients/sub_grad/tuple/control_dependency_1Identitygradients/sub_grad/Reshape_1$^gradients/sub_grad/tuple/group_deps*(
_output_shapes
:����������*
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
*gradients/sub_1_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/sub_1_grad/Shapegradients/sub_1_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
gradients/sub_1_grad/SumSum-gradients/Pow_7_grad/tuple/control_dependency*gradients/sub_1_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
gradients/sub_1_grad/ReshapeReshapegradients/sub_1_grad/Sumgradients/sub_1_grad/Shape*
T0*
Tshape0*(
_output_shapes
:����������
�
gradients/sub_1_grad/Sum_1Sum-gradients/Pow_7_grad/tuple/control_dependency,gradients/sub_1_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
^
gradients/sub_1_grad/NegNeggradients/sub_1_grad/Sum_1*
_output_shapes
:*
T0
�
gradients/sub_1_grad/Reshape_1Reshapegradients/sub_1_grad/Neggradients/sub_1_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:����������
m
%gradients/sub_1_grad/tuple/group_depsNoOp^gradients/sub_1_grad/Reshape^gradients/sub_1_grad/Reshape_1
�
-gradients/sub_1_grad/tuple/control_dependencyIdentitygradients/sub_1_grad/Reshape&^gradients/sub_1_grad/tuple/group_deps*(
_output_shapes
:����������*
T0*/
_class%
#!loc:@gradients/sub_1_grad/Reshape
�
/gradients/sub_1_grad/tuple/control_dependency_1Identitygradients/sub_1_grad/Reshape_1&^gradients/sub_1_grad/tuple/group_deps*1
_class'
%#loc:@gradients/sub_1_grad/Reshape_1*(
_output_shapes
:����������*
T0
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
:���������
�
gradients/AddNAddN+gradients/sub_grad/tuple/control_dependency-gradients/sub_1_grad/tuple/control_dependency*-
_class#
!loc:@gradients/sub_grad/Reshape*
N*(
_output_shapes
:����������*
T0
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
:���������
�
4gradients/model_2/Flatten/flatten/Reshape_grad/ShapeShapemodel_2/conv5/MaxPool2D/MaxPool*
_output_shapes
:*
T0*
out_type0
�
6gradients/model_2/Flatten/flatten/Reshape_grad/ReshapeReshape/gradients/sub_1_grad/tuple/control_dependency_14gradients/model_2/Flatten/flatten/Reshape_grad/Shape*
Tshape0*/
_output_shapes
:���������*
T0
�
:gradients/model_1/conv5/MaxPool2D/MaxPool_grad/MaxPoolGradMaxPoolGradmodel_1/conv5/BiasAddmodel_1/conv5/MaxPool2D/MaxPool6gradients/model_1/Flatten/flatten/Reshape_grad/Reshape*
ksize
*
paddingSAME*/
_output_shapes
:���������
*
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
:���������

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
:���������

�
0gradients/model_1/conv5/BiasAdd_grad/BiasAddGradBiasAddGrad:gradients/model_1/conv5/MaxPool2D/MaxPool_grad/MaxPoolGrad*
T0*
data_formatNHWC*
_output_shapes
:
�
5gradients/model_1/conv5/BiasAdd_grad/tuple/group_depsNoOp1^gradients/model_1/conv5/BiasAdd_grad/BiasAddGrad;^gradients/model_1/conv5/MaxPool2D/MaxPool_grad/MaxPoolGrad
�
=gradients/model_1/conv5/BiasAdd_grad/tuple/control_dependencyIdentity:gradients/model_1/conv5/MaxPool2D/MaxPool_grad/MaxPoolGrad6^gradients/model_1/conv5/BiasAdd_grad/tuple/group_deps*/
_output_shapes
:���������
*
T0*M
_classC
A?loc:@gradients/model_1/conv5/MaxPool2D/MaxPool_grad/MaxPoolGrad
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
:���������

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
=gradients/model_2/conv5/BiasAdd_grad/tuple/control_dependencyIdentity:gradients/model_2/conv5/MaxPool2D/MaxPool_grad/MaxPoolGrad6^gradients/model_2/conv5/BiasAdd_grad/tuple/group_deps*
T0*M
_classC
A?loc:@gradients/model_2/conv5/MaxPool2D/MaxPool_grad/MaxPoolGrad*/
_output_shapes
:���������

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
:���������
�*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(
�
8gradients/model_1/conv5/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFiltermodel_1/conv4/MaxPool2D/MaxPool,gradients/model_1/conv5/Conv2D_grad/ShapeN:1=gradients/model_1/conv5/BiasAdd_grad/tuple/control_dependency*
paddingSAME*'
_output_shapes
:�*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(
�
4gradients/model_1/conv5/Conv2D_grad/tuple/group_depsNoOp9^gradients/model_1/conv5/Conv2D_grad/Conv2DBackpropFilter8^gradients/model_1/conv5/Conv2D_grad/Conv2DBackpropInput
�
<gradients/model_1/conv5/Conv2D_grad/tuple/control_dependencyIdentity7gradients/model_1/conv5/Conv2D_grad/Conv2DBackpropInput5^gradients/model_1/conv5/Conv2D_grad/tuple/group_deps*
T0*J
_class@
><loc:@gradients/model_1/conv5/Conv2D_grad/Conv2DBackpropInput*0
_output_shapes
:���������
�
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
5gradients/model/conv5/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput(gradients/model/conv5/Conv2D_grad/ShapeNconv5/weights/read;gradients/model/conv5/BiasAdd_grad/tuple/control_dependency*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*0
_output_shapes
:���������
�
�
6gradients/model/conv5/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFiltermodel/conv4/MaxPool2D/MaxPool*gradients/model/conv5/Conv2D_grad/ShapeN:1;gradients/model/conv5/BiasAdd_grad/tuple/control_dependency*
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
2gradients/model/conv5/Conv2D_grad/tuple/group_depsNoOp7^gradients/model/conv5/Conv2D_grad/Conv2DBackpropFilter6^gradients/model/conv5/Conv2D_grad/Conv2DBackpropInput
�
:gradients/model/conv5/Conv2D_grad/tuple/control_dependencyIdentity5gradients/model/conv5/Conv2D_grad/Conv2DBackpropInput3^gradients/model/conv5/Conv2D_grad/tuple/group_deps*0
_output_shapes
:���������
�*
T0*H
_class>
<:loc:@gradients/model/conv5/Conv2D_grad/Conv2DBackpropInput
�
<gradients/model/conv5/Conv2D_grad/tuple/control_dependency_1Identity6gradients/model/conv5/Conv2D_grad/Conv2DBackpropFilter3^gradients/model/conv5/Conv2D_grad/tuple/group_deps*I
_class?
=;loc:@gradients/model/conv5/Conv2D_grad/Conv2DBackpropFilter*'
_output_shapes
:�*
T0
�
*gradients/model_2/conv5/Conv2D_grad/ShapeNShapeNmodel_2/conv4/MaxPool2D/MaxPoolconv5/weights/read*
N* 
_output_shapes
::*
T0*
out_type0
�
7gradients/model_2/conv5/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput*gradients/model_2/conv5/Conv2D_grad/ShapeNconv5/weights/read=gradients/model_2/conv5/BiasAdd_grad/tuple/control_dependency*
paddingSAME*0
_output_shapes
:���������
�*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
�
8gradients/model_2/conv5/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFiltermodel_2/conv4/MaxPool2D/MaxPool,gradients/model_2/conv5/Conv2D_grad/ShapeN:1=gradients/model_2/conv5/BiasAdd_grad/tuple/control_dependency*
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
4gradients/model_2/conv5/Conv2D_grad/tuple/group_depsNoOp9^gradients/model_2/conv5/Conv2D_grad/Conv2DBackpropFilter8^gradients/model_2/conv5/Conv2D_grad/Conv2DBackpropInput
�
<gradients/model_2/conv5/Conv2D_grad/tuple/control_dependencyIdentity7gradients/model_2/conv5/Conv2D_grad/Conv2DBackpropInput5^gradients/model_2/conv5/Conv2D_grad/tuple/group_deps*
T0*J
_class@
><loc:@gradients/model_2/conv5/Conv2D_grad/Conv2DBackpropInput*0
_output_shapes
:���������
�
�
>gradients/model_2/conv5/Conv2D_grad/tuple/control_dependency_1Identity8gradients/model_2/conv5/Conv2D_grad/Conv2DBackpropFilter5^gradients/model_2/conv5/Conv2D_grad/tuple/group_deps*
T0*K
_classA
?=loc:@gradients/model_2/conv5/Conv2D_grad/Conv2DBackpropFilter*'
_output_shapes
:�
�
gradients/AddN_1AddN?gradients/model_1/conv5/BiasAdd_grad/tuple/control_dependency_1=gradients/model/conv5/BiasAdd_grad/tuple/control_dependency_1?gradients/model_2/conv5/BiasAdd_grad/tuple/control_dependency_1*
_output_shapes
:*
T0*C
_class9
75loc:@gradients/model_1/conv5/BiasAdd_grad/BiasAddGrad*
N
�
:gradients/model_1/conv4/MaxPool2D/MaxPool_grad/MaxPoolGradMaxPoolGradmodel_1/conv4/conv4/Relumodel_1/conv4/MaxPool2D/MaxPool<gradients/model_1/conv5/Conv2D_grad/tuple/control_dependency*
data_formatNHWC*
strides
*
ksize
*
paddingSAME*0
_output_shapes
:����������*
T0
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
:����������
�
:gradients/model_2/conv4/MaxPool2D/MaxPool_grad/MaxPoolGradMaxPoolGradmodel_2/conv4/conv4/Relumodel_2/conv4/MaxPool2D/MaxPool<gradients/model_2/conv5/Conv2D_grad/tuple/control_dependency*
ksize
*
paddingSAME*0
_output_shapes
:����������*
T0*
data_formatNHWC*
strides

�
gradients/AddN_2AddN>gradients/model_1/conv5/Conv2D_grad/tuple/control_dependency_1<gradients/model/conv5/Conv2D_grad/tuple/control_dependency_1>gradients/model_2/conv5/Conv2D_grad/tuple/control_dependency_1*
T0*K
_classA
?=loc:@gradients/model_1/conv5/Conv2D_grad/Conv2DBackpropFilter*
N*'
_output_shapes
:�
�
0gradients/model_1/conv4/conv4/Relu_grad/ReluGradReluGrad:gradients/model_1/conv4/MaxPool2D/MaxPool_grad/MaxPoolGradmodel_1/conv4/conv4/Relu*
T0*0
_output_shapes
:����������
�
.gradients/model/conv4/conv4/Relu_grad/ReluGradReluGrad8gradients/model/conv4/MaxPool2D/MaxPool_grad/MaxPoolGradmodel/conv4/conv4/Relu*
T0*0
_output_shapes
:����������
�
0gradients/model_2/conv4/conv4/Relu_grad/ReluGradReluGrad:gradients/model_2/conv4/MaxPool2D/MaxPool_grad/MaxPoolGradmodel_2/conv4/conv4/Relu*
T0*0
_output_shapes
:����������
�
0gradients/model_1/conv4/BiasAdd_grad/BiasAddGradBiasAddGrad0gradients/model_1/conv4/conv4/Relu_grad/ReluGrad*
_output_shapes	
:�*
T0*
data_formatNHWC
�
5gradients/model_1/conv4/BiasAdd_grad/tuple/group_depsNoOp1^gradients/model_1/conv4/BiasAdd_grad/BiasAddGrad1^gradients/model_1/conv4/conv4/Relu_grad/ReluGrad
�
=gradients/model_1/conv4/BiasAdd_grad/tuple/control_dependencyIdentity0gradients/model_1/conv4/conv4/Relu_grad/ReluGrad6^gradients/model_1/conv4/BiasAdd_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/model_1/conv4/conv4/Relu_grad/ReluGrad*0
_output_shapes
:����������
�
?gradients/model_1/conv4/BiasAdd_grad/tuple/control_dependency_1Identity0gradients/model_1/conv4/BiasAdd_grad/BiasAddGrad6^gradients/model_1/conv4/BiasAdd_grad/tuple/group_deps*C
_class9
75loc:@gradients/model_1/conv4/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:�*
T0
�
.gradients/model/conv4/BiasAdd_grad/BiasAddGradBiasAddGrad.gradients/model/conv4/conv4/Relu_grad/ReluGrad*
_output_shapes	
:�*
T0*
data_formatNHWC
�
3gradients/model/conv4/BiasAdd_grad/tuple/group_depsNoOp/^gradients/model/conv4/BiasAdd_grad/BiasAddGrad/^gradients/model/conv4/conv4/Relu_grad/ReluGrad
�
;gradients/model/conv4/BiasAdd_grad/tuple/control_dependencyIdentity.gradients/model/conv4/conv4/Relu_grad/ReluGrad4^gradients/model/conv4/BiasAdd_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients/model/conv4/conv4/Relu_grad/ReluGrad*0
_output_shapes
:����������
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
:����������
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
:����������*
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
:����������
�
>gradients/model_1/conv4/Conv2D_grad/tuple/control_dependency_1Identity8gradients/model_1/conv4/Conv2D_grad/Conv2DBackpropFilter5^gradients/model_1/conv4/Conv2D_grad/tuple/group_deps*K
_classA
?=loc:@gradients/model_1/conv4/Conv2D_grad/Conv2DBackpropFilter*(
_output_shapes
:��*
T0
�
(gradients/model/conv4/Conv2D_grad/ShapeNShapeNmodel/conv3/MaxPool2D/MaxPoolconv4/weights/read*
out_type0*
N* 
_output_shapes
::*
T0
�
5gradients/model/conv4/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput(gradients/model/conv4/Conv2D_grad/ShapeNconv4/weights/read;gradients/model/conv4/BiasAdd_grad/tuple/control_dependency*0
_output_shapes
:����������*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME
�
6gradients/model/conv4/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFiltermodel/conv3/MaxPool2D/MaxPool*gradients/model/conv4/Conv2D_grad/ShapeN:1;gradients/model/conv4/BiasAdd_grad/tuple/control_dependency*
use_cudnn_on_gpu(*
paddingSAME*(
_output_shapes
:��*
	dilations
*
T0*
strides
*
data_formatNHWC
�
2gradients/model/conv4/Conv2D_grad/tuple/group_depsNoOp7^gradients/model/conv4/Conv2D_grad/Conv2DBackpropFilter6^gradients/model/conv4/Conv2D_grad/Conv2DBackpropInput
�
:gradients/model/conv4/Conv2D_grad/tuple/control_dependencyIdentity5gradients/model/conv4/Conv2D_grad/Conv2DBackpropInput3^gradients/model/conv4/Conv2D_grad/tuple/group_deps*
T0*H
_class>
<:loc:@gradients/model/conv4/Conv2D_grad/Conv2DBackpropInput*0
_output_shapes
:����������
�
<gradients/model/conv4/Conv2D_grad/tuple/control_dependency_1Identity6gradients/model/conv4/Conv2D_grad/Conv2DBackpropFilter3^gradients/model/conv4/Conv2D_grad/tuple/group_deps*(
_output_shapes
:��*
T0*I
_class?
=;loc:@gradients/model/conv4/Conv2D_grad/Conv2DBackpropFilter
�
*gradients/model_2/conv4/Conv2D_grad/ShapeNShapeNmodel_2/conv3/MaxPool2D/MaxPoolconv4/weights/read*
T0*
out_type0*
N* 
_output_shapes
::
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
:����������
�
8gradients/model_2/conv4/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFiltermodel_2/conv3/MaxPool2D/MaxPool,gradients/model_2/conv4/Conv2D_grad/ShapeN:1=gradients/model_2/conv4/BiasAdd_grad/tuple/control_dependency*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*(
_output_shapes
:��
�
4gradients/model_2/conv4/Conv2D_grad/tuple/group_depsNoOp9^gradients/model_2/conv4/Conv2D_grad/Conv2DBackpropFilter8^gradients/model_2/conv4/Conv2D_grad/Conv2DBackpropInput
�
<gradients/model_2/conv4/Conv2D_grad/tuple/control_dependencyIdentity7gradients/model_2/conv4/Conv2D_grad/Conv2DBackpropInput5^gradients/model_2/conv4/Conv2D_grad/tuple/group_deps*J
_class@
><loc:@gradients/model_2/conv4/Conv2D_grad/Conv2DBackpropInput*0
_output_shapes
:����������*
T0
�
>gradients/model_2/conv4/Conv2D_grad/tuple/control_dependency_1Identity8gradients/model_2/conv4/Conv2D_grad/Conv2DBackpropFilter5^gradients/model_2/conv4/Conv2D_grad/tuple/group_deps*
T0*K
_classA
?=loc:@gradients/model_2/conv4/Conv2D_grad/Conv2DBackpropFilter*(
_output_shapes
:��
�
gradients/AddN_3AddN?gradients/model_1/conv4/BiasAdd_grad/tuple/control_dependency_1=gradients/model/conv4/BiasAdd_grad/tuple/control_dependency_1?gradients/model_2/conv4/BiasAdd_grad/tuple/control_dependency_1*
N*
_output_shapes	
:�*
T0*C
_class9
75loc:@gradients/model_1/conv4/BiasAdd_grad/BiasAddGrad
�
:gradients/model_1/conv3/MaxPool2D/MaxPool_grad/MaxPoolGradMaxPoolGradmodel_1/conv3/conv3/Relumodel_1/conv3/MaxPool2D/MaxPool<gradients/model_1/conv4/Conv2D_grad/tuple/control_dependency*
T0*
data_formatNHWC*
strides
*
ksize
*
paddingSAME*0
_output_shapes
:���������&�
�
8gradients/model/conv3/MaxPool2D/MaxPool_grad/MaxPoolGradMaxPoolGradmodel/conv3/conv3/Relumodel/conv3/MaxPool2D/MaxPool:gradients/model/conv4/Conv2D_grad/tuple/control_dependency*
T0*
data_formatNHWC*
strides
*
ksize
*
paddingSAME*0
_output_shapes
:���������&�
�
:gradients/model_2/conv3/MaxPool2D/MaxPool_grad/MaxPoolGradMaxPoolGradmodel_2/conv3/conv3/Relumodel_2/conv3/MaxPool2D/MaxPool<gradients/model_2/conv4/Conv2D_grad/tuple/control_dependency*0
_output_shapes
:���������&�*
T0*
data_formatNHWC*
strides
*
ksize
*
paddingSAME
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
:���������&�
�
.gradients/model/conv3/conv3/Relu_grad/ReluGradReluGrad8gradients/model/conv3/MaxPool2D/MaxPool_grad/MaxPoolGradmodel/conv3/conv3/Relu*
T0*0
_output_shapes
:���������&�
�
0gradients/model_2/conv3/conv3/Relu_grad/ReluGradReluGrad:gradients/model_2/conv3/MaxPool2D/MaxPool_grad/MaxPoolGradmodel_2/conv3/conv3/Relu*
T0*0
_output_shapes
:���������&�
�
0gradients/model_1/conv3/BiasAdd_grad/BiasAddGradBiasAddGrad0gradients/model_1/conv3/conv3/Relu_grad/ReluGrad*
data_formatNHWC*
_output_shapes	
:�*
T0
�
5gradients/model_1/conv3/BiasAdd_grad/tuple/group_depsNoOp1^gradients/model_1/conv3/BiasAdd_grad/BiasAddGrad1^gradients/model_1/conv3/conv3/Relu_grad/ReluGrad
�
=gradients/model_1/conv3/BiasAdd_grad/tuple/control_dependencyIdentity0gradients/model_1/conv3/conv3/Relu_grad/ReluGrad6^gradients/model_1/conv3/BiasAdd_grad/tuple/group_deps*0
_output_shapes
:���������&�*
T0*C
_class9
75loc:@gradients/model_1/conv3/conv3/Relu_grad/ReluGrad
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
;gradients/model/conv3/BiasAdd_grad/tuple/control_dependencyIdentity.gradients/model/conv3/conv3/Relu_grad/ReluGrad4^gradients/model/conv3/BiasAdd_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients/model/conv3/conv3/Relu_grad/ReluGrad*0
_output_shapes
:���������&�
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
:���������&�
�
?gradients/model_2/conv3/BiasAdd_grad/tuple/control_dependency_1Identity0gradients/model_2/conv3/BiasAdd_grad/BiasAddGrad6^gradients/model_2/conv3/BiasAdd_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/model_2/conv3/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:�
�
*gradients/model_1/conv3/Conv2D_grad/ShapeNShapeNmodel_1/conv2/MaxPool2D/MaxPoolconv3/weights/read* 
_output_shapes
::*
T0*
out_type0*
N
�
7gradients/model_1/conv3/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput*gradients/model_1/conv3/Conv2D_grad/ShapeNconv3/weights/read=gradients/model_1/conv3/BiasAdd_grad/tuple/control_dependency*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*/
_output_shapes
:���������&@
�
8gradients/model_1/conv3/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFiltermodel_1/conv2/MaxPool2D/MaxPool,gradients/model_1/conv3/Conv2D_grad/ShapeN:1=gradients/model_1/conv3/BiasAdd_grad/tuple/control_dependency*'
_output_shapes
:@�*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME
�
4gradients/model_1/conv3/Conv2D_grad/tuple/group_depsNoOp9^gradients/model_1/conv3/Conv2D_grad/Conv2DBackpropFilter8^gradients/model_1/conv3/Conv2D_grad/Conv2DBackpropInput
�
<gradients/model_1/conv3/Conv2D_grad/tuple/control_dependencyIdentity7gradients/model_1/conv3/Conv2D_grad/Conv2DBackpropInput5^gradients/model_1/conv3/Conv2D_grad/tuple/group_deps*J
_class@
><loc:@gradients/model_1/conv3/Conv2D_grad/Conv2DBackpropInput*/
_output_shapes
:���������&@*
T0
�
>gradients/model_1/conv3/Conv2D_grad/tuple/control_dependency_1Identity8gradients/model_1/conv3/Conv2D_grad/Conv2DBackpropFilter5^gradients/model_1/conv3/Conv2D_grad/tuple/group_deps*
T0*K
_classA
?=loc:@gradients/model_1/conv3/Conv2D_grad/Conv2DBackpropFilter*'
_output_shapes
:@�
�
(gradients/model/conv3/Conv2D_grad/ShapeNShapeNmodel/conv2/MaxPool2D/MaxPoolconv3/weights/read*
out_type0*
N* 
_output_shapes
::*
T0
�
5gradients/model/conv3/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput(gradients/model/conv3/Conv2D_grad/ShapeNconv3/weights/read;gradients/model/conv3/BiasAdd_grad/tuple/control_dependency*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*/
_output_shapes
:���������&@*
	dilations

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
:gradients/model/conv3/Conv2D_grad/tuple/control_dependencyIdentity5gradients/model/conv3/Conv2D_grad/Conv2DBackpropInput3^gradients/model/conv3/Conv2D_grad/tuple/group_deps*
T0*H
_class>
<:loc:@gradients/model/conv3/Conv2D_grad/Conv2DBackpropInput*/
_output_shapes
:���������&@
�
<gradients/model/conv3/Conv2D_grad/tuple/control_dependency_1Identity6gradients/model/conv3/Conv2D_grad/Conv2DBackpropFilter3^gradients/model/conv3/Conv2D_grad/tuple/group_deps*
T0*I
_class?
=;loc:@gradients/model/conv3/Conv2D_grad/Conv2DBackpropFilter*'
_output_shapes
:@�
�
*gradients/model_2/conv3/Conv2D_grad/ShapeNShapeNmodel_2/conv2/MaxPool2D/MaxPoolconv3/weights/read*
out_type0*
N* 
_output_shapes
::*
T0
�
7gradients/model_2/conv3/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput*gradients/model_2/conv3/Conv2D_grad/ShapeNconv3/weights/read=gradients/model_2/conv3/BiasAdd_grad/tuple/control_dependency*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME*/
_output_shapes
:���������&@*
	dilations

�
8gradients/model_2/conv3/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFiltermodel_2/conv2/MaxPool2D/MaxPool,gradients/model_2/conv3/Conv2D_grad/ShapeN:1=gradients/model_2/conv3/BiasAdd_grad/tuple/control_dependency*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*'
_output_shapes
:@�*
	dilations
*
T0
�
4gradients/model_2/conv3/Conv2D_grad/tuple/group_depsNoOp9^gradients/model_2/conv3/Conv2D_grad/Conv2DBackpropFilter8^gradients/model_2/conv3/Conv2D_grad/Conv2DBackpropInput
�
<gradients/model_2/conv3/Conv2D_grad/tuple/control_dependencyIdentity7gradients/model_2/conv3/Conv2D_grad/Conv2DBackpropInput5^gradients/model_2/conv3/Conv2D_grad/tuple/group_deps*
T0*J
_class@
><loc:@gradients/model_2/conv3/Conv2D_grad/Conv2DBackpropInput*/
_output_shapes
:���������&@
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
:���������K@
�
8gradients/model/conv2/MaxPool2D/MaxPool_grad/MaxPoolGradMaxPoolGradmodel/conv2/conv2/Relumodel/conv2/MaxPool2D/MaxPool:gradients/model/conv3/Conv2D_grad/tuple/control_dependency*
ksize
*
paddingSAME*/
_output_shapes
:���������K@*
T0*
data_formatNHWC*
strides

�
:gradients/model_2/conv2/MaxPool2D/MaxPool_grad/MaxPoolGradMaxPoolGradmodel_2/conv2/conv2/Relumodel_2/conv2/MaxPool2D/MaxPool<gradients/model_2/conv3/Conv2D_grad/tuple/control_dependency*/
_output_shapes
:���������K@*
T0*
data_formatNHWC*
strides
*
ksize
*
paddingSAME
�
gradients/AddN_6AddN>gradients/model_1/conv3/Conv2D_grad/tuple/control_dependency_1<gradients/model/conv3/Conv2D_grad/tuple/control_dependency_1>gradients/model_2/conv3/Conv2D_grad/tuple/control_dependency_1*
T0*K
_classA
?=loc:@gradients/model_1/conv3/Conv2D_grad/Conv2DBackpropFilter*
N*'
_output_shapes
:@�
�
0gradients/model_1/conv2/conv2/Relu_grad/ReluGradReluGrad:gradients/model_1/conv2/MaxPool2D/MaxPool_grad/MaxPoolGradmodel_1/conv2/conv2/Relu*/
_output_shapes
:���������K@*
T0
�
.gradients/model/conv2/conv2/Relu_grad/ReluGradReluGrad8gradients/model/conv2/MaxPool2D/MaxPool_grad/MaxPoolGradmodel/conv2/conv2/Relu*/
_output_shapes
:���������K@*
T0
�
0gradients/model_2/conv2/conv2/Relu_grad/ReluGradReluGrad:gradients/model_2/conv2/MaxPool2D/MaxPool_grad/MaxPoolGradmodel_2/conv2/conv2/Relu*
T0*/
_output_shapes
:���������K@
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
:���������K@
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
;gradients/model/conv2/BiasAdd_grad/tuple/control_dependencyIdentity.gradients/model/conv2/conv2/Relu_grad/ReluGrad4^gradients/model/conv2/BiasAdd_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients/model/conv2/conv2/Relu_grad/ReluGrad*/
_output_shapes
:���������K@
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
:���������K@
�
?gradients/model_2/conv2/BiasAdd_grad/tuple/control_dependency_1Identity0gradients/model_2/conv2/BiasAdd_grad/BiasAddGrad6^gradients/model_2/conv2/BiasAdd_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/model_2/conv2/BiasAdd_grad/BiasAddGrad*
_output_shapes
:@
�
*gradients/model_1/conv2/Conv2D_grad/ShapeNShapeNmodel_1/conv1/MaxPool2D/MaxPoolconv2/weights/read*
out_type0*
N* 
_output_shapes
::*
T0
�
7gradients/model_1/conv2/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput*gradients/model_1/conv2/Conv2D_grad/ShapeNconv2/weights/read=gradients/model_1/conv2/BiasAdd_grad/tuple/control_dependency*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME*/
_output_shapes
:���������K 
�
8gradients/model_1/conv2/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFiltermodel_1/conv1/MaxPool2D/MaxPool,gradients/model_1/conv2/Conv2D_grad/ShapeN:1=gradients/model_1/conv2/BiasAdd_grad/tuple/control_dependency*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*&
_output_shapes
: @*
	dilations

�
4gradients/model_1/conv2/Conv2D_grad/tuple/group_depsNoOp9^gradients/model_1/conv2/Conv2D_grad/Conv2DBackpropFilter8^gradients/model_1/conv2/Conv2D_grad/Conv2DBackpropInput
�
<gradients/model_1/conv2/Conv2D_grad/tuple/control_dependencyIdentity7gradients/model_1/conv2/Conv2D_grad/Conv2DBackpropInput5^gradients/model_1/conv2/Conv2D_grad/tuple/group_deps*
T0*J
_class@
><loc:@gradients/model_1/conv2/Conv2D_grad/Conv2DBackpropInput*/
_output_shapes
:���������K 
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
5gradients/model/conv2/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput(gradients/model/conv2/Conv2D_grad/ShapeNconv2/weights/read;gradients/model/conv2/BiasAdd_grad/tuple/control_dependency*
use_cudnn_on_gpu(*
paddingSAME*/
_output_shapes
:���������K *
	dilations
*
T0*
strides
*
data_formatNHWC
�
6gradients/model/conv2/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFiltermodel/conv1/MaxPool2D/MaxPool*gradients/model/conv2/Conv2D_grad/ShapeN:1;gradients/model/conv2/BiasAdd_grad/tuple/control_dependency*
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
2gradients/model/conv2/Conv2D_grad/tuple/group_depsNoOp7^gradients/model/conv2/Conv2D_grad/Conv2DBackpropFilter6^gradients/model/conv2/Conv2D_grad/Conv2DBackpropInput
�
:gradients/model/conv2/Conv2D_grad/tuple/control_dependencyIdentity5gradients/model/conv2/Conv2D_grad/Conv2DBackpropInput3^gradients/model/conv2/Conv2D_grad/tuple/group_deps*
T0*H
_class>
<:loc:@gradients/model/conv2/Conv2D_grad/Conv2DBackpropInput*/
_output_shapes
:���������K 
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
7gradients/model_2/conv2/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput*gradients/model_2/conv2/Conv2D_grad/ShapeNconv2/weights/read=gradients/model_2/conv2/BiasAdd_grad/tuple/control_dependency*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*/
_output_shapes
:���������K *
	dilations

�
8gradients/model_2/conv2/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFiltermodel_2/conv1/MaxPool2D/MaxPool,gradients/model_2/conv2/Conv2D_grad/ShapeN:1=gradients/model_2/conv2/BiasAdd_grad/tuple/control_dependency*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME*&
_output_shapes
: @
�
4gradients/model_2/conv2/Conv2D_grad/tuple/group_depsNoOp9^gradients/model_2/conv2/Conv2D_grad/Conv2DBackpropFilter8^gradients/model_2/conv2/Conv2D_grad/Conv2DBackpropInput
�
<gradients/model_2/conv2/Conv2D_grad/tuple/control_dependencyIdentity7gradients/model_2/conv2/Conv2D_grad/Conv2DBackpropInput5^gradients/model_2/conv2/Conv2D_grad/tuple/group_deps*/
_output_shapes
:���������K *
T0*J
_class@
><loc:@gradients/model_2/conv2/Conv2D_grad/Conv2DBackpropInput
�
>gradients/model_2/conv2/Conv2D_grad/tuple/control_dependency_1Identity8gradients/model_2/conv2/Conv2D_grad/Conv2DBackpropFilter5^gradients/model_2/conv2/Conv2D_grad/tuple/group_deps*
T0*K
_classA
?=loc:@gradients/model_2/conv2/Conv2D_grad/Conv2DBackpropFilter*&
_output_shapes
: @
�
gradients/AddN_7AddN?gradients/model_1/conv2/BiasAdd_grad/tuple/control_dependency_1=gradients/model/conv2/BiasAdd_grad/tuple/control_dependency_1?gradients/model_2/conv2/BiasAdd_grad/tuple/control_dependency_1*
T0*C
_class9
75loc:@gradients/model_1/conv2/BiasAdd_grad/BiasAddGrad*
N*
_output_shapes
:@
�
:gradients/model_1/conv1/MaxPool2D/MaxPool_grad/MaxPoolGradMaxPoolGradmodel_1/conv1/conv1/Relumodel_1/conv1/MaxPool2D/MaxPool<gradients/model_1/conv2/Conv2D_grad/tuple/control_dependency*
ksize
*
paddingSAME*0
_output_shapes
:���������2� *
T0*
data_formatNHWC*
strides

�
8gradients/model/conv1/MaxPool2D/MaxPool_grad/MaxPoolGradMaxPoolGradmodel/conv1/conv1/Relumodel/conv1/MaxPool2D/MaxPool:gradients/model/conv2/Conv2D_grad/tuple/control_dependency*0
_output_shapes
:���������2� *
T0*
data_formatNHWC*
strides
*
ksize
*
paddingSAME
�
:gradients/model_2/conv1/MaxPool2D/MaxPool_grad/MaxPoolGradMaxPoolGradmodel_2/conv1/conv1/Relumodel_2/conv1/MaxPool2D/MaxPool<gradients/model_2/conv2/Conv2D_grad/tuple/control_dependency*
T0*
data_formatNHWC*
strides
*
ksize
*
paddingSAME*0
_output_shapes
:���������2� 
�
gradients/AddN_8AddN>gradients/model_1/conv2/Conv2D_grad/tuple/control_dependency_1<gradients/model/conv2/Conv2D_grad/tuple/control_dependency_1>gradients/model_2/conv2/Conv2D_grad/tuple/control_dependency_1*
T0*K
_classA
?=loc:@gradients/model_1/conv2/Conv2D_grad/Conv2DBackpropFilter*
N*&
_output_shapes
: @
�
0gradients/model_1/conv1/conv1/Relu_grad/ReluGradReluGrad:gradients/model_1/conv1/MaxPool2D/MaxPool_grad/MaxPoolGradmodel_1/conv1/conv1/Relu*0
_output_shapes
:���������2� *
T0
�
.gradients/model/conv1/conv1/Relu_grad/ReluGradReluGrad8gradients/model/conv1/MaxPool2D/MaxPool_grad/MaxPoolGradmodel/conv1/conv1/Relu*
T0*0
_output_shapes
:���������2� 
�
0gradients/model_2/conv1/conv1/Relu_grad/ReluGradReluGrad:gradients/model_2/conv1/MaxPool2D/MaxPool_grad/MaxPoolGradmodel_2/conv1/conv1/Relu*
T0*0
_output_shapes
:���������2� 
�
0gradients/model_1/conv1/BiasAdd_grad/BiasAddGradBiasAddGrad0gradients/model_1/conv1/conv1/Relu_grad/ReluGrad*
_output_shapes
: *
T0*
data_formatNHWC
�
5gradients/model_1/conv1/BiasAdd_grad/tuple/group_depsNoOp1^gradients/model_1/conv1/BiasAdd_grad/BiasAddGrad1^gradients/model_1/conv1/conv1/Relu_grad/ReluGrad
�
=gradients/model_1/conv1/BiasAdd_grad/tuple/control_dependencyIdentity0gradients/model_1/conv1/conv1/Relu_grad/ReluGrad6^gradients/model_1/conv1/BiasAdd_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/model_1/conv1/conv1/Relu_grad/ReluGrad*0
_output_shapes
:���������2� 
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
53loc:@gradients/model/conv1/conv1/Relu_grad/ReluGrad*0
_output_shapes
:���������2� 
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
75loc:@gradients/model_2/conv1/conv1/Relu_grad/ReluGrad*0
_output_shapes
:���������2� 
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
7gradients/model_1/conv1/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput*gradients/model_1/conv1/Conv2D_grad/ShapeNconv1/weights/read=gradients/model_1/conv1/BiasAdd_grad/tuple/control_dependency*
paddingSAME*0
_output_shapes
:���������2�*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
�
8gradients/model_1/conv1/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterpositive_input,gradients/model_1/conv1/Conv2D_grad/ShapeN:1=gradients/model_1/conv1/BiasAdd_grad/tuple/control_dependency*
paddingSAME*&
_output_shapes
: *
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(
�
4gradients/model_1/conv1/Conv2D_grad/tuple/group_depsNoOp9^gradients/model_1/conv1/Conv2D_grad/Conv2DBackpropFilter8^gradients/model_1/conv1/Conv2D_grad/Conv2DBackpropInput
�
<gradients/model_1/conv1/Conv2D_grad/tuple/control_dependencyIdentity7gradients/model_1/conv1/Conv2D_grad/Conv2DBackpropInput5^gradients/model_1/conv1/Conv2D_grad/tuple/group_deps*0
_output_shapes
:���������2�*
T0*J
_class@
><loc:@gradients/model_1/conv1/Conv2D_grad/Conv2DBackpropInput
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
5gradients/model/conv1/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput(gradients/model/conv1/Conv2D_grad/ShapeNconv1/weights/read;gradients/model/conv1/BiasAdd_grad/tuple/control_dependency*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME*0
_output_shapes
:���������2�
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
<:loc:@gradients/model/conv1/Conv2D_grad/Conv2DBackpropInput*0
_output_shapes
:���������2�
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
7gradients/model_2/conv1/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput*gradients/model_2/conv1/Conv2D_grad/ShapeNconv1/weights/read=gradients/model_2/conv1/BiasAdd_grad/tuple/control_dependency*0
_output_shapes
:���������2�*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME
�
8gradients/model_2/conv1/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilternegative_input,gradients/model_2/conv1/Conv2D_grad/ShapeN:1=gradients/model_2/conv1/BiasAdd_grad/tuple/control_dependency*&
_output_shapes
: *
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME
�
4gradients/model_2/conv1/Conv2D_grad/tuple/group_depsNoOp9^gradients/model_2/conv1/Conv2D_grad/Conv2DBackpropFilter8^gradients/model_2/conv1/Conv2D_grad/Conv2DBackpropInput
�
<gradients/model_2/conv1/Conv2D_grad/tuple/control_dependencyIdentity7gradients/model_2/conv1/Conv2D_grad/Conv2DBackpropInput5^gradients/model_2/conv1/Conv2D_grad/tuple/group_deps*
T0*J
_class@
><loc:@gradients/model_2/conv1/Conv2D_grad/Conv2DBackpropInput*0
_output_shapes
:���������2�
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
VariableV2*
dtype0*&
_output_shapes
: *
shared_name * 
_class
loc:@conv1/weights*
	container *
shape: 
�
conv1/weights/Momentum/AssignAssignconv1/weights/Momentum(conv1/weights/Momentum/Initializer/zeros*
validate_shape(*&
_output_shapes
: *
use_locking(*
T0* 
_class
loc:@conv1/weights
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
conv1/biases/Momentum/readIdentityconv1/biases/Momentum*
_class
loc:@conv1/biases*
_output_shapes
: *
T0
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
conv2/weights/Momentum/AssignAssignconv2/weights/Momentum(conv2/weights/Momentum/Initializer/zeros*&
_output_shapes
: @*
use_locking(*
T0* 
_class
loc:@conv2/weights*
validate_shape(
�
conv2/weights/Momentum/readIdentityconv2/weights/Momentum* 
_class
loc:@conv2/weights*&
_output_shapes
: @*
T0
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
conv2/biases/Momentum/readIdentityconv2/biases/Momentum*
_class
loc:@conv2/biases*
_output_shapes
:@*
T0
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
_class
loc:@conv3/weights*
valueB
 *    *
dtype0*
_output_shapes
: 
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
conv3/weights/Momentum/AssignAssignconv3/weights/Momentum(conv3/weights/Momentum/Initializer/zeros*
T0* 
_class
loc:@conv3/weights*
validate_shape(*'
_output_shapes
:@�*
use_locking(
�
conv3/weights/Momentum/readIdentityconv3/weights/Momentum* 
_class
loc:@conv3/weights*'
_output_shapes
:@�*
T0
�
'conv3/biases/Momentum/Initializer/zerosConst*
dtype0*
_output_shapes	
:�*
_class
loc:@conv3/biases*
valueB�*    
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
conv3/biases/Momentum/AssignAssignconv3/biases/Momentum'conv3/biases/Momentum/Initializer/zeros*
_class
loc:@conv3/biases*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0
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
_class
loc:@conv4/weights*
valueB
 *    *
dtype0*
_output_shapes
: 
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
VariableV2*
	container *
shape:��*
dtype0*(
_output_shapes
:��*
shared_name * 
_class
loc:@conv4/weights
�
conv4/weights/Momentum/AssignAssignconv4/weights/Momentum(conv4/weights/Momentum/Initializer/zeros*
use_locking(*
T0* 
_class
loc:@conv4/weights*
validate_shape(*(
_output_shapes
:��
�
conv4/weights/Momentum/readIdentityconv4/weights/Momentum*(
_output_shapes
:��*
T0* 
_class
loc:@conv4/weights
�
'conv4/biases/Momentum/Initializer/zerosConst*
_class
loc:@conv4/biases*
valueB�*    *
dtype0*
_output_shapes	
:�
�
conv4/biases/Momentum
VariableV2*
	container *
shape:�*
dtype0*
_output_shapes	
:�*
shared_name *
_class
loc:@conv4/biases
�
conv4/biases/Momentum/AssignAssignconv4/biases/Momentum'conv4/biases/Momentum/Initializer/zeros*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0*
_class
loc:@conv4/biases
�
conv4/biases/Momentum/readIdentityconv4/biases/Momentum*
T0*
_class
loc:@conv4/biases*
_output_shapes	
:�
�
8conv5/weights/Momentum/Initializer/zeros/shape_as_tensorConst*
_output_shapes
:* 
_class
loc:@conv5/weights*%
valueB"            *
dtype0
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
(conv5/weights/Momentum/Initializer/zerosFill8conv5/weights/Momentum/Initializer/zeros/shape_as_tensor.conv5/weights/Momentum/Initializer/zeros/Const*'
_output_shapes
:�*
T0* 
_class
loc:@conv5/weights*

index_type0
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
conv5/weights/Momentum/AssignAssignconv5/weights/Momentum(conv5/weights/Momentum/Initializer/zeros*
use_locking(*
T0* 
_class
loc:@conv5/weights*
validate_shape(*'
_output_shapes
:�
�
conv5/weights/Momentum/readIdentityconv5/weights/Momentum*'
_output_shapes
:�*
T0* 
_class
loc:@conv5/weights
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
+Momentum/update_conv1/weights/ApplyMomentumApplyMomentumconv1/weightsconv1/weights/MomentumMomentum/learning_rategradients/AddN_10Momentum/momentum*
use_locking( *
T0* 
_class
loc:@conv1/weights*
use_nesterov(*&
_output_shapes
: 
�
*Momentum/update_conv1/biases/ApplyMomentumApplyMomentumconv1/biasesconv1/biases/MomentumMomentum/learning_rategradients/AddN_9Momentum/momentum*
use_locking( *
T0*
_class
loc:@conv1/biases*
use_nesterov(*
_output_shapes
: 
�
+Momentum/update_conv2/weights/ApplyMomentumApplyMomentumconv2/weightsconv2/weights/MomentumMomentum/learning_rategradients/AddN_8Momentum/momentum*
use_nesterov(*&
_output_shapes
: @*
use_locking( *
T0* 
_class
loc:@conv2/weights
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
*Momentum/update_conv4/biases/ApplyMomentumApplyMomentumconv4/biasesconv4/biases/MomentumMomentum/learning_rategradients/AddN_3Momentum/momentum*
use_locking( *
T0*
_class
loc:@conv4/biases*
use_nesterov(*
_output_shapes	
:�
�
+Momentum/update_conv5/weights/ApplyMomentumApplyMomentumconv5/weightsconv5/weights/MomentumMomentum/learning_rategradients/AddN_2Momentum/momentum*
use_nesterov(*'
_output_shapes
:�*
use_locking( *
T0* 
_class
loc:@conv5/weights
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
Momentum/valueConst^Momentum/update*
value	B :*
_class
loc:@Variable*
dtype0*
_output_shapes
: 
�
Momentum	AssignAddVariableMomentum/value*
_output_shapes
: *
use_locking( *
T0*
_class
loc:@Variable
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
save/Const^save/SaveV2*
T0*
_class
loc:@save/Const*
_output_shapes
: 
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
save/AssignAssignVariablesave/RestoreV2*
_class
loc:@Variable*
validate_shape(*
_output_shapes
: *
use_locking(*
T0
�
save/Assign_1Assignconv1/biasessave/RestoreV2:1*
T0*
_class
loc:@conv1/biases*
validate_shape(*
_output_shapes
: *
use_locking(
�
save/Assign_2Assignconv1/biases/Momentumsave/RestoreV2:2*
use_locking(*
T0*
_class
loc:@conv1/biases*
validate_shape(*
_output_shapes
: 
�
save/Assign_3Assignconv1/weightssave/RestoreV2:3* 
_class
loc:@conv1/weights*
validate_shape(*&
_output_shapes
: *
use_locking(*
T0
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
save/Assign_6Assignconv2/biases/Momentumsave/RestoreV2:6*
_class
loc:@conv2/biases*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0
�
save/Assign_7Assignconv2/weightssave/RestoreV2:7*&
_output_shapes
: @*
use_locking(*
T0* 
_class
loc:@conv2/weights*
validate_shape(
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
save/Assign_11Assignconv3/weightssave/RestoreV2:11*'
_output_shapes
:@�*
use_locking(*
T0* 
_class
loc:@conv3/weights*
validate_shape(
�
save/Assign_12Assignconv3/weights/Momentumsave/RestoreV2:12*
T0* 
_class
loc:@conv3/weights*
validate_shape(*'
_output_shapes
:@�*
use_locking(
�
save/Assign_13Assignconv4/biasessave/RestoreV2:13*
use_locking(*
T0*
_class
loc:@conv4/biases*
validate_shape(*
_output_shapes	
:�
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
save/Assign_17Assignconv5/biasessave/RestoreV2:17*
_output_shapes
:*
use_locking(*
T0*
_class
loc:@conv5/biases*
validate_shape(
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
save/Assign_19Assignconv5/weightssave/RestoreV2:19*
T0* 
_class
loc:@conv5/weights*
validate_shape(*'
_output_shapes
:�*
use_locking(
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
stepScalarSummary	step/tagsVariable/read*
T0*
_output_shapes
: 
N
	loss/tagsConst*
valueB
 Bloss*
dtype0*
_output_shapes
: 
G
lossScalarSummary	loss/tagsMean*
_output_shapes
: *
T0
c
conv1/weights_1/tagConst* 
valueB Bconv1/weights_1*
dtype0*
_output_shapes
: 
m
conv1/weights_1HistogramSummaryconv1/weights_1/tagconv1/weights/read*
_output_shapes
: *
T0
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
valueB Bconv3/weights_1*
dtype0*
_output_shapes
: 
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
conv3/biases_1HistogramSummaryconv3/biases_1/tagconv3/biases/read*
_output_shapes
: *
T0
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
conv4/biases_1/tagConst*
_output_shapes
: *
valueB Bconv4/biases_1*
dtype0
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
conv5/weights_1HistogramSummaryconv5/weights_1/tagconv5/weights/read*
_output_shapes
: *
T0
a
conv5/biases_1/tagConst*
dtype0*
_output_shapes
: *
valueB Bconv5/biases_1
j
conv5/biases_1HistogramSummaryconv5/biases_1/tagconv5/biases/read*
T0*
_output_shapes
: 
�
Merge/MergeSummaryMergeSummarysteplossconv1/weights_1conv1/biases_1conv2/weights_1conv2/biases_1conv3/weights_1conv3/biases_1conv4/weights_1conv4/biases_1conv5/weights_1conv5/biases_1*
N*
_output_shapes
: "�/��:     =�t�	��.xX��AJ��
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
Ttype*1.13.12b'v1.13.0-rc2-5-g6612da8951'ԣ
�
anchor_inputPlaceholder*
dtype0*0
_output_shapes
:���������2�*%
shape:���������2�
�
positive_inputPlaceholder*%
shape:���������2�*
dtype0*0
_output_shapes
:���������2�
�
negative_inputPlaceholder*
dtype0*0
_output_shapes
:���������2�*%
shape:���������2�
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
_class
loc:@conv1/weights*
valueB
 *�Er�*
dtype0*
_output_shapes
: 
�
,conv1/weights/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: * 
_class
loc:@conv1/weights*
valueB
 *�Er=
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
(conv1/weights/Initializer/random_uniformAdd,conv1/weights/Initializer/random_uniform/mul,conv1/weights/Initializer/random_uniform/min*&
_output_shapes
: *
T0* 
_class
loc:@conv1/weights
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
conv1/weights/AssignAssignconv1/weights(conv1/weights/Initializer/random_uniform*
use_locking(*
T0* 
_class
loc:@conv1/weights*
validate_shape(*&
_output_shapes
: 
�
conv1/weights/readIdentityconv1/weights*&
_output_shapes
: *
T0* 
_class
loc:@conv1/weights
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
dtype0*
_output_shapes
:*
valueB"      
�
model/conv1/Conv2DConv2Danchor_inputconv1/weights/read*
paddingSAME*0
_output_shapes
:���������2� *
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
data_formatNHWC*0
_output_shapes
:���������2� 
n
model/conv1/conv1/ReluRelumodel/conv1/BiasAdd*
T0*0
_output_shapes
:���������2� 
�
model/conv1/MaxPool2D/MaxPoolMaxPoolmodel/conv1/conv1/Relu*
ksize
*
paddingSAME*/
_output_shapes
:���������K *
T0*
data_formatNHWC*
strides

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
_class
loc:@conv2/weights*
valueB
 *��L=*
dtype0*
_output_shapes
: 
�
6conv2/weights/Initializer/random_uniform/RandomUniformRandomUniform.conv2/weights/Initializer/random_uniform/shape*
seed2 *
dtype0*&
_output_shapes
: @*

seed *
T0* 
_class
loc:@conv2/weights
�
,conv2/weights/Initializer/random_uniform/subSub,conv2/weights/Initializer/random_uniform/max,conv2/weights/Initializer/random_uniform/min*
T0* 
_class
loc:@conv2/weights*
_output_shapes
: 
�
,conv2/weights/Initializer/random_uniform/mulMul6conv2/weights/Initializer/random_uniform/RandomUniform,conv2/weights/Initializer/random_uniform/sub*
T0* 
_class
loc:@conv2/weights*&
_output_shapes
: @
�
(conv2/weights/Initializer/random_uniformAdd,conv2/weights/Initializer/random_uniform/mul,conv2/weights/Initializer/random_uniform/min*&
_output_shapes
: @*
T0* 
_class
loc:@conv2/weights
�
conv2/weights
VariableV2* 
_class
loc:@conv2/weights*
	container *
shape: @*
dtype0*&
_output_shapes
: @*
shared_name 
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
_class
loc:@conv2/biases*
valueB@*    *
dtype0*
_output_shapes
:@
�
conv2/biases
VariableV2*
_class
loc:@conv2/biases*
	container *
shape:@*
dtype0*
_output_shapes
:@*
shared_name 
�
conv2/biases/AssignAssignconv2/biasesconv2/biases/Initializer/zeros*
T0*
_class
loc:@conv2/biases*
validate_shape(*
_output_shapes
:@*
use_locking(
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
model/conv2/Conv2DConv2Dmodel/conv1/MaxPool2D/MaxPoolconv2/weights/read*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*/
_output_shapes
:���������K@
�
model/conv2/BiasAddBiasAddmodel/conv2/Conv2Dconv2/biases/read*
T0*
data_formatNHWC*/
_output_shapes
:���������K@
m
model/conv2/conv2/ReluRelumodel/conv2/BiasAdd*/
_output_shapes
:���������K@*
T0
�
model/conv2/MaxPool2D/MaxPoolMaxPoolmodel/conv2/conv2/Relu*/
_output_shapes
:���������&@*
T0*
data_formatNHWC*
strides
*
ksize
*
paddingSAME
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
6conv3/weights/Initializer/random_uniform/RandomUniformRandomUniform.conv3/weights/Initializer/random_uniform/shape*
dtype0*'
_output_shapes
:@�*

seed *
T0* 
_class
loc:@conv3/weights*
seed2 
�
,conv3/weights/Initializer/random_uniform/subSub,conv3/weights/Initializer/random_uniform/max,conv3/weights/Initializer/random_uniform/min*
T0* 
_class
loc:@conv3/weights*
_output_shapes
: 
�
,conv3/weights/Initializer/random_uniform/mulMul6conv3/weights/Initializer/random_uniform/RandomUniform,conv3/weights/Initializer/random_uniform/sub*'
_output_shapes
:@�*
T0* 
_class
loc:@conv3/weights
�
(conv3/weights/Initializer/random_uniformAdd,conv3/weights/Initializer/random_uniform/mul,conv3/weights/Initializer/random_uniform/min*'
_output_shapes
:@�*
T0* 
_class
loc:@conv3/weights
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
VariableV2*
	container *
shape:�*
dtype0*
_output_shapes	
:�*
shared_name *
_class
loc:@conv3/biases
�
conv3/biases/AssignAssignconv3/biasesconv3/biases/Initializer/zeros*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0*
_class
loc:@conv3/biases
r
conv3/biases/readIdentityconv3/biases*
_output_shapes	
:�*
T0*
_class
loc:@conv3/biases
j
model/conv3/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
�
model/conv3/Conv2DConv2Dmodel/conv2/MaxPool2D/MaxPoolconv3/weights/read*0
_output_shapes
:���������&�*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME
�
model/conv3/BiasAddBiasAddmodel/conv3/Conv2Dconv3/biases/read*
T0*
data_formatNHWC*0
_output_shapes
:���������&�
n
model/conv3/conv3/ReluRelumodel/conv3/BiasAdd*
T0*0
_output_shapes
:���������&�
�
model/conv3/MaxPool2D/MaxPoolMaxPoolmodel/conv3/conv3/Relu*
T0*
data_formatNHWC*
strides
*
ksize
*
paddingSAME*0
_output_shapes
:����������
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
_class
loc:@conv4/weights*
valueB
 *   �*
dtype0*
_output_shapes
: 
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
6conv4/weights/Initializer/random_uniform/RandomUniformRandomUniform.conv4/weights/Initializer/random_uniform/shape*
T0* 
_class
loc:@conv4/weights*
seed2 *
dtype0*(
_output_shapes
:��*

seed 
�
,conv4/weights/Initializer/random_uniform/subSub,conv4/weights/Initializer/random_uniform/max,conv4/weights/Initializer/random_uniform/min*
_output_shapes
: *
T0* 
_class
loc:@conv4/weights
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
dtype0*(
_output_shapes
:��*
shared_name * 
_class
loc:@conv4/weights*
	container *
shape:��
�
conv4/weights/AssignAssignconv4/weights(conv4/weights/Initializer/random_uniform*
use_locking(*
T0* 
_class
loc:@conv4/weights*
validate_shape(*(
_output_shapes
:��
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
VariableV2*
	container *
shape:�*
dtype0*
_output_shapes	
:�*
shared_name *
_class
loc:@conv4/biases
�
conv4/biases/AssignAssignconv4/biasesconv4/biases/Initializer/zeros*
use_locking(*
T0*
_class
loc:@conv4/biases*
validate_shape(*
_output_shapes	
:�
r
conv4/biases/readIdentityconv4/biases*
_output_shapes	
:�*
T0*
_class
loc:@conv4/biases
j
model/conv4/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
�
model/conv4/Conv2DConv2Dmodel/conv3/MaxPool2D/MaxPoolconv4/weights/read*0
_output_shapes
:����������*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME
�
model/conv4/BiasAddBiasAddmodel/conv4/Conv2Dconv4/biases/read*
T0*
data_formatNHWC*0
_output_shapes
:����������
n
model/conv4/conv4/ReluRelumodel/conv4/BiasAdd*
T0*0
_output_shapes
:����������
�
model/conv4/MaxPool2D/MaxPoolMaxPoolmodel/conv4/conv4/Relu*
ksize
*
paddingSAME*0
_output_shapes
:���������
�*
T0*
data_formatNHWC*
strides

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
dtype0*
_output_shapes
: * 
_class
loc:@conv5/weights*
valueB
 *���
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
conv5/weights/AssignAssignconv5/weights(conv5/weights/Initializer/random_uniform*
validate_shape(*'
_output_shapes
:�*
use_locking(*
T0* 
_class
loc:@conv5/weights
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
conv5/biases/AssignAssignconv5/biasesconv5/biases/Initializer/zeros*
T0*
_class
loc:@conv5/biases*
validate_shape(*
_output_shapes
:*
use_locking(
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
model/conv5/Conv2DConv2Dmodel/conv4/MaxPool2D/MaxPoolconv5/weights/read*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME*/
_output_shapes
:���������

�
model/conv5/BiasAddBiasAddmodel/conv5/Conv2Dconv5/biases/read*
T0*
data_formatNHWC*/
_output_shapes
:���������

�
model/conv5/MaxPool2D/MaxPoolMaxPoolmodel/conv5/BiasAdd*
ksize
*
paddingSAME*/
_output_shapes
:���������*
T0*
data_formatNHWC*
strides

x
model/Flatten/flatten/ShapeShapemodel/conv5/MaxPool2D/MaxPool*
_output_shapes
:*
T0*
out_type0
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
#model/Flatten/flatten/strided_sliceStridedSlicemodel/Flatten/flatten/Shape)model/Flatten/flatten/strided_slice/stack+model/Flatten/flatten/strided_slice/stack_1+model/Flatten/flatten/strided_slice/stack_2*
T0*
Index0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: 
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
model/Flatten/flatten/ReshapeReshapemodel/conv5/MaxPool2D/MaxPool#model/Flatten/flatten/Reshape/shape*
T0*
Tshape0*(
_output_shapes
:����������
l
model_1/conv1/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
�
model_1/conv1/Conv2DConv2Dpositive_inputconv1/weights/read*0
_output_shapes
:���������2� *
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME
�
model_1/conv1/BiasAddBiasAddmodel_1/conv1/Conv2Dconv1/biases/read*
T0*
data_formatNHWC*0
_output_shapes
:���������2� 
r
model_1/conv1/conv1/ReluRelumodel_1/conv1/BiasAdd*
T0*0
_output_shapes
:���������2� 
�
model_1/conv1/MaxPool2D/MaxPoolMaxPoolmodel_1/conv1/conv1/Relu*
ksize
*
paddingSAME*/
_output_shapes
:���������K *
T0*
strides
*
data_formatNHWC
l
model_1/conv2/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
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
:���������K@
�
model_1/conv2/BiasAddBiasAddmodel_1/conv2/Conv2Dconv2/biases/read*
T0*
data_formatNHWC*/
_output_shapes
:���������K@
q
model_1/conv2/conv2/ReluRelumodel_1/conv2/BiasAdd*
T0*/
_output_shapes
:���������K@
�
model_1/conv2/MaxPool2D/MaxPoolMaxPoolmodel_1/conv2/conv2/Relu*/
_output_shapes
:���������&@*
T0*
strides
*
data_formatNHWC*
ksize
*
paddingSAME
l
model_1/conv3/dilation_rateConst*
dtype0*
_output_shapes
:*
valueB"      
�
model_1/conv3/Conv2DConv2Dmodel_1/conv2/MaxPool2D/MaxPoolconv3/weights/read*
paddingSAME*0
_output_shapes
:���������&�*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(
�
model_1/conv3/BiasAddBiasAddmodel_1/conv3/Conv2Dconv3/biases/read*
T0*
data_formatNHWC*0
_output_shapes
:���������&�
r
model_1/conv3/conv3/ReluRelumodel_1/conv3/BiasAdd*
T0*0
_output_shapes
:���������&�
�
model_1/conv3/MaxPool2D/MaxPoolMaxPoolmodel_1/conv3/conv3/Relu*
ksize
*
paddingSAME*0
_output_shapes
:����������*
T0*
data_formatNHWC*
strides

l
model_1/conv4/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
�
model_1/conv4/Conv2DConv2Dmodel_1/conv3/MaxPool2D/MaxPoolconv4/weights/read*0
_output_shapes
:����������*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME
�
model_1/conv4/BiasAddBiasAddmodel_1/conv4/Conv2Dconv4/biases/read*
T0*
data_formatNHWC*0
_output_shapes
:����������
r
model_1/conv4/conv4/ReluRelumodel_1/conv4/BiasAdd*0
_output_shapes
:����������*
T0
�
model_1/conv4/MaxPool2D/MaxPoolMaxPoolmodel_1/conv4/conv4/Relu*0
_output_shapes
:���������
�*
T0*
data_formatNHWC*
strides
*
ksize
*
paddingSAME
l
model_1/conv5/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
�
model_1/conv5/Conv2DConv2Dmodel_1/conv4/MaxPool2D/MaxPoolconv5/weights/read*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*/
_output_shapes
:���������
*
	dilations

�
model_1/conv5/BiasAddBiasAddmodel_1/conv5/Conv2Dconv5/biases/read*
T0*
data_formatNHWC*/
_output_shapes
:���������

�
model_1/conv5/MaxPool2D/MaxPoolMaxPoolmodel_1/conv5/BiasAdd*
data_formatNHWC*
strides
*
ksize
*
paddingSAME*/
_output_shapes
:���������*
T0
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
-model_1/Flatten/flatten/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
�
%model_1/Flatten/flatten/strided_sliceStridedSlicemodel_1/Flatten/flatten/Shape+model_1/Flatten/flatten/strided_slice/stack-model_1/Flatten/flatten/strided_slice/stack_1-model_1/Flatten/flatten/strided_slice/stack_2*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
Index0*
T0*
shrink_axis_mask
r
'model_1/Flatten/flatten/Reshape/shape/1Const*
dtype0*
_output_shapes
: *
valueB :
���������
�
%model_1/Flatten/flatten/Reshape/shapePack%model_1/Flatten/flatten/strided_slice'model_1/Flatten/flatten/Reshape/shape/1*
N*
_output_shapes
:*
T0*

axis 
�
model_1/Flatten/flatten/ReshapeReshapemodel_1/conv5/MaxPool2D/MaxPool%model_1/Flatten/flatten/Reshape/shape*
T0*
Tshape0*(
_output_shapes
:����������
l
model_2/conv1/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
�
model_2/conv1/Conv2DConv2Dnegative_inputconv1/weights/read*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME*0
_output_shapes
:���������2� *
	dilations

�
model_2/conv1/BiasAddBiasAddmodel_2/conv1/Conv2Dconv1/biases/read*
T0*
data_formatNHWC*0
_output_shapes
:���������2� 
r
model_2/conv1/conv1/ReluRelumodel_2/conv1/BiasAdd*0
_output_shapes
:���������2� *
T0
�
model_2/conv1/MaxPool2D/MaxPoolMaxPoolmodel_2/conv1/conv1/Relu*
ksize
*
paddingSAME*/
_output_shapes
:���������K *
T0*
strides
*
data_formatNHWC
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
:���������K@*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(
�
model_2/conv2/BiasAddBiasAddmodel_2/conv2/Conv2Dconv2/biases/read*
data_formatNHWC*/
_output_shapes
:���������K@*
T0
q
model_2/conv2/conv2/ReluRelumodel_2/conv2/BiasAdd*
T0*/
_output_shapes
:���������K@
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
:���������&@
l
model_2/conv3/dilation_rateConst*
dtype0*
_output_shapes
:*
valueB"      
�
model_2/conv3/Conv2DConv2Dmodel_2/conv2/MaxPool2D/MaxPoolconv3/weights/read*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME*0
_output_shapes
:���������&�
�
model_2/conv3/BiasAddBiasAddmodel_2/conv3/Conv2Dconv3/biases/read*
data_formatNHWC*0
_output_shapes
:���������&�*
T0
r
model_2/conv3/conv3/ReluRelumodel_2/conv3/BiasAdd*0
_output_shapes
:���������&�*
T0
�
model_2/conv3/MaxPool2D/MaxPoolMaxPoolmodel_2/conv3/conv3/Relu*
T0*
data_formatNHWC*
strides
*
ksize
*
paddingSAME*0
_output_shapes
:����������
l
model_2/conv4/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
�
model_2/conv4/Conv2DConv2Dmodel_2/conv3/MaxPool2D/MaxPoolconv4/weights/read*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*0
_output_shapes
:����������*
	dilations
*
T0
�
model_2/conv4/BiasAddBiasAddmodel_2/conv4/Conv2Dconv4/biases/read*
T0*
data_formatNHWC*0
_output_shapes
:����������
r
model_2/conv4/conv4/ReluRelumodel_2/conv4/BiasAdd*
T0*0
_output_shapes
:����������
�
model_2/conv4/MaxPool2D/MaxPoolMaxPoolmodel_2/conv4/conv4/Relu*0
_output_shapes
:���������
�*
T0*
data_formatNHWC*
strides
*
ksize
*
paddingSAME
l
model_2/conv5/dilation_rateConst*
dtype0*
_output_shapes
:*
valueB"      
�
model_2/conv5/Conv2DConv2Dmodel_2/conv4/MaxPool2D/MaxPoolconv5/weights/read*
paddingSAME*/
_output_shapes
:���������
*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(
�
model_2/conv5/BiasAddBiasAddmodel_2/conv5/Conv2Dconv5/biases/read*
T0*
data_formatNHWC*/
_output_shapes
:���������

�
model_2/conv5/MaxPool2D/MaxPoolMaxPoolmodel_2/conv5/BiasAdd*
T0*
strides
*
data_formatNHWC*
ksize
*
paddingSAME*/
_output_shapes
:���������
|
model_2/Flatten/flatten/ShapeShapemodel_2/conv5/MaxPool2D/MaxPool*
T0*
out_type0*
_output_shapes
:
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
-model_2/Flatten/flatten/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
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
model_2/Flatten/flatten/ReshapeReshapemodel_2/conv5/MaxPool2D/MaxPool%model_2/Flatten/flatten/Reshape/shape*
T0*
Tshape0*(
_output_shapes
:����������

mulMulmodel_1/Flatten/flatten/Reshapemodel_2/Flatten/flatten/Reshape*
T0*(
_output_shapes
:����������
W
Sum/reduction_indicesConst*
value	B :*
dtype0*
_output_shapes
: 
q
SumSummulSum/reduction_indices*
T0*#
_output_shapes
:���������*

Tidx0*
	keep_dims( 
J
Pow/yConst*
valueB
 *   @*
dtype0*
_output_shapes
: 
e
PowPowmodel_1/Flatten/flatten/ReshapePow/y*(
_output_shapes
:����������*
T0
Y
Sum_1/reduction_indicesConst*
value	B :*
dtype0*
_output_shapes
: 
u
Sum_1SumPowSum_1/reduction_indices*

Tidx0*
	keep_dims( *
T0*#
_output_shapes
:���������
A
SqrtSqrtSum_1*
T0*#
_output_shapes
:���������
L
Pow_1/yConst*
dtype0*
_output_shapes
: *
valueB
 *   @
i
Pow_1Powmodel_2/Flatten/flatten/ReshapePow_1/y*
T0*(
_output_shapes
:����������
Y
Sum_2/reduction_indicesConst*
value	B :*
dtype0*
_output_shapes
: 
w
Sum_2SumPow_1Sum_2/reduction_indices*

Tidx0*
	keep_dims( *
T0*#
_output_shapes
:���������
C
Sqrt_1SqrtSum_2*
T0*#
_output_shapes
:���������
H
mul_1MulSqrtSqrt_1*
T0*#
_output_shapes
:���������
L
truedivRealDivSummul_1*#
_output_shapes
:���������*
T0

mul_2Mulmodel_1/Flatten/flatten/Reshapemodel/Flatten/flatten/Reshape*
T0*(
_output_shapes
:����������
Y
Sum_3/reduction_indicesConst*
value	B :*
dtype0*
_output_shapes
: 
w
Sum_3Summul_2Sum_3/reduction_indices*#
_output_shapes
:���������*

Tidx0*
	keep_dims( *
T0
L
Pow_2/yConst*
dtype0*
_output_shapes
: *
valueB
 *   @
i
Pow_2Powmodel_1/Flatten/flatten/ReshapePow_2/y*(
_output_shapes
:����������*
T0
Y
Sum_4/reduction_indicesConst*
dtype0*
_output_shapes
: *
value	B :
w
Sum_4SumPow_2Sum_4/reduction_indices*
T0*#
_output_shapes
:���������*

Tidx0*
	keep_dims( 
C
Sqrt_2SqrtSum_4*#
_output_shapes
:���������*
T0
L
Pow_3/yConst*
valueB
 *   @*
dtype0*
_output_shapes
: 
g
Pow_3Powmodel/Flatten/flatten/ReshapePow_3/y*
T0*(
_output_shapes
:����������
Y
Sum_5/reduction_indicesConst*
dtype0*
_output_shapes
: *
value	B :
w
Sum_5SumPow_3Sum_5/reduction_indices*#
_output_shapes
:���������*

Tidx0*
	keep_dims( *
T0
C
Sqrt_3SqrtSum_5*
T0*#
_output_shapes
:���������
J
mul_3MulSqrt_2Sqrt_3*
T0*#
_output_shapes
:���������
P
	truediv_1RealDivSum_3mul_3*
T0*#
_output_shapes
:���������

mul_4Mulmodel_2/Flatten/flatten/Reshapemodel/Flatten/flatten/Reshape*
T0*(
_output_shapes
:����������
Y
Sum_6/reduction_indicesConst*
value	B :*
dtype0*
_output_shapes
: 
w
Sum_6Summul_4Sum_6/reduction_indices*
T0*#
_output_shapes
:���������*

Tidx0*
	keep_dims( 
L
Pow_4/yConst*
valueB
 *   @*
dtype0*
_output_shapes
: 
i
Pow_4Powmodel_2/Flatten/flatten/ReshapePow_4/y*
T0*(
_output_shapes
:����������
Y
Sum_7/reduction_indicesConst*
value	B :*
dtype0*
_output_shapes
: 
w
Sum_7SumPow_4Sum_7/reduction_indices*

Tidx0*
	keep_dims( *
T0*#
_output_shapes
:���������
C
Sqrt_4SqrtSum_7*
T0*#
_output_shapes
:���������
L
Pow_5/yConst*
valueB
 *   @*
dtype0*
_output_shapes
: 
g
Pow_5Powmodel/Flatten/flatten/ReshapePow_5/y*(
_output_shapes
:����������*
T0
Y
Sum_8/reduction_indicesConst*
value	B :*
dtype0*
_output_shapes
: 
w
Sum_8SumPow_5Sum_8/reduction_indices*
T0*#
_output_shapes
:���������*

Tidx0*
	keep_dims( 
C
Sqrt_5SqrtSum_8*
T0*#
_output_shapes
:���������
J
mul_5MulSqrt_4Sqrt_5*
T0*#
_output_shapes
:���������
P
	truediv_2RealDivSum_6mul_5*#
_output_shapes
:���������*
T0
}
subSubmodel/Flatten/flatten/Reshapemodel_1/Flatten/flatten/Reshape*
T0*(
_output_shapes
:����������
L
Pow_6/yConst*
valueB
 *   @*
dtype0*
_output_shapes
: 
M
Pow_6PowsubPow_6/y*
T0*(
_output_shapes
:����������
Y
Sum_9/reduction_indicesConst*
value	B :*
dtype0*
_output_shapes
: 
{
Sum_9SumPow_6Sum_9/reduction_indices*
T0*'
_output_shapes
:���������*

Tidx0*
	keep_dims(
G
Sqrt_6SqrtSum_9*'
_output_shapes
:���������*
T0

sub_1Submodel/Flatten/flatten/Reshapemodel_2/Flatten/flatten/Reshape*
T0*(
_output_shapes
:����������
L
Pow_7/yConst*
valueB
 *   @*
dtype0*
_output_shapes
: 
O
Pow_7Powsub_1Pow_7/y*
T0*(
_output_shapes
:����������
Z
Sum_10/reduction_indicesConst*
value	B :*
dtype0*
_output_shapes
: 
}
Sum_10SumPow_7Sum_10/reduction_indices*'
_output_shapes
:���������*

Tidx0*
	keep_dims(*
T0
H
Sqrt_7SqrtSum_10*'
_output_shapes
:���������*
T0
N
sub_2SubSqrt_6Sqrt_7*
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
addAddsub_2add/y*'
_output_shapes
:���������*
T0
N
	Maximum/yConst*
dtype0*
_output_shapes
: *
valueB
 *    
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
MeanMeanMaximumConst*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
X
Variable/initial_valueConst*
value	B : *
dtype0*
_output_shapes
: 
l
Variable
VariableV2*
shape: *
shared_name *
dtype0*
	container *
_output_shapes
: 
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
gradients/ShapeConst*
dtype0*
_output_shapes
: *
valueB 
X
gradients/grad_ys_0Const*
valueB
 *  �?*
dtype0*
_output_shapes
: 
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
gradients/Mean_grad/TileTilegradients/Mean_grad/Reshapegradients/Mean_grad/Shape*'
_output_shapes
:���������*

Tmultiples0*
T0
b
gradients/Mean_grad/Shape_1ShapeMaximum*
_output_shapes
:*
T0*
out_type0
^
gradients/Mean_grad/Shape_2Const*
dtype0*
_output_shapes
: *
valueB 
c
gradients/Mean_grad/ConstConst*
valueB: *
dtype0*
_output_shapes
:
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
gradients/Mean_grad/Maximum/yConst*
dtype0*
_output_shapes
: *
value	B :
�
gradients/Mean_grad/MaximumMaximumgradients/Mean_grad/Prod_1gradients/Mean_grad/Maximum/y*
_output_shapes
: *
T0
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
gradients/Maximum_grad/Shape_2Shapegradients/Mean_grad/truediv*
_output_shapes
:*
T0*
out_type0
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
#gradients/Maximum_grad/GreaterEqualGreaterEqualadd	Maximum/y*
T0*'
_output_shapes
:���������
�
,gradients/Maximum_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/Maximum_grad/Shapegradients/Maximum_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
gradients/Maximum_grad/SelectSelect#gradients/Maximum_grad/GreaterEqualgradients/Mean_grad/truedivgradients/Maximum_grad/zeros*
T0*'
_output_shapes
:���������
�
gradients/Maximum_grad/Select_1Select#gradients/Maximum_grad/GreaterEqualgradients/Maximum_grad/zerosgradients/Mean_grad/truediv*'
_output_shapes
:���������*
T0
�
gradients/Maximum_grad/SumSumgradients/Maximum_grad/Select,gradients/Maximum_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
gradients/Maximum_grad/ReshapeReshapegradients/Maximum_grad/Sumgradients/Maximum_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
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
/gradients/Maximum_grad/tuple/control_dependencyIdentitygradients/Maximum_grad/Reshape(^gradients/Maximum_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/Maximum_grad/Reshape*'
_output_shapes
:���������
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
gradients/add_grad/Shape_1Const*
dtype0*
_output_shapes
: *
valueB 
�
(gradients/add_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_grad/Shapegradients/add_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
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
gradients/add_grad/Sum_1Sum/gradients/Maximum_grad/tuple/control_dependency*gradients/add_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
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
-gradients/add_grad/tuple/control_dependency_1Identitygradients/add_grad/Reshape_1$^gradients/add_grad/tuple/group_deps*
_output_shapes
: *
T0*/
_class%
#!loc:@gradients/add_grad/Reshape_1
`
gradients/sub_2_grad/ShapeShapeSqrt_6*
T0*
out_type0*
_output_shapes
:
b
gradients/sub_2_grad/Shape_1ShapeSqrt_7*
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
gradients/sub_2_grad/SumSum+gradients/add_grad/tuple/control_dependency*gradients/sub_2_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
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
:*

Tidx0*
	keep_dims( 
^
gradients/sub_2_grad/NegNeggradients/sub_2_grad/Sum_1*
T0*
_output_shapes
:
�
gradients/sub_2_grad/Reshape_1Reshapegradients/sub_2_grad/Neggradients/sub_2_grad/Shape_1*'
_output_shapes
:���������*
T0*
Tshape0
m
%gradients/sub_2_grad/tuple/group_depsNoOp^gradients/sub_2_grad/Reshape^gradients/sub_2_grad/Reshape_1
�
-gradients/sub_2_grad/tuple/control_dependencyIdentitygradients/sub_2_grad/Reshape&^gradients/sub_2_grad/tuple/group_deps*'
_output_shapes
:���������*
T0*/
_class%
#!loc:@gradients/sub_2_grad/Reshape
�
/gradients/sub_2_grad/tuple/control_dependency_1Identitygradients/sub_2_grad/Reshape_1&^gradients/sub_2_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/sub_2_grad/Reshape_1*'
_output_shapes
:���������
�
gradients/Sqrt_6_grad/SqrtGradSqrtGradSqrt_6-gradients/sub_2_grad/tuple/control_dependency*
T0*'
_output_shapes
:���������
�
gradients/Sqrt_7_grad/SqrtGradSqrtGradSqrt_7/gradients/sub_2_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:���������
_
gradients/Sum_9_grad/ShapeShapePow_6*
T0*
out_type0*
_output_shapes
:
�
gradients/Sum_9_grad/SizeConst*-
_class#
!loc:@gradients/Sum_9_grad/Shape*
value	B :*
dtype0*
_output_shapes
: 
�
gradients/Sum_9_grad/addAddSum_9/reduction_indicesgradients/Sum_9_grad/Size*
T0*-
_class#
!loc:@gradients/Sum_9_grad/Shape*
_output_shapes
: 
�
gradients/Sum_9_grad/modFloorModgradients/Sum_9_grad/addgradients/Sum_9_grad/Size*
T0*-
_class#
!loc:@gradients/Sum_9_grad/Shape*
_output_shapes
: 
�
gradients/Sum_9_grad/Shape_1Const*-
_class#
!loc:@gradients/Sum_9_grad/Shape*
valueB *
dtype0*
_output_shapes
: 
�
 gradients/Sum_9_grad/range/startConst*-
_class#
!loc:@gradients/Sum_9_grad/Shape*
value	B : *
dtype0*
_output_shapes
: 
�
 gradients/Sum_9_grad/range/deltaConst*-
_class#
!loc:@gradients/Sum_9_grad/Shape*
value	B :*
dtype0*
_output_shapes
: 
�
gradients/Sum_9_grad/rangeRange gradients/Sum_9_grad/range/startgradients/Sum_9_grad/Size gradients/Sum_9_grad/range/delta*-
_class#
!loc:@gradients/Sum_9_grad/Shape*
_output_shapes
:*

Tidx0
�
gradients/Sum_9_grad/Fill/valueConst*
dtype0*
_output_shapes
: *-
_class#
!loc:@gradients/Sum_9_grad/Shape*
value	B :
�
gradients/Sum_9_grad/FillFillgradients/Sum_9_grad/Shape_1gradients/Sum_9_grad/Fill/value*
T0*-
_class#
!loc:@gradients/Sum_9_grad/Shape*

index_type0*
_output_shapes
: 
�
"gradients/Sum_9_grad/DynamicStitchDynamicStitchgradients/Sum_9_grad/rangegradients/Sum_9_grad/modgradients/Sum_9_grad/Shapegradients/Sum_9_grad/Fill*
T0*-
_class#
!loc:@gradients/Sum_9_grad/Shape*
N*
_output_shapes
:
�
gradients/Sum_9_grad/Maximum/yConst*-
_class#
!loc:@gradients/Sum_9_grad/Shape*
value	B :*
dtype0*
_output_shapes
: 
�
gradients/Sum_9_grad/MaximumMaximum"gradients/Sum_9_grad/DynamicStitchgradients/Sum_9_grad/Maximum/y*
T0*-
_class#
!loc:@gradients/Sum_9_grad/Shape*
_output_shapes
:
�
gradients/Sum_9_grad/floordivFloorDivgradients/Sum_9_grad/Shapegradients/Sum_9_grad/Maximum*
T0*-
_class#
!loc:@gradients/Sum_9_grad/Shape*
_output_shapes
:
�
gradients/Sum_9_grad/ReshapeReshapegradients/Sqrt_6_grad/SqrtGrad"gradients/Sum_9_grad/DynamicStitch*0
_output_shapes
:������������������*
T0*
Tshape0
�
gradients/Sum_9_grad/TileTilegradients/Sum_9_grad/Reshapegradients/Sum_9_grad/floordiv*

Tmultiples0*
T0*(
_output_shapes
:����������
`
gradients/Sum_10_grad/ShapeShapePow_7*
_output_shapes
:*
T0*
out_type0
�
gradients/Sum_10_grad/SizeConst*.
_class$
" loc:@gradients/Sum_10_grad/Shape*
value	B :*
dtype0*
_output_shapes
: 
�
gradients/Sum_10_grad/addAddSum_10/reduction_indicesgradients/Sum_10_grad/Size*
_output_shapes
: *
T0*.
_class$
" loc:@gradients/Sum_10_grad/Shape
�
gradients/Sum_10_grad/modFloorModgradients/Sum_10_grad/addgradients/Sum_10_grad/Size*
T0*.
_class$
" loc:@gradients/Sum_10_grad/Shape*
_output_shapes
: 
�
gradients/Sum_10_grad/Shape_1Const*.
_class$
" loc:@gradients/Sum_10_grad/Shape*
valueB *
dtype0*
_output_shapes
: 
�
!gradients/Sum_10_grad/range/startConst*
dtype0*
_output_shapes
: *.
_class$
" loc:@gradients/Sum_10_grad/Shape*
value	B : 
�
!gradients/Sum_10_grad/range/deltaConst*.
_class$
" loc:@gradients/Sum_10_grad/Shape*
value	B :*
dtype0*
_output_shapes
: 
�
gradients/Sum_10_grad/rangeRange!gradients/Sum_10_grad/range/startgradients/Sum_10_grad/Size!gradients/Sum_10_grad/range/delta*.
_class$
" loc:@gradients/Sum_10_grad/Shape*
_output_shapes
:*

Tidx0
�
 gradients/Sum_10_grad/Fill/valueConst*
dtype0*
_output_shapes
: *.
_class$
" loc:@gradients/Sum_10_grad/Shape*
value	B :
�
gradients/Sum_10_grad/FillFillgradients/Sum_10_grad/Shape_1 gradients/Sum_10_grad/Fill/value*
T0*.
_class$
" loc:@gradients/Sum_10_grad/Shape*

index_type0*
_output_shapes
: 
�
#gradients/Sum_10_grad/DynamicStitchDynamicStitchgradients/Sum_10_grad/rangegradients/Sum_10_grad/modgradients/Sum_10_grad/Shapegradients/Sum_10_grad/Fill*
T0*.
_class$
" loc:@gradients/Sum_10_grad/Shape*
N*
_output_shapes
:
�
gradients/Sum_10_grad/Maximum/yConst*.
_class$
" loc:@gradients/Sum_10_grad/Shape*
value	B :*
dtype0*
_output_shapes
: 
�
gradients/Sum_10_grad/MaximumMaximum#gradients/Sum_10_grad/DynamicStitchgradients/Sum_10_grad/Maximum/y*
_output_shapes
:*
T0*.
_class$
" loc:@gradients/Sum_10_grad/Shape
�
gradients/Sum_10_grad/floordivFloorDivgradients/Sum_10_grad/Shapegradients/Sum_10_grad/Maximum*
T0*.
_class$
" loc:@gradients/Sum_10_grad/Shape*
_output_shapes
:
�
gradients/Sum_10_grad/ReshapeReshapegradients/Sqrt_7_grad/SqrtGrad#gradients/Sum_10_grad/DynamicStitch*
T0*
Tshape0*0
_output_shapes
:������������������
�
gradients/Sum_10_grad/TileTilegradients/Sum_10_grad/Reshapegradients/Sum_10_grad/floordiv*
T0*(
_output_shapes
:����������*

Tmultiples0
]
gradients/Pow_6_grad/ShapeShapesub*
_output_shapes
:*
T0*
out_type0
_
gradients/Pow_6_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
�
*gradients/Pow_6_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/Pow_6_grad/Shapegradients/Pow_6_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
v
gradients/Pow_6_grad/mulMulgradients/Sum_9_grad/TilePow_6/y*
T0*(
_output_shapes
:����������
_
gradients/Pow_6_grad/sub/yConst*
dtype0*
_output_shapes
: *
valueB
 *  �?
e
gradients/Pow_6_grad/subSubPow_6/ygradients/Pow_6_grad/sub/y*
_output_shapes
: *
T0
q
gradients/Pow_6_grad/PowPowsubgradients/Pow_6_grad/sub*(
_output_shapes
:����������*
T0
�
gradients/Pow_6_grad/mul_1Mulgradients/Pow_6_grad/mulgradients/Pow_6_grad/Pow*
T0*(
_output_shapes
:����������
�
gradients/Pow_6_grad/SumSumgradients/Pow_6_grad/mul_1*gradients/Pow_6_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
�
gradients/Pow_6_grad/ReshapeReshapegradients/Pow_6_grad/Sumgradients/Pow_6_grad/Shape*(
_output_shapes
:����������*
T0*
Tshape0
c
gradients/Pow_6_grad/Greater/yConst*
valueB
 *    *
dtype0*
_output_shapes
: 

gradients/Pow_6_grad/GreaterGreatersubgradients/Pow_6_grad/Greater/y*(
_output_shapes
:����������*
T0
g
$gradients/Pow_6_grad/ones_like/ShapeShapesub*
T0*
out_type0*
_output_shapes
:
i
$gradients/Pow_6_grad/ones_like/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *  �?
�
gradients/Pow_6_grad/ones_likeFill$gradients/Pow_6_grad/ones_like/Shape$gradients/Pow_6_grad/ones_like/Const*
T0*

index_type0*(
_output_shapes
:����������
�
gradients/Pow_6_grad/SelectSelectgradients/Pow_6_grad/Greatersubgradients/Pow_6_grad/ones_like*
T0*(
_output_shapes
:����������
o
gradients/Pow_6_grad/LogLoggradients/Pow_6_grad/Select*
T0*(
_output_shapes
:����������
d
gradients/Pow_6_grad/zeros_like	ZerosLikesub*(
_output_shapes
:����������*
T0
�
gradients/Pow_6_grad/Select_1Selectgradients/Pow_6_grad/Greatergradients/Pow_6_grad/Loggradients/Pow_6_grad/zeros_like*
T0*(
_output_shapes
:����������
v
gradients/Pow_6_grad/mul_2Mulgradients/Sum_9_grad/TilePow_6*
T0*(
_output_shapes
:����������
�
gradients/Pow_6_grad/mul_3Mulgradients/Pow_6_grad/mul_2gradients/Pow_6_grad/Select_1*
T0*(
_output_shapes
:����������
�
gradients/Pow_6_grad/Sum_1Sumgradients/Pow_6_grad/mul_3,gradients/Pow_6_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
gradients/Pow_6_grad/Reshape_1Reshapegradients/Pow_6_grad/Sum_1gradients/Pow_6_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
m
%gradients/Pow_6_grad/tuple/group_depsNoOp^gradients/Pow_6_grad/Reshape^gradients/Pow_6_grad/Reshape_1
�
-gradients/Pow_6_grad/tuple/control_dependencyIdentitygradients/Pow_6_grad/Reshape&^gradients/Pow_6_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/Pow_6_grad/Reshape*(
_output_shapes
:����������
�
/gradients/Pow_6_grad/tuple/control_dependency_1Identitygradients/Pow_6_grad/Reshape_1&^gradients/Pow_6_grad/tuple/group_deps*
_output_shapes
: *
T0*1
_class'
%#loc:@gradients/Pow_6_grad/Reshape_1
_
gradients/Pow_7_grad/ShapeShapesub_1*
T0*
out_type0*
_output_shapes
:
_
gradients/Pow_7_grad/Shape_1Const*
_output_shapes
: *
valueB *
dtype0
�
*gradients/Pow_7_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/Pow_7_grad/Shapegradients/Pow_7_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
w
gradients/Pow_7_grad/mulMulgradients/Sum_10_grad/TilePow_7/y*(
_output_shapes
:����������*
T0
_
gradients/Pow_7_grad/sub/yConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
e
gradients/Pow_7_grad/subSubPow_7/ygradients/Pow_7_grad/sub/y*
T0*
_output_shapes
: 
s
gradients/Pow_7_grad/PowPowsub_1gradients/Pow_7_grad/sub*
T0*(
_output_shapes
:����������
�
gradients/Pow_7_grad/mul_1Mulgradients/Pow_7_grad/mulgradients/Pow_7_grad/Pow*
T0*(
_output_shapes
:����������
�
gradients/Pow_7_grad/SumSumgradients/Pow_7_grad/mul_1*gradients/Pow_7_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
gradients/Pow_7_grad/ReshapeReshapegradients/Pow_7_grad/Sumgradients/Pow_7_grad/Shape*
Tshape0*(
_output_shapes
:����������*
T0
c
gradients/Pow_7_grad/Greater/yConst*
_output_shapes
: *
valueB
 *    *
dtype0
�
gradients/Pow_7_grad/GreaterGreatersub_1gradients/Pow_7_grad/Greater/y*
T0*(
_output_shapes
:����������
i
$gradients/Pow_7_grad/ones_like/ShapeShapesub_1*
out_type0*
_output_shapes
:*
T0
i
$gradients/Pow_7_grad/ones_like/ConstConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
gradients/Pow_7_grad/ones_likeFill$gradients/Pow_7_grad/ones_like/Shape$gradients/Pow_7_grad/ones_like/Const*
T0*

index_type0*(
_output_shapes
:����������
�
gradients/Pow_7_grad/SelectSelectgradients/Pow_7_grad/Greatersub_1gradients/Pow_7_grad/ones_like*
T0*(
_output_shapes
:����������
o
gradients/Pow_7_grad/LogLoggradients/Pow_7_grad/Select*
T0*(
_output_shapes
:����������
f
gradients/Pow_7_grad/zeros_like	ZerosLikesub_1*
T0*(
_output_shapes
:����������
�
gradients/Pow_7_grad/Select_1Selectgradients/Pow_7_grad/Greatergradients/Pow_7_grad/Loggradients/Pow_7_grad/zeros_like*(
_output_shapes
:����������*
T0
w
gradients/Pow_7_grad/mul_2Mulgradients/Sum_10_grad/TilePow_7*
T0*(
_output_shapes
:����������
�
gradients/Pow_7_grad/mul_3Mulgradients/Pow_7_grad/mul_2gradients/Pow_7_grad/Select_1*
T0*(
_output_shapes
:����������
�
gradients/Pow_7_grad/Sum_1Sumgradients/Pow_7_grad/mul_3,gradients/Pow_7_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
�
gradients/Pow_7_grad/Reshape_1Reshapegradients/Pow_7_grad/Sum_1gradients/Pow_7_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
m
%gradients/Pow_7_grad/tuple/group_depsNoOp^gradients/Pow_7_grad/Reshape^gradients/Pow_7_grad/Reshape_1
�
-gradients/Pow_7_grad/tuple/control_dependencyIdentitygradients/Pow_7_grad/Reshape&^gradients/Pow_7_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/Pow_7_grad/Reshape*(
_output_shapes
:����������
�
/gradients/Pow_7_grad/tuple/control_dependency_1Identitygradients/Pow_7_grad/Reshape_1&^gradients/Pow_7_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/Pow_7_grad/Reshape_1*
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
gradients/sub_grad/SumSum-gradients/Pow_6_grad/tuple/control_dependency(gradients/sub_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
gradients/sub_grad/ReshapeReshapegradients/sub_grad/Sumgradients/sub_grad/Shape*
T0*
Tshape0*(
_output_shapes
:����������
�
gradients/sub_grad/Sum_1Sum-gradients/Pow_6_grad/tuple/control_dependency*gradients/sub_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
Z
gradients/sub_grad/NegNeggradients/sub_grad/Sum_1*
_output_shapes
:*
T0
�
gradients/sub_grad/Reshape_1Reshapegradients/sub_grad/Neggradients/sub_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:����������
g
#gradients/sub_grad/tuple/group_depsNoOp^gradients/sub_grad/Reshape^gradients/sub_grad/Reshape_1
�
+gradients/sub_grad/tuple/control_dependencyIdentitygradients/sub_grad/Reshape$^gradients/sub_grad/tuple/group_deps*
T0*-
_class#
!loc:@gradients/sub_grad/Reshape*(
_output_shapes
:����������
�
-gradients/sub_grad/tuple/control_dependency_1Identitygradients/sub_grad/Reshape_1$^gradients/sub_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/sub_grad/Reshape_1*(
_output_shapes
:����������
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
gradients/sub_1_grad/SumSum-gradients/Pow_7_grad/tuple/control_dependency*gradients/sub_1_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
gradients/sub_1_grad/ReshapeReshapegradients/sub_1_grad/Sumgradients/sub_1_grad/Shape*
T0*
Tshape0*(
_output_shapes
:����������
�
gradients/sub_1_grad/Sum_1Sum-gradients/Pow_7_grad/tuple/control_dependency,gradients/sub_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
^
gradients/sub_1_grad/NegNeggradients/sub_1_grad/Sum_1*
_output_shapes
:*
T0
�
gradients/sub_1_grad/Reshape_1Reshapegradients/sub_1_grad/Neggradients/sub_1_grad/Shape_1*(
_output_shapes
:����������*
T0*
Tshape0
m
%gradients/sub_1_grad/tuple/group_depsNoOp^gradients/sub_1_grad/Reshape^gradients/sub_1_grad/Reshape_1
�
-gradients/sub_1_grad/tuple/control_dependencyIdentitygradients/sub_1_grad/Reshape&^gradients/sub_1_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/sub_1_grad/Reshape*(
_output_shapes
:����������
�
/gradients/sub_1_grad/tuple/control_dependency_1Identitygradients/sub_1_grad/Reshape_1&^gradients/sub_1_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/sub_1_grad/Reshape_1*(
_output_shapes
:����������
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
:���������
�
gradients/AddNAddN+gradients/sub_grad/tuple/control_dependency-gradients/sub_1_grad/tuple/control_dependency*-
_class#
!loc:@gradients/sub_grad/Reshape*
N*(
_output_shapes
:����������*
T0
�
2gradients/model/Flatten/flatten/Reshape_grad/ShapeShapemodel/conv5/MaxPool2D/MaxPool*
out_type0*
_output_shapes
:*
T0
�
4gradients/model/Flatten/flatten/Reshape_grad/ReshapeReshapegradients/AddN2gradients/model/Flatten/flatten/Reshape_grad/Shape*/
_output_shapes
:���������*
T0*
Tshape0
�
4gradients/model_2/Flatten/flatten/Reshape_grad/ShapeShapemodel_2/conv5/MaxPool2D/MaxPool*
T0*
out_type0*
_output_shapes
:
�
6gradients/model_2/Flatten/flatten/Reshape_grad/ReshapeReshape/gradients/sub_1_grad/tuple/control_dependency_14gradients/model_2/Flatten/flatten/Reshape_grad/Shape*
T0*
Tshape0*/
_output_shapes
:���������
�
:gradients/model_1/conv5/MaxPool2D/MaxPool_grad/MaxPoolGradMaxPoolGradmodel_1/conv5/BiasAddmodel_1/conv5/MaxPool2D/MaxPool6gradients/model_1/Flatten/flatten/Reshape_grad/Reshape*
ksize
*
paddingSAME*/
_output_shapes
:���������
*
T0*
data_formatNHWC*
strides

�
8gradients/model/conv5/MaxPool2D/MaxPool_grad/MaxPoolGradMaxPoolGradmodel/conv5/BiasAddmodel/conv5/MaxPool2D/MaxPool4gradients/model/Flatten/flatten/Reshape_grad/Reshape*
ksize
*
paddingSAME*/
_output_shapes
:���������
*
T0*
strides
*
data_formatNHWC
�
:gradients/model_2/conv5/MaxPool2D/MaxPool_grad/MaxPoolGradMaxPoolGradmodel_2/conv5/BiasAddmodel_2/conv5/MaxPool2D/MaxPool6gradients/model_2/Flatten/flatten/Reshape_grad/Reshape*/
_output_shapes
:���������
*
T0*
data_formatNHWC*
strides
*
ksize
*
paddingSAME
�
0gradients/model_1/conv5/BiasAdd_grad/BiasAddGradBiasAddGrad:gradients/model_1/conv5/MaxPool2D/MaxPool_grad/MaxPoolGrad*
T0*
data_formatNHWC*
_output_shapes
:
�
5gradients/model_1/conv5/BiasAdd_grad/tuple/group_depsNoOp1^gradients/model_1/conv5/BiasAdd_grad/BiasAddGrad;^gradients/model_1/conv5/MaxPool2D/MaxPool_grad/MaxPoolGrad
�
=gradients/model_1/conv5/BiasAdd_grad/tuple/control_dependencyIdentity:gradients/model_1/conv5/MaxPool2D/MaxPool_grad/MaxPoolGrad6^gradients/model_1/conv5/BiasAdd_grad/tuple/group_deps*/
_output_shapes
:���������
*
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
:���������

�
=gradients/model/conv5/BiasAdd_grad/tuple/control_dependency_1Identity.gradients/model/conv5/BiasAdd_grad/BiasAddGrad4^gradients/model/conv5/BiasAdd_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients/model/conv5/BiasAdd_grad/BiasAddGrad*
_output_shapes
:
�
0gradients/model_2/conv5/BiasAdd_grad/BiasAddGradBiasAddGrad:gradients/model_2/conv5/MaxPool2D/MaxPool_grad/MaxPoolGrad*
data_formatNHWC*
_output_shapes
:*
T0
�
5gradients/model_2/conv5/BiasAdd_grad/tuple/group_depsNoOp1^gradients/model_2/conv5/BiasAdd_grad/BiasAddGrad;^gradients/model_2/conv5/MaxPool2D/MaxPool_grad/MaxPoolGrad
�
=gradients/model_2/conv5/BiasAdd_grad/tuple/control_dependencyIdentity:gradients/model_2/conv5/MaxPool2D/MaxPool_grad/MaxPoolGrad6^gradients/model_2/conv5/BiasAdd_grad/tuple/group_deps*/
_output_shapes
:���������
*
T0*M
_classC
A?loc:@gradients/model_2/conv5/MaxPool2D/MaxPool_grad/MaxPoolGrad
�
?gradients/model_2/conv5/BiasAdd_grad/tuple/control_dependency_1Identity0gradients/model_2/conv5/BiasAdd_grad/BiasAddGrad6^gradients/model_2/conv5/BiasAdd_grad/tuple/group_deps*C
_class9
75loc:@gradients/model_2/conv5/BiasAdd_grad/BiasAddGrad*
_output_shapes
:*
T0
�
*gradients/model_1/conv5/Conv2D_grad/ShapeNShapeNmodel_1/conv4/MaxPool2D/MaxPoolconv5/weights/read*
T0*
out_type0*
N* 
_output_shapes
::
�
7gradients/model_1/conv5/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput*gradients/model_1/conv5/Conv2D_grad/ShapeNconv5/weights/read=gradients/model_1/conv5/BiasAdd_grad/tuple/control_dependency*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*0
_output_shapes
:���������
�*
	dilations
*
T0
�
8gradients/model_1/conv5/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFiltermodel_1/conv4/MaxPool2D/MaxPool,gradients/model_1/conv5/Conv2D_grad/ShapeN:1=gradients/model_1/conv5/BiasAdd_grad/tuple/control_dependency*
use_cudnn_on_gpu(*
paddingSAME*'
_output_shapes
:�*
	dilations
*
T0*
data_formatNHWC*
strides

�
4gradients/model_1/conv5/Conv2D_grad/tuple/group_depsNoOp9^gradients/model_1/conv5/Conv2D_grad/Conv2DBackpropFilter8^gradients/model_1/conv5/Conv2D_grad/Conv2DBackpropInput
�
<gradients/model_1/conv5/Conv2D_grad/tuple/control_dependencyIdentity7gradients/model_1/conv5/Conv2D_grad/Conv2DBackpropInput5^gradients/model_1/conv5/Conv2D_grad/tuple/group_deps*0
_output_shapes
:���������
�*
T0*J
_class@
><loc:@gradients/model_1/conv5/Conv2D_grad/Conv2DBackpropInput
�
>gradients/model_1/conv5/Conv2D_grad/tuple/control_dependency_1Identity8gradients/model_1/conv5/Conv2D_grad/Conv2DBackpropFilter5^gradients/model_1/conv5/Conv2D_grad/tuple/group_deps*
T0*K
_classA
?=loc:@gradients/model_1/conv5/Conv2D_grad/Conv2DBackpropFilter*'
_output_shapes
:�
�
(gradients/model/conv5/Conv2D_grad/ShapeNShapeNmodel/conv4/MaxPool2D/MaxPoolconv5/weights/read*
N* 
_output_shapes
::*
T0*
out_type0
�
5gradients/model/conv5/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput(gradients/model/conv5/Conv2D_grad/ShapeNconv5/weights/read;gradients/model/conv5/BiasAdd_grad/tuple/control_dependency*0
_output_shapes
:���������
�*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME
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
:���������
�
�
<gradients/model/conv5/Conv2D_grad/tuple/control_dependency_1Identity6gradients/model/conv5/Conv2D_grad/Conv2DBackpropFilter3^gradients/model/conv5/Conv2D_grad/tuple/group_deps*
T0*I
_class?
=;loc:@gradients/model/conv5/Conv2D_grad/Conv2DBackpropFilter*'
_output_shapes
:�
�
*gradients/model_2/conv5/Conv2D_grad/ShapeNShapeNmodel_2/conv4/MaxPool2D/MaxPoolconv5/weights/read*
T0*
out_type0*
N* 
_output_shapes
::
�
7gradients/model_2/conv5/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput*gradients/model_2/conv5/Conv2D_grad/ShapeNconv5/weights/read=gradients/model_2/conv5/BiasAdd_grad/tuple/control_dependency*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*0
_output_shapes
:���������
�
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
<gradients/model_2/conv5/Conv2D_grad/tuple/control_dependencyIdentity7gradients/model_2/conv5/Conv2D_grad/Conv2DBackpropInput5^gradients/model_2/conv5/Conv2D_grad/tuple/group_deps*0
_output_shapes
:���������
�*
T0*J
_class@
><loc:@gradients/model_2/conv5/Conv2D_grad/Conv2DBackpropInput
�
>gradients/model_2/conv5/Conv2D_grad/tuple/control_dependency_1Identity8gradients/model_2/conv5/Conv2D_grad/Conv2DBackpropFilter5^gradients/model_2/conv5/Conv2D_grad/tuple/group_deps*
T0*K
_classA
?=loc:@gradients/model_2/conv5/Conv2D_grad/Conv2DBackpropFilter*'
_output_shapes
:�
�
gradients/AddN_1AddN?gradients/model_1/conv5/BiasAdd_grad/tuple/control_dependency_1=gradients/model/conv5/BiasAdd_grad/tuple/control_dependency_1?gradients/model_2/conv5/BiasAdd_grad/tuple/control_dependency_1*
T0*C
_class9
75loc:@gradients/model_1/conv5/BiasAdd_grad/BiasAddGrad*
N*
_output_shapes
:
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
:����������
�
8gradients/model/conv4/MaxPool2D/MaxPool_grad/MaxPoolGradMaxPoolGradmodel/conv4/conv4/Relumodel/conv4/MaxPool2D/MaxPool:gradients/model/conv5/Conv2D_grad/tuple/control_dependency*
paddingSAME*0
_output_shapes
:����������*
T0*
data_formatNHWC*
strides
*
ksize

�
:gradients/model_2/conv4/MaxPool2D/MaxPool_grad/MaxPoolGradMaxPoolGradmodel_2/conv4/conv4/Relumodel_2/conv4/MaxPool2D/MaxPool<gradients/model_2/conv5/Conv2D_grad/tuple/control_dependency*
ksize
*
paddingSAME*0
_output_shapes
:����������*
T0*
strides
*
data_formatNHWC
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
:����������*
T0
�
.gradients/model/conv4/conv4/Relu_grad/ReluGradReluGrad8gradients/model/conv4/MaxPool2D/MaxPool_grad/MaxPoolGradmodel/conv4/conv4/Relu*
T0*0
_output_shapes
:����������
�
0gradients/model_2/conv4/conv4/Relu_grad/ReluGradReluGrad:gradients/model_2/conv4/MaxPool2D/MaxPool_grad/MaxPoolGradmodel_2/conv4/conv4/Relu*
T0*0
_output_shapes
:����������
�
0gradients/model_1/conv4/BiasAdd_grad/BiasAddGradBiasAddGrad0gradients/model_1/conv4/conv4/Relu_grad/ReluGrad*
_output_shapes	
:�*
T0*
data_formatNHWC
�
5gradients/model_1/conv4/BiasAdd_grad/tuple/group_depsNoOp1^gradients/model_1/conv4/BiasAdd_grad/BiasAddGrad1^gradients/model_1/conv4/conv4/Relu_grad/ReluGrad
�
=gradients/model_1/conv4/BiasAdd_grad/tuple/control_dependencyIdentity0gradients/model_1/conv4/conv4/Relu_grad/ReluGrad6^gradients/model_1/conv4/BiasAdd_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/model_1/conv4/conv4/Relu_grad/ReluGrad*0
_output_shapes
:����������
�
?gradients/model_1/conv4/BiasAdd_grad/tuple/control_dependency_1Identity0gradients/model_1/conv4/BiasAdd_grad/BiasAddGrad6^gradients/model_1/conv4/BiasAdd_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/model_1/conv4/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:�
�
.gradients/model/conv4/BiasAdd_grad/BiasAddGradBiasAddGrad.gradients/model/conv4/conv4/Relu_grad/ReluGrad*
_output_shapes	
:�*
T0*
data_formatNHWC
�
3gradients/model/conv4/BiasAdd_grad/tuple/group_depsNoOp/^gradients/model/conv4/BiasAdd_grad/BiasAddGrad/^gradients/model/conv4/conv4/Relu_grad/ReluGrad
�
;gradients/model/conv4/BiasAdd_grad/tuple/control_dependencyIdentity.gradients/model/conv4/conv4/Relu_grad/ReluGrad4^gradients/model/conv4/BiasAdd_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients/model/conv4/conv4/Relu_grad/ReluGrad*0
_output_shapes
:����������
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
:����������
�
?gradients/model_2/conv4/BiasAdd_grad/tuple/control_dependency_1Identity0gradients/model_2/conv4/BiasAdd_grad/BiasAddGrad6^gradients/model_2/conv4/BiasAdd_grad/tuple/group_deps*
_output_shapes	
:�*
T0*C
_class9
75loc:@gradients/model_2/conv4/BiasAdd_grad/BiasAddGrad
�
*gradients/model_1/conv4/Conv2D_grad/ShapeNShapeNmodel_1/conv3/MaxPool2D/MaxPoolconv4/weights/read* 
_output_shapes
::*
T0*
out_type0*
N
�
7gradients/model_1/conv4/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput*gradients/model_1/conv4/Conv2D_grad/ShapeNconv4/weights/read=gradients/model_1/conv4/BiasAdd_grad/tuple/control_dependency*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*0
_output_shapes
:����������*
	dilations
*
T0
�
8gradients/model_1/conv4/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFiltermodel_1/conv3/MaxPool2D/MaxPool,gradients/model_1/conv4/Conv2D_grad/ShapeN:1=gradients/model_1/conv4/BiasAdd_grad/tuple/control_dependency*
paddingSAME*(
_output_shapes
:��*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(
�
4gradients/model_1/conv4/Conv2D_grad/tuple/group_depsNoOp9^gradients/model_1/conv4/Conv2D_grad/Conv2DBackpropFilter8^gradients/model_1/conv4/Conv2D_grad/Conv2DBackpropInput
�
<gradients/model_1/conv4/Conv2D_grad/tuple/control_dependencyIdentity7gradients/model_1/conv4/Conv2D_grad/Conv2DBackpropInput5^gradients/model_1/conv4/Conv2D_grad/tuple/group_deps*
T0*J
_class@
><loc:@gradients/model_1/conv4/Conv2D_grad/Conv2DBackpropInput*0
_output_shapes
:����������
�
>gradients/model_1/conv4/Conv2D_grad/tuple/control_dependency_1Identity8gradients/model_1/conv4/Conv2D_grad/Conv2DBackpropFilter5^gradients/model_1/conv4/Conv2D_grad/tuple/group_deps*(
_output_shapes
:��*
T0*K
_classA
?=loc:@gradients/model_1/conv4/Conv2D_grad/Conv2DBackpropFilter
�
(gradients/model/conv4/Conv2D_grad/ShapeNShapeNmodel/conv3/MaxPool2D/MaxPoolconv4/weights/read*
T0*
out_type0*
N* 
_output_shapes
::
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
:����������
�
6gradients/model/conv4/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFiltermodel/conv3/MaxPool2D/MaxPool*gradients/model/conv4/Conv2D_grad/ShapeN:1;gradients/model/conv4/BiasAdd_grad/tuple/control_dependency*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*(
_output_shapes
:��
�
2gradients/model/conv4/Conv2D_grad/tuple/group_depsNoOp7^gradients/model/conv4/Conv2D_grad/Conv2DBackpropFilter6^gradients/model/conv4/Conv2D_grad/Conv2DBackpropInput
�
:gradients/model/conv4/Conv2D_grad/tuple/control_dependencyIdentity5gradients/model/conv4/Conv2D_grad/Conv2DBackpropInput3^gradients/model/conv4/Conv2D_grad/tuple/group_deps*0
_output_shapes
:����������*
T0*H
_class>
<:loc:@gradients/model/conv4/Conv2D_grad/Conv2DBackpropInput
�
<gradients/model/conv4/Conv2D_grad/tuple/control_dependency_1Identity6gradients/model/conv4/Conv2D_grad/Conv2DBackpropFilter3^gradients/model/conv4/Conv2D_grad/tuple/group_deps*
T0*I
_class?
=;loc:@gradients/model/conv4/Conv2D_grad/Conv2DBackpropFilter*(
_output_shapes
:��
�
*gradients/model_2/conv4/Conv2D_grad/ShapeNShapeNmodel_2/conv3/MaxPool2D/MaxPoolconv4/weights/read*
out_type0*
N* 
_output_shapes
::*
T0
�
7gradients/model_2/conv4/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput*gradients/model_2/conv4/Conv2D_grad/ShapeNconv4/weights/read=gradients/model_2/conv4/BiasAdd_grad/tuple/control_dependency*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*0
_output_shapes
:����������*
	dilations
*
T0
�
8gradients/model_2/conv4/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFiltermodel_2/conv3/MaxPool2D/MaxPool,gradients/model_2/conv4/Conv2D_grad/ShapeN:1=gradients/model_2/conv4/BiasAdd_grad/tuple/control_dependency*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*(
_output_shapes
:��*
	dilations

�
4gradients/model_2/conv4/Conv2D_grad/tuple/group_depsNoOp9^gradients/model_2/conv4/Conv2D_grad/Conv2DBackpropFilter8^gradients/model_2/conv4/Conv2D_grad/Conv2DBackpropInput
�
<gradients/model_2/conv4/Conv2D_grad/tuple/control_dependencyIdentity7gradients/model_2/conv4/Conv2D_grad/Conv2DBackpropInput5^gradients/model_2/conv4/Conv2D_grad/tuple/group_deps*
T0*J
_class@
><loc:@gradients/model_2/conv4/Conv2D_grad/Conv2DBackpropInput*0
_output_shapes
:����������
�
>gradients/model_2/conv4/Conv2D_grad/tuple/control_dependency_1Identity8gradients/model_2/conv4/Conv2D_grad/Conv2DBackpropFilter5^gradients/model_2/conv4/Conv2D_grad/tuple/group_deps*(
_output_shapes
:��*
T0*K
_classA
?=loc:@gradients/model_2/conv4/Conv2D_grad/Conv2DBackpropFilter
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
:���������&�*
T0*
data_formatNHWC*
strides
*
ksize
*
paddingSAME
�
8gradients/model/conv3/MaxPool2D/MaxPool_grad/MaxPoolGradMaxPoolGradmodel/conv3/conv3/Relumodel/conv3/MaxPool2D/MaxPool:gradients/model/conv4/Conv2D_grad/tuple/control_dependency*
ksize
*
paddingSAME*0
_output_shapes
:���������&�*
T0*
strides
*
data_formatNHWC
�
:gradients/model_2/conv3/MaxPool2D/MaxPool_grad/MaxPoolGradMaxPoolGradmodel_2/conv3/conv3/Relumodel_2/conv3/MaxPool2D/MaxPool<gradients/model_2/conv4/Conv2D_grad/tuple/control_dependency*
ksize
*
paddingSAME*0
_output_shapes
:���������&�*
T0*
data_formatNHWC*
strides

�
gradients/AddN_4AddN>gradients/model_1/conv4/Conv2D_grad/tuple/control_dependency_1<gradients/model/conv4/Conv2D_grad/tuple/control_dependency_1>gradients/model_2/conv4/Conv2D_grad/tuple/control_dependency_1*(
_output_shapes
:��*
T0*K
_classA
?=loc:@gradients/model_1/conv4/Conv2D_grad/Conv2DBackpropFilter*
N
�
0gradients/model_1/conv3/conv3/Relu_grad/ReluGradReluGrad:gradients/model_1/conv3/MaxPool2D/MaxPool_grad/MaxPoolGradmodel_1/conv3/conv3/Relu*
T0*0
_output_shapes
:���������&�
�
.gradients/model/conv3/conv3/Relu_grad/ReluGradReluGrad8gradients/model/conv3/MaxPool2D/MaxPool_grad/MaxPoolGradmodel/conv3/conv3/Relu*0
_output_shapes
:���������&�*
T0
�
0gradients/model_2/conv3/conv3/Relu_grad/ReluGradReluGrad:gradients/model_2/conv3/MaxPool2D/MaxPool_grad/MaxPoolGradmodel_2/conv3/conv3/Relu*0
_output_shapes
:���������&�*
T0
�
0gradients/model_1/conv3/BiasAdd_grad/BiasAddGradBiasAddGrad0gradients/model_1/conv3/conv3/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes	
:�
�
5gradients/model_1/conv3/BiasAdd_grad/tuple/group_depsNoOp1^gradients/model_1/conv3/BiasAdd_grad/BiasAddGrad1^gradients/model_1/conv3/conv3/Relu_grad/ReluGrad
�
=gradients/model_1/conv3/BiasAdd_grad/tuple/control_dependencyIdentity0gradients/model_1/conv3/conv3/Relu_grad/ReluGrad6^gradients/model_1/conv3/BiasAdd_grad/tuple/group_deps*0
_output_shapes
:���������&�*
T0*C
_class9
75loc:@gradients/model_1/conv3/conv3/Relu_grad/ReluGrad
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
;gradients/model/conv3/BiasAdd_grad/tuple/control_dependencyIdentity.gradients/model/conv3/conv3/Relu_grad/ReluGrad4^gradients/model/conv3/BiasAdd_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients/model/conv3/conv3/Relu_grad/ReluGrad*0
_output_shapes
:���������&�
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
:���������&�
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
:���������&@*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME
�
8gradients/model_1/conv3/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFiltermodel_1/conv2/MaxPool2D/MaxPool,gradients/model_1/conv3/Conv2D_grad/ShapeN:1=gradients/model_1/conv3/BiasAdd_grad/tuple/control_dependency*'
_output_shapes
:@�*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME
�
4gradients/model_1/conv3/Conv2D_grad/tuple/group_depsNoOp9^gradients/model_1/conv3/Conv2D_grad/Conv2DBackpropFilter8^gradients/model_1/conv3/Conv2D_grad/Conv2DBackpropInput
�
<gradients/model_1/conv3/Conv2D_grad/tuple/control_dependencyIdentity7gradients/model_1/conv3/Conv2D_grad/Conv2DBackpropInput5^gradients/model_1/conv3/Conv2D_grad/tuple/group_deps*J
_class@
><loc:@gradients/model_1/conv3/Conv2D_grad/Conv2DBackpropInput*/
_output_shapes
:���������&@*
T0
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
:���������&@*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(
�
6gradients/model/conv3/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFiltermodel/conv2/MaxPool2D/MaxPool*gradients/model/conv3/Conv2D_grad/ShapeN:1;gradients/model/conv3/BiasAdd_grad/tuple/control_dependency*
use_cudnn_on_gpu(*
paddingSAME*'
_output_shapes
:@�*
	dilations
*
T0*
data_formatNHWC*
strides

�
2gradients/model/conv3/Conv2D_grad/tuple/group_depsNoOp7^gradients/model/conv3/Conv2D_grad/Conv2DBackpropFilter6^gradients/model/conv3/Conv2D_grad/Conv2DBackpropInput
�
:gradients/model/conv3/Conv2D_grad/tuple/control_dependencyIdentity5gradients/model/conv3/Conv2D_grad/Conv2DBackpropInput3^gradients/model/conv3/Conv2D_grad/tuple/group_deps*
T0*H
_class>
<:loc:@gradients/model/conv3/Conv2D_grad/Conv2DBackpropInput*/
_output_shapes
:���������&@
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
7gradients/model_2/conv3/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput*gradients/model_2/conv3/Conv2D_grad/ShapeNconv3/weights/read=gradients/model_2/conv3/BiasAdd_grad/tuple/control_dependency*
use_cudnn_on_gpu(*
paddingSAME*/
_output_shapes
:���������&@*
	dilations
*
T0*
data_formatNHWC*
strides

�
8gradients/model_2/conv3/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFiltermodel_2/conv2/MaxPool2D/MaxPool,gradients/model_2/conv3/Conv2D_grad/ShapeN:1=gradients/model_2/conv3/BiasAdd_grad/tuple/control_dependency*
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
4gradients/model_2/conv3/Conv2D_grad/tuple/group_depsNoOp9^gradients/model_2/conv3/Conv2D_grad/Conv2DBackpropFilter8^gradients/model_2/conv3/Conv2D_grad/Conv2DBackpropInput
�
<gradients/model_2/conv3/Conv2D_grad/tuple/control_dependencyIdentity7gradients/model_2/conv3/Conv2D_grad/Conv2DBackpropInput5^gradients/model_2/conv3/Conv2D_grad/tuple/group_deps*
T0*J
_class@
><loc:@gradients/model_2/conv3/Conv2D_grad/Conv2DBackpropInput*/
_output_shapes
:���������&@
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
:gradients/model_1/conv2/MaxPool2D/MaxPool_grad/MaxPoolGradMaxPoolGradmodel_1/conv2/conv2/Relumodel_1/conv2/MaxPool2D/MaxPool<gradients/model_1/conv3/Conv2D_grad/tuple/control_dependency*
ksize
*
paddingSAME*/
_output_shapes
:���������K@*
T0*
data_formatNHWC*
strides

�
8gradients/model/conv2/MaxPool2D/MaxPool_grad/MaxPoolGradMaxPoolGradmodel/conv2/conv2/Relumodel/conv2/MaxPool2D/MaxPool:gradients/model/conv3/Conv2D_grad/tuple/control_dependency*/
_output_shapes
:���������K@*
T0*
data_formatNHWC*
strides
*
ksize
*
paddingSAME
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
:���������K@
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
:���������K@
�
.gradients/model/conv2/conv2/Relu_grad/ReluGradReluGrad8gradients/model/conv2/MaxPool2D/MaxPool_grad/MaxPoolGradmodel/conv2/conv2/Relu*
T0*/
_output_shapes
:���������K@
�
0gradients/model_2/conv2/conv2/Relu_grad/ReluGradReluGrad:gradients/model_2/conv2/MaxPool2D/MaxPool_grad/MaxPoolGradmodel_2/conv2/conv2/Relu*
T0*/
_output_shapes
:���������K@
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
:���������K@
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
;gradients/model/conv2/BiasAdd_grad/tuple/control_dependencyIdentity.gradients/model/conv2/conv2/Relu_grad/ReluGrad4^gradients/model/conv2/BiasAdd_grad/tuple/group_deps*A
_class7
53loc:@gradients/model/conv2/conv2/Relu_grad/ReluGrad*/
_output_shapes
:���������K@*
T0
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
:���������K@
�
?gradients/model_2/conv2/BiasAdd_grad/tuple/control_dependency_1Identity0gradients/model_2/conv2/BiasAdd_grad/BiasAddGrad6^gradients/model_2/conv2/BiasAdd_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/model_2/conv2/BiasAdd_grad/BiasAddGrad*
_output_shapes
:@
�
*gradients/model_1/conv2/Conv2D_grad/ShapeNShapeNmodel_1/conv1/MaxPool2D/MaxPoolconv2/weights/read*
out_type0*
N* 
_output_shapes
::*
T0
�
7gradients/model_1/conv2/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput*gradients/model_1/conv2/Conv2D_grad/ShapeNconv2/weights/read=gradients/model_1/conv2/BiasAdd_grad/tuple/control_dependency*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*/
_output_shapes
:���������K *
	dilations
*
T0
�
8gradients/model_1/conv2/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFiltermodel_1/conv1/MaxPool2D/MaxPool,gradients/model_1/conv2/Conv2D_grad/ShapeN:1=gradients/model_1/conv2/BiasAdd_grad/tuple/control_dependency*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*&
_output_shapes
: @*
	dilations

�
4gradients/model_1/conv2/Conv2D_grad/tuple/group_depsNoOp9^gradients/model_1/conv2/Conv2D_grad/Conv2DBackpropFilter8^gradients/model_1/conv2/Conv2D_grad/Conv2DBackpropInput
�
<gradients/model_1/conv2/Conv2D_grad/tuple/control_dependencyIdentity7gradients/model_1/conv2/Conv2D_grad/Conv2DBackpropInput5^gradients/model_1/conv2/Conv2D_grad/tuple/group_deps*
T0*J
_class@
><loc:@gradients/model_1/conv2/Conv2D_grad/Conv2DBackpropInput*/
_output_shapes
:���������K 
�
>gradients/model_1/conv2/Conv2D_grad/tuple/control_dependency_1Identity8gradients/model_1/conv2/Conv2D_grad/Conv2DBackpropFilter5^gradients/model_1/conv2/Conv2D_grad/tuple/group_deps*
T0*K
_classA
?=loc:@gradients/model_1/conv2/Conv2D_grad/Conv2DBackpropFilter*&
_output_shapes
: @
�
(gradients/model/conv2/Conv2D_grad/ShapeNShapeNmodel/conv1/MaxPool2D/MaxPoolconv2/weights/read*
out_type0*
N* 
_output_shapes
::*
T0
�
5gradients/model/conv2/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput(gradients/model/conv2/Conv2D_grad/ShapeNconv2/weights/read;gradients/model/conv2/BiasAdd_grad/tuple/control_dependency*/
_output_shapes
:���������K *
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME
�
6gradients/model/conv2/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFiltermodel/conv1/MaxPool2D/MaxPool*gradients/model/conv2/Conv2D_grad/ShapeN:1;gradients/model/conv2/BiasAdd_grad/tuple/control_dependency*
use_cudnn_on_gpu(*
paddingSAME*&
_output_shapes
: @*
	dilations
*
T0*
strides
*
data_formatNHWC
�
2gradients/model/conv2/Conv2D_grad/tuple/group_depsNoOp7^gradients/model/conv2/Conv2D_grad/Conv2DBackpropFilter6^gradients/model/conv2/Conv2D_grad/Conv2DBackpropInput
�
:gradients/model/conv2/Conv2D_grad/tuple/control_dependencyIdentity5gradients/model/conv2/Conv2D_grad/Conv2DBackpropInput3^gradients/model/conv2/Conv2D_grad/tuple/group_deps*
T0*H
_class>
<:loc:@gradients/model/conv2/Conv2D_grad/Conv2DBackpropInput*/
_output_shapes
:���������K 
�
<gradients/model/conv2/Conv2D_grad/tuple/control_dependency_1Identity6gradients/model/conv2/Conv2D_grad/Conv2DBackpropFilter3^gradients/model/conv2/Conv2D_grad/tuple/group_deps*
T0*I
_class?
=;loc:@gradients/model/conv2/Conv2D_grad/Conv2DBackpropFilter*&
_output_shapes
: @
�
*gradients/model_2/conv2/Conv2D_grad/ShapeNShapeNmodel_2/conv1/MaxPool2D/MaxPoolconv2/weights/read*
out_type0*
N* 
_output_shapes
::*
T0
�
7gradients/model_2/conv2/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput*gradients/model_2/conv2/Conv2D_grad/ShapeNconv2/weights/read=gradients/model_2/conv2/BiasAdd_grad/tuple/control_dependency*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*/
_output_shapes
:���������K *
	dilations
*
T0
�
8gradients/model_2/conv2/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFiltermodel_2/conv1/MaxPool2D/MaxPool,gradients/model_2/conv2/Conv2D_grad/ShapeN:1=gradients/model_2/conv2/BiasAdd_grad/tuple/control_dependency*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME*&
_output_shapes
: @*
	dilations

�
4gradients/model_2/conv2/Conv2D_grad/tuple/group_depsNoOp9^gradients/model_2/conv2/Conv2D_grad/Conv2DBackpropFilter8^gradients/model_2/conv2/Conv2D_grad/Conv2DBackpropInput
�
<gradients/model_2/conv2/Conv2D_grad/tuple/control_dependencyIdentity7gradients/model_2/conv2/Conv2D_grad/Conv2DBackpropInput5^gradients/model_2/conv2/Conv2D_grad/tuple/group_deps*/
_output_shapes
:���������K *
T0*J
_class@
><loc:@gradients/model_2/conv2/Conv2D_grad/Conv2DBackpropInput
�
>gradients/model_2/conv2/Conv2D_grad/tuple/control_dependency_1Identity8gradients/model_2/conv2/Conv2D_grad/Conv2DBackpropFilter5^gradients/model_2/conv2/Conv2D_grad/tuple/group_deps*K
_classA
?=loc:@gradients/model_2/conv2/Conv2D_grad/Conv2DBackpropFilter*&
_output_shapes
: @*
T0
�
gradients/AddN_7AddN?gradients/model_1/conv2/BiasAdd_grad/tuple/control_dependency_1=gradients/model/conv2/BiasAdd_grad/tuple/control_dependency_1?gradients/model_2/conv2/BiasAdd_grad/tuple/control_dependency_1*
T0*C
_class9
75loc:@gradients/model_1/conv2/BiasAdd_grad/BiasAddGrad*
N*
_output_shapes
:@
�
:gradients/model_1/conv1/MaxPool2D/MaxPool_grad/MaxPoolGradMaxPoolGradmodel_1/conv1/conv1/Relumodel_1/conv1/MaxPool2D/MaxPool<gradients/model_1/conv2/Conv2D_grad/tuple/control_dependency*
ksize
*
paddingSAME*0
_output_shapes
:���������2� *
T0*
data_formatNHWC*
strides

�
8gradients/model/conv1/MaxPool2D/MaxPool_grad/MaxPoolGradMaxPoolGradmodel/conv1/conv1/Relumodel/conv1/MaxPool2D/MaxPool:gradients/model/conv2/Conv2D_grad/tuple/control_dependency*
T0*
data_formatNHWC*
strides
*
ksize
*
paddingSAME*0
_output_shapes
:���������2� 
�
:gradients/model_2/conv1/MaxPool2D/MaxPool_grad/MaxPoolGradMaxPoolGradmodel_2/conv1/conv1/Relumodel_2/conv1/MaxPool2D/MaxPool<gradients/model_2/conv2/Conv2D_grad/tuple/control_dependency*
ksize
*
paddingSAME*0
_output_shapes
:���������2� *
T0*
data_formatNHWC*
strides

�
gradients/AddN_8AddN>gradients/model_1/conv2/Conv2D_grad/tuple/control_dependency_1<gradients/model/conv2/Conv2D_grad/tuple/control_dependency_1>gradients/model_2/conv2/Conv2D_grad/tuple/control_dependency_1*
T0*K
_classA
?=loc:@gradients/model_1/conv2/Conv2D_grad/Conv2DBackpropFilter*
N*&
_output_shapes
: @
�
0gradients/model_1/conv1/conv1/Relu_grad/ReluGradReluGrad:gradients/model_1/conv1/MaxPool2D/MaxPool_grad/MaxPoolGradmodel_1/conv1/conv1/Relu*
T0*0
_output_shapes
:���������2� 
�
.gradients/model/conv1/conv1/Relu_grad/ReluGradReluGrad8gradients/model/conv1/MaxPool2D/MaxPool_grad/MaxPoolGradmodel/conv1/conv1/Relu*
T0*0
_output_shapes
:���������2� 
�
0gradients/model_2/conv1/conv1/Relu_grad/ReluGradReluGrad:gradients/model_2/conv1/MaxPool2D/MaxPool_grad/MaxPoolGradmodel_2/conv1/conv1/Relu*
T0*0
_output_shapes
:���������2� 
�
0gradients/model_1/conv1/BiasAdd_grad/BiasAddGradBiasAddGrad0gradients/model_1/conv1/conv1/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes
: 
�
5gradients/model_1/conv1/BiasAdd_grad/tuple/group_depsNoOp1^gradients/model_1/conv1/BiasAdd_grad/BiasAddGrad1^gradients/model_1/conv1/conv1/Relu_grad/ReluGrad
�
=gradients/model_1/conv1/BiasAdd_grad/tuple/control_dependencyIdentity0gradients/model_1/conv1/conv1/Relu_grad/ReluGrad6^gradients/model_1/conv1/BiasAdd_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/model_1/conv1/conv1/Relu_grad/ReluGrad*0
_output_shapes
:���������2� 
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
53loc:@gradients/model/conv1/conv1/Relu_grad/ReluGrad*0
_output_shapes
:���������2� 
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
=gradients/model_2/conv1/BiasAdd_grad/tuple/control_dependencyIdentity0gradients/model_2/conv1/conv1/Relu_grad/ReluGrad6^gradients/model_2/conv1/BiasAdd_grad/tuple/group_deps*C
_class9
75loc:@gradients/model_2/conv1/conv1/Relu_grad/ReluGrad*0
_output_shapes
:���������2� *
T0
�
?gradients/model_2/conv1/BiasAdd_grad/tuple/control_dependency_1Identity0gradients/model_2/conv1/BiasAdd_grad/BiasAddGrad6^gradients/model_2/conv1/BiasAdd_grad/tuple/group_deps*
_output_shapes
: *
T0*C
_class9
75loc:@gradients/model_2/conv1/BiasAdd_grad/BiasAddGrad
�
*gradients/model_1/conv1/Conv2D_grad/ShapeNShapeNpositive_inputconv1/weights/read*
T0*
out_type0*
N* 
_output_shapes
::
�
7gradients/model_1/conv1/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput*gradients/model_1/conv1/Conv2D_grad/ShapeNconv1/weights/read=gradients/model_1/conv1/BiasAdd_grad/tuple/control_dependency*
use_cudnn_on_gpu(*
paddingSAME*0
_output_shapes
:���������2�*
	dilations
*
T0*
data_formatNHWC*
strides

�
8gradients/model_1/conv1/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterpositive_input,gradients/model_1/conv1/Conv2D_grad/ShapeN:1=gradients/model_1/conv1/BiasAdd_grad/tuple/control_dependency*
use_cudnn_on_gpu(*
paddingSAME*&
_output_shapes
: *
	dilations
*
T0*
data_formatNHWC*
strides

�
4gradients/model_1/conv1/Conv2D_grad/tuple/group_depsNoOp9^gradients/model_1/conv1/Conv2D_grad/Conv2DBackpropFilter8^gradients/model_1/conv1/Conv2D_grad/Conv2DBackpropInput
�
<gradients/model_1/conv1/Conv2D_grad/tuple/control_dependencyIdentity7gradients/model_1/conv1/Conv2D_grad/Conv2DBackpropInput5^gradients/model_1/conv1/Conv2D_grad/tuple/group_deps*0
_output_shapes
:���������2�*
T0*J
_class@
><loc:@gradients/model_1/conv1/Conv2D_grad/Conv2DBackpropInput
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
5gradients/model/conv1/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput(gradients/model/conv1/Conv2D_grad/ShapeNconv1/weights/read;gradients/model/conv1/BiasAdd_grad/tuple/control_dependency*
use_cudnn_on_gpu(*
paddingSAME*0
_output_shapes
:���������2�*
	dilations
*
T0*
strides
*
data_formatNHWC
�
6gradients/model/conv1/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilteranchor_input*gradients/model/conv1/Conv2D_grad/ShapeN:1;gradients/model/conv1/BiasAdd_grad/tuple/control_dependency*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*&
_output_shapes
: *
	dilations
*
T0
�
2gradients/model/conv1/Conv2D_grad/tuple/group_depsNoOp7^gradients/model/conv1/Conv2D_grad/Conv2DBackpropFilter6^gradients/model/conv1/Conv2D_grad/Conv2DBackpropInput
�
:gradients/model/conv1/Conv2D_grad/tuple/control_dependencyIdentity5gradients/model/conv1/Conv2D_grad/Conv2DBackpropInput3^gradients/model/conv1/Conv2D_grad/tuple/group_deps*
T0*H
_class>
<:loc:@gradients/model/conv1/Conv2D_grad/Conv2DBackpropInput*0
_output_shapes
:���������2�
�
<gradients/model/conv1/Conv2D_grad/tuple/control_dependency_1Identity6gradients/model/conv1/Conv2D_grad/Conv2DBackpropFilter3^gradients/model/conv1/Conv2D_grad/tuple/group_deps*&
_output_shapes
: *
T0*I
_class?
=;loc:@gradients/model/conv1/Conv2D_grad/Conv2DBackpropFilter
�
*gradients/model_2/conv1/Conv2D_grad/ShapeNShapeNnegative_inputconv1/weights/read*
N* 
_output_shapes
::*
T0*
out_type0
�
7gradients/model_2/conv1/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput*gradients/model_2/conv1/Conv2D_grad/ShapeNconv1/weights/read=gradients/model_2/conv1/BiasAdd_grad/tuple/control_dependency*0
_output_shapes
:���������2�*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME
�
8gradients/model_2/conv1/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilternegative_input,gradients/model_2/conv1/Conv2D_grad/ShapeN:1=gradients/model_2/conv1/BiasAdd_grad/tuple/control_dependency*
paddingSAME*&
_output_shapes
: *
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(
�
4gradients/model_2/conv1/Conv2D_grad/tuple/group_depsNoOp9^gradients/model_2/conv1/Conv2D_grad/Conv2DBackpropFilter8^gradients/model_2/conv1/Conv2D_grad/Conv2DBackpropInput
�
<gradients/model_2/conv1/Conv2D_grad/tuple/control_dependencyIdentity7gradients/model_2/conv1/Conv2D_grad/Conv2DBackpropInput5^gradients/model_2/conv1/Conv2D_grad/tuple/group_deps*
T0*J
_class@
><loc:@gradients/model_2/conv1/Conv2D_grad/Conv2DBackpropInput*0
_output_shapes
:���������2�
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
T0*K
_classA
?=loc:@gradients/model_1/conv1/Conv2D_grad/Conv2DBackpropFilter*
N*&
_output_shapes
: 
�
8conv1/weights/Momentum/Initializer/zeros/shape_as_tensorConst*%
valueB"             * 
_class
loc:@conv1/weights*
dtype0*
_output_shapes
:
�
.conv1/weights/Momentum/Initializer/zeros/ConstConst*
valueB
 *    * 
_class
loc:@conv1/weights*
dtype0*
_output_shapes
: 
�
(conv1/weights/Momentum/Initializer/zerosFill8conv1/weights/Momentum/Initializer/zeros/shape_as_tensor.conv1/weights/Momentum/Initializer/zeros/Const*&
_output_shapes
: *
T0*

index_type0* 
_class
loc:@conv1/weights
�
conv1/weights/Momentum
VariableV2*
dtype0*&
_output_shapes
: *
shared_name * 
_class
loc:@conv1/weights*
	container *
shape: 
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
conv1/biases/Momentum/readIdentityconv1/biases/Momentum*
_output_shapes
: *
T0*
_class
loc:@conv1/biases
�
8conv2/weights/Momentum/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*%
valueB"          @   * 
_class
loc:@conv2/weights
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
(conv2/weights/Momentum/Initializer/zerosFill8conv2/weights/Momentum/Initializer/zeros/shape_as_tensor.conv2/weights/Momentum/Initializer/zeros/Const*&
_output_shapes
: @*
T0*

index_type0* 
_class
loc:@conv2/weights
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
conv2/biases/Momentum/readIdentityconv2/biases/Momentum*
_output_shapes
:@*
T0*
_class
loc:@conv2/biases
�
8conv3/weights/Momentum/Initializer/zeros/shape_as_tensorConst*%
valueB"      @   �   * 
_class
loc:@conv3/weights*
dtype0*
_output_shapes
:
�
.conv3/weights/Momentum/Initializer/zeros/ConstConst*
valueB
 *    * 
_class
loc:@conv3/weights*
dtype0*
_output_shapes
: 
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
conv3/weights/Momentum/AssignAssignconv3/weights/Momentum(conv3/weights/Momentum/Initializer/zeros*
use_locking(*
T0* 
_class
loc:@conv3/weights*
validate_shape(*'
_output_shapes
:@�
�
conv3/weights/Momentum/readIdentityconv3/weights/Momentum*
T0* 
_class
loc:@conv3/weights*'
_output_shapes
:@�
�
'conv3/biases/Momentum/Initializer/zerosConst*
valueB�*    *
_class
loc:@conv3/biases*
dtype0*
_output_shapes	
:�
�
conv3/biases/Momentum
VariableV2*
shape:�*
dtype0*
_output_shapes	
:�*
shared_name *
_class
loc:@conv3/biases*
	container 
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
conv3/biases/Momentum/readIdentityconv3/biases/Momentum*
_output_shapes	
:�*
T0*
_class
loc:@conv3/biases
�
8conv4/weights/Momentum/Initializer/zeros/shape_as_tensorConst*%
valueB"      �      * 
_class
loc:@conv4/weights*
dtype0*
_output_shapes
:
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
(conv4/weights/Momentum/Initializer/zerosFill8conv4/weights/Momentum/Initializer/zeros/shape_as_tensor.conv4/weights/Momentum/Initializer/zeros/Const*

index_type0* 
_class
loc:@conv4/weights*(
_output_shapes
:��*
T0
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
conv4/weights/Momentum/AssignAssignconv4/weights/Momentum(conv4/weights/Momentum/Initializer/zeros*
use_locking(*
T0* 
_class
loc:@conv4/weights*
validate_shape(*(
_output_shapes
:��
�
conv4/weights/Momentum/readIdentityconv4/weights/Momentum*
T0* 
_class
loc:@conv4/weights*(
_output_shapes
:��
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
_class
loc:@conv4/biases*
	container *
shape:�*
dtype0*
_output_shapes	
:�*
shared_name 
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
conv4/biases/Momentum/readIdentityconv4/biases/Momentum*
T0*
_class
loc:@conv4/biases*
_output_shapes	
:�
�
8conv5/weights/Momentum/Initializer/zeros/shape_as_tensorConst*%
valueB"            * 
_class
loc:@conv5/weights*
dtype0*
_output_shapes
:
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
(conv5/weights/Momentum/Initializer/zerosFill8conv5/weights/Momentum/Initializer/zeros/shape_as_tensor.conv5/weights/Momentum/Initializer/zeros/Const*
T0*

index_type0* 
_class
loc:@conv5/weights*'
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
conv5/weights/Momentum/AssignAssignconv5/weights/Momentum(conv5/weights/Momentum/Initializer/zeros*
use_locking(*
T0* 
_class
loc:@conv5/weights*
validate_shape(*'
_output_shapes
:�
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
+Momentum/update_conv1/weights/ApplyMomentumApplyMomentumconv1/weightsconv1/weights/MomentumMomentum/learning_rategradients/AddN_10Momentum/momentum*
use_locking( *
T0* 
_class
loc:@conv1/weights*
use_nesterov(*&
_output_shapes
: 
�
*Momentum/update_conv1/biases/ApplyMomentumApplyMomentumconv1/biasesconv1/biases/MomentumMomentum/learning_rategradients/AddN_9Momentum/momentum*
use_locking( *
T0*
_class
loc:@conv1/biases*
use_nesterov(*
_output_shapes
: 
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
*Momentum/update_conv2/biases/ApplyMomentumApplyMomentumconv2/biasesconv2/biases/MomentumMomentum/learning_rategradients/AddN_7Momentum/momentum*
T0*
_class
loc:@conv2/biases*
use_nesterov(*
_output_shapes
:@*
use_locking( 
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
*Momentum/update_conv3/biases/ApplyMomentumApplyMomentumconv3/biasesconv3/biases/MomentumMomentum/learning_rategradients/AddN_5Momentum/momentum*
use_locking( *
T0*
_class
loc:@conv3/biases*
use_nesterov(*
_output_shapes	
:�
�
+Momentum/update_conv4/weights/ApplyMomentumApplyMomentumconv4/weightsconv4/weights/MomentumMomentum/learning_rategradients/AddN_4Momentum/momentum*
T0* 
_class
loc:@conv4/weights*
use_nesterov(*(
_output_shapes
:��*
use_locking( 
�
*Momentum/update_conv4/biases/ApplyMomentumApplyMomentumconv4/biasesconv4/biases/MomentumMomentum/learning_rategradients/AddN_3Momentum/momentum*
use_locking( *
T0*
_class
loc:@conv4/biases*
use_nesterov(*
_output_shapes	
:�
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
Momentum/valueConst^Momentum/update*
_output_shapes
: *
_class
loc:@Variable*
value	B :*
dtype0
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
shape: *
dtype0*
_output_shapes
: 
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
save/Const^save/SaveV2*
T0*
_class
loc:@save/Const*
_output_shapes
: 
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
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices"/device:CPU:0*#
dtypes
2*h
_output_shapesV
T:::::::::::::::::::::
�
save/AssignAssignVariablesave/RestoreV2*
use_locking(*
T0*
_class
loc:@Variable*
validate_shape(*
_output_shapes
: 
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
save/Assign_2Assignconv1/biases/Momentumsave/RestoreV2:2*
use_locking(*
T0*
_class
loc:@conv1/biases*
validate_shape(*
_output_shapes
: 
�
save/Assign_3Assignconv1/weightssave/RestoreV2:3*
validate_shape(*&
_output_shapes
: *
use_locking(*
T0* 
_class
loc:@conv1/weights
�
save/Assign_4Assignconv1/weights/Momentumsave/RestoreV2:4*&
_output_shapes
: *
use_locking(*
T0* 
_class
loc:@conv1/weights*
validate_shape(
�
save/Assign_5Assignconv2/biasessave/RestoreV2:5*
T0*
_class
loc:@conv2/biases*
validate_shape(*
_output_shapes
:@*
use_locking(
�
save/Assign_6Assignconv2/biases/Momentumsave/RestoreV2:6*
use_locking(*
T0*
_class
loc:@conv2/biases*
validate_shape(*
_output_shapes
:@
�
save/Assign_7Assignconv2/weightssave/RestoreV2:7* 
_class
loc:@conv2/weights*
validate_shape(*&
_output_shapes
: @*
use_locking(*
T0
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
save/Assign_9Assignconv3/biasessave/RestoreV2:9*
_class
loc:@conv3/biases*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0
�
save/Assign_10Assignconv3/biases/Momentumsave/RestoreV2:10*
T0*
_class
loc:@conv3/biases*
validate_shape(*
_output_shapes	
:�*
use_locking(
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
save/Assign_12Assignconv3/weights/Momentumsave/RestoreV2:12* 
_class
loc:@conv3/weights*
validate_shape(*'
_output_shapes
:@�*
use_locking(*
T0
�
save/Assign_13Assignconv4/biasessave/RestoreV2:13*
use_locking(*
T0*
_class
loc:@conv4/biases*
validate_shape(*
_output_shapes	
:�
�
save/Assign_14Assignconv4/biases/Momentumsave/RestoreV2:14*
_class
loc:@conv4/biases*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0
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
save/Assign_16Assignconv4/weights/Momentumsave/RestoreV2:16*(
_output_shapes
:��*
use_locking(*
T0* 
_class
loc:@conv4/weights*
validate_shape(
�
save/Assign_17Assignconv5/biasessave/RestoreV2:17*
use_locking(*
T0*
_class
loc:@conv5/biases*
validate_shape(*
_output_shapes
:
�
save/Assign_18Assignconv5/biases/Momentumsave/RestoreV2:18*
_output_shapes
:*
use_locking(*
T0*
_class
loc:@conv5/biases*
validate_shape(
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
save/Assign_20Assignconv5/weights/Momentumsave/RestoreV2:20*'
_output_shapes
:�*
use_locking(*
T0* 
_class
loc:@conv5/weights*
validate_shape(
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
conv4/weights_1HistogramSummaryconv4/weights_1/tagconv4/weights/read*
_output_shapes
: *
T0
a
conv4/biases_1/tagConst*
_output_shapes
: *
valueB Bconv4/biases_1*
dtype0
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
dtype0*
_output_shapes
: *
valueB Bconv5/biases_1
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
conv5/biases/Momentum:0conv5/biases/Momentum/Assignconv5/biases/Momentum/read:02)conv5/biases/Momentum/Initializer/zeros:0\;��=9      �;�	AFcyX��A*�r

step    

loss��>
�
conv1/weights_1*�	   �>��   ��H�?     `�@!   H���)R�,��@2�����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !��vV�R9��T7���f�ʜ�7
������6�]����ߊ4F��>})�l a�>��d�r?�5�i}1?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?�������:�             �G@      j@     �i@     �d@      d@     �b@      _@     �`@     @Y@     �Y@     �Q@      U@     @R@     �O@     �M@      N@      F@      C@     �G@      @@     �C@      6@      @@      ?@      4@      8@      =@      "@      2@      .@      .@      ,@      $@      $@      "@      @      @      @      $@      @      @      @      @      @       @      @      @              @       @      @      @       @       @      �?              �?      �?      �?      �?              �?      �?      �?              �?              �?      �?              �?              �?              �?       @      �?      �?               @              @       @      �?      �?      @      @              @              @      @       @      �?      @      @       @      $@      @      $@      @      *@      @      $@      $@      @      @      0@      5@      5@      ;@      4@      <@      ;@      =@      ?@      G@      E@      L@     �N@      K@     @R@     @Q@     @R@     �V@     @W@     �X@     @Z@      _@     �]@     �`@     �d@     �e@     �d@     �h@      O@        
O
conv1/biases_1*=      @@2        �-���q=�������:              @@        
�
conv2/weights_1*�	   �����   �L��?      �@!  Ȑt7@)lC�d�UE@2����g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ���(��澢f�����uE���⾮��%ᾙѩ�-߾�iD*L�پ�_�T�l׾X$�z��
�}����[#=�؏�������~�;�"�q�>['�?��>�_�T�l�>�iD*L��>a�Ϭ(�>8K�ߝ�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?�������:�             �@     �@     �@     ,�@     ��@     ��@     ,�@     \�@     ��@     ��@     `�@     ��@     ��@     ��@     @�@     `�@     @     p}@     �{@     �y@     0w@     �u@     `r@     pr@      n@     �l@      g@     �f@     �c@     �b@     �c@     �`@     @]@     @\@     @Y@     �X@      U@      O@      Q@      K@     �D@     �C@      H@      C@     �C@      <@      =@      :@      7@      4@      ;@      3@      7@      (@      ,@      (@      $@      (@       @      @      @      "@       @      @      "@      @      @      @      @      @       @      @      �?               @      @      @      �?      @       @      �?       @              �?      �?       @      �?              �?      �?              �?               @              �?              �?              �?              �?              �?              �?      �?      �?       @       @      @      @      @      @      �?       @       @      @      @       @      "@      @      @       @      @       @      .@      ,@      ,@      (@      @      .@      &@      5@      3@      .@      6@      3@      :@      4@      A@     �B@     �C@      D@      K@     �J@     �J@     �L@     @S@      K@     �Q@     �T@     �W@     �\@      ^@     @^@     �_@      b@     �g@     �h@     @g@     `m@     Pp@     �p@     �r@     u@     Pw@     0y@      |@     ��@     �@     ��@     �@     P�@     ��@     ��@     ��@     ��@     T�@     d�@     ��@     ��@     ��@     ��@     H�@     ¡@     T�@        
O
conv2/biases_1*=      P@2        �-���q=�������:              P@        
�
conv3/weights_1*�	    �*��   �a+�?      �@!  ���1�)63]~\U@2�����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%�E��a�Wܾ�iD*L�پ['�?�;;�"�qʾ����ž�XQ�þ5�"�g���0�6�/n��豪}0ڰ������;9��R���5�L�����m!#���
�%W���MZ��K�>��|�~�>��~���>�XQ��>�����>
�/eq
�>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?�������:�             `�@     ��@     ��@     @�@     �@     ҡ@     t�@      �@     �@     <�@     ��@     <�@     �@     (�@     ��@     @�@     ��@     ��@     ؃@     �@     ��@     ~@     0{@     pz@     x@     �u@     pq@     @q@     �p@      m@     @k@      h@      d@     �b@     @a@      b@     �Z@      [@     @W@     �X@     @U@     @T@      Q@      P@     �H@     �F@      H@      A@      @@     �@@      A@     �C@      :@      :@      =@      8@      .@      2@      3@      3@      *@       @      $@      "@      @      &@      @      "@      @      @      $@      �?       @      @       @      @      �?      @       @       @       @      �?      �?              �?      @       @       @               @               @              �?               @               @              �?              �?              �?              �?              �?              �?              �?              �?              �?      �?      �?              �?              �?              �?      �?      �?               @      @      �?      @      �?      @              @      �?      @      @       @       @      �?      �?      @      @      �?      @      $@      @      @       @      "@       @       @      3@      .@      8@      &@      ,@      0@      =@      8@      ;@      9@     �C@      D@     �F@      D@     �L@      K@      N@     @R@     �L@     �S@      T@     �X@     @Y@     �Z@     �^@      b@      d@     �f@     �d@     �l@      l@     �n@     pq@      t@     `w@     �t@     `x@     0}@     �{@     P�@     `�@     �@     �@     ��@     Љ@     ��@     Ȑ@     ܑ@     Г@     �@     �@     x�@     �@     �@     .�@     ��@     ��@     Ƨ@     ��@     �@        
O
conv3/biases_1*=      `@2        �-���q=�������:              `@        
�
conv4/weights_1*�	   ����   ����?      �@!  ��QM1�)	��A9e@2�Ӗ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
���Zr[v��I��P=���ߊ4F��h���`f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾw`f���n>ہkVl�p>�FF�G ?��[�?6�]��?����?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�������:�             `b@     ��@     ��@     ؒ@     X�@     $�@     ��@     ��@     8�@     �@     P�@      �@     p�@     �|@     |@     �y@     `v@     �s@     Pq@     q@     �m@     �h@     `e@      h@     @d@     @e@     `b@     @^@     @V@     �Z@     @Y@      T@      Q@     �V@     �G@     �H@      J@      J@      >@      A@      D@     �@@      5@      ;@      1@      1@      &@      &@      .@      (@      "@      $@      &@      @      1@      @      "@      "@      &@      @      @      @       @      �?      @              �?       @      @      @       @      @       @              �?      @       @      �?               @      �?      �?      �?              �?              �?              �?              �?      �?              �?              �?              �?              �?               @       @              �?       @       @      @       @       @               @      @       @      @      @      @      @      @      @      @      @      @       @      @      @      (@      5@      "@      3@      (@      0@      .@      <@      8@      9@     �A@     �A@     �D@     �B@      I@     �H@      R@      P@      J@     �Q@     �Q@      [@      S@      W@     �_@      ^@     @c@     �c@     �d@     �e@     �i@      j@     �o@     `q@     `r@     �u@     �v@     �x@     �|@     �{@     ��@     0�@     x�@     �@     ��@      �@     ��@     H�@     ��@     D�@     ��@     ��@     �`@        
O
conv4/biases_1*=      p@2        �-���q=�������:              p@        
�
conv5/weights_1*�	   ��¿   `���?      �@!  �[�1@)��qg�I@2��?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���VlQ.��7Kaa+��S�F !�ji6�9����d�r�x?�x���XQ��>�����>�S�F !?�[^:��"?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%>��:?d�\D�X=?���#@?�!�A?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?�������:�             `n@      t@     `p@      l@     �k@     �k@     �g@     �g@     �c@      ]@     �`@     @_@     �[@     @U@     �T@     �Q@     @R@     @Q@      O@     �D@     �N@     �I@     �G@      H@      ?@      <@      :@      9@      1@      3@      <@      4@      (@      .@      "@      &@      &@      @       @      $@      "@      @      �?      @      @       @       @      @      @      @      @      @       @      @       @              �?      �?      �?      �?               @      �?      @      �?              �?              �?              �?              �?              �?              �?      �?              �?              �?              �?               @       @      @       @      �?      �?       @       @      @      @      @      @      $@      $@       @      @      "@      "@      .@      "@      3@      2@      1@      ,@      3@      3@      @@      8@     �C@      @@      2@     �B@     �E@      E@      G@      H@      N@      Q@     �R@      R@      X@     �V@     �[@     �\@     �`@      b@     `c@     `f@     �e@     `m@     �l@     0p@     Pq@      t@      n@        
O
conv5/biases_1*=      <@2        �-���q=�������:              <@        @l�dHU      ��u	b�zX��A*��

step  �?

loss��>
�
conv1/weights_1*�	   @y��    Hw�?     `�@! ��/{C�)ō�F9�@2�����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&���ڋ��vV�R9��T7����5�i}1���d�r�x?�x��I��P=�>��Zr[v�>f�ʜ�7
?>h�'�?�.�?ji6�9�?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?�������:�              F@     �j@     `i@      e@     �c@     `c@     �_@      `@     �Y@      Z@     �Q@     �U@      Q@     �N@     �P@     �L@      F@     �B@     �H@      >@      E@      :@      :@      >@      ;@      3@      7@      3@      0@      *@      1@      $@      *@      @      (@      @      &@      @      "@      @      @      @      @      �?      @      @      @      �?      @      @       @       @      �?       @      �?      �?      �?       @              �?              �?              �?              �?              �?              �?              �?              �?               @      �?              �?      @              @              �?      @       @      @               @       @       @      @      @      @      @      @      @      @       @      &@      @      @      @      @      &@      *@      @      3@      3@      5@      7@      :@      ?@      8@      =@      <@      F@      G@     �L@      K@      K@     �S@     �Q@     �P@      W@     @X@     �X@      [@     �]@      ]@      `@     �e@     �e@     �d@     �h@      O@        
�
conv1/biases_1*�	    V�4�   @��N?      @@!  �F�ii?)�mlC�>2���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�U�4@@�$��[^:��"��S�F !���ڋ��vV�R9��5�i}1���d�r�f�ʜ�7
��������[���FF�G �O�ʗ�����Zr[v��pz�w�7��})�l a��ߊ4F��8K�ߝ�a�Ϭ(�jqs&\��>��~]�[�>���%�>�uE����>�f����>��(���>6�]��?����?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?��%>��:?d�\D�X=?���#@?�!�A?IcD���L?k�1^�sO?�������:�              �?      �?              �?              �?              �?      �?              �?              �?              �?              �?              �?              �?      �?              �?              �?              �?      �?      �?              �?               @              �?      �?      �?              �?      �?      �?              �?      �?      �?               @        
�
conv2/weights_1*�	   `6���   ���?      �@! 9�m�b7@)����#VE@2����g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a�h���`�8K�ߝ�a�Ϭ(���(��澢f���侮��%ᾙѩ�-߾E��a�Wܾjqs&\�ѾK+�E��Ͼ['�?�;�[�=�k���*��ڽ�K+�E���>jqs&\��>��~]�[�>�_�T�l�>�iD*L��>�ѩ�-�>���%�>�f����>��(���>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?�������:�             ��@     �@     ��@     �@     ��@     p�@     H�@     8�@     ȑ@     ��@     ��@     ؉@     ��@     ��@     x�@     8�@     �@     �|@     �{@     �y@      w@     �u@     �r@     �q@     �o@      l@     �g@      g@      c@     �c@     �b@     �a@     @\@     �[@     �Z@     �V@     �V@      O@     �P@     �K@      E@      C@     �H@      D@      ?@      ?@      >@      <@      2@      5@      :@      7@      7@      .@      *@      $@      "@      &@       @      $@      @      &@       @      @      @       @      @              �?      @      @       @      �?      �?      @      @      @       @      �?       @              @              �?              �?               @      @              @      �?              �?      �?              �?              �?      �?              �?              �?              �?              �?      �?      �?              �?              @       @      �?      @      @      @       @      �?      �?      @      @      @      @      @      @      @      @      @      &@      *@      ,@      (@      "@      (@      0@      *@      0@      0@      .@      9@      8@      3@      9@      @@     �C@      D@     �F@      G@      L@      J@     �L@     @S@     �J@      R@     @T@     @V@      ^@     @_@     �[@     �_@     `b@      g@     �h@     �g@     @m@      p@     Pp@      s@     �t@     @w@     @y@      |@     ��@     �@     ��@     �@     P�@     ��@     ��@     ��@     ��@     T�@     T�@     �@     ��@     ��@     �@     0�@     ��@     x�@        
�	
conv2/biases_1*�	   ���:�   `<]K?      P@!  @���*?)�{��/�>2�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲��FF�G �>�?�s����f�����uE���⾞[�=�k���*��ڽ��*��ڽ>�[�=�k�>���%�>�uE����>})�l a�>pz�w�7�>I��P=�>��Zr[v�>>�?�s��>�FF�G ?��[�?1��a˲?����?f�ʜ�7
?>h�'�?�T7��?�vV�R9?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?�T���C?a�$��{E?�qU���I?IcD���L?�������:�              �?      �?      �?              @      �?      @               @               @      �?       @               @              �?      �?       @              �?      @      �?              @              �?              �?              �?              �?              �?              �?      �?      �?              �?               @              �?      �?               @               @      @      �?               @              �?              �?              @      �?      �?              �?              �?        
�
conv3/weights_1*�	   `7,��   �O>�?      �@!�v'1�)����C\U@2�����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%��_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;�XQ�þ��~��¾�[�=�k���*��ڽ�;9��R���5�L����|�~���MZ��K��X$�z��
�}�����*��ڽ>�[�=�k�>��~���>�XQ��>;�"�q�>['�?��>K+�E���>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>��(���>a�Ϭ(�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?�������:�             X�@     ��@     ��@     <�@     Σ@     �@     X�@     �@     ,�@     L�@     ܖ@     P�@     �@     $�@     ��@     X�@     `�@     ��@     ��@     �@     ��@     �}@     `{@     pz@      x@     �u@     �q@     pq@     �p@     �l@     `k@     `h@     �d@     �a@     @b@     �`@     @[@     �[@     �X@     @V@     �T@     �U@     �Q@     �M@     �H@      H@      E@     �C@      >@     �B@      ?@     �B@      ;@      :@      <@      7@      4@      *@      4@      ,@      .@      1@       @       @       @      *@      @      @       @       @      @              @      �?              @               @       @       @              @              @      �?      �?      @      �?       @      �?               @              @              �?      �?              �?      �?              �?              �?              �?              �?              �?              �?              �?              �?      �?              �?      �?      �?              �?              �?              �?      �?      �?      �?       @       @      �?      �?      �?       @       @      @      @      @      @      �?      @      @      @      @      @      "@      "@      @      "@      &@      &@      @      3@      0@      1@      0@      &@      6@      9@      >@      5@      =@     �@@      G@     �B@     �B@     �O@     �K@     �L@      Q@     �M@      T@     @V@      V@     �Y@     @\@     @_@     �a@     �c@     �f@     @e@      l@     @k@     �n@     �q@     �s@     �w@     @t@     �x@     �|@     �{@     `�@     p�@     ȃ@     �@     ��@     ��@     �@     А@     ȑ@     ��@     0�@     ԗ@     l�@     0�@     �@     D�@     �@     ��@     ħ@     ©@     H�@        
�
conv3/biases_1*�	   @��9�   �=�R?      `@!  ̂~m?)����F�>2���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1�>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%�        �-���q=�u`P+d�>0�6�/n�>
�/eq
�>;�"�q�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?��%�V6?uܬ�@8?��%>��:?d�\D�X=?�!�A?�T���C?nK���LQ?�lDZrS?�������:�              �?              �?       @       @              �?       @      @      �?      @      @       @              @      �?              �?      �?              @      �?      �?               @      �?      �?      �?       @              �?      @      �?      �?      �?              �?              @               @              �?              �?      �?      �?       @               @      �?              �?              �?              @      �?       @              @      @      �?       @      �?      @       @      @       @              @      �?      @      �?       @      @       @      @      @      �?       @              @              �?              �?              �?              �?        
�
conv4/weights_1*�	    ����   �&�?      �@! w�121�)b�\9e@2�Ӗ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �8K�ߝ�a�Ϭ(龢f�����uE���⾄iD*L�پ�_�T�l׾��~]�[Ӿjqs&\�Ѿa�Ϭ(�>8K�ߝ�>pz�w�7�>I��P=�>�FF�G ?��[�?6�]��?����?>h�'�?x?�x�?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�������:�             @b@     ��@     ��@     ؒ@     P�@     $�@     ��@     ��@     (�@     (�@     H�@     (�@     `�@     �|@     |@     py@     `v@     �s@     `q@     �p@      n@     �h@     �e@     �g@     @d@     @e@      b@      _@      V@      Z@     �Y@     @T@     @Q@     �U@     �H@      H@      K@     �H@     �@@      A@      C@     �@@      7@      9@      2@      1@      $@      (@      &@      2@      @      (@       @      "@      ,@       @      @      (@      "@       @      @       @       @      �?      @      �?       @               @      @       @       @      @      �?       @       @      �?      �?      �?       @      �?              �?      �?              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?       @      �?      �?       @      @      @      �?      �?      �?      �?      @       @       @      @      @      @      @       @      @      @      @      @      @      @      &@      4@      $@      1@      *@      0@      1@      9@      :@      ;@     �@@      A@     �C@     �C@      I@     �I@      Q@     �P@      J@     �Q@     �Q@     �Z@     @S@      W@     �_@      ^@      c@     �c@     �d@     �e@      j@     �i@     �o@     `q@     `r@     �u@     Pv@     y@     p|@     �{@     ��@     (�@     x�@      �@     �@      �@     ��@     P�@     ��@     @�@     ��@     x�@     �`@        
�
conv4/biases_1*�	   @��A�    R!P?      p@!�+���G�)C8��;�>2��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a�a�Ϭ(���(��澢f���侮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;;�"�qʾ
�/eq
Ⱦ����ž�XQ�þ�[�=�k���*��ڽ�5�"�g���0�6�/n���u`P+d����������?�ګ��u��gr��R%��������ӤP�����z!�?��%�����i
�k�Łt�=	���R����y�+pm��mm7&c��`���nx6�X� ��f׽r����tO����f;H�\Q������%���9�e�����-��J�'j��p���1���i@4[���Qu�R"����X>ؽ��
"
ֽ�|86	Խ(�+y�6ҽ;3���н��.4Nν�!p/�^˽�Į#�������/��        �-���q=����/�=�Į#��=K?�\���=�b1��=�!p/�^�=��.4N�=;3����=(�+y�6�=���X>�=H�����=PæҭU�=�Qu�R"�=i@4[��=z�����=ݟ��uy�=�/�4��==��]���=��1���='j��p�=��-��J�=�K���=�9�e��=����%�=f;H�\Q�=�tO���=�f׽r��=nx6�X� >�`��>y�+pm>RT��+�>���">Z�TA[�>�#���j>�4[_>��>
�}���>�MZ��K�>��|�~�>���]���>�5�L�>;9��R�>���?�ګ>����>�u`P+d�>0�6�/n�>�[�=�k�>��~���>;�"�q�>['�?��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��bȬ�0?��82?�u�w74?�!�A?�T���C?k�1^�sO?nK���LQ?�������:�              �?              �?              �?               @              �?      �?              �?              @      @      @      @      �?      @      �?      @      @              �?      �?       @      @      �?      �?      @       @      @       @      �?      �?      �?               @      �?              @      �?       @               @      �?              �?      �?              �?              �?              �?      �?              �?              �?              �?              �?              �?              �?       @      �?       @               @              �?              �?       @              �?              �?      �?              �?       @      �?              �?             �D@              �?               @              �?              �?              �?              �?               @              �?      @      �?      �?      �?      �?      @       @               @      @       @       @              �?              �?      �?              �?              �?              �?      �?              @              �?              �?              �?               @      �?              �?       @      �?      �?      �?      �?      �?      �?      @      �?      �?      �?       @      �?       @      @      �?      �?       @       @       @      �?      @              @       @      �?      �?       @       @      �?       @       @      @      �?       @      �?              �?      �?              �?              �?        
�
conv5/weights_1*�	   ��¿   �m��?      �@! �}Z~@)�gAշI@2��?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���VlQ.��7Kaa+�ji6�9���.��})�l a��ߊ4F��XQ��>�����>I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?d�\D�X=?���#@?�!�A?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?�������:�             @n@     t@     `p@      l@     �k@     �k@     �g@     �g@     �c@     @]@     �`@     �_@     �[@     �U@     @T@     �Q@     @R@     @Q@      O@     �D@     �N@      I@      H@      G@      @@      =@      :@      :@      0@      4@      ;@      2@      ,@      .@      "@      $@      (@      @      "@      "@      "@      @      �?      @      @      @      @      @      @      @       @      @      �?      @      �?              �?      �?      �?      �?               @               @      @              �?              �?              �?              �?              �?              �?               @              �?      �?               @              �?      @       @      �?      �?       @      @       @       @       @      @      $@      $@       @      @      @      &@      .@      "@      3@      2@      1@      ,@      3@      2@      >@      <@      C@     �@@      1@     �B@     �E@      E@     �F@      H@     �N@     @Q@      R@     �R@     �W@      V@     @\@     @\@     �`@      b@     `c@     @f@     �e@     `m@     �l@     @p@     @q@      t@     @n@        
�
conv5/biases_1*�	   ����    ��>      <@!  �j�.>)�%��eX<2��#���j�Z�TA[���tO����f;H�\Q���9�e����K���'j��p���1���=��]����Qu�R"�PæҭUݽ��
"
ֽ�|86	Խ�b1�ĽK?�\��½        �-���q=�|86	�=��
"
�=���X>�=H�����=�Qu�R"�=i@4[��=z�����=ݟ��uy�=��1���='j��p�=��-��J�=�K���=�9�e��=����%�=f;H�\Q�=���">Z�TA[�>�#���j>�J>2!K�R�>��R���>�������:�              �?              �?              �?               @      �?               @              �?              �?              �?              �?              �?              �?      �?      �?               @       @      �?       @      �?      �?              �?              �?              �?        `�jʘW      ��-l	r �{X��A*��

step   @

loss���>
�
conv1/weights_1*�	    Ю�   @���?     `�@!  mY�j�)� ��@2�����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���VlQ.��7Kaa+�U�4@@�$��[^:��"�ji6�9���.����ڋ��vV�R9��_�T�l׾��>M|Kվ��[�?1��a˲?6�]��?�.�?ji6�9�?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?�������:�             �I@     �i@     �i@      e@      c@     �c@      `@     @_@     �Z@     �Y@      Q@     @V@     �P@      O@     �P@      M@      D@      E@      F@      A@      C@      =@      8@      ?@      9@      9@      6@      $@      3@      ,@      0@      &@      (@       @      &@      @      *@      @       @      @      @      @      @      �?       @      @      @       @      @       @      @      �?      �?              @       @      �?               @              �?              �?              �?              �?              �?      �?              �?              �?      �?      �?      �?      �?       @       @      �?              �?       @      �?      @      @      �?      �?      @      �?      @       @      @      "@      "@      @      @      &@       @      @      @      @      .@      $@      $@      (@      4@      6@      9@      <@      4@      =@      =@      >@      E@      G@      L@      L@      M@     �S@     �Q@     �P@     �V@     �W@     �X@      Z@      `@     @[@     �`@      e@      f@     `d@      h@     @P@        
�
conv1/biases_1*�	   `�:C�   @I3T?      @@!  ��By?)����>2��T���C��!�A���%>��:�uܬ�@8���%�V6���82���bȬ�0���VlQ.��S�F !�ji6�9���.����ڋ��vV�R9���d�r�x?�x��pz�w�7��})�l a�['�?�;;�"�qʾ0�6�/n���u`P+d��['�?��>K+�E���>I��P=�>��Zr[v�>��d�r?�5�i}1?�T7��?ji6�9�?�S�F !?�[^:��"?�7Kaa+?��VlQ.?��bȬ�0?��82?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?�������:�               @              �?      �?              �?      �?              �?      �?      �?      �?              �?              �?              �?              �?              �?              �?              �?      �?              �?      �?              �?              �?              �?      �?      �?      �?              �?              �?              �?              @        
�
conv2/weights_1*�	   �D©�   `©?      �@! oT�7@)�e��VE@2����g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ뾢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾�[�=�k���*��ڽ��i����v>E'�/��x>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?�������:�             ؐ@     ��@     ��@     ��@     ؙ@     L�@     d�@     D�@     ��@     ��@     ��@     ��@     Ȉ@     ��@     x�@     0�@      �@     �|@     p{@     �y@     Pw@     �u@     �r@     �q@     �n@     @k@      i@      f@      d@     �c@     @b@     ``@     �]@      Z@     �[@     �X@     @V@     �Q@      M@      K@      B@      G@     �G@      F@     �@@      @@      @@      2@      5@      ;@      1@      6@      4@      &@      ,@      *@      .@      "@      &@      @      @      @      "@      @      @      @      @       @      �?      @      @              @       @      �?      @      @      �?      �?      �?      �?       @      �?       @       @       @      �?               @      �?              �?              �?              �?              �?              �?      �?      �?              �?      �?              �?              �?      �?               @       @      @      @      �?      @      �?       @      @      �?       @      @      @      @      @      @      @      @       @      @      @       @      (@       @      (@      *@      .@      0@      *@      ,@      &@      7@      3@      9@      7@      8@      A@     �C@      A@     �G@      J@     �I@     �I@     @P@      Q@      O@      P@     �T@     �T@     @`@      `@     �Y@     �`@     �b@     �f@     �g@     `i@      l@     pp@     �p@     �r@     �t@      w@     �y@     |@     ��@     Ё@     ��@     �@     H�@     ��@     ��@      �@     ��@      �@     ��@     ��@     �@     x�@     ,�@     ,�@     ��@     ��@        
�	
conv2/biases_1*�		   ���K�   `Y R?      P@!  �S1~P?)�%2���>2�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��S�F !�ji6�9���.����ڋ��vV�R9��T7��������6�]�����[���FF�G ��h���`�8K�ߝ뾕XQ�þ��~��¾���]���>�5�L�>�iD*L��>E��a�W�>�FF�G ?��[�?1��a˲?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?�������:�               @      �?      �?       @              �?      �?      �?      �?              �?      @              �?      �?              �?       @              �?      �?      �?              @              �?              �?              �?              �?              �?              �?              �?       @              �?      �?              �?               @       @       @              �?      �?               @      �?      �?              �?               @      �?      �?              �?      �?       @               @              �?               @        
�
conv3/weights_1*�	   �7��   `xY�?      �@! F��&�/�)�Ϟ�\U@2�����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �O�ʗ�����Zr[v��I��P=��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾄiD*L�پ�_�T�l׾��>M|Kվ��~]�[ӾK+�E��Ͼ['�?�;
�/eq
Ⱦ����ž�XQ�þG&�$��5�"�g���豪}0ڰ���������?�ګ�;9��R���MZ��K���u��gr��.��fc��>39W$:��>;9��R�>���?�ګ>�u`P+d�>0�6�/n�>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?�������:�             0�@     ��@     Ƨ@     4�@     ֣@     ֡@     `�@     �@     �@     d�@     Ԗ@     p�@     �@     $�@     �@     8�@     ��@     �@     ��@     X�@     h�@     �}@     `{@     �z@     �w@     �u@     Pq@     pq@     �p@     �m@     @j@     @i@     �c@     �b@      a@     @a@     �[@     @[@     �W@      W@      T@     �S@     �S@     �L@     �H@     �G@      I@      C@      ?@     �B@     �D@      =@      5@      ;@      B@      5@      3@      "@      .@      0@      ,@      0@      @      $@      (@      *@      &@       @      @      @      @       @      @      @              @      �?       @      @      �?              �?      �?              @       @              �?              �?      �?      �?               @              �?              �?              �?       @              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?       @      �?              �?              �?       @      �?              �?       @               @      �?       @       @      @      @      @      @      @      @      �?      @      �?      @      @      @       @      @      @      &@      $@      &@      &@      0@      .@      0@      ,@      1@      6@      9@      7@      >@      7@      =@      C@     �H@      D@      J@     �N@     �J@      R@     �O@      R@     @V@     �X@     �U@     @]@     �_@     �a@     @e@     �e@     �e@     �l@     �j@     �m@     0r@     �s@     �w@     �t@     px@      }@     �{@     ��@     8�@     �@     8�@     x�@     ��@     (�@     Đ@     ��@     ܓ@     0�@     ��@     @�@     X�@     Ԟ@     L�@     ��@     ��@     ��@     ��@     ��@        
�
conv3/biases_1*�	    �wI�   ��%X?      `@!  ��L�?)����>2��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7�����d�r�x?�x��>h�'�������6�]�����[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a�8K�ߝ�a�Ϭ(���(��澢f������~]�[Ӿjqs&\�ѾK+�E��Ͼ�*��ڽ�G&�$����n�����豪}0ڰ�        �-���q=豪}0ڰ>��n����>G&�$�>�*��ڽ>�uE����>�f����>a�Ϭ(�>8K�ߝ�>�ߊ4F��>})�l a�>��Zr[v�>O�ʗ��>>�?�s��>��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�������:�              �?      �?               @      �?               @      @      @              �?      �?               @              @               @      �?      @       @      �?               @              @      �?              �?              �?      �?              �?      �?               @              �?               @              �?      �?              �?              �?              @              �?              �?              �?              �?              �?              �?      �?              �?       @       @      �?              �?      @              �?       @      �?      @      @      �?      @      �?      �?              @      @      @              @      �?      @       @       @      �?      @      @      �?      �?              �?               @              �?        
�
conv4/weights_1*�	   ����   @J�?      �@! ��][1�)��Փ�9e@2�Ӗ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
��FF�G �>�?�s���pz�w�7��})�l a�a�Ϭ(���(��澢f�����uE���⾄iD*L�پ�_�T�l׾�[�=�k�>��~���>�FF�G ?��[�?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�������:�             `b@     ��@     ��@     ܒ@     \�@     �@     ��@     ȉ@     �@     (�@     `�@     �@     `�@     �|@     |@     `y@     �v@     �s@     pq@      q@     �m@     �h@     �e@     �g@     �d@     �d@      b@     �_@     @V@      Y@     @Z@     @T@     �Q@     �U@     �H@     �G@      K@     �I@     �@@      A@      C@      @@      =@      3@      3@      0@      $@      (@      (@      .@      &@      &@      @      $@      *@      @      $@      $@      $@      @      @      $@       @      �?      @      �?       @      �?       @       @      �?      @       @              �?      �?      @      �?              �?               @      �?              �?              �?              �?      �?      �?              �?              �?              �?              �?               @      �?              �?      �?      �?       @      �?      �?      @      �?      @      �?      �?      �?      @      @      @       @      @      @      @      @      @      @      @      @      @      @      &@      4@      &@      1@      *@      3@      (@      <@      ;@      9@     �A@      @@     �B@     �D@     �H@      J@     �P@      Q@     �I@     @R@     �Q@     �Z@     @S@     �V@     @`@     �]@      c@     �c@     �d@     @f@     �i@     �i@     �o@     pq@     @r@     �u@     Pv@      y@     p|@     �{@     ��@      �@     ��@     ��@     �@     �@     ��@     H�@     �@     0�@     �@     t�@     �`@        
�
conv4/biases_1*�	   ���W�   @�c?      p@!`�H�W�)Sı�V��>2���bB�SY�ܗ�SsW�nK���LQ�k�1^�sO�IcD���L��qU���I��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;;�"�qʾ
�/eq
Ⱦ����ž�XQ�þ��~��¾5�"�g���0�6�/n���u`P+d���5�L�����]����E'�/��x��i����v�4��evk'���o�kJ%�4�e|�Z#�%�����i
�k�2!K�R���J��#���j����"�RT��+��y�+pm��mm7&c��`���nx6�X� ��f׽r����tO����f;H�\Q������%���9�e����K���'j��p���1���PæҭUݽH����ڽ;3���н��.4Nν        �-���q=;3����=(�+y�6�=��
"
�=���X>�=H�����=PæҭU�=z�����=ݟ��uy�==��]���=��1���=�K���=�9�e��=����%�=f;H�\Q�=�tO���=�f׽r��=nx6�X� >�`��>�mm7&c>y�+pm>RT��+�>���">Z�TA[�>�#���j>2!K�R�>��R���>Łt�=	>�i
�k>%���>4�e|�Z#>��o�kJ%>=�.^ol>w`f���n>R%�����>�u��gr�>�MZ��K�>��|�~�>;9��R�>���?�ګ>����>豪}0ڰ>�u`P+d�>0�6�/n�>�XQ��>�����>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>�FF�G ?��[�?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?�qU���I?IcD���L?���%��b?5Ucv0ed?�������:�              �?              �?      �?      �?              �?      �?               @      �?              �?      @      @      @      @      @      @      �?              @      �?      @       @              �?       @       @      �?               @       @      �?      �?      �?       @      �?      �?      �?              �?              �?              �?       @      �?      @              �?      �?      �?      �?              �?      @              �?              �?      �?              �?               @      �?              �?              �?              �?      �?              �?              �?      @               @      �?      �?       @              �?              �?       @              �?               @              �?              �?              A@              �?              �?              �?               @              �?               @      @       @               @              �?      �?      @              @              @               @      �?              �?              �?              �?              �?              �?               @              �?              �?              �?               @       @               @      �?              �?              �?              �?       @      �?              �?       @       @      �?       @               @               @       @       @       @      �?      @       @      �?      @       @      �?      @       @      �?      @      �?       @       @       @       @       @      �?               @               @      �?      �?      �?              �?              �?        
�
conv5/weights_1*�	    ��¿    b��?      �@! �u�K�@)+i��|�I@2�	�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���VlQ.��7Kaa+�I�I�)�(�ji6�9���.��
�/eq
�>;�"�q�>��d�r?�5�i}1?+A�F�&?I�I�)�(?��VlQ.?��bȬ�0?��82?�u�w74?d�\D�X=?���#@?�!�A?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?�������:�	              n@     0t@     pp@      l@     �k@     �k@     �g@     �g@     �c@     @]@     �`@     �_@     �[@     �U@     �T@     @Q@     @R@     @Q@      O@     �D@     �N@     �H@     �H@     �G@      @@      <@      9@      ;@      .@      5@      <@      1@      1@      *@      "@      "@      *@      @      "@      @      "@      @      �?      @      @       @      @      @      @      @       @      @      @      @      �?      �?              �?       @      �?              �?              �?       @      �?              �?      �?              �?              �?              �?              �?              �?              �?              �?       @              �?      �?               @       @       @       @      �?      @      �?      @      @       @      @      "@      &@       @      @      @      *@      *@       @      3@      3@      0@      ,@      3@      2@      ?@      <@     �C@      @@      2@      B@     �D@      F@      F@      H@     �N@     �Q@     �Q@     �R@     �W@     @V@     @\@      \@     �`@     `b@      c@     `f@     �e@     @m@      m@      p@     @q@      t@      n@        
�
conv5/biases_1*�	   �lt!�   @G�%>      <@!   Æ�<>)�y|	~<2���-�z�!�%������f��p�Łt�=	��J��#���j�RT��+��y�+pm��mm7&c��`����tO����f;H�\Q����1���=��]���;3���н��.4NνK?�\���=�b1��=�|86	�=��
"
�=�/�4��==��]���=��1���='j��p�=��-��J�=�K���=�9�e��=f;H�\Q�=�tO���=�f׽r��=�`��>�mm7&c>y�+pm>RT��+�>���">Z�TA[�>�#���j>�J>2!K�R�>��R���>Łt�=	>��f��p>�i
�k>��o�kJ%>4��evk'>�������:�              �?              �?              �?              �?               @              �?              �?              �?              �?              �?              �?              �?              �?      �?               @      �?              �?      �?      �?              �?              �?      �?      �?      �?              �?              �?        9�U      ����	��,}X��A*��

step  @@

losst9�>
�
conv1/weights_1*�	   @e��   ���?     `�@! �!e��)�����@2�	����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���bȬ�0���VlQ.��7Kaa+�U�4@@�$��[^:��"��S�F !�ji6�9���.��x?�x��>h�'��f�ʜ�7
������1��a˲���[���FF�G �>�?�s�����[�?1��a˲?����?f�ʜ�7
?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?�������:�	              K@     �i@     �h@     `e@     �c@     �c@     �^@     @`@      [@     �W@      R@     �U@      R@      M@      O@      N@     �E@      E@     �C@     �C@      D@      9@      :@      <@      =@      ;@      1@      0@      0@      *@      *@      &@      (@      "@      0@      "@      @      @       @      @      @      @      @              �?      @      @       @      @              �?      �?      �?       @               @       @              �?      �?      �?       @              �?              �?              �?              �?              �?               @              �?      �?      �?              �?      �?              �?              �?              @              �?               @      �?       @       @      @      �?      @      @      @       @      @      @      @      @      $@      @      "@      @      @      ,@      (@       @      *@      1@      ,@      5@      ?@      9@      2@      A@      :@     �@@      B@      J@      I@      M@      P@     �R@     @P@     @Q@     @V@     �X@      X@     @Z@     �_@      \@      a@     @d@     �f@     @d@     �g@     �P@        
�
conv1/biases_1*�	    ��S�   �E�d?      @@!  h�A��?)�ZƙO�>2�<DKc��T��lDZrS�
����G�a�$��{E��T���C����#@�d�\D�X=���%>��:���82���bȬ�0��7Kaa+�I�I�)�(�+A�F�&��.����ڋ��XQ�þ��~��¾��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?>h�'�?x?�x�?�[^:��"?U�4@@�$?��bȬ�0?��82?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?IcD���L?k�1^�sO?nK���LQ?�lDZrS?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?5Ucv0ed?Tw��Nof?�������:�              �?              �?      �?               @      �?               @              �?      �?              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?       @               @      �?       @              �?               @      �?              �?        
�
conv2/weights_1*�	   �⩿   �&�?      �@! Ɉ�Z�8@)��CA�WE@2����g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`f�����uE����;9��R�>���?�ګ>��n����>�u`P+d�>['�?��>K+�E���>���%�>�uE����>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?�������:�             ��@     �@     Ԟ@     �@     ԙ@      �@     ��@     0�@     ؑ@     D�@      �@     ��@     ��@     ؄@     8�@     P�@     �@     @}@      {@     Py@     Pw@      v@     �r@     pq@      o@     @k@     �h@     `g@      d@     �c@     �a@     �_@     �^@     �\@      X@     @Z@     �V@     @R@     �M@     �K@      ?@     �E@      J@     �A@      G@      =@      @@      7@      6@      8@      4@      "@      5@      0@      *@      (@      $@      (@      @      @      (@       @      @       @      @      @      @       @      @      @       @      @       @       @      �?       @      �?      �?       @      �?      @      �?      �?      @               @              �?              �?              �?              �?              �?               @      �?       @              @       @               @      �?      @              @      @      @      @      �?      "@      "@      @      @      *@      $@      @      &@      $@      $@      @      $@      .@      &@      4@      4@      (@      =@      8@      6@      >@      6@      C@      G@     �D@     �I@      H@     �J@     �N@     @R@     @P@      M@     �U@     �U@     �]@      a@     �Z@      _@      c@     @f@      i@     �g@     �m@     �n@     q@     0r@     �u@     �v@     �y@     �|@     0�@     8�@     ؀@      �@     �@     ��@     P�@     �@     |�@     0�@     ��@     p�@     (�@     ��@     <�@     �@     ��@     ��@        
�
conv2/biases_1*�	    �Y�   @��`?      P@!  �!��Y?)�k���@ ?2��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��!�A����#@���%>��:�uܬ�@8���%�V6��u�w74���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !���ڋ��vV�R9�1��a˲���[���FF�G �a�Ϭ(���(���})�l a�>pz�w�7�>�FF�G ?��[�?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?E��{��^?�l�P�`?�������:�              �?      �?       @      �?              �?      �?       @               @               @              �?       @      �?              �?              �?       @      �?       @      �?              �?              �?      �?              �?              �?               @              �?      �?              �?              @              �?       @       @      �?      �?      �?       @      �?      �?              �?      �?               @      �?               @      @      �?       @              �?        
�
conv3/weights_1*�	   �3J��   ��}�?      �@!�Fֆ?3,�)��/]U@2�����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(����uE���⾮��%�E��a�Wܾ�iD*L�پK+�E��Ͼ['�?�;
�/eq
Ⱦ����ž�XQ�þ��~��¾5�"�g���0�6�/n������>豪}0ڰ>��n����>�u`P+d�>�XQ��>�����>
�/eq
�>K+�E���>jqs&\��>��~]�[�>�_�T�l�>�iD*L��>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?�������:�             ��@     ��@     ��@     F�@     ԣ@     Ρ@     L�@     H�@     ��@     T�@     Ȗ@     ��@     ܐ@     �@     0�@     8�@     ��@     І@     ��@     @�@     ��@     �}@     �z@     {@      x@     �u@     �q@     �q@     `p@     @n@      i@      j@     �c@     @b@      c@      `@     @\@     �X@      X@     @W@     �T@     �Q@     �S@     �O@      G@      H@     �K@     �@@      >@     �D@     �C@      C@      @@      1@      <@      3@      2@      "@      3@      3@      *@      $@      @      (@      @      &@      &@      @      @      @      @       @      @       @      @      @      @      �?      �?       @      @      �?      �?               @      @      �?              �?       @      �?              �?              �?               @              �?               @              �?              �?              �?              �?      �?              �?      �?              @              �?              �?      �?       @      �?              �?      @      �?      @      �?      �?              @       @       @      @      @              @      @      @      @      @      @      @      @      @      *@      (@      @      (@      8@      .@      0@      ,@      ,@      7@      :@      7@      =@      5@      ?@      B@     �D@      B@     �K@      L@      P@     �O@      R@     �T@     �Q@      [@      T@     �[@     �a@      a@     `e@     �e@     �f@      l@     `j@     �m@     �r@     �s@     �w@     pt@     @x@     �|@      |@     x�@     ��@     ��@     ��@     ��@     x�@     P�@     ��@     ��@     ē@     <�@     �@     4�@     D�@     ܞ@     `�@     �@     ��@     ��@     ̩@     �@        
�
conv3/biases_1*�	   @��S�   @[�e?      `@!  �aa�?)$���R�?2�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9���d�r�x?�x��>h�'��f�ʜ�7
������1��a˲���[���FF�G �>�?�s�����Zr[v��I��P=��8K�ߝ�a�Ϭ(龢f�����uE����K���7��[#=�؏��        �-���q=���]���>�5�L�>�_�T�l�>�iD*L��>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>})�l a�>pz�w�7�>��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?��d�r?�5�i}1?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?�������:�              �?      @              �?              �?      @      �?       @       @      �?              �?      �?       @       @              �?      �?       @       @       @              �?      �?      �?              �?              �?      �?       @      �?              �?              �?               @              �?              �?              �?              @              �?              �?              �?      �?       @              �?              �?      @              �?      �?              �?              @      �?      @       @               @      �?      �?      �?      @      @      @      @      @       @      @      @       @       @       @      @       @      �?       @       @      @               @               @              �?              �?              �?        
�
conv4/weights_1*�	   ����   �_�?      �@! �G���0�),NM�9e@2�Ӗ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�>h�'��f�ʜ�7
��FF�G �>�?�s���})�l a��ߊ4F��f�����uE���⾙ѩ�-߾E��a�Wܾ���?�ګ>����>���%�>�uE����>�f����>��(���>�FF�G ?��[�?����?f�ʜ�7
?>h�'�?x?�x�?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�������:�             �b@     ܗ@     ��@     ��@     X�@     $�@     ��@     ��@     �@      �@     p�@     �@     p�@     �|@     �{@     �y@     0v@     �s@     `q@     0q@     �m@     �h@     @e@      h@     @d@     �d@     �b@     �_@     �U@      Y@     �Z@     @S@     �R@     @T@     �I@     �G@     �K@      I@     �@@     �A@      B@      @@      @@      2@      2@      2@       @      .@      $@      (@      &@      *@       @      "@      (@      &@      $@      @      $@      @      @       @       @       @      @              @      �?      @       @      �?      @               @      �?      �?      �?       @              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?               @              �?              �?      @      �?      @      �?       @      @      @      �?       @      @      @      @      @      @      @      @      @      "@      @       @      @      @      "@      4@      $@      0@      0@      6@      (@      ;@      9@      8@     �B@      ?@      C@     �D@      G@     �J@     �P@      P@     �L@     �Q@      R@     �Z@      S@      W@      `@     �]@      c@     �c@     �d@      f@     �i@     @j@      o@     �q@     pr@     pu@     Pv@     py@     P|@     �{@     ��@     (�@     ��@     ��@     (�@     ��@     ��@     h�@     ��@     0�@     �@     \�@     �a@        
�
conv4/biases_1*�	   �*ed�   @�'r?      p@!�90=�`�)f�_�?2�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7�����d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f���侙ѩ�-߾E��a�Wܾ�iD*L�پK+�E��Ͼ['�?�;;�"�qʾ
�/eq
Ⱦ�[�=�k���*��ڽ�G&�$����������?�ګ��MZ��K���u��gr��39W$:���.��fc����z��6��so쩾4�6NK��2�_"s�$1�7'_��+/��'v�V,���o�kJ%�4�e|�Z#��i
�k���f��p�Łt�=	���R����2!K�R���J��#���j�RT��+��y�+pm��mm7&c��`���nx6�X� �f;H�\Q������%���9�e����d7���Ƚ��؜�ƽ�Į#�������/��        �-���q=��.4N�=;3����=��1���='j��p�=�9�e��=����%�=f;H�\Q�=�tO���=�f׽r��=�`��>�mm7&c>y�+pm>RT��+�>���">Z�TA[�>�#���j>�J>2!K�R�>��R���>Łt�=	>��f��p>4�e|�Z#>��o�kJ%>4��evk'>�'v�V,>7'_��+/>�H5�8�t>�i����v>39W$:��>R%�����>��|�~�>���]���>��n����>�u`P+d�>0�6�/n�>�*��ڽ>�[�=�k�>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>���%�>�uE����>�f����>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?<DKc��T?ܗ�SsW?uWy��r?hyO�s?�������:�              �?              �?      �?              �?               @      �?      �?      �?       @      �?      @              @       @              �?      @       @      @      �?      �?      @       @       @       @       @       @              @       @              �?       @       @       @              @       @      �?              �?      �?      �?      �?      �?               @      �?              �?      �?              �?       @               @              �?              �?      �?              �?              �?              �?              �?              �?              �?              �?              �?      �?       @      �?      @      �?              �?      �?      �?       @              �?      �?              �?              �?             �@@              �?              �?               @       @              @              �?      �?      �?      @              �?       @      �?      �?      @       @              �?      �?              �?              �?              �?              �?              �?      @              �?              �?      �?      �?              @              �?       @               @      �?      �?              �?      �?      �?      �?      �?      �?              �?      �?      �?              �?      �?      @      �?      @       @      �?      @      @               @       @       @       @      @       @      �?       @      �?      �?      �?      @      �?              @       @              �?              �?        
�
conv5/weights_1*�	   @y�¿   @���?      �@! �n�x@)��F/��I@2�	�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�ji6�9���.��K+�E���>jqs&\��>I��P=�>��Zr[v�>�[^:��"?U�4@@�$?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?�������:�	             �m@     @t@     �p@     �k@     �k@     �k@     �g@     �g@     �c@      ]@     �`@     �_@     �[@     �U@     �T@     �Q@     �Q@     @Q@      O@     �D@     �N@      I@      H@      H@      ?@      >@      7@      =@      ,@      5@      ;@      2@      1@      (@      $@      "@      (@      "@       @      @       @      @      �?      @       @      @      @      @       @      @       @      @       @      @      �?       @      �?       @      �?              �?              �?              �?       @              �?              �?              �?              �?              �?              �?      �?              �?               @              �?               @      �?      �?       @       @      @       @      �?      �?      @      @      @      @      $@       @      $@      @      @      *@      ,@      "@      1@      4@      0@      ,@      4@      2@      @@      :@      D@      @@      2@     �A@      E@     �E@     �F@      H@      N@     �Q@     �Q@     �R@     �W@      V@     �\@     @\@     �`@     @b@     @c@     `f@     �e@      m@      m@     �o@     0q@     Pt@      n@        
�
conv5/biases_1*�	   ��[4�   ��%>      <@!   ���;>)˼Di��<2��so쩾4�6NK��2�4�e|�Z#���-�z�!�%������f��p�Łt�=	��J��#���j�nx6�X� ��f׽r���f;H�\Q������%���/�4��ݟ��uy�i@4[���Qu�R"�5%����G�L���=��]���=��1���=��-��J�=�K���=�`��>�mm7&c>y�+pm>RT��+�>���">Z�TA[�>2!K�R�>��R���>Łt�=	>��f��p>%���>��-�z�!>4�e|�Z#>��o�kJ%>�������:�              �?              �?      �?              �?              �?              �?              �?              �?              �?              �?              �?              �?              @      �?      �?      �?      �?              @      �?      �?              �?               @        F���V      ��%	5n~X��A*��

step  �@

loss^Π>
�
conv1/weights_1*�	    3��    �&�?     `�@! ���1��);F��s�@2�����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0��7Kaa+�I�I�)�(�+A�F�&�ji6�9���.����ڋ�})�l a��ߊ4F����n�����豪}0ڰ�K+�E���>jqs&\��>�.�?ji6�9�?�S�F !?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?�������:�             �L@     �i@     �h@     �e@     �b@     `c@     �`@      _@     @[@     �V@     �R@     �T@     @R@      N@     �L@      P@     �F@     �D@     �C@     �D@     �B@      :@      <@      7@      @@      @@      (@      2@      (@      ,@       @      $@      0@      *@      "@      $@       @      @      "@      @      @      @      @      @      �?      @      @              �?      �?      �?       @      �?              @      �?              �?              �?      �?              �?      �?              �?              �?              �?              �?      �?              �?               @      @      @      @      �?               @       @      @       @       @              @              @      @      "@      @      @       @      @      @      &@      @      @      @      (@      $@      1@      ,@      @      2@      5@      >@      ;@      ,@     �A@      9@     �C@     �B@      H@      H@     �M@      Q@     �Q@      N@      S@     �U@     �X@     �W@     �Y@     �^@     @^@      a@     @d@     �f@     �c@      g@     @R@        
�
conv1/biases_1*�	   @��Z�    ~k?      @@!  o-ד?)!�.��]?2��m9�H�[���bB�SY�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@��u�w74���82���bȬ�0�I�I�)�(�+A�F�&��S�F !�ji6�9��E��a�Wܾ�iD*L�پ����?f�ʜ�7
?>h�'�?�vV�R9?��ڋ?���#@?�!�A?a�$��{E?
����G?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�������:�              �?              �?      �?      �?              �?              �?              �?       @              �?              �?              �?              �?      �?               @              �?               @              �?      �?      �?              �?       @      �?               @              �?       @              �?        
�
conv2/weights_1*�	   @a��   `J�?      �@! a����9@)�%��YE@2����g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�_�T�l׾��>M|Kվ��~]�[ӾK+�E��Ͼ['�?�;�5�L�>;9��R�>�u`P+d�>0�6�/n�>��>M|K�>�_�T�l�>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?�������:�             ��@     ��@     Ȟ@     <�@     ��@     �@     ��@     $�@     �@     h�@      �@     ��@     ��@     (�@     ��@     �@     �@     `}@     @{@     �x@     �x@     �u@     pr@     `r@     �l@      l@     `g@      i@     �c@      e@     ``@     @_@     �]@     �^@     @W@     �X@     �X@     �R@      N@     �F@      D@     �D@      H@     �E@      G@      >@      9@      :@      ;@      5@      3@      (@      .@      .@      ,@      .@      (@      $@      @      @       @      @      @      @      @      @      @      �?      @       @       @      @       @      @      �?      @              @      @       @       @      @      �?       @      �?               @              �?      �?              �?      �?              �?      �?              �?              �?              �?              �?              �?      �?               @              �?      �?              �?      �?              �?      @       @       @      @       @      @      @      @      @      @      @      @      @      @      @      @      "@      $@      0@      *@      1@      "@      (@      4@      2@      .@      :@      ;@      6@      ?@      =@     �@@     �A@      I@     �I@     �G@      H@      M@      S@     �Q@     �O@      S@     �W@     @^@     �]@     �^@      _@     �b@     �f@     �i@      h@     @m@     @o@     �p@     0r@     �t@     pw@     z@     �|@     �@     @�@     ��@     H�@      �@     Ȉ@     ��@     ��@     ��@     �@     ��@     H�@     H�@     ��@     <�@     �@     ȡ@     Б@        
�	
conv2/biases_1*�		   ���a�   �dcg?      P@!  �w�no?)&�ߵ�?2����%��b��l�P�`�E��{��^���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7���1��a˲���[��jqs&\��>��~]�[�>a�Ϭ(�>8K�ߝ�>�ߊ4F��>})�l a�>�T7��?�vV�R9?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?Tw��Nof?P}���h?�������:�               @       @               @              �?       @               @              �?              �?      �?              �?      �?              �?      �?      �?              �?              �?       @              �?      �?              �?              �?              �?              �?              �?              �?              �?              �?              �?      �?      @      �?               @      �?              �?       @       @      �?      �?       @       @      �?              @       @      �?              �?               @              �?        
�
conv3/weights_1*�	   �VP��    e��?      �@! V�FI'�)R,�^U@2�����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ�[�=�k���*��ڽ�0�6�/n���u`P+d����z!�?��T�L<��R%�����>�u��gr�>��|�~�>���]���>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>�ѩ�-�>���%�>�f����>��(���>a�Ϭ(�>8K�ߝ�>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?�������:�             ��@     ��@     ��@     8�@     ̣@     ҡ@     |�@     �@     8�@     D�@     Ж@     X�@     ��@     �@      �@     @�@     �@     ��@     �@     ��@     h�@     �~@     �z@     @z@     px@      v@     0q@     pq@      p@     @p@      i@     �h@     `d@     �a@      b@     `a@     �\@     �X@     @V@      W@     �S@     �U@     �O@     �K@     �J@     �K@     �I@     �F@     �A@      >@      E@     �A@      ?@      7@      7@      2@      5@      3@      (@      7@      $@      @      "@       @      $@      $@      "@      @      @      @      @      @      @       @      �?      @       @              �?      �?       @              @      �?      @      �?              @              �?      �?       @       @      �?      �?      �?              �?              �?              �?              �?              �?              �?              �?              �?      �?              �?              �?      �?               @       @              �?              �?      �?      �?               @      �?              @       @      �?              @       @      �?       @              @      "@      @      @      @      @      @      $@      @      @      $@      (@      (@      ,@      6@      *@      0@      3@      &@      :@      4@      ;@      :@      @@      >@      <@      <@     �E@      K@      O@     �M@     �Q@      N@     @V@     @S@     �X@     �U@      ^@      `@     �a@     �e@      f@      f@     `l@     �i@      m@     �r@     �s@     �v@     �u@     �w@     }@     P|@     �@     ��@     ��@     h�@     Ј@     P�@     ��@     ��@     P�@     ̓@     l�@     �@     h�@     0�@     ��@     `�@     �@     ��@     ��@     ک@     ��@        
�
conv3/biases_1*�	    �3_�    n?      `@!  80�V�?)�y!���?2��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !��.����ڋ��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[��>�?�s���O�ʗ���        �-���q=��n����>�u`P+d�>��~]�[�>��>M|K�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?P}���h?ߤ�(g%k?�N�W�m?;8�clp?�������:�              �?       @      �?              �?       @              @              �?      @      �?      @      �?       @      �?      �?      �?               @      �?       @      �?      �?              �?      @              �?              �?              �?      �?      �?      �?      �?              �?              �?               @              �?              �?              �?               @      �?              �?              �?               @      @              �?      �?      �?      �?       @      �?      @              @      @              �?       @      @      @       @      �?       @       @       @       @      @       @      @       @       @      @      �?      �?       @       @      @      �?              �?              �?        
�
conv4/weights_1*�	   ����   `�,�?      �@!����l0�)�S�0:e@2�Ӗ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
���Zr[v��I��P=���f�����uE���⾙ѩ�-߾E��a�Wܾ�[�=�k���*��ڽ��ߊ4F��>})�l a�>O�ʗ��>>�?�s��>�FF�G ?��[�?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�������:�             �b@     ԗ@     ��@     ��@     X�@     �@     ��@     ؉@     �@     �@     h�@     �@     p�@     �|@     �{@     @y@     `v@     t@     0q@     �p@     �n@     `h@     @e@     �g@     �d@     �d@     `b@      `@      V@     �X@      [@     �R@     �R@     �S@      L@      G@     �J@     �I@     �@@     �B@      A@      ;@      B@      3@      3@      1@      "@      *@      &@      *@       @      ,@      @       @      ,@      (@      "@      @      (@      @      @      @       @      �?      @      �?      @      �?      �?      �?      @       @      �?       @               @      �?      �?      �?      �?               @      �?              �?              �?               @              �?              �?              �?              �?              �?               @       @              �?       @      �?      �?              @      �?       @      �?               @      @      @      @      @       @      @      @       @      @      &@      @      @      @      @      "@      4@      (@      ,@      .@      6@      &@     �A@      .@      <@     �B@      @@      D@     �D@     �D@     �L@      O@      Q@      K@     �Q@      S@     @Z@     @S@     @W@     �_@     @^@     �b@      d@      d@     `f@      i@     �j@     �n@     �q@     pr@     `u@     pv@     Py@     �|@     �{@     ��@     8�@     ��@     ؅@     (�@     �@     p�@     ��@     �@     D�@     �@     l�@     `a@        
�
conv4/biases_1*�	   ��4p�    �{?      p@!����Md�)��N<!?2�;8�clp��N�W�m�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��S�F !�ji6�9���.����ڋ��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
�1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a�h���`�8K�ߝ���(��澢f�����uE���⾮��%ᾙѩ�-߾�_�T�l׾��>M|Kվjqs&\�ѾK+�E��Ͼ����ž�XQ�þ��~��¾�[�=�k���*��ڽ��u`P+d����n�����;9��R���5�L�����]������|�~��39W$:���.��fc���p��Dp�@�����W_>�p
T~�;�u 5�9��z��6�7'_��+/��'v�V,���o�kJ%�4�e|�Z#���-�z�!�%�����i
�k���f��p�Łt�=	���R����2!K�R���#���j�Z�TA[��RT��+��y�+pm��mm7&c��`���nx6�X� ��f׽r���f;H�\Q������%���8�4L���<QGEԬ�        �-���q=��s�=������=�|86	�=��
"
�=H�����=PæҭU�=�Qu�R"�=i@4[��=�/�4��==��]���=f;H�\Q�=�tO���=�f׽r��=y�+pm>RT��+�>���">Z�TA[�>�#���j>�J>2!K�R�>��R���>Łt�=	>��f��p>�i
�k>%���>��-�z�!>4�e|�Z#>��o�kJ%>4��evk'>���<�)>_"s�$1>6NK��2>�so쩾4>f^��`{>�����~>�u��gr�>�MZ��K�>�5�L�>;9��R�>5�"�g��>G&�$�>['�?��>K+�E���>E��a�W�>�ѩ�-�>���%�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>��[�?1��a˲?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?o��5sz?���T}?�������:�              �?              �?      �?              �?              �?               @      @      �?              �?      @       @      @       @      @       @      �?      �?      �?      �?      �?       @      @      @       @      �?      �?      �?              @      �?      �?               @      �?      �?       @      @              �?      @       @      �?               @       @      @              �?              �?              �?       @              �?              �?              �?      �?      �?      �?              �?              �?      �?      �?              �?              �?      �?              �?              �?              �?      �?              �?       @       @              �?               @              @      �?              �?      �?              �?              �?              @@              �?              �?              �?              �?              �?              �?      �?              �?              �?              �?              @      �?       @       @       @       @      �?      �?      �?      �?              �?      �?              �?              �?              �?               @              �?              �?      �?              �?              �?      �?       @       @      �?      �?              �?               @              @      �?               @              �?              �?       @       @               @      @       @      @      �?              �?      @               @      �?       @       @      �?       @      �?               @       @      @       @              �?              �?      �?              �?        
�
conv5/weights_1*�	   ���¿   ����?      �@! �� N@)�]� �I@2��?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���VlQ.��7Kaa+�I�I�)�(�ji6�9���.��jqs&\��>��~]�[�>�.�?ji6�9�?�7Kaa+?��VlQ.?��82?�u�w74?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?�������:�             �m@     `t@     pp@     �k@     �k@     �k@     �g@     �g@     �c@     �]@     �`@      `@     @[@     �U@      U@     @Q@     �P@     @R@     �O@      D@      N@      J@      G@      I@      =@      ?@      7@      <@      1@      5@      9@      2@      0@      ,@      $@      "@      &@      $@      "@      @      "@       @       @      @      @      @      @      @      @      @      @      @       @      �?      @               @      �?              �?      �?              �?              �?               @      �?              �?              �?              �?              �?               @               @       @              �?      �?              �?       @      �?      @      @      �?       @      @      @      @      @      "@       @      "@      @      @      *@      ,@      $@      1@      4@      0@      *@      6@      0@      ?@      <@      D@      @@      3@     �@@     �E@      F@      F@     �G@      N@     �Q@     �Q@      S@     �W@     �V@      \@      \@     �`@     �b@      c@     �f@     �e@     @m@      m@     �o@     0q@     @t@     `n@        
�
conv5/biases_1*�	   �|�:�   ��t,>      <@!  ��G>)Ƹ�ż]�<2�p
T~�;�u 5�9�7'_��+/��'v�V,����<�)�4��evk'���-�z�!�%������f��p�Łt�=	����"�RT��+��y�+pm�PæҭUݽH����ڽPæҭU�=�Qu�R"�=z�����=ݟ��uy�='j��p�=��-��J�=�mm7&c>y�+pm>RT��+�>���">Z�TA[�>�#���j>�J>��R���>Łt�=	>��f��p>�i
�k>%���>��-�z�!>4�e|�Z#>��o�kJ%>���<�)>�'v�V,>7'_��+/>�������:�              �?              �?              �?              �?              �?              �?      �?              �?              �?              �?              �?               @              �?              �?       @              �?               @      @              �?      �?              �?      �?        f8�U      ��q�	̚�X��A*ɫ

step  �@

loss���>
�
conv1/weights_1*�	   `����    �T�?     `�@!  C�$���)cRj��@2�����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���VlQ.��7Kaa+�I�I�)�(�+A�F�&��[^:��"��S�F !�ji6�9���.��})�l a��ߊ4F�𾂼d�r?�5�i}1?�T7��?�vV�R9?ji6�9�?�S�F !?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?�������:�             �N@      i@     �i@     �d@      c@     �c@      `@     �_@     �Z@     @W@      S@     �S@      R@     �N@      L@     �M@     �G@     �K@      A@     �A@      D@      6@      <@      ;@      B@      7@      0@      ,@      *@      ,@       @      *@      (@      (@      "@      @      ,@      @      @      @      @      "@      @      @      @      @              @       @       @       @       @      �?      @       @      �?      �?              �?              �?              �?              �?              �?              �?              �?              �?              �?       @              @      �?      �?      �?      @      �?       @       @      �?       @      @      @      @               @      @       @      @      @      @      @      @      @      @      @      "@      @      .@      0@      ,@      *@      .@      5@      9@      4@      ;@      :@      ?@     �E@      A@     �F@     �H@     �N@     �P@      R@     �N@     �R@     @V@      X@     �V@     �Z@     �]@     @^@     �a@     @d@      f@     �d@     �f@     �S@        
�
conv1/biases_1*�	   ���W�   ��p?      @@!  �籟?)Dَņ,?2���bB�SY�ܗ�SsW�<DKc��T�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=��u�w74���82���bȬ�0���VlQ.�pz�w�7��})�l a�6�]��?����?x?�x�?��d�r?��bȬ�0?��82?�!�A?�T���C?
����G?�qU���I?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?�������:�              �?      �?               @      �?      �?      �?      �?              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?              @              �?      �?      @      �?              �?       @      �?              �?        
�
conv2/weights_1*�	   �R��    
?�?      �@! T���Q;@)i�?�[E@2����g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�a�Ϭ(���(��澢f���侮��%ᾙѩ�-߾X$�z�>.��fc��>��>M|K�>�_�T�l�>�iD*L��>�uE����>�f����>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?�������:�             l�@     �@     �@     �@     ș@      �@     x�@     H�@     ��@     @�@     ��@     (�@     `�@     ��@     ��@     0�@     �@     �}@     pz@     �x@     Px@      v@     0r@     s@      m@     �i@     �h@     �g@     �d@      d@     @a@     ``@     �[@     �^@      W@      Y@     @W@      R@      O@     �H@      E@     �A@     �I@      H@     �E@      =@      >@      >@      6@      5@      6@      (@      4@      (@      $@      &@      "@      $@      @      @       @      @       @       @      @      @      �?      @      "@      @      @      @       @      @      �?      �?      @      @      �?              �?      @              �?      �?      @              �?      �?              �?              �?              �?      �?              �?              �?       @      �?               @      �?              �?       @      @      @      @      �?      @      �?      �?      @      @      @      @      @      @      @      @      $@      (@      *@       @      0@       @      $@      .@      ,@      2@      1@      ;@      6@      9@      =@      ?@      ?@     �B@      D@      L@     �I@     �D@      N@      O@     @U@     �P@     �R@     �Z@     �\@     �_@     �\@     �^@     �b@     `e@     `i@     �h@     �k@     0p@     �p@     �q@     �u@     �v@     �z@      |@     ��@     ��@     h�@     X�@     8�@     (�@     �@     ��@     ��@     ��@     Д@     \�@     (�@     �@     �@     (�@     ¡@     �@        
�
conv2/biases_1*�	    ��g�   �Uyk?      P@!  �Q�u?)r�?շ?2�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��!�A����#@�d�\D�X=�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��S�F !�ji6�9���5�i}1���d�r����%ᾙѩ�-߾I��P=�>��Zr[v�>O�ʗ��>��ڋ?�.�?ji6�9�?��bȬ�0?��82?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?�������:�              @      �?              �?      �?      �?      �?      �?      �?      �?              �?              �?      �?               @       @              �?       @      �?      �?       @              �?              �?               @              �?      �?              �?      �?               @              �?      �?      @              �?       @      �?      �?      @      �?               @       @       @       @               @      �?              �?      �?      �?              �?        
�
conv3/weights_1*�	   `P~��   �ڮ?      �@! ��=!�)%��9_U@2�����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ���pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[ӾK+�E��Ͼ['�?�;����ž�XQ�þ�*��ڽ�G&�$��;9��R���5�L�����]����[#=�؏�������~��u`P+d�>0�6�/n�>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>�f����>��(���>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?�������:�             h�@     ��@     ��@     L�@     ģ@     ơ@     ��@     ��@     L�@     �@     Ж@     ��@     ��@     �@     ��@     ��@     P�@     @�@      �@     ��@     ��@      ~@     Pz@     @z@     `x@     �u@     �p@     `r@     �o@     �n@     @k@     �h@     @d@     �a@     �a@     @a@     @\@      Y@     �W@     �T@     @V@     �P@     �R@     �J@      H@     �I@     �L@      D@     �@@     �E@     �B@     �D@      :@      <@      7@      4@      5@      1@      *@      .@      1@      @      @       @       @      &@      "@      "@      "@      @      @      @      @      @      @      @      @      @      �?       @      �?      �?               @              �?      �?              �?               @       @              �?              �?              �?              �?              �?              �?      �?              �?              �?              �?              �?               @               @      �?      �?       @              �?              �?              �?               @      �?       @              �?       @       @       @      @      �?      @       @      �?              @       @              @      $@      @      @      @      (@      (@      &@      .@      1@      4@      .@      1@      2@      6@      4@      :@      9@      >@      :@     �A@     �@@      E@      F@     �O@      L@     �S@     @P@     @S@     @S@     �Y@      W@     �`@      ]@      a@     �d@      g@     @f@     �l@     �k@      k@     �r@     @s@      w@     �t@     �x@     p|@     �|@     8�@     ��@     ��@     �@     �@     ��@     ؎@     ��@     h�@     ��@     <�@     ��@     t�@     4�@     ؞@     N�@     �@     ~�@     ��@     ©@     ��@        
�
conv3/biases_1*�	   ��d�    ��r?      `@!  ��V>�?)d���`�(?2�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&��[^:��"��S�F !��.����ڋ��vV�R9��T7����5�i}1�>h�'��f�ʜ�7
�6�]���1��a˲��FF�G �>�?�s���pz�w�7��})�l a�        �-���q=�f����>��(���>>�?�s��>�FF�G ?��[�?6�]��?����?f�ʜ�7
?x?�x�?��d�r?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?uWy��r?hyO�s?�������:�               @      �?      �?       @      �?      �?      �?      �?               @      @      �?       @      �?      @       @               @      �?      �?      @              �?              @      �?      �?      �?               @              �?      �?      �?      �?              �?              �?              �?              �?               @              �?              �?      �?              �?       @              @              �?              �?      �?              @      �?      @       @              @      �?      @       @              @      @      �?      �?      @      @       @      @       @       @      �?      @      �?              �?      @      @      �?      �?               @        
�
conv4/weights_1*�	   ���   ��A�?      �@! ��%��/�)�g}�:e@2�Ӗ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]����FF�G �>�?�s���O�ʗ���})�l a��ߊ4F���_�T�l׾��>M|Kվ['�?��>K+�E���>�_�T�l�>�iD*L��>I��P=�>��Zr[v�>O�ʗ��>�FF�G ?��[�?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�������:�              b@     ؗ@     ��@     �@     `�@     ��@     ��@     ȉ@     �@     (�@     X�@     �@     h�@     �|@     �{@     �y@     0v@     �s@     pq@      q@     �n@     �h@     @e@      g@     �d@      e@     @b@      _@     �W@     �X@     �Z@      S@     @Q@     �T@      K@     �H@      L@     �F@     �@@      D@     �A@      7@     �A@      6@      5@      0@      $@      ,@      &@      &@      "@      &@      $@      @      (@      *@      &@      @      "@      @      @       @               @      @      �?       @      �?      �?       @       @      @      �?      �?      �?      @      �?      @               @      �?              �?               @              �?      �?              �?              �?              �?              �?               @      �?              �?              �?              �?      �?       @      �?      �?      �?      �?      �?       @      @       @      @              @       @      @      @      @      @      @      @      @      $@      @      @       @      @      $@      1@      $@      0@      .@      4@      .@      <@      4@      :@     �B@      ?@      D@     �C@      E@      M@     �O@     �P@      L@      R@     �S@      X@     �T@      W@     �^@      _@      c@     �c@     �c@     @f@     �i@     �j@     �n@     �q@     pr@     pu@     Pv@     Py@     p|@     �{@     ��@     P�@     ��@     �@     �@     ��@     ��@     p�@     ��@     (�@     �@     h�@      b@        
�
conv4/biases_1*�	   �
vu�   �I'�?      p@!�����K�)�K���1?2�&b՞
�u�hyO�s��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'�������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=���ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%�E��a�Wܾ�iD*L�پ����ž�XQ�þ��~��¾�[�=�k���*��ڽ�G&�$��0�6�/n���u`P+d��豪}0ڰ������E'�/��x��i����v���Ő�;F��`�}6D�/�p`B�p��Dp�@�����W_>��z��6��so쩾4�_"s�$1�7'_��+/�4��evk'���o�kJ%�4�e|�Z#���-�z�!�%�����i
�k�Łt�=	���R����2!K�R���J�Z�TA[�����"��mm7&c��`���nx6�X� �����%���9�e�����1���=��]���z�����i@4[��PæҭUݽH����ڽ        �-���q=5%���=�Bb�!�=�|86	�=��
"
�=ݟ��uy�=�/�4��==��]���=��1���=�K���=�9�e��=����%�=f;H�\Q�=y�+pm>RT��+�>Z�TA[�>�#���j>Łt�=	>��f��p>�i
�k>%���>��-�z�!>4�e|�Z#>��o�kJ%>4��evk'>���<�)>�'v�V,>7'_��+/>_"s�$1>�z��6>u 5�9>[#=�؏�>K���7�>��ӤP��>�
�%W�>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>K+�E���>jqs&\��>��>M|K�>�_�T�l�>�uE����>�f����>��(���>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?6�]��?����?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?���J�\�?-Ա�L�?�������:�              �?              �?              �?       @              �?       @      �?      @               @       @      @              @      @      @              �?      �?       @      @       @              �?      @      �?       @      �?               @              �?      �?      �?       @      �?      �?              �?       @      �?              �?              �?      �?      �?      �?      �?      @              �?              �?       @      �?              �?              �?              �?              �?              �?              �?              �?              �?              �?      �?              �?              �?              �?              �?              �?      �?      �?              @               @              �?              �?      �?              �?              �?              �?              �?              >@              �?              �?              �?              �?              �?              �?              �?              �?              �?      @      �?       @       @      �?      �?      @              �?      �?               @              �?              �?              �?              �?              �?      �?              �?              �?              �?      �?              �?       @              �?              �?      @      �?              �?              �?      �?       @      �?              @              �?      �?               @       @      �?       @       @      @      �?       @      @      �?              @      �?      @      �?       @       @      @       @               @       @       @      @      @      �?              �?      �?      �?              �?        
�
conv5/weights_1*�	   �e�¿   @��?      �@! H��a@)�s�2�I@2�	�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=�uܬ�@8���%�V6��u�w74���82���VlQ.��7Kaa+�I�I�)�(��.����ڋ��f�����uE������~]�[�>��>M|K�>�5�i}1?�T7��?I�I�)�(?�7Kaa+?��82?�u�w74?uܬ�@8?��%>��:?���#@?�!�A?�T���C?a�$��{E?
����G?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?�������:�	             �m@      t@     �p@      l@     �k@     �k@     �g@     `g@     �c@     �]@     �`@     �_@     @[@     �V@     �T@     �P@     �P@     @Q@      Q@      D@     �M@     �J@      G@     �H@      >@      >@      7@      <@      0@      9@      6@      4@      ,@      .@      $@      $@      "@      "@      &@      @      "@       @       @      @      @       @      @      @      @      @      @      @      @      @       @      �?      �?      �?      �?              �?               @              �?              �?      �?               @              �?              �?              �?              �?              �?              �?              �?      �?       @      �?               @       @      �?       @      @      �?      @       @      @      �?      "@      "@       @       @      @      @      (@      *@      $@      2@      2@      3@      (@      6@      2@      ;@      =@     �D@      @@      4@     �@@      D@      G@     �F@     �G@      M@     �Q@     �Q@     �S@     �W@     @V@      ]@     �[@     �`@     �b@     @c@     �f@     �e@      m@      m@     �o@     @q@     0t@     �n@        
�
conv5/biases_1*�	    �.A�    A84>      <@!  ��w�Q>)�H����<2�/�p`B�p��Dp�@�6NK��2�_"s�$1���o�kJ%�4�e|�Z#�%�����i
�k�Z�TA[�����"�f;H�\Q������%���b1�ĽK?�\��½H�����=PæҭU�=z�����=ݟ��uy�=����%�=f;H�\Q�=RT��+�>���">Z�TA[�>�#���j>�J>��R���>Łt�=	>��f��p>�i
�k>%���>��-�z�!>4�e|�Z#>��o�kJ%>4��evk'>���<�)>�'v�V,>_"s�$1>6NK��2>�so쩾4>�������:�              �?               @              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?      �?              �?       @      �?               @      �?      �?       @               @              �?      �?        r�z@�T      �?� 	���X��A*�

step  �@

loss��R>
�
conv1/weights_1*�	   ����   ���?     `�@!  �g��)�|��@2�	����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !��.����ڋ��T7����5�i}1���[���FF�G �I��P=��pz�w�7��
�/eq
�>;�"�q�>I��P=�>��Zr[v�>����?f�ʜ�7
?�T7��?�vV�R9?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?�������:�	             @Q@     �i@     �i@     �c@     �c@     `c@     �`@     �^@     @Z@     �V@     �T@      T@     �Q@     �L@     �K@      P@     �G@      L@      ;@      B@      B@      ;@      >@     �@@      :@      4@      0@      ,@      5@      $@      "@      .@      ,@       @      @      $@      $@      @      @      @      @      @      @      @      @      @      @      @      @              �?               @              �?              �?      �?      �?       @               @              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?      �?      �?       @              �?       @      @       @      @      �?      @              �?      @       @      @      @      @      @      @      @       @      @      @      @      "@      "@      0@      .@      3@      1@      *@      <@      7@      4@      @@      =@     �E@     �A@      I@     �I@      I@      Q@     @R@      O@     �S@     �U@     @W@     �V@     �[@     �^@     @\@      b@      d@     �e@     �d@      f@     �T@        
�
conv1/biases_1*�	   �
�X�   ��2t?      @@!  ��?)尖��%?2���bB�SY�ܗ�SsW�IcD���L��qU���I�a�$��{E��T���C��!�A�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0��FF�G �>�?�s�����Zr[v�>O�ʗ��>�5�i}1?�T7��?���#@?�!�A?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?�������:�               @               @              �?      �?              �?              �?              �?      �?              �?              �?              �?              �?              @      �?      �?      �?      �?              @      �?      �?      �?      �?      �?      �?      �?        
�
conv2/weights_1*�	   �Y���   ��n�?      �@! DR�8=@)>���^E@2����g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s�����Zr[v��I��P=��pz�w�7��})�l a�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ��>M|Kվ��~]�[ӾG&�$�>�*��ڽ>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?�������:�             <�@     ԡ@      �@     �@     ؙ@     �@     ��@     @�@     �@     $�@     ��@     p�@     ��@     8�@      �@      �@     �@     �}@     �z@     �w@     Px@     �u@     pr@     pr@     �l@      l@     �g@     �g@     �d@     `c@     �b@     ``@     �\@     @\@     �X@     �W@      V@     @R@     �O@     �F@     �J@      E@     �H@      E@      >@      <@      D@      <@      =@      5@      3@      5@      0@      ,@      ,@      .@      @      ,@      @      "@      @       @      @       @      @       @      @      @      @      @      @       @       @      �?      @               @      �?       @              �?              @               @              �?      �?              �?               @              �?              �?      @              �?      �?              �?              �?      �?              �?      �?      @      �?       @      @      @       @      @      @      @      @      @      @      @       @      @       @       @      *@      @      (@      .@      *@      ,@      (@      *@      5@      <@      8@      4@      7@      >@      =@      E@     �F@     �H@     �H@      L@      G@     �O@     @Q@     �T@      S@      Z@      ^@     �]@      `@     @\@     `c@      e@      j@     �g@     �n@     �m@     Pq@      q@     @v@      v@     �z@     @|@     P�@     ��@     ؀@     p�@     8�@     ��@     x�@     ��@     Đ@     Ȑ@     ��@     ��@     �@     ț@     �@     d�@     ��@     ��@        
�
conv2/biases_1*�	   ���m�   ���p?      P@!  �ĿE�?)F�H:.J(?2��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�uܬ�@8���%�V6��u�w74���82���bȬ�0�I�I�)�(�+A�F�&�U�4@@�$���Zr[v��I��P=��a�Ϭ(�>8K�ߝ�>��d�r?�5�i}1?�[^:��"?U�4@@�$?��VlQ.?��bȬ�0?uܬ�@8?��%>��:?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?�������:�              @      �?               @      �?      �?      �?               @              �?              �?              �?      �?               @      �?      �?               @       @       @      �?              �?      �?              �?              �?              �?              �?              @               @              �?       @      @               @      @              �?       @      �?       @       @      �?               @      �?              �?               @        
�
conv3/weights_1*�	   `����    �?      �@! ��.��)΄S��`U@2�����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[��>�?�s���O�ʗ�����Zr[v��I��P=��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(龢f�����uE����E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ;�"�qʾ
�/eq
Ⱦ��~��¾�[�=�k��5�"�g���0�6�/n���u`P+d����n����������>
�/eq
�>['�?��>K+�E���>��~]�[�>��>M|K�>�_�T�l�>�ѩ�-�>���%�>�uE����>�f����>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?�������:�             ؃@     ��@     ��@     J�@     ��@     ʡ@     ��@     ��@     t�@     ԗ@     �@     l�@     �@     �@     ��@     ��@     ��@     @�@     H�@     Ђ@     Ȁ@     ~@     �z@     z@     0x@     �u@     pp@     �r@     �n@     �o@     �j@     �i@     �c@     �`@     �b@     �`@     @^@     @W@     �V@     �W@     @S@     �Q@     �P@      N@     �I@     �K@     �G@     �B@     �B@      C@      D@     �F@      @@      9@      5@      4@      2@      (@      4@      2@      &@       @      @      @       @       @      @      "@      @      @      @      @      �?      @       @      �?      @      @      �?              �?              @              �?       @              �?               @               @      �?       @              �?              �?              �?              �?              �?               @              �?      �?              �?      �?      �?              �?      @       @       @      �?      @      �?      @      @      @       @      @      �?      �?      @      �?      @      �?      �?      @      �?      @      &@      @      @      "@      $@       @      &@      .@      *@      6@      (@      4@      7@      7@      8@      6@      ;@      <@      <@     �A@     �A@      C@     �F@     �M@     �P@      R@     @P@     �T@     �U@      U@      Z@     @^@     �]@     `b@     �c@     `f@     �f@     �m@     �j@     �j@     �r@      s@     �w@     `t@     �x@     �{@     �|@     ��@     p�@     �@     h�@     �@     ��@     @�@     �@     4�@     ؓ@     <�@     ��@     ��@     P�@     ��@     t�@     �@     n�@     ��@     ʩ@     ��@        
�
conv3/biases_1*�	   ��i�   @��w?      `@!  �.�1�?)�C7^��4?2�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82�I�I�)�(�+A�F�&�U�4@@�$��[^:��"�ji6�9���.��x?�x��>h�'��1��a˲���[��pz�w�7��})�l a��ߊ4F��h���`�        �-���q=�uE����>�f����>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>O�ʗ��>>�?�s��>�FF�G ?��d�r?�5�i}1?�T7��?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?�������:�              �?      �?      �?      �?      �?       @              @              �?      @               @               @       @      @      �?      @               @       @      �?               @       @              �?      �?      �?              @              �?              �?              �?              �?               @              �?              �?              �?              �?      �?              �?      �?              �?              �?              �?              �?      �?       @              @      �?      �?               @               @       @      �?      @      @      @      @       @       @       @      @      �?      �?       @       @      �?      �?      �?               @       @      @              �?               @        
�
conv4/weights_1*�	    ���   �KW�?      �@! X�A1�.�)� Ǭ\;e@2�Ӗ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !��.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������1��a˲���[���FF�G �>�?�s����iD*L�پ�_�T�l׾jqs&\�ѾK+�E��Ͼ�iD*L��>E��a�W�>�ѩ�-�>���%�>>�?�s��>�FF�G ?��[�?6�]��?����?f�ʜ�7
?>h�'�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�������:�              b@     З@     ��@     �@     d�@     �@     p�@     ��@     �@     ��@     ��@     �@     X�@     �|@     p{@     �y@     �u@     t@     @q@     `q@     �n@     �g@     @e@     �f@     �d@     �e@     @b@      _@     @X@     @X@     �Y@     @T@     �P@     �S@     �L@      J@      M@      F@      @@      B@     �C@      7@      @@      8@      ,@      6@      .@      *@      *@      $@      @      (@       @      "@      *@      "@      $@      @       @      @      @      @      �?       @      @      @              @       @      @      �?      �?              @      �?       @              �?              �?      �?      �?               @              �?              �?              �?              �?              �?               @      �?              �?              @              �?               @               @      �?              @      @       @               @      @       @       @      @      @      @      @      @      "@       @      @      @      @      @       @      &@      1@      &@      1@      .@      3@      *@      =@      6@      7@     �B@      A@     �E@      A@     �E@      J@     �Q@     �O@      K@     @S@     �R@     @X@     �S@      Y@     �^@     @^@     �c@     �b@      d@     �f@      i@     @k@      n@     r@     pr@     `u@     �u@     �y@     @|@     �{@     ��@     @�@     x�@     ؅@     �@     Њ@     ��@     ��@     ܐ@     @�@     �@     h�@     �b@        
�
conv4/biases_1*�	    ��z�   ��؊?      p@!pS0�E�t?)��z�
�=?2����T}�o��5sz�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=���ߊ4F��h���`�a�Ϭ(���(����uE���⾮��%ᾙѩ�-߾E��a�Wܾ
�/eq
Ⱦ����ž�[�=�k���*��ڽ�G&�$��5�"�g���0�6�/n���u`P+d��6��>?�J���8"uH���Ő�;F��`�}6D�/�p`B�p
T~�;�u 5�9����<�)�4��evk'�4�e|�Z#���-�z�!�%�����i
�k���f��p�Łt�=	���R�����J��#���j�y�+pm��mm7&c��`���nx6�X� ��f׽r����tO����f;H�\Q������%���9�e����K����Qu�R"�PæҭUݽ��
"
ֽ�|86	Խ�Bb�!澽5%����        �-���q=�|86	�=��
"
�=f;H�\Q�=�tO���=y�+pm>RT��+�>�#���j>�J>2!K�R�>Łt�=	>��f��p>�i
�k>%���>��-�z�!>4�e|�Z#>��o�kJ%>4��evk'>���<�)>�'v�V,>7'_��+/>6NK��2>�so쩾4>p
T~�;>����W_>>p��Dp�@>u��6
�>T�L<�>R%�����>�u��gr�>0�6�/n�>5�"�g��>��~���>�XQ��>�����>�iD*L��>E��a�W�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?���%��b?5Ucv0ed?Tw��Nof?�N�W�m?;8�clp?uWy��r?hyO�s?#�+(�ŉ?�7c_XY�?�������:�              �?              �?               @      �?               @       @               @      �?       @      @       @      @      @       @      �?      �?      @      �?      �?              @      @              @      �?               @               @       @      �?       @      �?               @              �?      �?      �?      �?              �?              �?      @              �?      �?      �?              @              �?              �?               @              �?              �?               @              �?              �?              �?              �?      �?              �?              �?               @               @      �?      �?      �?      �?      �?               @              �?              �?              �?              �?              �?              �?              �?              �?              >@              �?              �?               @              �?      �?              �?      �?      �?      @      �?              @      �?      @       @              �?              �?      �?              �?              �?              �?              �?      �?               @              �?              �?              �?              @       @       @      �?      �?              �?       @              �?      �?       @              �?       @      �?              �?              �?      �?      @      @      �?      @      @      �?      @      @              @       @       @      @       @       @       @              @      �?      @       @              �?      @              �?      �?      �?              �?        
�
conv5/weights_1*�	   ���¿    }�?      �@! 0�꬏@)�w�g��I@2��?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=�uܬ�@8���%�V6��u�w74��7Kaa+�I�I�)�(��vV�R9��T7����iD*L��>E��a�W�>O�ʗ��>>�?�s��>�FF�G ?��[�?+A�F�&?I�I�)�(?��VlQ.?��bȬ�0?��82?�u�w74?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?�������:�             �m@      t@     �p@     �k@     �k@     �k@      h@      g@     �c@     �]@     �`@     �_@     �Z@     �V@      T@     �Q@     @P@     �Q@     �Q@     �D@     �M@      H@     �I@     �F@      @@      ;@      ;@      9@      0@      =@      3@      5@      *@      1@      (@       @      "@      @      &@      @      "@       @      @      @      @       @      @      @      @      @      @      @       @      �?       @       @              @      �?      �?      �?              �?      �?               @              �?              �?              �?              �?              �?              �?              �?              @      �?              �?               @       @       @      �?      �?      �?      @      @      @      @      @      @      $@      @       @       @      @      .@      *@       @      4@      .@      6@      &@      3@      3@      >@      9@      F@      >@      8@      ?@      C@      H@     �G@      G@     �K@      R@      Q@     �T@      W@     �V@     �\@     �[@     @`@     �b@     �c@     �f@     �e@     �m@     �l@     �o@     @q@      t@     �n@        
�
conv5/biases_1*�	   @`+E�   ��:>      <@!  @uͨT>)��<���<2���Ő�;F��`�}6D�p
T~�;�u 5�9��z��6��so쩾4���o�kJ%�4�e|�Z#���-�z�!�2!K�R���J��tO����f;H�\Q������%��ݟ��uy�=�/�4��=��-��J�=�K���=�f׽r��=nx6�X� >���">Z�TA[�>�#���j>�J>2!K�R�>��f��p>�i
�k>%���>��-�z�!>4�e|�Z#>��o�kJ%>4��evk'>���<�)>�'v�V,>7'_��+/>_"s�$1>6NK��2>�so쩾4>�z��6>u 5�9>p
T~�;>�������:�              �?              �?              �?              �?      �?              �?              �?      �?              �?              �?              �?              �?              �?      �?               @      �?      �?               @      �?      �?      �?               @      �?              �?              �?        y�td�U      �
�	�f�X��A*��

step  �@

lossv�
>
�
conv1/weights_1*�	   �H��   �O�?     `�@!  w;����)���9*@2�����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��vV�R9��T7���>�?�s��>�FF�G ?f�ʜ�7
?>h�'�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?�������:�             �P@     �j@     �i@     �c@     �c@     �b@      a@     �^@     @Z@      V@      T@      U@     �P@      M@      N@      N@      K@     �G@      9@      E@     �A@      9@      B@      :@      5@      8@      3@      .@      5@       @      (@      $@      $@      *@      $@      @      "@      @      "@      @      @      @      @       @      @      @       @      @       @      �?              �?              �?      @      �?      �?      �?       @      �?      �?       @      �?      �?               @              �?              �?              �?              �?              �?      �?       @      �?      �?              �?              �?               @      @      �?               @      �?       @       @      @      @      @       @      @       @      @      $@      @      @       @      @      @      .@      .@      3@      4@      .@      5@      6@      =@      =@      :@     �I@      @@      F@      J@     �J@      R@      Q@     �M@      T@     �V@     �U@     �W@     �[@     �^@     �Z@     �c@     �c@      f@      d@      f@     �U@        
�
conv1/biases_1*�	    �Z�   �D(y?      @@!  ��r̮?)g,��2?2��m9�H�[���bB�SY�ܗ�SsW�<DKc��T�nK���LQ�k�1^�sO��qU���I�
����G����#@�d�\D�X=���%>��:���82���bȬ�0�I�I�)�(�+A�F�&������6�]������%�>�uE����>x?�x�?��d�r?��ڋ?�.�?�qU���I?IcD���L?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?�������:�              �?               @              �?              �?              �?      �?              �?              �?              �?              �?              �?              �?              �?               @      �?      �?              �?      �?       @       @      �?       @      �?      �?      �?       @        
�
conv2/weights_1*�	   �)���   `���?      �@!�N�l~?@)��LjbE@2����g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��8K�ߝ�a�Ϭ(���(��澢f���侙ѩ�-߾E��a�Wܾ�iD*L�پ�*��ڽ�G&�$��0�6�/n���u`P+d��
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>�iD*L��>E��a�W�>�uE����>�f����>a�Ϭ(�>8K�ߝ�>�h���`�>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?�������:�             �@     ء@     �@     �@     ę@     ��@     ��@      �@     ؑ@     h�@     ��@     ��@     P�@     ��@     �@     @�@     �@     �}@     �z@      w@     y@     �u@     �r@     �q@     @n@     �k@     �h@      g@     �e@     �`@     �c@     �a@     �Y@     �_@     @W@      U@     �V@     @T@      N@     �F@      H@     �G@     �H@      F@      @@      C@      >@      <@     �@@      8@      1@      0@      ,@      $@      &@      ,@      @      (@      *@       @      "@      @      (@      @      @      @      �?      �?      @      @      @      @      @      @      @              �?      �?      �?               @      �?      �?       @              �?       @      �?              �?      �?              �?              �?              �?              �?              �?              �?               @              �?      �?              �?              �?      �?      �?               @       @      @      �?      @              �?       @      @      @      "@      @      @      @      @      @      @      &@      0@       @      &@      $@      3@      1@      (@      $@      2@      8@      9@      0@      5@      ?@      B@      B@      G@     �E@     �J@     �K@     �J@      Q@      Q@     �P@     �S@     @Z@     �]@     @_@     �`@      \@     �c@     @d@      i@     @h@     �n@      n@     pq@     0q@     �u@      v@     �z@     �|@      �@     ��@     �@     H�@     ��@     ��@     ��@     ��@     ��@     ̐@     ��@     ��@     @�@     ��@     0�@     X�@     ��@     �@        
�
conv2/biases_1*�	   @�?r�   ��v?      P@!  �F ?�?)����x2?2�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A���%>��:�uܬ�@8���82���bȬ�0�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��.����ڋ��T7����5�i}1�x?�x�?��d�r?ji6�9�?�S�F !?U�4@@�$?+A�F�&?I�I�)�(?�u�w74?��%�V6?uܬ�@8?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?�������:�              �?       @               @              �?       @      �?              �?      @              @       @      �?      �?      �?              @              �?              �?              �?              �?              �?              �?              �?              �?      �?               @      �?              �?              �?      �?      �?              �?      @       @      @      �?       @      �?       @       @      �?              �?       @      �?      �?              �?        
�
conv3/weights_1*�	   ���   ��Z�?      �@! (�O>�)�^�n�bU@2�����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �O�ʗ�����Zr[v��I��P=��pz�w�7���ߊ4F��h���`�a�Ϭ(���(��澢f�����uE����E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[ӾG&�$��5�"�g���
�}���>X$�z�>�MZ��K�>��|�~�>5�"�g��>G&�$�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?�������:�             @�@     ��@     |�@     \�@     ��@     ơ@     ğ@     �@     p�@     ��@     �@     L�@     `�@     �@     (�@     ȋ@     ؇@     h�@     h�@     �@     ��@     @~@     �z@     z@     �w@     `v@     �p@     �q@     �o@     p@     �j@     `h@     �e@     @a@     ``@     �a@     �]@      Z@      V@     �U@      U@     @P@     @R@     �K@      H@      K@     �G@     �G@      7@     �D@     �D@      =@      @@      =@      ;@      ;@      2@      .@      $@      4@      ,@      ,@      $@      "@      @      @      @      @      "@      @       @       @      @      @      @       @      @       @              �?              �?      �?      �?              �?              @              �?              �?      @              �?              �?              �?              �?              �?              �?              �?              �?              �?      �?              �?      �?               @      �?               @       @      �?      @              @      @      @              @      @       @      @      @      @      @      @      @      @      $@      @      1@      *@      (@      *@      .@      4@      2@      .@      2@      4@      >@      ;@      :@      <@      B@     �@@     �A@     �E@     �H@      K@     �J@     �R@      P@     �T@      W@     �V@     �Y@     �[@     �`@     �`@     `d@     �f@      f@     �l@     @l@     �k@     �q@     @s@     �v@     �t@     `x@     �|@     P|@     ȁ@     @�@     @�@      �@     H�@     Ȉ@     p�@     ��@     �@     ȓ@     \�@     ��@     ��@     �@     �@     h�@     �@     f�@     ��@     �@     ȉ@        
�
conv3/biases_1*�	    ��n�   @��|?      `@!  ��^r�?)���Y
@?2�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���VlQ.��7Kaa+�U�4@@�$��[^:��"��S�F !�ji6�9���T7����5�i}1���d�r�x?�x��f�ʜ�7
������1��a˲���[��O�ʗ�����Zr[v��        �-���q=�_�T�l�>�iD*L��>})�l a�>pz�w�7�>�FF�G ?��[�?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?Tw��Nof?P}���h?ߤ�(g%k?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?�������:�              �?      �?      �?      �?              @       @       @       @              �?      �?       @      �?       @       @       @      �?               @      �?      @              @       @               @               @               @              �?              �?              �?              �?              �?              �?               @              �?              �?              �?              �?              �?              �?      �?              �?              �?              @      �?               @      �?              @       @      �?       @               @              @      @      �?      @      @      @      @       @       @       @      @              �?       @              @      @      �?              �?       @        
�
conv4/weights_1*�	   `�!��   �Ao�?      �@! 4#i%7-�)Ili�!<e@2�Ӗ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[��a�Ϭ(���(�������ž�XQ�þ})�l a�>pz�w�7�>�FF�G ?��[�?1��a˲?��d�r?�5�i}1?�T7��?�vV�R9?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�������:�             �a@     ؗ@     ��@     �@     \�@     �@     ��@     ��@     ��@     ؅@     ��@     0�@     8�@      }@     P{@     �y@     �u@     Pt@     0q@     0q@      o@      h@      e@      g@     �d@     @e@     @b@     �_@     @X@      W@     �Z@     �S@     �P@      T@     �L@     �G@     �O@     �F@      @@      A@     �@@      >@      =@      8@      0@      5@      1@      ,@      (@      (@      "@      $@       @      @      *@      &@       @      @       @      @       @      $@              @      @      @       @      @      @      @       @       @              �?               @       @       @               @      �?              �?      �?       @              �?              �?              �?              �?              �?       @               @              �?               @       @      @      @      �?      �?              @      @      �?      @      @      @      @       @      @      @      @      @      @      "@      @      (@      2@      $@      2@      .@      6@      ,@      9@      3@      8@     �B@     �A@      D@     �B@      E@      J@     �P@     �P@     �K@     @S@     �R@     �X@     �S@     �Y@     @]@     �_@     �c@     �b@     �c@      f@     @i@      k@     �n@     �q@     �r@     `u@     �u@     z@     �{@     �{@     ��@     `�@     p�@     ��@     0�@     ��@     Ў@     ��@     �@     �@     �@     t�@     �b@        
�
conv4/biases_1*�	   �z�   �}#�?      p@!�\����?)>`U>�*H?2�>	� �����T}�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�>h�'��f�ʜ�7
��������[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a�h���`�8K�ߝ�a�Ϭ(���(��澙ѩ�-߾E��a�WܾK+�E��Ͼ['�?�;����ž�XQ�þ�*��ڽ�G&�$��5�"�g����
L�v�Q�28���FP�������M�6��>?�J���8"uH�����W_>�p
T~�;�7'_��+/��'v�V,����<�)�4��evk'���o�kJ%�4�e|�Z#���-�z�!��i
�k���f��p�2!K�R���J��#���j�Z�TA[�����"�RT��+��y�+pm��mm7&c�f;H�\Q������%���9�e����K���z�����i@4[��PæҭUݽH����ڽ        �-���q=��.4N�=;3����=�Qu�R"�=i@4[��=�`��>�mm7&c>y�+pm>���">Z�TA[�>2!K�R�>��R���>Łt�=	>��-�z�!>4�e|�Z#>��o�kJ%>4��evk'>���<�)>�'v�V,>7'_��+/>_"s�$1>6NK��2>�so쩾4>����W_>>p��Dp�@>/�p`B>�`�}6D>��z!�?�>��ӤP��>�5�L�>;9��R�>����>豪}0ڰ>�XQ��>�����>
�/eq
�>jqs&\��>��~]�[�>�iD*L��>E��a�W�>�ѩ�-�>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?���T}?>	� �?�#�h/�?���&�?�������:�              �?              �?               @      �?      �?       @              �?      @      �?              @      �?       @      @               @      @      �?      �?      �?       @      �?      @       @               @      @               @      @      �?               @              �?       @      �?      @              �?               @              �?              �?       @              �?      @       @              �?               @              �?              �?              �?              �?              �?              �?      �?              �?              �?      �?              �?              �?      �?      �?       @               @              �?               @      �?              �?      �?              �?              �?      �?      �?              �?              �?              ;@              �?              �?              �?      �?              �?              �?      �?              �?      �?       @      @      �?       @      @              �?              �?              �?              �?              �?              �?              �?      �?              �?              �?      �?              �?              �?      �?      �?              �?      �?              �?      �?              �?       @              �?       @      �?       @      @      �?              �?      �?      �?              �?       @      �?      �?       @      @       @      �?      @       @      @              @      @      @      �?      �?      �?       @      @       @       @       @      @              �?               @      �?              �?              �?              �?              �?        
�
conv5/weights_1*�	    �¿   @�!�?      �@! 0A��s @)�p2��I@2�	�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=�uܬ�@8���%�V6��u�w74���82��7Kaa+�I�I�)�(�+A�F�&�ji6�9���.���5�i}1���d�r��h���`�8K�ߝ뾙ѩ�-�>���%�>��ڋ?�.�?�[^:��"?U�4@@�$?+A�F�&?��82?�u�w74?��%�V6?uܬ�@8?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?�������:�	             �m@      t@     �p@      l@     �k@     �k@     �g@      g@     �c@     �\@     �`@      `@     �Z@      W@     �S@     �Q@     �P@     @Q@     �P@     �F@      N@      G@     �J@     �E@     �A@      :@      ;@      :@      0@      ;@      7@      2@      (@      2@      *@       @       @      "@       @      @      "@       @      @      @      @      @      @      @      @      @       @      @      �?      �?       @      �?      �?      �?      �?               @               @              �?              �?      �?              �?              �?              �?              �?              �?              �?      �?              �?              �?              �?               @      �?              �?       @      �?      �?      �?       @      @       @      @      @      @      @      $@      @      $@      @      $@      "@      ,@      $@      1@      3@      2@      .@      0@      4@      8@      :@      H@      >@      8@     �@@     �A@     �G@      H@      G@      M@     �Q@     �P@     �U@      V@     �V@     �\@     @\@     @`@     �b@     �c@     @f@     �e@     `m@     �l@     �o@     0q@     @t@     �n@        
�
conv5/biases_1*�	   ��I�   ���>>      <@!  �t�VV>)����r�<2�6��>?�J���8"uH�p��Dp�@�����W_>�p
T~�;�u 5�9����<�)�4��evk'���o�kJ%�Łt�=	���R�����#���j�Z�TA[���`���nx6�X� �z�����=ݟ��uy�=nx6�X� >�`��>�mm7&c>y�+pm>�J>2!K�R�>��R���>Łt�=	>��f��p>�i
�k>��-�z�!>4�e|�Z#>��o�kJ%>4��evk'>���<�)>�'v�V,>7'_��+/>_"s�$1>6NK��2>�so쩾4>�z��6>u 5�9>����W_>>p��Dp�@>�������:�              �?              �?              �?              �?      �?              �?              �?              �?              �?              �?              �?              �?              �?      �?      �?               @       @               @      �?               @               @              �?              �?        �[]U      ���	lϿ�X��A*��

step   A

loss�J>
�
conv1/weights_1*�	   ��J��    1�?     `�@!  N09��)$����H@2���]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0�+A�F�&�U�4@@�$��vV�R9��T7���>h�'��f�ʜ�7
�O�ʗ��>>�?�s��>ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?�������:�              �?     �Q@     @j@      j@     �c@      d@     �b@      a@      ^@     @X@     �V@     @T@     �T@     �P@      P@     �N@     �N@      G@      G@      @@     �@@      D@      ;@      >@      7@     �@@      1@      0@      2@      ,@      .@      (@      .@       @       @      $@      @      &@      @      "@      @      @      @      @      @      @      @      @      @      �?              �?       @       @       @               @      �?              �?              �?               @              �?              �?               @              �?              �?       @              @       @      �?       @      �?       @      @              �?      @      @      @      @       @      @      @      @              @       @      @      "@       @      @      0@      $@      ,@      .@      5@      1@      3@      5@      @@      9@     �A@      F@      B@      F@     �H@      M@     �M@     �S@     �N@     @R@     �W@     �T@     @W@     @\@      `@     @Y@      c@     �c@     �e@      d@      f@     �W@        
�
conv1/biases_1*�	   ���_�   �<�?      @@!  ,�#�?)B���s>?2��l�P�`�E��{��^��m9�H�[�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L�
����G�a�$��{E�I�I�)�(�+A�F�&�x?�x��>h�'��f�ʜ�7
�����?f�ʜ�7
?��ڋ?�.�?ji6�9�?�T���C?a�$��{E?�lDZrS?<DKc��T?E��{��^?�l�P�`?5Ucv0ed?Tw��Nof?P}���h?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?�������:�              �?      �?              �?      �?              �?              �?              �?              �?      �?              �?              �?      �?              �?              �?              �?              �?       @              @      �?       @       @       @      �?      �?      �?      �?        
�
conv2/weights_1*�	   `s骿   �&�?      �@!`/ O�A@)�g��wfE@2����g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[��>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ���(��澢f�����uE���⾮��%ᾄiD*L�پ�_�T�l׾�4[_>��>
�}���>�[�=�k�>��~���>�XQ��>�����>E��a�W�>�ѩ�-�>�f����>��(���>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?�������:�             X�@     ֡@     (�@     �@     ș@     ��@     ��@     (�@     ��@     P�@     �@      �@     P�@     0�@     ��@     ��@     �@     �}@     0{@     �v@     �x@     Pv@     �r@     �q@      p@      i@      j@      f@      e@      c@      b@     @`@     �]@     �[@     @Z@      U@      U@     �R@     �Q@      J@      I@      I@     �E@     �E@      ?@      E@      ;@      9@      :@      8@      1@      2@      0@      &@      ,@      0@      @      *@      "@      $@      @      "@       @       @      @       @      @      @      @      @      @      @      @      �?      �?      �?       @              �?      @      @              �?       @      �?      �?              �?              �?              �?              �?              �?               @              �?              �?               @       @               @       @               @              @       @      @              @       @      @      @      �?      $@       @       @      @      @      @      @      &@      *@      &@      ,@      (@      (@      &@      0@      $@      *@      7@      8@      3@      8@      B@     �A@     �E@     �A@      D@     �F@      K@      M@     @S@     �T@     �L@     �P@     �Y@      \@     �a@      ]@      a@     �b@     `d@     �h@     @h@     �m@      n@     0q@     pr@     0u@     `v@     �y@     �}@     ��@     ؀@     H�@      �@      �@     0�@      �@     �@     ��@     ��@     P�@     ��@     T�@     P�@     8�@     t�@     ��@     |�@       @        
�	
conv2/biases_1*�	   ��
v�   ��:|?      P@!  ��!Q�?)��{J:?2�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:���82���bȬ�0��5�i}1���d�r�>�?�s���O�ʗ���6�]��?����?�5�i}1?�T7��?�[^:��"?U�4@@�$?I�I�)�(?�7Kaa+?�u�w74?��%�V6?uܬ�@8?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?�������:�              �?      �?      �?      �?      �?              �?       @      �?       @      �?      �?              �?       @      �?      �?               @               @      �?              �?      �?              �?              �?              �?               @              �?              �?              �?              �?      �?              �?              �?      �?       @              �?       @      @      @      �?       @              @      �?      �?               @               @      �?              �?        
�
conv3/weights_1*�	   ��6��   �_��?      �@! &Q]6�@)tj��dU@2�����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F����(��澢f�����uE���⾮��%��_�T�l׾��>M|Kվ5�"�g���0�6�/n��=�.^ol>w`f���n>[#=�؏�>K���7�>�[�=�k�>��~���>
�/eq
�>;�"�q�>['�?��>jqs&\��>��~]�[�>�_�T�l�>�iD*L��>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>pz�w�7�>I��P=�>��Zr[v�>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?�������:�             �@     ��@     v�@     B�@     ��@     С@     �@     ��@     ��@     ��@     �@     x�@     x�@     ��@     ��@     h�@      �@     ��@     ��@     p�@     0�@      @     �z@     �x@      x@     pv@      q@     �r@      n@      p@     �j@     @f@     `g@     �a@      a@      `@     �_@     �Y@     �V@     �V@      S@     �P@     �P@      K@     �L@     �J@      J@     �C@      B@      @@     �@@     �A@      A@      <@      5@      .@      7@      (@      "@      2@      ,@      0@       @      (@       @      $@      @       @      @      @      @      @       @      @      @       @      @      @       @       @      �?      �?      �?              �?       @      �?              �?       @       @              �?              �?              �?              �?              �?              �?       @              �?              �?              �?              �?      �?              �?               @       @               @       @      @      @       @      @       @      @       @      @      @      @      &@       @       @      (@      ,@      0@      $@      4@      *@      0@      &@      6@      2@      8@      4@      ;@      @@      9@      A@      C@      B@      G@      D@     �I@     �M@      R@     @P@     �U@      V@     �Y@      X@      ^@     @\@     @c@     �b@     �e@     �g@     @k@      m@     �l@     pq@     �r@     �v@     @u@     �w@     �|@     �|@     P�@     Ȃ@     (�@     ��@     X�@     �@     p�@     ؐ@     (�@     ��@     ��@     ��@     ��@     4�@     Ȟ@     ��@     ¢@     j�@     ��@     �@     p�@        
�
conv3/biases_1*�	   ��r�   ��V�?      `@! ̨��;�?)�Zt��G?2�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@���%>��:�uܬ�@8���%�V6��u�w74��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$���ڋ��vV�R9���d�r�x?�x��>h�'�������6�]���jqs&\�ѾK+�E��ϾT�L<��u��6
��        �-���q=�FF�G ?��[�?1��a˲?6�]��?��d�r?�5�i}1?�vV�R9?��ڋ?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?ߤ�(g%k?�N�W�m?;8�clp?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?�������:�              �?      �?              �?      @      �?      �?      @       @              �?       @      �?      �?       @      @              �?      �?       @              @      �?              �?       @      �?              �?       @      �?              �?              �?      �?               @              �?              �?              �?               @              �?              �?               @              �?       @              �?              �?       @      @       @               @       @       @       @      �?              �?       @       @      �?       @       @      @      @      @      @      @      �?      @       @              �?       @              �?      @       @      �?              �?      �?        
�
conv4/weights_1*�	   �0��    ��?      �@! ���*�+�)I���<e@2�Ӗ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
�6�]���1��a˲��FF�G �>�?�s�����Zr[v��I��P=���f�����uE����['�?�;;�"�qʾ�_�T�l�>�iD*L��>�uE����>�f����>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�������:�             �a@     ��@     ��@      �@     P�@     �@     x�@     ��@     �@     Ѕ@     Ȅ@     `�@     �@     �|@     p{@     �y@     �u@     t@     @q@     `q@     �n@     @h@     @e@     `f@      e@     `e@     �a@     �_@     @Y@     �W@     �X@     @T@     @P@     �T@     �K@     �J@     �M@      E@      C@     �@@      =@      =@      @@      5@      5@      1@      1@      .@      *@      &@      &@      *@      @      $@      &@      &@      "@      @      @      @      @       @      @      @      @       @      �?      @      @      �?       @      �?       @              �?       @      �?       @               @              �?      �?               @              �?              �?              �?              �?              �?              �?              �?      �?              �?              �?              �?      �?      �?              �?              �?      �?      @      @       @      �?      �?      �?      @      �?      @       @      @      @      @      @      @      @      @      @      @      "@      @       @      5@      (@      3@      *@      6@      (@      :@      0@      ;@      C@     �A@     �D@     �@@     �F@      I@     �M@     @S@     �K@      R@     �S@     �W@      S@     @[@     @\@     ``@     �c@     �b@      d@     `e@     @i@     `k@      o@     �q@     �r@     pu@     `u@      z@     �{@     �{@     ��@     h�@     x�@     ��@     �@     Њ@     Ў@     ��@     �@     �@      �@     h�@      d@        
�
conv4/biases_1*�	   `�2��   `�	�?      p@!$�q��x�?)ţ�&5R?2����J�\������=���*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7���x?�x��>h�'�������6�]���1��a˲���[���FF�G ���Zr[v��I��P=��pz�w�7��})�l a�h���`�8K�ߝ���>M|Kվ��~]�[Ӿ;�"�qʾ
�/eq
Ⱦ��~��¾�[�=�k���*��ڽ�G&�$���MZ��K���u��gr��H��'ϱS��
L�v�Q�28���FP�������M�p��Dp�@�����W_>�6NK��2�_"s�$1�7'_��+/��'v�V,����<�)�4��evk'���o�kJ%�4�e|�Z#���-�z�!�%����2!K�R���J��#���j�Z�TA[�����"�RT��+���mm7&c��`���nx6�X� ��f׽r����9�e����K���z�����i@4[��        �-���q=i@4[��=z�����=����%�=f;H�\Q�=�tO���=�f׽r��=nx6�X� >�#���j>�J>��R���>Łt�=	>4�e|�Z#>��o�kJ%>4��evk'>���<�)>�'v�V,>7'_��+/>_"s�$1>6NK��2>�so쩾4>�z��6>u 5�9>p��Dp�@>/�p`B>�`�}6D>��Ő�;F>��ӤP��>�
�%W�>����>豪}0ڰ>�*��ڽ>�[�=�k�>�����>
�/eq
�>;�"�q�>['�?��>�f����>��(���>�ߊ4F��>})�l a�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?-Ա�L�?eiS�m�?^�S���?�"�uԖ?�������:�              �?              �?               @       @       @              �?      @      �?              �?      @       @      @      �?      @       @       @              �?      �?      �?      �?      @      @       @       @      @               @              �?               @       @               @      �?       @       @       @      �?              �?              @              �?              @       @              @      �?      �?              �?              �?              �?              �?              �?              �?              �?               @              �?              �?      �?               @      �?      �?              �?      �?              �?       @              �?      �?              �?      �?      �?              �?              �?              ;@              �?              �?              �?      �?              �?               @              �?      �?              �?      @      �?      @      �?      �?      �?              �?              �?              �?              �?              �?              �?              �?              �?               @              �?      �?      �?      �?              �?       @              �?      �?              �?      �?      @               @       @      �?      @              �?       @               @      �?      �?      �?              �?      @       @       @       @      @       @      �?      @      @      @       @       @               @      @      @      @      @      �?              �?              �?      �?      �?      �?      �?              �?              �?        
�
conv5/weights_1*�	   @��¿   ��B�?      �@! p5^<"!@)�ŧU&�I@2�	�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���bȬ�0���VlQ.��7Kaa+�I�I�)�(�x?�x��>h�'��f�ʜ�7
�6�]���1��a˲����%�>�uE����>��d�r?�5�i}1?ji6�9�?�S�F !?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�������:�	             `m@      t@     �p@      l@     �k@     �k@     �g@     �g@      c@      ]@     �`@     �`@     @[@      V@      T@      R@     �P@     �P@     @P@      H@     �L@      H@      K@      F@     �@@      9@      ;@      9@      2@      9@      ;@      1@      &@      3@      ,@      @       @      "@      "@      @      "@       @      @      @      @      @       @      @      @      @      @      @      �?       @      �?              �?       @      �?              �?              �?      �?      �?              �?              �?              �?      �?              �?              �?              �?              �?              �?              �?       @              �?      �?      @              �?              �?      �?               @      @      @      @       @      @              "@      @      @      (@      @      &@      $@      (@      (@      0@      1@      5@      ,@      2@      2@      7@      >@     �E@      ?@      :@     �@@      A@     �G@     �G@     �G@     �L@     �Q@     �P@     �U@      U@      X@     �[@     �\@     �`@     �b@     �c@     @f@     �e@      m@     �l@     @p@     q@     @t@     �n@      �?        
�
conv5/biases_1*�	    k�L�   ���?>      <@!  �RX>)�	�Fp�<2�������M�6��>?�J��`�}6D�/�p`B�p��Dp�@�����W_>��'v�V,����<�)�4��evk'��i
�k���f��p��J��#���j����"�RT��+���Qu�R"�=i@4[��=�mm7&c>y�+pm>RT��+�>���">�J>2!K�R�>��f��p>�i
�k>%���>4�e|�Z#>��o�kJ%>4��evk'>���<�)>�'v�V,>7'_��+/>_"s�$1>6NK��2>�z��6>u 5�9>p
T~�;>����W_>>p��Dp�@>�������:�              �?              �?              �?              �?      �?              �?              �?              �?              �?              �?              �?              �?              �?       @              �?       @              @              �?      �?              �?      �?      �?      �?        �}~�V      ��v_	�*�X��A	*��

step  A

loss�q>
�
conv1/weights_1*�	   ೭��    �j�?     `�@! ������)ԯ�ȁj@2���]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.�>h�'��f�ʜ�7
�O�ʗ��>>�?�s��>�FF�G ?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�������:�              @     �R@      j@     @j@      d@     `c@     �b@     �`@      ^@     �W@      Y@      R@     @T@     @P@      R@      P@     �L@      D@     �G@      C@      @@      C@      A@      8@      8@     �@@      (@      6@      (@      *@      .@      3@      ,@      @      &@      @      @      @      ,@      @      @      �?      @       @      @      @      @      �?      @      @       @       @              �?      @       @      �?      @       @      �?              �?              �?      �?              �?              �?      �?       @              �?      @       @       @      �?       @       @      @       @      @      @      @      @       @      @      @      @      $@      @      @      "@      &@      $@      (@      .@      ,@      3@      1@      6@      9@      8@     �@@      @@      C@     �D@     �E@     �F@     �O@      P@     @R@     �O@      S@     �S@      Y@     �V@      Z@      `@     �Y@     �b@      c@     `f@      d@      f@     �X@      �?        
�
conv1/biases_1*�	    �1d�    "�?      @@!  ���5�?)	E?��F?2�5Ucv0ed����%��b��l�P�`�E��{��^�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��S�F !�ji6�9���5�i}1���d�r�>h�'�?x?�x�?�[^:��"?U�4@@�$?��%>��:?d�\D�X=?�!�A?�T���C?�lDZrS?<DKc��T?�m9�H�[?E��{��^?�l�P�`?���%��b?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?�������:�              �?              �?              �?              �?      �?      �?              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?      �?      �?       @      �?      @      �?      �?       @               @      �?        
�
conv2/weights_1*�	   `@��   �n�?      �@! 
���SB@)آ�Z�jE@2�I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f���侄iD*L�پ�_�T�l׾豪}0ڰ��������~���>�XQ��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?�������:�              �?     ��@     ܡ@     �@     �@     ��@      �@     �@     �@     ��@     4�@     �@     8�@     ��@     �@     x�@     p�@     8�@     �|@     �z@     �v@      x@     �u@     �r@      r@     �o@     @k@     �h@     �e@     @e@      c@     `c@     @^@     �\@     @Y@     �Y@     �V@      U@     �R@      Q@     �I@     �P@     �E@     �K@      B@      B@      B@      <@      8@      9@      ?@      *@      *@      *@      0@      $@      &@      "@      (@      .@      @      $@      @      @      &@      @      @      @       @      �?      @      @      @      �?              @      �?      �?       @       @      �?      �?      �?              @      �?              �?      �?              @              �?              �?              �?              �?               @      �?              �?              �?      �?      �?              �?              �?      �?              �?      �?      �?      @      �?      @       @      @      @              �?      @       @      �?      "@       @      @      @      @      &@      ,@      *@      @      (@      @      &@      *@      (@      0@      ,@       @      ;@      3@      7@      9@      =@      @@     �F@      B@     �C@     �L@      G@     �K@     �N@     �X@     @P@     �Q@     �U@     �_@      ]@     @^@     �a@     �c@     `e@     �f@     �g@     `m@      n@     �q@     �r@     �t@     �u@     �y@     `~@     `�@     ��@     ��@     �@     �@      �@     P�@     ��@     X�@     T�@     T�@     l�@     �@     �@     �@     \�@     r�@     0�@      @        
�	
conv2/biases_1*�		   ��y�   @��?      P@!  @���?)W����A?2�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A���82���bȬ�0�U�4@@�$��[^:��"��S�F !�ji6�9��f�ʜ�7
�������iD*L�پ�_�T�l׾ji6�9�?�S�F !?�[^:��"?U�4@@�$?I�I�)�(?�7Kaa+?��VlQ.?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?hyO�s?&b՞
�u?*QH�x?o��5sz?>	� �?����=��?�������:�              �?       @              �?      �?      �?              �?      @              �?              �?              �?      �?      �?      �?      �?      �?      �?       @      �?              �?      �?              �?              �?              �?              �?              �?              �?              �?              �?      �?              �?      �?              �?               @      �?      �?               @       @      @       @       @              @      �?      �?      �?               @              @              �?        
�
conv3/weights_1*�	   �u��   @��?      �@! Y>f�@)(�BgU@2�����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f���侮��%ᾙѩ�-߾E��a�Wܾ
�/eq
Ⱦ����ž�XQ�þ��~��¾�u`P+d����n�����豪}0ڰ��������|�~�>���]���>
�/eq
�>;�"�q�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>��(���>a�Ϭ(�>8K�ߝ�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?�������:�             ��@     T�@     V�@     n�@     ��@     ��@     "�@     Ĝ@     t�@     ��@     ��@     ��@     T�@     p�@     0�@     P�@     ��@      �@     X�@     Ѓ@     8�@      @     pz@     Py@     Pw@     pv@     �q@      r@     �m@     `p@     @k@     �f@     @f@     �a@     �`@     @`@     �^@     @[@     �W@     �V@     @P@     @S@      P@     �I@      J@     �K@      I@      D@      ?@      A@     �B@      <@     �@@      6@      :@      :@      2@      *@      *@      (@      ,@      .@      @      @      @      0@      @      @       @      "@       @      @      @      @      @       @              @      @               @      �?      @       @               @              �?      �?      �?      �?              �?      �?              �?              �?              �?              �?              �?               @              �?       @      �?      @              �?      �?              �?       @      �?      �?      @      �?              �?      @      @      @      @              @      @      @      @      "@      @      @      @      "@      *@      .@      &@      0@      .@      *@      2@      ,@      .@      8@      2@      8@      >@      <@      =@      =@     �C@     �D@     �A@     �F@      K@      M@      R@      Q@     �T@     @X@     @W@     @W@     �]@     �`@     @b@     `c@     �e@     �f@     `l@     �k@     �k@     �q@     `r@     0x@     �s@     x@      |@     �|@     h�@     �@     ��@     Ѕ@     ��@     ��@     Ў@     ��@     D�@     ��@     T�@     ̗@     X�@     ��@     X�@     ġ@     ��@     \�@     ��@     ܩ@     �@        
�
conv3/biases_1*�	   `l�u�   �@��?      `@! ���2�?)�]t�IP?2�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��S�F !�ji6�9����ڋ��vV�R9���d�r�x?�x��>h�'��f�ʜ�7
����m!#���
�%W��        �-���q=})�l a�>pz�w�7�>>�?�s��>�FF�G ?����?f�ʜ�7
?>h�'�?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?;8�clp?uWy��r?hyO�s?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?�������:�              �?      �?              @      �?      �?       @      @      �?      �?       @      �?              @      @              �?       @              �?       @       @              �?              �?      �?              �?              �?               @              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?       @              �?      �?              �?      �?              �?      �?      �?              �?      �?       @      �?      �?      �?      @               @      @              �?      @      �?      �?       @       @       @      @      @      @      @       @      @      @      @      �?               @      �?               @      @       @      �?      �?              �?        
�
conv4/weights_1*�	   ��>��   `ޮ�?      �@!��G�*�)�"���=e@2�Ӗ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�>h�'��f�ʜ�7
���[���FF�G ���Zr[v��I��P=��pz�w�7����~]�[Ӿjqs&\�Ѿ['�?�;;�"�qʾ��~��¾�[�=�k���FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�������:�             �a@     ��@     ��@     �@     h�@     ��@     ��@     �@     ��@     ��@     ��@     x�@     X�@     p|@     {@     �z@     �u@     �s@     pq@     Pq@     �n@     �h@      e@     �f@     �d@     �d@      b@     �_@     �X@     @Y@     @W@     �S@      P@     @V@      M@      H@      O@      B@     �C@     �B@      <@      >@      9@      <@      3@      4@      ,@      1@      *@      &@      (@      $@      @      ,@       @      (@      @      @      @      @      @      $@       @      @      @       @      �?       @       @      �?       @       @      �?      �?              �?      �?      �?      �?       @              �?              �?              �?      �?              �?              �?              �?              �?              �?      �?      �?       @              �?      �?      �?      �?       @      @      @      @       @      �?       @       @      �?       @      @       @      @       @      "@      @      @      @      @      @       @      (@      1@      0@      2@      (@      3@      0@      6@      3@      8@      C@      C@     �C@      B@      E@      K@     �K@     �S@     �K@     @R@     �R@     �X@      S@     �Z@     �]@     �^@     �c@     �b@      d@     �d@     �i@     �k@     @o@     �q@     pr@     �u@     pu@     �y@     �{@     0|@     ��@     P�@     ��@     ��@      �@     ��@     ��@     p�@     ��@     �@     �@     ��@     @d@        
�
conv4/biases_1*�	   �����   @�ݘ?      p@!�Yy`�ٛ?)($����Y?2�-Ա�L�����J�\�����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��5�i}1���d�r�x?�x��>h�'��f�ʜ�7
�6�]���1��a˲���[��>�?�s���O�ʗ�����Zr[v��I��P=��a�Ϭ(���(����uE���⾮��%ᾄiD*L�پ�_�T�l׾K+�E��Ͼ['�?�;;�"�qʾ�XQ�þ��~��¾�[�=�k���*��ڽ�Fixі�W���x��U�H��'ϱS��
L�v�Q�28���FP�/�p`B�p��Dp�@��z��6��so쩾4�6NK��2�_"s�$1�7'_��+/��'v�V,����<�)�4��evk'�4�e|�Z#���-�z�!�%������f��p�Łt�=	���R����2!K�R��Z�TA[�����"�RT��+��y�+pm�nx6�X� ��f׽r�������%���9�e���=��]����/�4������/���EDPq��        �-���q=��-��J�=�K���=�f׽r��=nx6�X� >RT��+�>���">�#���j>�J>Łt�=	>��f��p>��o�kJ%>4��evk'>���<�)>�'v�V,>7'_��+/>_"s�$1>6NK��2>�so쩾4>�z��6>u 5�9>p
T~�;>����W_>>/�p`B>�`�}6D>��8"uH>6��>?�J>�
�%W�>���m!#�>
�}���>X$�z�>��n����>�u`P+d�>G&�$�>�*��ڽ>�XQ��>�����>
�/eq
�>;�"�q�>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>��Zr[v�>O�ʗ��>��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?#�+(�ŉ?�7c_XY�?�"�uԖ?}Y�4j�?�������:�              �?              �?               @      �?      @      �?              @       @              @      @      @      �?       @      @              @              �?       @              �?      @       @      @       @      �?      �?       @      �?      �?      �?               @      �?       @      �?               @      �?      �?      �?      �?      �?              �?      @      �?       @              @      �?              �?      �?      �?              �?              �?              �?              �?      �?              �?              �?              �?              �?      �?              �?              �?      �?      �?              �?      �?      �?              �?      �?              �?              �?              �?               @              �?              �?              �?              �?              ;@              �?              �?              �?              �?               @              �?              �?       @               @      @       @      �?      �?      �?              �?              �?              �?              �?              �?              �?              �?      �?      �?              �?              �?              �?              @              �?       @      �?              �?              �?      �?      �?              �?       @      �?      �?       @               @              �?              �?      �?      @               @              �?       @      @      �?      �?       @      @       @      @      �?      @       @      @       @       @       @       @      @      �?      @       @              �?              �?      �?      �?      �?      �?              �?              �?        
�
conv5/weights_1*�	   @��¿    'd�?      �@! ̎��!@)8����I@2�	�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��.����ڋ���[���FF�G ��f����>��(���>�T7��?�vV�R9?�.�?ji6�9�?��82?�u�w74?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�������:�	             `m@      t@     �p@      k@     �k@      k@     �g@     �g@     �c@      \@     �`@     �`@     �Z@     �V@      T@      R@     �P@     @Q@      P@     �G@     �K@     �I@     �K@     �D@      A@      ;@      :@      8@      2@      :@      8@      5@      &@      2@      ,@      @      "@       @      &@      @      "@      @       @      @      @      @      @      @      @       @      @      @       @      �?      �?              @      �?              �?      �?              �?              �?              �?      �?      �?              �?              �?              �?              �?              �?              �?              @      @               @              �?              �?       @      �?       @       @      �?      @              @      @      @      @      @      "@      @      ,@      &@      "@      .@      .@      4@      .@      .@      6@      .@      7@      <@      F@      @@      ;@      ?@      B@      G@     �H@     �F@     �L@      R@     @P@     �V@     @T@     @X@     �[@      ]@      `@     @c@     @c@     `f@     �e@     �l@     �l@     @p@     �p@     �s@     �o@       @        
�
conv5/biases_1*�	   �1.P�   ��?>      <@!  �+DhY>)*�����<2�28���FP�������M���Ő�;F��`�}6D�/�p`B�7'_��+/��'v�V,�Łt�=	���R�������"�RT��+��nx6�X� ��f׽r����|86	�=��
"
�=���">Z�TA[�>�J>2!K�R�>�i
�k>%���>��-�z�!>4�e|�Z#>4��evk'>���<�)>�'v�V,>7'_��+/>_"s�$1>6NK��2>�so쩾4>p
T~�;>����W_>>p��Dp�@>�������:�              �?              �?      �?               @              �?              �?              �?              �?               @              �?              �?      �?       @              @      �?       @      �?              �?               @       @        ��"#