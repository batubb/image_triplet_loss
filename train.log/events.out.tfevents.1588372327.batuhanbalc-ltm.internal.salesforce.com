       �K"	  ��(��Abrain.Event:2��ۯ     �:E	t}��(��A"��
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

seed *
T0* 
_class
loc:@conv1/weights*
seed2 *
dtype0*&
_output_shapes
: 
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
conv1/biases/AssignAssignconv1/biasesconv1/biases/Initializer/zeros*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*
_class
loc:@conv1/biases
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
model/conv1/Conv2DConv2Danchor_inputconv1/weights/read*0
_output_shapes
:���������2� *
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME
�
model/conv1/BiasAddBiasAddmodel/conv1/Conv2Dconv1/biases/read*
data_formatNHWC*0
_output_shapes
:���������2� *
T0
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
dtype0*
_output_shapes
:*%
valueB"          @   * 
_class
loc:@conv2/weights
�
,conv2/weights/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *
valueB
 *��L�* 
_class
loc:@conv2/weights
�
,conv2/weights/Initializer/random_uniform/maxConst*
valueB
 *��L=* 
_class
loc:@conv2/weights*
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
(conv2/weights/Initializer/random_uniformAdd,conv2/weights/Initializer/random_uniform/mul,conv2/weights/Initializer/random_uniform/min*
T0* 
_class
loc:@conv2/weights*&
_output_shapes
: @
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
conv2/biases/AssignAssignconv2/biasesconv2/biases/Initializer/zeros*
use_locking(*
T0*
_class
loc:@conv2/biases*
validate_shape(*
_output_shapes
:@
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
model/conv2/Conv2DConv2Dmodel/conv1/MaxPool2D/MaxPoolconv2/weights/read*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME*/
_output_shapes
:���������K@*
	dilations
*
T0
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
conv3/biases/AssignAssignconv3/biasesconv3/biases/Initializer/zeros*
use_locking(*
T0*
_class
loc:@conv3/biases*
validate_shape(*
_output_shapes	
:�
r
conv3/biases/readIdentityconv3/biases*
_output_shapes	
:�*
T0*
_class
loc:@conv3/biases
j
model/conv3/dilation_rateConst*
dtype0*
_output_shapes
:*
valueB"      
�
model/conv3/Conv2DConv2Dmodel/conv2/MaxPool2D/MaxPoolconv3/weights/read*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME*0
_output_shapes
:���������&�*
	dilations

�
model/conv3/BiasAddBiasAddmodel/conv3/Conv2Dconv3/biases/read*
T0*
data_formatNHWC*0
_output_shapes
:���������&�
n
model/conv3/conv3/ReluRelumodel/conv3/BiasAdd*0
_output_shapes
:���������&�*
T0
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
6conv4/weights/Initializer/random_uniform/RandomUniformRandomUniform.conv4/weights/Initializer/random_uniform/shape*

seed *
T0* 
_class
loc:@conv4/weights*
seed2 *
dtype0*(
_output_shapes
:��
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
_class
loc:@conv4/biases*
	container *
shape:�*
dtype0*
_output_shapes	
:�*
shared_name 
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
model/conv4/Conv2DConv2Dmodel/conv3/MaxPool2D/MaxPoolconv4/weights/read*
paddingSAME*0
_output_shapes
:����������*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(
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
model/conv4/MaxPool2D/MaxPoolMaxPoolmodel/conv4/conv4/Relu*0
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
6conv5/weights/Initializer/random_uniform/RandomUniformRandomUniform.conv5/weights/Initializer/random_uniform/shape*
T0* 
_class
loc:@conv5/weights*
seed2 *
dtype0*'
_output_shapes
:�*

seed 
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
(conv5/weights/Initializer/random_uniformAdd,conv5/weights/Initializer/random_uniform/mul,conv5/weights/Initializer/random_uniform/min*
T0* 
_class
loc:@conv5/weights*'
_output_shapes
:�
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
+model/Flatten/flatten/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
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
dtype0*
_output_shapes
:*
valueB"      
�
model_1/conv1/Conv2DConv2Dpositive_inputconv1/weights/read*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME*0
_output_shapes
:���������2� 
�
model_1/conv1/BiasAddBiasAddmodel_1/conv1/Conv2Dconv1/biases/read*
T0*
data_formatNHWC*0
_output_shapes
:���������2� 
r
model_1/conv1/conv1/ReluRelumodel_1/conv1/BiasAdd*0
_output_shapes
:���������2� *
T0
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
T0*
data_formatNHWC*
strides
*
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
model_1/conv2/MaxPool2D/MaxPoolMaxPoolmodel_1/conv2/conv2/Relu*
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
model_1/conv3/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
�
model_1/conv3/Conv2DConv2Dmodel_1/conv2/MaxPool2D/MaxPoolconv3/weights/read*
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
model_1/conv3/BiasAddBiasAddmodel_1/conv3/Conv2Dconv3/biases/read*
data_formatNHWC*0
_output_shapes
:���������&�*
T0
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
T0*
data_formatNHWC*
strides
*
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
model_1/conv4/MaxPool2D/MaxPoolMaxPoolmodel_1/conv4/conv4/Relu*
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
l
model_1/conv5/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
�
model_1/conv5/Conv2DConv2Dmodel_1/conv4/MaxPool2D/MaxPoolconv5/weights/read*
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
-model_1/Flatten/flatten/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB:
w
-model_1/Flatten/flatten/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
�
%model_1/Flatten/flatten/strided_sliceStridedSlicemodel_1/Flatten/flatten/Shape+model_1/Flatten/flatten/strided_slice/stack-model_1/Flatten/flatten/strided_slice/stack_1-model_1/Flatten/flatten/strided_slice/stack_2*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
Index0*
T0
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
model_1/Flatten/flatten/ReshapeReshapemodel_1/conv5/MaxPool2D/MaxPool%model_1/Flatten/flatten/Reshape/shape*(
_output_shapes
:����������*
T0*
Tshape0
l
model_2/conv1/dilation_rateConst*
dtype0*
_output_shapes
:*
valueB"      
�
model_2/conv1/Conv2DConv2Dnegative_inputconv1/weights/read*
paddingSAME*0
_output_shapes
:���������2� *
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(
�
model_2/conv1/BiasAddBiasAddmodel_2/conv1/Conv2Dconv1/biases/read*
data_formatNHWC*0
_output_shapes
:���������2� *
T0
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
model_2/conv2/Conv2DConv2Dmodel_2/conv1/MaxPool2D/MaxPoolconv2/weights/read*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME*/
_output_shapes
:���������K@*
	dilations
*
T0
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
model_2/conv3/MaxPool2D/MaxPoolMaxPoolmodel_2/conv3/conv3/Relu*0
_output_shapes
:����������*
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
model_2/conv4/Conv2DConv2Dmodel_2/conv3/MaxPool2D/MaxPoolconv4/weights/read*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*0
_output_shapes
:����������*
	dilations

�
model_2/conv4/BiasAddBiasAddmodel_2/conv4/Conv2Dconv4/biases/read*
T0*
data_formatNHWC*0
_output_shapes
:����������
r
model_2/conv4/conv4/ReluRelumodel_2/conv4/BiasAdd*0
_output_shapes
:����������*
T0
�
model_2/conv4/MaxPool2D/MaxPoolMaxPoolmodel_2/conv4/conv4/Relu*
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
l
model_2/conv5/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
�
model_2/conv5/Conv2DConv2Dmodel_2/conv4/MaxPool2D/MaxPoolconv5/weights/read*/
_output_shapes
:���������
*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME
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
%model_2/Flatten/flatten/strided_sliceStridedSlicemodel_2/Flatten/flatten/Shape+model_2/Flatten/flatten/strided_slice/stack-model_2/Flatten/flatten/strided_slice/stack_1-model_2/Flatten/flatten/strided_slice/stack_2*

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
SqrtSqrtSum_1*
T0*#
_output_shapes
:���������
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
Sum_2/reduction_indicesConst*
dtype0*
_output_shapes
: *
value	B :
w
Sum_2SumPow_1Sum_2/reduction_indices*
T0*#
_output_shapes
:���������*
	keep_dims( *

Tidx0
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
H
divRealDivSummul_1*
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
Pow_2/yConst*
valueB
 *   @*
dtype0*
_output_shapes
: 
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
Sqrt_2SqrtSum_4*
T0*#
_output_shapes
:���������
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
Sqrt_3SqrtSum_5*#
_output_shapes
:���������*
T0
J
mul_3MulSqrt_2Sqrt_3*
T0*#
_output_shapes
:���������
L
div_1RealDivSum_3mul_3*
T0*#
_output_shapes
:���������
}
subSubmodel/Flatten/flatten/Reshapemodel_1/Flatten/flatten/Reshape*
T0*(
_output_shapes
:����������
L
Pow_4/yConst*
valueB
 *   @*
dtype0*
_output_shapes
: 
M
Pow_4PowsubPow_4/y*
T0*(
_output_shapes
:����������
Y
Sum_6/reduction_indicesConst*
value	B :*
dtype0*
_output_shapes
: 
{
Sum_6SumPow_4Sum_6/reduction_indices*
T0*'
_output_shapes
:���������*
	keep_dims(*

Tidx0
G
Sqrt_4SqrtSum_6*
T0*'
_output_shapes
:���������

sub_1Submodel/Flatten/flatten/Reshapemodel_2/Flatten/flatten/Reshape*
T0*(
_output_shapes
:����������
L
Pow_5/yConst*
valueB
 *   @*
dtype0*
_output_shapes
: 
O
Pow_5Powsub_1Pow_5/y*
T0*(
_output_shapes
:����������
Y
Sum_7/reduction_indicesConst*
dtype0*
_output_shapes
: *
value	B :
{
Sum_7SumPow_5Sum_7/reduction_indices*'
_output_shapes
:���������*
	keep_dims(*

Tidx0*
T0
G
Sqrt_5SqrtSum_7*
T0*'
_output_shapes
:���������
N
sub_2SubSqrt_4Sqrt_5*
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
valueB"       *
dtype0*
_output_shapes
:
Z
MeanMeanMaximumConst*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
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
dtype0*
_output_shapes
: *
	container 
�
Variable/AssignAssignVariableVariable/initial_value*
T0*
_class
loc:@Variable*
validate_shape(*
_output_shapes
: *
use_locking(
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
Truncate( *
_output_shapes
: *

DstT0*

SrcT0
�
gradients/Mean_grad/truedivRealDivgradients/Mean_grad/Tilegradients/Mean_grad/Cast*'
_output_shapes
:���������*
T0
_
gradients/Maximum_grad/ShapeShapeadd*
_output_shapes
:*
T0*
out_type0
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
gradients/Maximum_grad/ReshapeReshapegradients/Maximum_grad/Sumgradients/Maximum_grad/Shape*'
_output_shapes
:���������*
T0*
Tshape0
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
(gradients/add_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_grad/Shapegradients/add_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
gradients/add_grad/SumSum/gradients/Maximum_grad/tuple/control_dependency(gradients/add_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
gradients/add_grad/ReshapeReshapegradients/add_grad/Sumgradients/add_grad/Shape*'
_output_shapes
:���������*
T0*
Tshape0
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
+gradients/add_grad/tuple/control_dependencyIdentitygradients/add_grad/Reshape$^gradients/add_grad/tuple/group_deps*'
_output_shapes
:���������*
T0*-
_class#
!loc:@gradients/add_grad/Reshape
�
-gradients/add_grad/tuple/control_dependency_1Identitygradients/add_grad/Reshape_1$^gradients/add_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/add_grad/Reshape_1*
_output_shapes
: 
`
gradients/sub_2_grad/ShapeShapeSqrt_4*
T0*
out_type0*
_output_shapes
:
b
gradients/sub_2_grad/Shape_1ShapeSqrt_5*
T0*
out_type0*
_output_shapes
:
�
*gradients/sub_2_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/sub_2_grad/Shapegradients/sub_2_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
gradients/sub_2_grad/SumSum+gradients/add_grad/tuple/control_dependency*gradients/sub_2_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
gradients/sub_2_grad/ReshapeReshapegradients/sub_2_grad/Sumgradients/sub_2_grad/Shape*'
_output_shapes
:���������*
T0*
Tshape0
�
gradients/sub_2_grad/Sum_1Sum+gradients/add_grad/tuple/control_dependency,gradients/sub_2_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
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
gradients/Sqrt_4_grad/SqrtGradSqrtGradSqrt_4-gradients/sub_2_grad/tuple/control_dependency*
T0*'
_output_shapes
:���������
�
gradients/Sqrt_5_grad/SqrtGradSqrtGradSqrt_5/gradients/sub_2_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:���������
_
gradients/Sum_6_grad/ShapeShapePow_4*
T0*
out_type0*
_output_shapes
:
�
gradients/Sum_6_grad/SizeConst*
dtype0*
_output_shapes
: *
value	B :*-
_class#
!loc:@gradients/Sum_6_grad/Shape
�
gradients/Sum_6_grad/addAddSum_6/reduction_indicesgradients/Sum_6_grad/Size*
T0*-
_class#
!loc:@gradients/Sum_6_grad/Shape*
_output_shapes
: 
�
gradients/Sum_6_grad/modFloorModgradients/Sum_6_grad/addgradients/Sum_6_grad/Size*
T0*-
_class#
!loc:@gradients/Sum_6_grad/Shape*
_output_shapes
: 
�
gradients/Sum_6_grad/Shape_1Const*
valueB *-
_class#
!loc:@gradients/Sum_6_grad/Shape*
dtype0*
_output_shapes
: 
�
 gradients/Sum_6_grad/range/startConst*
value	B : *-
_class#
!loc:@gradients/Sum_6_grad/Shape*
dtype0*
_output_shapes
: 
�
 gradients/Sum_6_grad/range/deltaConst*
value	B :*-
_class#
!loc:@gradients/Sum_6_grad/Shape*
dtype0*
_output_shapes
: 
�
gradients/Sum_6_grad/rangeRange gradients/Sum_6_grad/range/startgradients/Sum_6_grad/Size gradients/Sum_6_grad/range/delta*-
_class#
!loc:@gradients/Sum_6_grad/Shape*
_output_shapes
:*

Tidx0
�
gradients/Sum_6_grad/Fill/valueConst*
value	B :*-
_class#
!loc:@gradients/Sum_6_grad/Shape*
dtype0*
_output_shapes
: 
�
gradients/Sum_6_grad/FillFillgradients/Sum_6_grad/Shape_1gradients/Sum_6_grad/Fill/value*
T0*

index_type0*-
_class#
!loc:@gradients/Sum_6_grad/Shape*
_output_shapes
: 
�
"gradients/Sum_6_grad/DynamicStitchDynamicStitchgradients/Sum_6_grad/rangegradients/Sum_6_grad/modgradients/Sum_6_grad/Shapegradients/Sum_6_grad/Fill*
T0*-
_class#
!loc:@gradients/Sum_6_grad/Shape*
N*
_output_shapes
:
�
gradients/Sum_6_grad/Maximum/yConst*
value	B :*-
_class#
!loc:@gradients/Sum_6_grad/Shape*
dtype0*
_output_shapes
: 
�
gradients/Sum_6_grad/MaximumMaximum"gradients/Sum_6_grad/DynamicStitchgradients/Sum_6_grad/Maximum/y*
_output_shapes
:*
T0*-
_class#
!loc:@gradients/Sum_6_grad/Shape
�
gradients/Sum_6_grad/floordivFloorDivgradients/Sum_6_grad/Shapegradients/Sum_6_grad/Maximum*
T0*-
_class#
!loc:@gradients/Sum_6_grad/Shape*
_output_shapes
:
�
gradients/Sum_6_grad/ReshapeReshapegradients/Sqrt_4_grad/SqrtGrad"gradients/Sum_6_grad/DynamicStitch*
T0*
Tshape0*0
_output_shapes
:������������������
�
gradients/Sum_6_grad/TileTilegradients/Sum_6_grad/Reshapegradients/Sum_6_grad/floordiv*

Tmultiples0*
T0*(
_output_shapes
:����������
_
gradients/Sum_7_grad/ShapeShapePow_5*
T0*
out_type0*
_output_shapes
:
�
gradients/Sum_7_grad/SizeConst*
value	B :*-
_class#
!loc:@gradients/Sum_7_grad/Shape*
dtype0*
_output_shapes
: 
�
gradients/Sum_7_grad/addAddSum_7/reduction_indicesgradients/Sum_7_grad/Size*
T0*-
_class#
!loc:@gradients/Sum_7_grad/Shape*
_output_shapes
: 
�
gradients/Sum_7_grad/modFloorModgradients/Sum_7_grad/addgradients/Sum_7_grad/Size*
T0*-
_class#
!loc:@gradients/Sum_7_grad/Shape*
_output_shapes
: 
�
gradients/Sum_7_grad/Shape_1Const*
dtype0*
_output_shapes
: *
valueB *-
_class#
!loc:@gradients/Sum_7_grad/Shape
�
 gradients/Sum_7_grad/range/startConst*
dtype0*
_output_shapes
: *
value	B : *-
_class#
!loc:@gradients/Sum_7_grad/Shape
�
 gradients/Sum_7_grad/range/deltaConst*
dtype0*
_output_shapes
: *
value	B :*-
_class#
!loc:@gradients/Sum_7_grad/Shape
�
gradients/Sum_7_grad/rangeRange gradients/Sum_7_grad/range/startgradients/Sum_7_grad/Size gradients/Sum_7_grad/range/delta*
_output_shapes
:*

Tidx0*-
_class#
!loc:@gradients/Sum_7_grad/Shape
�
gradients/Sum_7_grad/Fill/valueConst*
dtype0*
_output_shapes
: *
value	B :*-
_class#
!loc:@gradients/Sum_7_grad/Shape
�
gradients/Sum_7_grad/FillFillgradients/Sum_7_grad/Shape_1gradients/Sum_7_grad/Fill/value*
T0*

index_type0*-
_class#
!loc:@gradients/Sum_7_grad/Shape*
_output_shapes
: 
�
"gradients/Sum_7_grad/DynamicStitchDynamicStitchgradients/Sum_7_grad/rangegradients/Sum_7_grad/modgradients/Sum_7_grad/Shapegradients/Sum_7_grad/Fill*
T0*-
_class#
!loc:@gradients/Sum_7_grad/Shape*
N*
_output_shapes
:
�
gradients/Sum_7_grad/Maximum/yConst*
value	B :*-
_class#
!loc:@gradients/Sum_7_grad/Shape*
dtype0*
_output_shapes
: 
�
gradients/Sum_7_grad/MaximumMaximum"gradients/Sum_7_grad/DynamicStitchgradients/Sum_7_grad/Maximum/y*
T0*-
_class#
!loc:@gradients/Sum_7_grad/Shape*
_output_shapes
:
�
gradients/Sum_7_grad/floordivFloorDivgradients/Sum_7_grad/Shapegradients/Sum_7_grad/Maximum*
T0*-
_class#
!loc:@gradients/Sum_7_grad/Shape*
_output_shapes
:
�
gradients/Sum_7_grad/ReshapeReshapegradients/Sqrt_5_grad/SqrtGrad"gradients/Sum_7_grad/DynamicStitch*
T0*
Tshape0*0
_output_shapes
:������������������
�
gradients/Sum_7_grad/TileTilegradients/Sum_7_grad/Reshapegradients/Sum_7_grad/floordiv*

Tmultiples0*
T0*(
_output_shapes
:����������
]
gradients/Pow_4_grad/ShapeShapesub*
T0*
out_type0*
_output_shapes
:
_
gradients/Pow_4_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
�
*gradients/Pow_4_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/Pow_4_grad/Shapegradients/Pow_4_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
v
gradients/Pow_4_grad/mulMulgradients/Sum_6_grad/TilePow_4/y*
T0*(
_output_shapes
:����������
_
gradients/Pow_4_grad/sub/yConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
e
gradients/Pow_4_grad/subSubPow_4/ygradients/Pow_4_grad/sub/y*
_output_shapes
: *
T0
q
gradients/Pow_4_grad/PowPowsubgradients/Pow_4_grad/sub*(
_output_shapes
:����������*
T0
�
gradients/Pow_4_grad/mul_1Mulgradients/Pow_4_grad/mulgradients/Pow_4_grad/Pow*(
_output_shapes
:����������*
T0
�
gradients/Pow_4_grad/SumSumgradients/Pow_4_grad/mul_1*gradients/Pow_4_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
gradients/Pow_4_grad/ReshapeReshapegradients/Pow_4_grad/Sumgradients/Pow_4_grad/Shape*
T0*
Tshape0*(
_output_shapes
:����������
c
gradients/Pow_4_grad/Greater/yConst*
dtype0*
_output_shapes
: *
valueB
 *    

gradients/Pow_4_grad/GreaterGreatersubgradients/Pow_4_grad/Greater/y*
T0*(
_output_shapes
:����������
g
$gradients/Pow_4_grad/ones_like/ShapeShapesub*
T0*
out_type0*
_output_shapes
:
i
$gradients/Pow_4_grad/ones_like/ConstConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
gradients/Pow_4_grad/ones_likeFill$gradients/Pow_4_grad/ones_like/Shape$gradients/Pow_4_grad/ones_like/Const*(
_output_shapes
:����������*
T0*

index_type0
�
gradients/Pow_4_grad/SelectSelectgradients/Pow_4_grad/Greatersubgradients/Pow_4_grad/ones_like*
T0*(
_output_shapes
:����������
o
gradients/Pow_4_grad/LogLoggradients/Pow_4_grad/Select*
T0*(
_output_shapes
:����������
d
gradients/Pow_4_grad/zeros_like	ZerosLikesub*
T0*(
_output_shapes
:����������
�
gradients/Pow_4_grad/Select_1Selectgradients/Pow_4_grad/Greatergradients/Pow_4_grad/Loggradients/Pow_4_grad/zeros_like*
T0*(
_output_shapes
:����������
v
gradients/Pow_4_grad/mul_2Mulgradients/Sum_6_grad/TilePow_4*
T0*(
_output_shapes
:����������
�
gradients/Pow_4_grad/mul_3Mulgradients/Pow_4_grad/mul_2gradients/Pow_4_grad/Select_1*
T0*(
_output_shapes
:����������
�
gradients/Pow_4_grad/Sum_1Sumgradients/Pow_4_grad/mul_3,gradients/Pow_4_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
gradients/Pow_4_grad/Reshape_1Reshapegradients/Pow_4_grad/Sum_1gradients/Pow_4_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
m
%gradients/Pow_4_grad/tuple/group_depsNoOp^gradients/Pow_4_grad/Reshape^gradients/Pow_4_grad/Reshape_1
�
-gradients/Pow_4_grad/tuple/control_dependencyIdentitygradients/Pow_4_grad/Reshape&^gradients/Pow_4_grad/tuple/group_deps*(
_output_shapes
:����������*
T0*/
_class%
#!loc:@gradients/Pow_4_grad/Reshape
�
/gradients/Pow_4_grad/tuple/control_dependency_1Identitygradients/Pow_4_grad/Reshape_1&^gradients/Pow_4_grad/tuple/group_deps*
_output_shapes
: *
T0*1
_class'
%#loc:@gradients/Pow_4_grad/Reshape_1
_
gradients/Pow_5_grad/ShapeShapesub_1*
T0*
out_type0*
_output_shapes
:
_
gradients/Pow_5_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
�
*gradients/Pow_5_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/Pow_5_grad/Shapegradients/Pow_5_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
v
gradients/Pow_5_grad/mulMulgradients/Sum_7_grad/TilePow_5/y*
T0*(
_output_shapes
:����������
_
gradients/Pow_5_grad/sub/yConst*
dtype0*
_output_shapes
: *
valueB
 *  �?
e
gradients/Pow_5_grad/subSubPow_5/ygradients/Pow_5_grad/sub/y*
_output_shapes
: *
T0
s
gradients/Pow_5_grad/PowPowsub_1gradients/Pow_5_grad/sub*
T0*(
_output_shapes
:����������
�
gradients/Pow_5_grad/mul_1Mulgradients/Pow_5_grad/mulgradients/Pow_5_grad/Pow*(
_output_shapes
:����������*
T0
�
gradients/Pow_5_grad/SumSumgradients/Pow_5_grad/mul_1*gradients/Pow_5_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
gradients/Pow_5_grad/ReshapeReshapegradients/Pow_5_grad/Sumgradients/Pow_5_grad/Shape*
T0*
Tshape0*(
_output_shapes
:����������
c
gradients/Pow_5_grad/Greater/yConst*
dtype0*
_output_shapes
: *
valueB
 *    
�
gradients/Pow_5_grad/GreaterGreatersub_1gradients/Pow_5_grad/Greater/y*(
_output_shapes
:����������*
T0
i
$gradients/Pow_5_grad/ones_like/ShapeShapesub_1*
T0*
out_type0*
_output_shapes
:
i
$gradients/Pow_5_grad/ones_like/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *  �?
�
gradients/Pow_5_grad/ones_likeFill$gradients/Pow_5_grad/ones_like/Shape$gradients/Pow_5_grad/ones_like/Const*(
_output_shapes
:����������*
T0*

index_type0
�
gradients/Pow_5_grad/SelectSelectgradients/Pow_5_grad/Greatersub_1gradients/Pow_5_grad/ones_like*
T0*(
_output_shapes
:����������
o
gradients/Pow_5_grad/LogLoggradients/Pow_5_grad/Select*
T0*(
_output_shapes
:����������
f
gradients/Pow_5_grad/zeros_like	ZerosLikesub_1*
T0*(
_output_shapes
:����������
�
gradients/Pow_5_grad/Select_1Selectgradients/Pow_5_grad/Greatergradients/Pow_5_grad/Loggradients/Pow_5_grad/zeros_like*
T0*(
_output_shapes
:����������
v
gradients/Pow_5_grad/mul_2Mulgradients/Sum_7_grad/TilePow_5*(
_output_shapes
:����������*
T0
�
gradients/Pow_5_grad/mul_3Mulgradients/Pow_5_grad/mul_2gradients/Pow_5_grad/Select_1*
T0*(
_output_shapes
:����������
�
gradients/Pow_5_grad/Sum_1Sumgradients/Pow_5_grad/mul_3,gradients/Pow_5_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
gradients/Pow_5_grad/Reshape_1Reshapegradients/Pow_5_grad/Sum_1gradients/Pow_5_grad/Shape_1*
_output_shapes
: *
T0*
Tshape0
m
%gradients/Pow_5_grad/tuple/group_depsNoOp^gradients/Pow_5_grad/Reshape^gradients/Pow_5_grad/Reshape_1
�
-gradients/Pow_5_grad/tuple/control_dependencyIdentitygradients/Pow_5_grad/Reshape&^gradients/Pow_5_grad/tuple/group_deps*(
_output_shapes
:����������*
T0*/
_class%
#!loc:@gradients/Pow_5_grad/Reshape
�
/gradients/Pow_5_grad/tuple/control_dependency_1Identitygradients/Pow_5_grad/Reshape_1&^gradients/Pow_5_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/Pow_5_grad/Reshape_1*
_output_shapes
: 
u
gradients/sub_grad/ShapeShapemodel/Flatten/flatten/Reshape*
_output_shapes
:*
T0*
out_type0
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
gradients/sub_grad/SumSum-gradients/Pow_4_grad/tuple/control_dependency(gradients/sub_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
gradients/sub_grad/ReshapeReshapegradients/sub_grad/Sumgradients/sub_grad/Shape*(
_output_shapes
:����������*
T0*
Tshape0
�
gradients/sub_grad/Sum_1Sum-gradients/Pow_4_grad/tuple/control_dependency*gradients/sub_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
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
gradients/sub_1_grad/SumSum-gradients/Pow_5_grad/tuple/control_dependency*gradients/sub_1_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
gradients/sub_1_grad/ReshapeReshapegradients/sub_1_grad/Sumgradients/sub_1_grad/Shape*(
_output_shapes
:����������*
T0*
Tshape0
�
gradients/sub_1_grad/Sum_1Sum-gradients/Pow_5_grad/tuple/control_dependency,gradients/sub_1_grad/BroadcastGradientArgs:1*
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
Tshape0*(
_output_shapes
:����������
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
/gradients/sub_1_grad/tuple/control_dependency_1Identitygradients/sub_1_grad/Reshape_1&^gradients/sub_1_grad/tuple/group_deps*(
_output_shapes
:����������*
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
:���������
�
gradients/AddNAddN+gradients/sub_grad/tuple/control_dependency-gradients/sub_1_grad/tuple/control_dependency*
T0*-
_class#
!loc:@gradients/sub_grad/Reshape*
N*(
_output_shapes
:����������
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
:gradients/model_1/conv5/MaxPool2D/MaxPool_grad/MaxPoolGradMaxPoolGradmodel_1/conv5/BiasAddmodel_1/conv5/MaxPool2D/MaxPool6gradients/model_1/Flatten/flatten/Reshape_grad/Reshape*/
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
=gradients/model_1/conv5/BiasAdd_grad/tuple/control_dependencyIdentity:gradients/model_1/conv5/MaxPool2D/MaxPool_grad/MaxPoolGrad6^gradients/model_1/conv5/BiasAdd_grad/tuple/group_deps*
T0*M
_classC
A?loc:@gradients/model_1/conv5/MaxPool2D/MaxPool_grad/MaxPoolGrad*/
_output_shapes
:���������

�
?gradients/model_1/conv5/BiasAdd_grad/tuple/control_dependency_1Identity0gradients/model_1/conv5/BiasAdd_grad/BiasAddGrad6^gradients/model_1/conv5/BiasAdd_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/model_1/conv5/BiasAdd_grad/BiasAddGrad*
_output_shapes
:
�
.gradients/model/conv5/BiasAdd_grad/BiasAddGradBiasAddGrad8gradients/model/conv5/MaxPool2D/MaxPool_grad/MaxPoolGrad*
data_formatNHWC*
_output_shapes
:*
T0
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
N* 
_output_shapes
::*
T0*
out_type0
�
7gradients/model_1/conv5/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput*gradients/model_1/conv5/Conv2D_grad/ShapeNconv5/weights/read=gradients/model_1/conv5/BiasAdd_grad/tuple/control_dependency*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME*0
_output_shapes
:���������
�
�
8gradients/model_1/conv5/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFiltermodel_1/conv4/MaxPool2D/MaxPool,gradients/model_1/conv5/Conv2D_grad/ShapeN:1=gradients/model_1/conv5/BiasAdd_grad/tuple/control_dependency*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME*'
_output_shapes
:�*
	dilations
*
T0
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
>gradients/model_1/conv5/Conv2D_grad/tuple/control_dependency_1Identity8gradients/model_1/conv5/Conv2D_grad/Conv2DBackpropFilter5^gradients/model_1/conv5/Conv2D_grad/tuple/group_deps*'
_output_shapes
:�*
T0*K
_classA
?=loc:@gradients/model_1/conv5/Conv2D_grad/Conv2DBackpropFilter
�
(gradients/model/conv5/Conv2D_grad/ShapeNShapeNmodel/conv4/MaxPool2D/MaxPoolconv5/weights/read*
N* 
_output_shapes
::*
T0*
out_type0
�
5gradients/model/conv5/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput(gradients/model/conv5/Conv2D_grad/ShapeNconv5/weights/read;gradients/model/conv5/BiasAdd_grad/tuple/control_dependency*
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
6gradients/model/conv5/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFiltermodel/conv4/MaxPool2D/MaxPool*gradients/model/conv5/Conv2D_grad/ShapeN:1;gradients/model/conv5/BiasAdd_grad/tuple/control_dependency*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME*'
_output_shapes
:�
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
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME*0
_output_shapes
:���������
�
�
8gradients/model_2/conv5/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFiltermodel_2/conv4/MaxPool2D/MaxPool,gradients/model_2/conv5/Conv2D_grad/ShapeN:1=gradients/model_2/conv5/BiasAdd_grad/tuple/control_dependency*
T0*
data_formatNHWC*
strides
*
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
gradients/AddN_1AddN?gradients/model_1/conv5/BiasAdd_grad/tuple/control_dependency_1=gradients/model/conv5/BiasAdd_grad/tuple/control_dependency_1?gradients/model_2/conv5/BiasAdd_grad/tuple/control_dependency_1*
T0*C
_class9
75loc:@gradients/model_1/conv5/BiasAdd_grad/BiasAddGrad*
N*
_output_shapes
:
�
:gradients/model_1/conv4/MaxPool2D/MaxPool_grad/MaxPoolGradMaxPoolGradmodel_1/conv4/conv4/Relumodel_1/conv4/MaxPool2D/MaxPool<gradients/model_1/conv5/Conv2D_grad/tuple/control_dependency*0
_output_shapes
:����������*
T0*
data_formatNHWC*
strides
*
ksize
*
paddingSAME
�
8gradients/model/conv4/MaxPool2D/MaxPool_grad/MaxPoolGradMaxPoolGradmodel/conv4/conv4/Relumodel/conv4/MaxPool2D/MaxPool:gradients/model/conv5/Conv2D_grad/tuple/control_dependency*
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
:gradients/model_2/conv4/MaxPool2D/MaxPool_grad/MaxPoolGradMaxPoolGradmodel_2/conv4/conv4/Relumodel_2/conv4/MaxPool2D/MaxPool<gradients/model_2/conv5/Conv2D_grad/tuple/control_dependency*
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
:����������*
T0*C
_class9
75loc:@gradients/model_1/conv4/conv4/Relu_grad/ReluGrad
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
T0*
out_type0*
N* 
_output_shapes
::
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
<gradients/model_1/conv4/Conv2D_grad/tuple/control_dependencyIdentity7gradients/model_1/conv4/Conv2D_grad/Conv2DBackpropInput5^gradients/model_1/conv4/Conv2D_grad/tuple/group_deps*0
_output_shapes
:����������*
T0*J
_class@
><loc:@gradients/model_1/conv4/Conv2D_grad/Conv2DBackpropInput
�
>gradients/model_1/conv4/Conv2D_grad/tuple/control_dependency_1Identity8gradients/model_1/conv4/Conv2D_grad/Conv2DBackpropFilter5^gradients/model_1/conv4/Conv2D_grad/tuple/group_deps*
T0*K
_classA
?=loc:@gradients/model_1/conv4/Conv2D_grad/Conv2DBackpropFilter*(
_output_shapes
:��
�
(gradients/model/conv4/Conv2D_grad/ShapeNShapeNmodel/conv3/MaxPool2D/MaxPoolconv4/weights/read*
T0*
out_type0*
N* 
_output_shapes
::
�
5gradients/model/conv4/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput(gradients/model/conv4/Conv2D_grad/ShapeNconv4/weights/read;gradients/model/conv4/BiasAdd_grad/tuple/control_dependency*
paddingSAME*0
_output_shapes
:����������*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(
�
6gradients/model/conv4/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFiltermodel/conv3/MaxPool2D/MaxPool*gradients/model/conv4/Conv2D_grad/ShapeN:1;gradients/model/conv4/BiasAdd_grad/tuple/control_dependency*(
_output_shapes
:��*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME
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
<gradients/model/conv4/Conv2D_grad/tuple/control_dependency_1Identity6gradients/model/conv4/Conv2D_grad/Conv2DBackpropFilter3^gradients/model/conv4/Conv2D_grad/tuple/group_deps*(
_output_shapes
:��*
T0*I
_class?
=;loc:@gradients/model/conv4/Conv2D_grad/Conv2DBackpropFilter
�
*gradients/model_2/conv4/Conv2D_grad/ShapeNShapeNmodel_2/conv3/MaxPool2D/MaxPoolconv4/weights/read*
N* 
_output_shapes
::*
T0*
out_type0
�
7gradients/model_2/conv4/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput*gradients/model_2/conv4/Conv2D_grad/ShapeNconv4/weights/read=gradients/model_2/conv4/BiasAdd_grad/tuple/control_dependency*0
_output_shapes
:����������*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME
�
8gradients/model_2/conv4/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFiltermodel_2/conv3/MaxPool2D/MaxPool,gradients/model_2/conv4/Conv2D_grad/ShapeN:1=gradients/model_2/conv4/BiasAdd_grad/tuple/control_dependency*(
_output_shapes
:��*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME
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
:gradients/model_2/conv3/MaxPool2D/MaxPool_grad/MaxPoolGradMaxPoolGradmodel_2/conv3/conv3/Relumodel_2/conv3/MaxPool2D/MaxPool<gradients/model_2/conv4/Conv2D_grad/tuple/control_dependency*
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
gradients/AddN_4AddN>gradients/model_1/conv4/Conv2D_grad/tuple/control_dependency_1<gradients/model/conv4/Conv2D_grad/tuple/control_dependency_1>gradients/model_2/conv4/Conv2D_grad/tuple/control_dependency_1*
T0*K
_classA
?=loc:@gradients/model_1/conv4/Conv2D_grad/Conv2DBackpropFilter*
N*(
_output_shapes
:��
�
0gradients/model_1/conv3/conv3/Relu_grad/ReluGradReluGrad:gradients/model_1/conv3/MaxPool2D/MaxPool_grad/MaxPoolGradmodel_1/conv3/conv3/Relu*0
_output_shapes
:���������&�*
T0
�
.gradients/model/conv3/conv3/Relu_grad/ReluGradReluGrad8gradients/model/conv3/MaxPool2D/MaxPool_grad/MaxPoolGradmodel/conv3/conv3/Relu*
T0*0
_output_shapes
:���������&�
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
=gradients/model_1/conv3/BiasAdd_grad/tuple/control_dependencyIdentity0gradients/model_1/conv3/conv3/Relu_grad/ReluGrad6^gradients/model_1/conv3/BiasAdd_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/model_1/conv3/conv3/Relu_grad/ReluGrad*0
_output_shapes
:���������&�
�
?gradients/model_1/conv3/BiasAdd_grad/tuple/control_dependency_1Identity0gradients/model_1/conv3/BiasAdd_grad/BiasAddGrad6^gradients/model_1/conv3/BiasAdd_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/model_1/conv3/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:�
�
.gradients/model/conv3/BiasAdd_grad/BiasAddGradBiasAddGrad.gradients/model/conv3/conv3/Relu_grad/ReluGrad*
data_formatNHWC*
_output_shapes	
:�*
T0
�
3gradients/model/conv3/BiasAdd_grad/tuple/group_depsNoOp/^gradients/model/conv3/BiasAdd_grad/BiasAddGrad/^gradients/model/conv3/conv3/Relu_grad/ReluGrad
�
;gradients/model/conv3/BiasAdd_grad/tuple/control_dependencyIdentity.gradients/model/conv3/conv3/Relu_grad/ReluGrad4^gradients/model/conv3/BiasAdd_grad/tuple/group_deps*0
_output_shapes
:���������&�*
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
0gradients/model_2/conv3/BiasAdd_grad/BiasAddGradBiasAddGrad0gradients/model_2/conv3/conv3/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes	
:�
�
5gradients/model_2/conv3/BiasAdd_grad/tuple/group_depsNoOp1^gradients/model_2/conv3/BiasAdd_grad/BiasAddGrad1^gradients/model_2/conv3/conv3/Relu_grad/ReluGrad
�
=gradients/model_2/conv3/BiasAdd_grad/tuple/control_dependencyIdentity0gradients/model_2/conv3/conv3/Relu_grad/ReluGrad6^gradients/model_2/conv3/BiasAdd_grad/tuple/group_deps*0
_output_shapes
:���������&�*
T0*C
_class9
75loc:@gradients/model_2/conv3/conv3/Relu_grad/ReluGrad
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
7gradients/model_1/conv3/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput*gradients/model_1/conv3/Conv2D_grad/ShapeNconv3/weights/read=gradients/model_1/conv3/BiasAdd_grad/tuple/control_dependency*
paddingSAME*/
_output_shapes
:���������&@*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
�
8gradients/model_1/conv3/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFiltermodel_1/conv2/MaxPool2D/MaxPool,gradients/model_1/conv3/Conv2D_grad/ShapeN:1=gradients/model_1/conv3/BiasAdd_grad/tuple/control_dependency*
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
4gradients/model_1/conv3/Conv2D_grad/tuple/group_depsNoOp9^gradients/model_1/conv3/Conv2D_grad/Conv2DBackpropFilter8^gradients/model_1/conv3/Conv2D_grad/Conv2DBackpropInput
�
<gradients/model_1/conv3/Conv2D_grad/tuple/control_dependencyIdentity7gradients/model_1/conv3/Conv2D_grad/Conv2DBackpropInput5^gradients/model_1/conv3/Conv2D_grad/tuple/group_deps*
T0*J
_class@
><loc:@gradients/model_1/conv3/Conv2D_grad/Conv2DBackpropInput*/
_output_shapes
:���������&@
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
:gradients/model/conv3/Conv2D_grad/tuple/control_dependencyIdentity5gradients/model/conv3/Conv2D_grad/Conv2DBackpropInput3^gradients/model/conv3/Conv2D_grad/tuple/group_deps*/
_output_shapes
:���������&@*
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
N* 
_output_shapes
::*
T0*
out_type0
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
<gradients/model_2/conv3/Conv2D_grad/tuple/control_dependencyIdentity7gradients/model_2/conv3/Conv2D_grad/Conv2DBackpropInput5^gradients/model_2/conv3/Conv2D_grad/tuple/group_deps*/
_output_shapes
:���������&@*
T0*J
_class@
><loc:@gradients/model_2/conv3/Conv2D_grad/Conv2DBackpropInput
�
>gradients/model_2/conv3/Conv2D_grad/tuple/control_dependency_1Identity8gradients/model_2/conv3/Conv2D_grad/Conv2DBackpropFilter5^gradients/model_2/conv3/Conv2D_grad/tuple/group_deps*
T0*K
_classA
?=loc:@gradients/model_2/conv3/Conv2D_grad/Conv2DBackpropFilter*'
_output_shapes
:@�
�
gradients/AddN_5AddN?gradients/model_1/conv3/BiasAdd_grad/tuple/control_dependency_1=gradients/model/conv3/BiasAdd_grad/tuple/control_dependency_1?gradients/model_2/conv3/BiasAdd_grad/tuple/control_dependency_1*
N*
_output_shapes	
:�*
T0*C
_class9
75loc:@gradients/model_1/conv3/BiasAdd_grad/BiasAddGrad
�
:gradients/model_1/conv2/MaxPool2D/MaxPool_grad/MaxPoolGradMaxPoolGradmodel_1/conv2/conv2/Relumodel_1/conv2/MaxPool2D/MaxPool<gradients/model_1/conv3/Conv2D_grad/tuple/control_dependency*/
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
:gradients/model_2/conv2/MaxPool2D/MaxPool_grad/MaxPoolGradMaxPoolGradmodel_2/conv2/conv2/Relumodel_2/conv2/MaxPool2D/MaxPool<gradients/model_2/conv3/Conv2D_grad/tuple/control_dependency*
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
gradients/AddN_6AddN>gradients/model_1/conv3/Conv2D_grad/tuple/control_dependency_1<gradients/model/conv3/Conv2D_grad/tuple/control_dependency_1>gradients/model_2/conv3/Conv2D_grad/tuple/control_dependency_1*'
_output_shapes
:@�*
T0*K
_classA
?=loc:@gradients/model_1/conv3/Conv2D_grad/Conv2DBackpropFilter*
N
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
?gradients/model_1/conv2/BiasAdd_grad/tuple/control_dependency_1Identity0gradients/model_1/conv2/BiasAdd_grad/BiasAddGrad6^gradients/model_1/conv2/BiasAdd_grad/tuple/group_deps*C
_class9
75loc:@gradients/model_1/conv2/BiasAdd_grad/BiasAddGrad*
_output_shapes
:@*
T0
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
:���������K@*
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
:���������K@
�
?gradients/model_2/conv2/BiasAdd_grad/tuple/control_dependency_1Identity0gradients/model_2/conv2/BiasAdd_grad/BiasAddGrad6^gradients/model_2/conv2/BiasAdd_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/model_2/conv2/BiasAdd_grad/BiasAddGrad*
_output_shapes
:@
�
*gradients/model_1/conv2/Conv2D_grad/ShapeNShapeNmodel_1/conv1/MaxPool2D/MaxPoolconv2/weights/read*
T0*
out_type0*
N* 
_output_shapes
::
�
7gradients/model_1/conv2/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput*gradients/model_1/conv2/Conv2D_grad/ShapeNconv2/weights/read=gradients/model_1/conv2/BiasAdd_grad/tuple/control_dependency*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME*/
_output_shapes
:���������K *
	dilations
*
T0
�
8gradients/model_1/conv2/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFiltermodel_1/conv1/MaxPool2D/MaxPool,gradients/model_1/conv2/Conv2D_grad/ShapeN:1=gradients/model_1/conv2/BiasAdd_grad/tuple/control_dependency*
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
4gradients/model_1/conv2/Conv2D_grad/tuple/group_depsNoOp9^gradients/model_1/conv2/Conv2D_grad/Conv2DBackpropFilter8^gradients/model_1/conv2/Conv2D_grad/Conv2DBackpropInput
�
<gradients/model_1/conv2/Conv2D_grad/tuple/control_dependencyIdentity7gradients/model_1/conv2/Conv2D_grad/Conv2DBackpropInput5^gradients/model_1/conv2/Conv2D_grad/tuple/group_deps*/
_output_shapes
:���������K *
T0*J
_class@
><loc:@gradients/model_1/conv2/Conv2D_grad/Conv2DBackpropInput
�
>gradients/model_1/conv2/Conv2D_grad/tuple/control_dependency_1Identity8gradients/model_1/conv2/Conv2D_grad/Conv2DBackpropFilter5^gradients/model_1/conv2/Conv2D_grad/tuple/group_deps*K
_classA
?=loc:@gradients/model_1/conv2/Conv2D_grad/Conv2DBackpropFilter*&
_output_shapes
: @*
T0
�
(gradients/model/conv2/Conv2D_grad/ShapeNShapeNmodel/conv1/MaxPool2D/MaxPoolconv2/weights/read*
N* 
_output_shapes
::*
T0*
out_type0
�
5gradients/model/conv2/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput(gradients/model/conv2/Conv2D_grad/ShapeNconv2/weights/read;gradients/model/conv2/BiasAdd_grad/tuple/control_dependency*
paddingSAME*/
_output_shapes
:���������K *
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(
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
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME*/
_output_shapes
:���������K *
	dilations

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
:���������K 
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
:gradients/model_1/conv1/MaxPool2D/MaxPool_grad/MaxPoolGradMaxPoolGradmodel_1/conv1/conv1/Relumodel_1/conv1/MaxPool2D/MaxPool<gradients/model_1/conv2/Conv2D_grad/tuple/control_dependency*0
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
:gradients/model_2/conv1/MaxPool2D/MaxPool_grad/MaxPoolGradMaxPoolGradmodel_2/conv1/conv1/Relumodel_2/conv1/MaxPool2D/MaxPool<gradients/model_2/conv2/Conv2D_grad/tuple/control_dependency*
data_formatNHWC*
strides
*
ksize
*
paddingSAME*0
_output_shapes
:���������2� *
T0
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
=gradients/model_1/conv1/BiasAdd_grad/tuple/control_dependencyIdentity0gradients/model_1/conv1/conv1/Relu_grad/ReluGrad6^gradients/model_1/conv1/BiasAdd_grad/tuple/group_deps*0
_output_shapes
:���������2� *
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
53loc:@gradients/model/conv1/conv1/Relu_grad/ReluGrad*0
_output_shapes
:���������2� 
�
=gradients/model/conv1/BiasAdd_grad/tuple/control_dependency_1Identity.gradients/model/conv1/BiasAdd_grad/BiasAddGrad4^gradients/model/conv1/BiasAdd_grad/tuple/group_deps*
_output_shapes
: *
T0*A
_class7
53loc:@gradients/model/conv1/BiasAdd_grad/BiasAddGrad
�
0gradients/model_2/conv1/BiasAdd_grad/BiasAddGradBiasAddGrad0gradients/model_2/conv1/conv1/Relu_grad/ReluGrad*
_output_shapes
: *
T0*
data_formatNHWC
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
7gradients/model_1/conv1/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput*gradients/model_1/conv1/Conv2D_grad/ShapeNconv1/weights/read=gradients/model_1/conv1/BiasAdd_grad/tuple/control_dependency*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME*0
_output_shapes
:���������2�*
	dilations
*
T0
�
8gradients/model_1/conv1/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterpositive_input,gradients/model_1/conv1/Conv2D_grad/ShapeN:1=gradients/model_1/conv1/BiasAdd_grad/tuple/control_dependency*
	dilations
*
T0*
data_formatNHWC*
strides
*
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
><loc:@gradients/model_1/conv1/Conv2D_grad/Conv2DBackpropInput*0
_output_shapes
:���������2�
�
>gradients/model_1/conv1/Conv2D_grad/tuple/control_dependency_1Identity8gradients/model_1/conv1/Conv2D_grad/Conv2DBackpropFilter5^gradients/model_1/conv1/Conv2D_grad/tuple/group_deps*K
_classA
?=loc:@gradients/model_1/conv1/Conv2D_grad/Conv2DBackpropFilter*&
_output_shapes
: *
T0
�
(gradients/model/conv1/Conv2D_grad/ShapeNShapeNanchor_inputconv1/weights/read*
T0*
out_type0*
N* 
_output_shapes
::
�
5gradients/model/conv1/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput(gradients/model/conv1/Conv2D_grad/ShapeNconv1/weights/read;gradients/model/conv1/BiasAdd_grad/tuple/control_dependency*0
_output_shapes
:���������2�*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME
�
6gradients/model/conv1/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilteranchor_input*gradients/model/conv1/Conv2D_grad/ShapeN:1;gradients/model/conv1/BiasAdd_grad/tuple/control_dependency*
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
*gradients/model_2/conv1/Conv2D_grad/ShapeNShapeNnegative_inputconv1/weights/read* 
_output_shapes
::*
T0*
out_type0*
N
�
7gradients/model_2/conv1/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput*gradients/model_2/conv1/Conv2D_grad/ShapeNconv1/weights/read=gradients/model_2/conv1/BiasAdd_grad/tuple/control_dependency*
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
8gradients/model_2/conv1/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilternegative_input,gradients/model_2/conv1/Conv2D_grad/ShapeN:1=gradients/model_2/conv1/BiasAdd_grad/tuple/control_dependency*&
_output_shapes
: *
	dilations
*
T0*
data_formatNHWC*
strides
*
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
8conv1/weights/Momentum/Initializer/zeros/shape_as_tensorConst*
_output_shapes
:* 
_class
loc:@conv1/weights*%
valueB"             *
dtype0
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
'conv1/biases/Momentum/Initializer/zerosConst*
_class
loc:@conv1/biases*
valueB *    *
dtype0*
_output_shapes
: 
�
conv1/biases/Momentum
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
(conv2/weights/Momentum/Initializer/zerosFill8conv2/weights/Momentum/Initializer/zeros/shape_as_tensor.conv2/weights/Momentum/Initializer/zeros/Const*&
_output_shapes
: @*
T0* 
_class
loc:@conv2/weights*

index_type0
�
conv2/weights/Momentum
VariableV2*
shape: @*
dtype0*&
_output_shapes
: @*
shared_name * 
_class
loc:@conv2/weights*
	container 
�
conv2/weights/Momentum/AssignAssignconv2/weights/Momentum(conv2/weights/Momentum/Initializer/zeros*
validate_shape(*&
_output_shapes
: @*
use_locking(*
T0* 
_class
loc:@conv2/weights
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
conv2/biases/Momentum/AssignAssignconv2/biases/Momentum'conv2/biases/Momentum/Initializer/zeros*
T0*
_class
loc:@conv2/biases*
validate_shape(*
_output_shapes
:@*
use_locking(
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
conv3/weights/Momentum/readIdentityconv3/weights/Momentum*'
_output_shapes
:@�*
T0* 
_class
loc:@conv3/weights
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
conv4/biases/Momentum/AssignAssignconv4/biases/Momentum'conv4/biases/Momentum/Initializer/zeros*
use_locking(*
T0*
_class
loc:@conv4/biases*
validate_shape(*
_output_shapes	
:�
�
conv4/biases/Momentum/readIdentityconv4/biases/Momentum*
T0*
_class
loc:@conv4/biases*
_output_shapes	
:�
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
(conv5/weights/Momentum/Initializer/zerosFill8conv5/weights/Momentum/Initializer/zeros/shape_as_tensor.conv5/weights/Momentum/Initializer/zeros/Const* 
_class
loc:@conv5/weights*

index_type0*'
_output_shapes
:�*
T0
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
conv5/weights/Momentum/AssignAssignconv5/weights/Momentum(conv5/weights/Momentum/Initializer/zeros*'
_output_shapes
:�*
use_locking(*
T0* 
_class
loc:@conv5/weights*
validate_shape(
�
conv5/weights/Momentum/readIdentityconv5/weights/Momentum*
T0* 
_class
loc:@conv5/weights*'
_output_shapes
:�
�
'conv5/biases/Momentum/Initializer/zerosConst*
_output_shapes
:*
_class
loc:@conv5/biases*
valueB*    *
dtype0
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
conv5/biases/Momentum/AssignAssignconv5/biases/Momentum'conv5/biases/Momentum/Initializer/zeros*
use_locking(*
T0*
_class
loc:@conv5/biases*
validate_shape(*
_output_shapes
:
�
conv5/biases/Momentum/readIdentityconv5/biases/Momentum*
_class
loc:@conv5/biases*
_output_shapes
:*
T0
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
+Momentum/update_conv1/weights/ApplyMomentumApplyMomentumconv1/weightsconv1/weights/MomentumMomentum/learning_rategradients/AddN_10Momentum/momentum*
use_nesterov(*&
_output_shapes
: *
use_locking( *
T0* 
_class
loc:@conv1/weights
�
*Momentum/update_conv1/biases/ApplyMomentumApplyMomentumconv1/biasesconv1/biases/MomentumMomentum/learning_rategradients/AddN_9Momentum/momentum*
_output_shapes
: *
use_locking( *
T0*
_class
loc:@conv1/biases*
use_nesterov(
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
*Momentum/update_conv2/biases/ApplyMomentumApplyMomentumconv2/biasesconv2/biases/MomentumMomentum/learning_rategradients/AddN_7Momentum/momentum*
_output_shapes
:@*
use_locking( *
T0*
_class
loc:@conv2/biases*
use_nesterov(
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
+Momentum/update_conv4/weights/ApplyMomentumApplyMomentumconv4/weightsconv4/weights/MomentumMomentum/learning_rategradients/AddN_4Momentum/momentum* 
_class
loc:@conv4/weights*
use_nesterov(*(
_output_shapes
:��*
use_locking( *
T0
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
Momentum/valueConst^Momentum/update*
value	B :*
_class
loc:@Variable*
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
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices"/device:CPU:0*h
_output_shapesV
T:::::::::::::::::::::*#
dtypes
2
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
save/Assign_6Assignconv2/biases/Momentumsave/RestoreV2:6*
_class
loc:@conv2/biases*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0
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
save/Assign_8Assignconv2/weights/Momentumsave/RestoreV2:8*
T0* 
_class
loc:@conv2/weights*
validate_shape(*&
_output_shapes
: @*
use_locking(
�
save/Assign_9Assignconv3/biasessave/RestoreV2:9*
_output_shapes	
:�*
use_locking(*
T0*
_class
loc:@conv3/biases*
validate_shape(
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
save/Assign_12Assignconv3/weights/Momentumsave/RestoreV2:12* 
_class
loc:@conv3/weights*
validate_shape(*'
_output_shapes
:@�*
use_locking(*
T0
�
save/Assign_13Assignconv4/biasessave/RestoreV2:13*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0*
_class
loc:@conv4/biases
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
save/Assign_15Assignconv4/weightssave/RestoreV2:15* 
_class
loc:@conv4/weights*
validate_shape(*(
_output_shapes
:��*
use_locking(*
T0
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
conv2/weights_1HistogramSummaryconv2/weights_1/tagconv2/weights/read*
_output_shapes
: *
T0
a
conv2/biases_1/tagConst*
_output_shapes
: *
valueB Bconv2/biases_1*
dtype0
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
dtype0*
_output_shapes
: *
valueB Bconv3/biases_1
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
conv4/weights_1HistogramSummaryconv4/weights_1/tagconv4/weights/read*
T0*
_output_shapes
: 
a
conv4/biases_1/tagConst*
dtype0*
_output_shapes
: *
valueB Bconv4/biases_1
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
: "Q$
5     ��B�	~���(��AJ��
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
Ttype*1.13.12v1.13.0-rc2-5-g6612da8951��
�
anchor_inputPlaceholder*%
shape:���������2�*
dtype0*0
_output_shapes
:���������2�
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
conv1/biases/AssignAssignconv1/biasesconv1/biases/Initializer/zeros*
T0*
_class
loc:@conv1/biases*
validate_shape(*
_output_shapes
: *
use_locking(
q
conv1/biases/readIdentityconv1/biases*
_output_shapes
: *
T0*
_class
loc:@conv1/biases
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
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME*0
_output_shapes
:���������2� 
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
model/conv1/MaxPool2D/MaxPoolMaxPoolmodel/conv1/conv1/Relu*/
_output_shapes
:���������K *
T0*
data_formatNHWC*
strides
*
ksize
*
paddingSAME
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
VariableV2*
shared_name *
_class
loc:@conv2/biases*
	container *
shape:@*
dtype0*
_output_shapes
:@
�
conv2/biases/AssignAssignconv2/biasesconv2/biases/Initializer/zeros*
use_locking(*
T0*
_class
loc:@conv2/biases*
validate_shape(*
_output_shapes
:@
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
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME*/
_output_shapes
:���������K@*
	dilations
*
T0
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
,conv3/weights/Initializer/random_uniform/subSub,conv3/weights/Initializer/random_uniform/max,conv3/weights/Initializer/random_uniform/min*
_output_shapes
: *
T0* 
_class
loc:@conv3/weights
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
conv3/weights/AssignAssignconv3/weights(conv3/weights/Initializer/random_uniform*
validate_shape(*'
_output_shapes
:@�*
use_locking(*
T0* 
_class
loc:@conv3/weights
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
dtype0*
_output_shapes	
:�*
shared_name *
_class
loc:@conv3/biases*
	container *
shape:�
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
model/conv3/Conv2DConv2Dmodel/conv2/MaxPool2D/MaxPoolconv3/weights/read*
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
_class
loc:@conv4/weights*%
valueB"      �      *
dtype0*
_output_shapes
:
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
dtype0*
_output_shapes
:*
valueB"      
�
model/conv4/Conv2DConv2Dmodel/conv3/MaxPool2D/MaxPoolconv4/weights/read*
paddingSAME*0
_output_shapes
:����������*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(
�
model/conv4/BiasAddBiasAddmodel/conv4/Conv2Dconv4/biases/read*
T0*
data_formatNHWC*0
_output_shapes
:����������
n
model/conv4/conv4/ReluRelumodel/conv4/BiasAdd*0
_output_shapes
:����������*
T0
�
model/conv4/MaxPool2D/MaxPoolMaxPoolmodel/conv4/conv4/Relu*
ksize
*
paddingSAME*0
_output_shapes
:���������
�*
T0*
strides
*
data_formatNHWC
�
.conv5/weights/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:* 
_class
loc:@conv5/weights*%
valueB"            
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
6conv5/weights/Initializer/random_uniform/RandomUniformRandomUniform.conv5/weights/Initializer/random_uniform/shape*
T0* 
_class
loc:@conv5/weights*
seed2 *
dtype0*'
_output_shapes
:�*

seed 
�
,conv5/weights/Initializer/random_uniform/subSub,conv5/weights/Initializer/random_uniform/max,conv5/weights/Initializer/random_uniform/min*
T0* 
_class
loc:@conv5/weights*
_output_shapes
: 
�
,conv5/weights/Initializer/random_uniform/mulMul6conv5/weights/Initializer/random_uniform/RandomUniform,conv5/weights/Initializer/random_uniform/sub*'
_output_shapes
:�*
T0* 
_class
loc:@conv5/weights
�
(conv5/weights/Initializer/random_uniformAdd,conv5/weights/Initializer/random_uniform/mul,conv5/weights/Initializer/random_uniform/min*
T0* 
_class
loc:@conv5/weights*'
_output_shapes
:�
�
conv5/weights
VariableV2*
shape:�*
dtype0*'
_output_shapes
:�*
shared_name * 
_class
loc:@conv5/weights*
	container 
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
conv5/weights/readIdentityconv5/weights*'
_output_shapes
:�*
T0* 
_class
loc:@conv5/weights
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
model/conv5/Conv2DConv2Dmodel/conv4/MaxPool2D/MaxPoolconv5/weights/read*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME*/
_output_shapes
:���������
*
	dilations

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
+model/Flatten/flatten/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
�
#model/Flatten/flatten/strided_sliceStridedSlicemodel/Flatten/flatten/Shape)model/Flatten/flatten/strided_slice/stack+model/Flatten/flatten/strided_slice/stack_1+model/Flatten/flatten/strided_slice/stack_2*
Index0*
T0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: 
p
%model/Flatten/flatten/Reshape/shape/1Const*
dtype0*
_output_shapes
: *
valueB :
���������
�
#model/Flatten/flatten/Reshape/shapePack#model/Flatten/flatten/strided_slice%model/Flatten/flatten/Reshape/shape/1*
T0*

axis *
N*
_output_shapes
:
�
model/Flatten/flatten/ReshapeReshapemodel/conv5/MaxPool2D/MaxPool#model/Flatten/flatten/Reshape/shape*(
_output_shapes
:����������*
T0*
Tshape0
l
model_1/conv1/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
�
model_1/conv1/Conv2DConv2Dpositive_inputconv1/weights/read*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME*0
_output_shapes
:���������2� *
	dilations
*
T0
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
model_1/conv1/MaxPool2D/MaxPoolMaxPoolmodel_1/conv1/conv1/Relu*/
_output_shapes
:���������K *
T0*
data_formatNHWC*
strides
*
ksize
*
paddingSAME
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
T0*
data_formatNHWC*
strides
*
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
model_1/conv2/conv2/ReluRelumodel_1/conv2/BiasAdd*/
_output_shapes
:���������K@*
T0
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
dtype0*
_output_shapes
:*
valueB"      
�
model_1/conv3/Conv2DConv2Dmodel_1/conv2/MaxPool2D/MaxPoolconv3/weights/read*
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
model_1/conv3/MaxPool2D/MaxPoolMaxPoolmodel_1/conv3/conv3/Relu*
strides
*
data_formatNHWC*
ksize
*
paddingSAME*0
_output_shapes
:����������*
T0
l
model_1/conv4/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
�
model_1/conv4/Conv2DConv2Dmodel_1/conv3/MaxPool2D/MaxPoolconv4/weights/read*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME*0
_output_shapes
:����������*
	dilations
*
T0
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
model_1/conv4/MaxPool2D/MaxPoolMaxPoolmodel_1/conv4/conv4/Relu*
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
l
model_1/conv5/dilation_rateConst*
dtype0*
_output_shapes
:*
valueB"      
�
model_1/conv5/Conv2DConv2Dmodel_1/conv4/MaxPool2D/MaxPoolconv5/weights/read*
T0*
data_formatNHWC*
strides
*
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
-model_1/Flatten/flatten/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
�
%model_1/Flatten/flatten/strided_sliceStridedSlicemodel_1/Flatten/flatten/Shape+model_1/Flatten/flatten/strided_slice/stack-model_1/Flatten/flatten/strided_slice/stack_1-model_1/Flatten/flatten/strided_slice/stack_2*
end_mask *
_output_shapes
: *
T0*
Index0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask 
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
model_1/Flatten/flatten/ReshapeReshapemodel_1/conv5/MaxPool2D/MaxPool%model_1/Flatten/flatten/Reshape/shape*(
_output_shapes
:����������*
T0*
Tshape0
l
model_2/conv1/dilation_rateConst*
dtype0*
_output_shapes
:*
valueB"      
�
model_2/conv1/Conv2DConv2Dnegative_inputconv1/weights/read*
paddingSAME*0
_output_shapes
:���������2� *
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(
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
model_2/conv1/MaxPool2D/MaxPoolMaxPoolmodel_2/conv1/conv1/Relu*
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
model_2/conv2/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
�
model_2/conv2/Conv2DConv2Dmodel_2/conv1/MaxPool2D/MaxPoolconv2/weights/read*/
_output_shapes
:���������K@*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME
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
model_2/conv2/MaxPool2D/MaxPoolMaxPoolmodel_2/conv2/conv2/Relu*/
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
model_2/conv3/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
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
model_2/conv3/BiasAddBiasAddmodel_2/conv3/Conv2Dconv3/biases/read*
T0*
data_formatNHWC*0
_output_shapes
:���������&�
r
model_2/conv3/conv3/ReluRelumodel_2/conv3/BiasAdd*0
_output_shapes
:���������&�*
T0
�
model_2/conv3/MaxPool2D/MaxPoolMaxPoolmodel_2/conv3/conv3/Relu*
data_formatNHWC*
strides
*
ksize
*
paddingSAME*0
_output_shapes
:����������*
T0
l
model_2/conv4/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
�
model_2/conv4/Conv2DConv2Dmodel_2/conv3/MaxPool2D/MaxPoolconv4/weights/read*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*0
_output_shapes
:����������*
	dilations

�
model_2/conv4/BiasAddBiasAddmodel_2/conv4/Conv2Dconv4/biases/read*
T0*
data_formatNHWC*0
_output_shapes
:����������
r
model_2/conv4/conv4/ReluRelumodel_2/conv4/BiasAdd*0
_output_shapes
:����������*
T0
�
model_2/conv4/MaxPool2D/MaxPoolMaxPoolmodel_2/conv4/conv4/Relu*0
_output_shapes
:���������
�*
T0*
strides
*
data_formatNHWC*
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
:���������

�
model_2/conv5/BiasAddBiasAddmodel_2/conv5/Conv2Dconv5/biases/read*
T0*
data_formatNHWC*/
_output_shapes
:���������

�
model_2/conv5/MaxPool2D/MaxPoolMaxPoolmodel_2/conv5/BiasAdd*/
_output_shapes
:���������*
T0*
data_formatNHWC*
strides
*
ksize
*
paddingSAME
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
'model_2/Flatten/flatten/Reshape/shape/1Const*
dtype0*
_output_shapes
: *
valueB :
���������
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
SumSummulSum/reduction_indices*

Tidx0*
	keep_dims( *
T0*#
_output_shapes
:���������
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
Sum_1SumPowSum_1/reduction_indices*
T0*#
_output_shapes
:���������*

Tidx0*
	keep_dims( 
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
Sum_2/reduction_indicesConst*
dtype0*
_output_shapes
: *
value	B :
w
Sum_2SumPow_1Sum_2/reduction_indices*
T0*#
_output_shapes
:���������*

Tidx0*
	keep_dims( 
C
Sqrt_1SqrtSum_2*#
_output_shapes
:���������*
T0
H
mul_1MulSqrtSqrt_1*
T0*#
_output_shapes
:���������
H
divRealDivSummul_1*
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
:���������*

Tidx0*
	keep_dims( 
L
Pow_2/yConst*
valueB
 *   @*
dtype0*
_output_shapes
: 
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
:���������*

Tidx0*
	keep_dims( 
C
Sqrt_2SqrtSum_4*
T0*#
_output_shapes
:���������
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
L
div_1RealDivSum_3mul_3*#
_output_shapes
:���������*
T0
}
subSubmodel/Flatten/flatten/Reshapemodel_1/Flatten/flatten/Reshape*
T0*(
_output_shapes
:����������
L
Pow_4/yConst*
valueB
 *   @*
dtype0*
_output_shapes
: 
M
Pow_4PowsubPow_4/y*
T0*(
_output_shapes
:����������
Y
Sum_6/reduction_indicesConst*
dtype0*
_output_shapes
: *
value	B :
{
Sum_6SumPow_4Sum_6/reduction_indices*'
_output_shapes
:���������*

Tidx0*
	keep_dims(*
T0
G
Sqrt_4SqrtSum_6*
T0*'
_output_shapes
:���������

sub_1Submodel/Flatten/flatten/Reshapemodel_2/Flatten/flatten/Reshape*(
_output_shapes
:����������*
T0
L
Pow_5/yConst*
valueB
 *   @*
dtype0*
_output_shapes
: 
O
Pow_5Powsub_1Pow_5/y*
T0*(
_output_shapes
:����������
Y
Sum_7/reduction_indicesConst*
value	B :*
dtype0*
_output_shapes
: 
{
Sum_7SumPow_5Sum_7/reduction_indices*
T0*'
_output_shapes
:���������*

Tidx0*
	keep_dims(
G
Sqrt_5SqrtSum_7*
T0*'
_output_shapes
:���������
N
sub_2SubSqrt_4Sqrt_5*
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
Variable/initial_valueConst*
dtype0*
_output_shapes
: *
value	B : 
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
gradients/Mean_grad/ShapeShapeMaximum*
_output_shapes
:*
T0*
out_type0
�
gradients/Mean_grad/TileTilegradients/Mean_grad/Reshapegradients/Mean_grad/Shape*'
_output_shapes
:���������*

Tmultiples0*
T0
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
gradients/Mean_grad/ConstConst*
valueB: *
dtype0*
_output_shapes
:
�
gradients/Mean_grad/ProdProdgradients/Mean_grad/Shape_1gradients/Mean_grad/Const*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
e
gradients/Mean_grad/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
�
gradients/Mean_grad/Prod_1Prodgradients/Mean_grad/Shape_2gradients/Mean_grad/Const_1*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
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
Truncate( *

DstT0*
_output_shapes
: 
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
gradients/Maximum_grad/Shape_1Const*
dtype0*
_output_shapes
: *
valueB 
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
gradients/Maximum_grad/ReshapeReshapegradients/Maximum_grad/Sumgradients/Maximum_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
gradients/Maximum_grad/Sum_1Sumgradients/Maximum_grad/Select_1.gradients/Maximum_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
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
(gradients/add_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_grad/Shapegradients/add_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
gradients/add_grad/SumSum/gradients/Maximum_grad/tuple/control_dependency(gradients/add_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
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
gradients/add_grad/Reshape_1Reshapegradients/add_grad/Sum_1gradients/add_grad/Shape_1*
_output_shapes
: *
T0*
Tshape0
g
#gradients/add_grad/tuple/group_depsNoOp^gradients/add_grad/Reshape^gradients/add_grad/Reshape_1
�
+gradients/add_grad/tuple/control_dependencyIdentitygradients/add_grad/Reshape$^gradients/add_grad/tuple/group_deps*'
_output_shapes
:���������*
T0*-
_class#
!loc:@gradients/add_grad/Reshape
�
-gradients/add_grad/tuple/control_dependency_1Identitygradients/add_grad/Reshape_1$^gradients/add_grad/tuple/group_deps*
_output_shapes
: *
T0*/
_class%
#!loc:@gradients/add_grad/Reshape_1
`
gradients/sub_2_grad/ShapeShapeSqrt_4*
T0*
out_type0*
_output_shapes
:
b
gradients/sub_2_grad/Shape_1ShapeSqrt_5*
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
gradients/Sqrt_4_grad/SqrtGradSqrtGradSqrt_4-gradients/sub_2_grad/tuple/control_dependency*
T0*'
_output_shapes
:���������
�
gradients/Sqrt_5_grad/SqrtGradSqrtGradSqrt_5/gradients/sub_2_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:���������
_
gradients/Sum_6_grad/ShapeShapePow_4*
T0*
out_type0*
_output_shapes
:
�
gradients/Sum_6_grad/SizeConst*-
_class#
!loc:@gradients/Sum_6_grad/Shape*
value	B :*
dtype0*
_output_shapes
: 
�
gradients/Sum_6_grad/addAddSum_6/reduction_indicesgradients/Sum_6_grad/Size*
T0*-
_class#
!loc:@gradients/Sum_6_grad/Shape*
_output_shapes
: 
�
gradients/Sum_6_grad/modFloorModgradients/Sum_6_grad/addgradients/Sum_6_grad/Size*
T0*-
_class#
!loc:@gradients/Sum_6_grad/Shape*
_output_shapes
: 
�
gradients/Sum_6_grad/Shape_1Const*-
_class#
!loc:@gradients/Sum_6_grad/Shape*
valueB *
dtype0*
_output_shapes
: 
�
 gradients/Sum_6_grad/range/startConst*-
_class#
!loc:@gradients/Sum_6_grad/Shape*
value	B : *
dtype0*
_output_shapes
: 
�
 gradients/Sum_6_grad/range/deltaConst*
dtype0*
_output_shapes
: *-
_class#
!loc:@gradients/Sum_6_grad/Shape*
value	B :
�
gradients/Sum_6_grad/rangeRange gradients/Sum_6_grad/range/startgradients/Sum_6_grad/Size gradients/Sum_6_grad/range/delta*-
_class#
!loc:@gradients/Sum_6_grad/Shape*
_output_shapes
:*

Tidx0
�
gradients/Sum_6_grad/Fill/valueConst*
dtype0*
_output_shapes
: *-
_class#
!loc:@gradients/Sum_6_grad/Shape*
value	B :
�
gradients/Sum_6_grad/FillFillgradients/Sum_6_grad/Shape_1gradients/Sum_6_grad/Fill/value*
_output_shapes
: *
T0*-
_class#
!loc:@gradients/Sum_6_grad/Shape*

index_type0
�
"gradients/Sum_6_grad/DynamicStitchDynamicStitchgradients/Sum_6_grad/rangegradients/Sum_6_grad/modgradients/Sum_6_grad/Shapegradients/Sum_6_grad/Fill*
T0*-
_class#
!loc:@gradients/Sum_6_grad/Shape*
N*
_output_shapes
:
�
gradients/Sum_6_grad/Maximum/yConst*-
_class#
!loc:@gradients/Sum_6_grad/Shape*
value	B :*
dtype0*
_output_shapes
: 
�
gradients/Sum_6_grad/MaximumMaximum"gradients/Sum_6_grad/DynamicStitchgradients/Sum_6_grad/Maximum/y*
T0*-
_class#
!loc:@gradients/Sum_6_grad/Shape*
_output_shapes
:
�
gradients/Sum_6_grad/floordivFloorDivgradients/Sum_6_grad/Shapegradients/Sum_6_grad/Maximum*
T0*-
_class#
!loc:@gradients/Sum_6_grad/Shape*
_output_shapes
:
�
gradients/Sum_6_grad/ReshapeReshapegradients/Sqrt_4_grad/SqrtGrad"gradients/Sum_6_grad/DynamicStitch*
T0*
Tshape0*0
_output_shapes
:������������������
�
gradients/Sum_6_grad/TileTilegradients/Sum_6_grad/Reshapegradients/Sum_6_grad/floordiv*
T0*(
_output_shapes
:����������*

Tmultiples0
_
gradients/Sum_7_grad/ShapeShapePow_5*
T0*
out_type0*
_output_shapes
:
�
gradients/Sum_7_grad/SizeConst*-
_class#
!loc:@gradients/Sum_7_grad/Shape*
value	B :*
dtype0*
_output_shapes
: 
�
gradients/Sum_7_grad/addAddSum_7/reduction_indicesgradients/Sum_7_grad/Size*
T0*-
_class#
!loc:@gradients/Sum_7_grad/Shape*
_output_shapes
: 
�
gradients/Sum_7_grad/modFloorModgradients/Sum_7_grad/addgradients/Sum_7_grad/Size*
T0*-
_class#
!loc:@gradients/Sum_7_grad/Shape*
_output_shapes
: 
�
gradients/Sum_7_grad/Shape_1Const*-
_class#
!loc:@gradients/Sum_7_grad/Shape*
valueB *
dtype0*
_output_shapes
: 
�
 gradients/Sum_7_grad/range/startConst*-
_class#
!loc:@gradients/Sum_7_grad/Shape*
value	B : *
dtype0*
_output_shapes
: 
�
 gradients/Sum_7_grad/range/deltaConst*-
_class#
!loc:@gradients/Sum_7_grad/Shape*
value	B :*
dtype0*
_output_shapes
: 
�
gradients/Sum_7_grad/rangeRange gradients/Sum_7_grad/range/startgradients/Sum_7_grad/Size gradients/Sum_7_grad/range/delta*-
_class#
!loc:@gradients/Sum_7_grad/Shape*
_output_shapes
:*

Tidx0
�
gradients/Sum_7_grad/Fill/valueConst*
dtype0*
_output_shapes
: *-
_class#
!loc:@gradients/Sum_7_grad/Shape*
value	B :
�
gradients/Sum_7_grad/FillFillgradients/Sum_7_grad/Shape_1gradients/Sum_7_grad/Fill/value*
T0*-
_class#
!loc:@gradients/Sum_7_grad/Shape*

index_type0*
_output_shapes
: 
�
"gradients/Sum_7_grad/DynamicStitchDynamicStitchgradients/Sum_7_grad/rangegradients/Sum_7_grad/modgradients/Sum_7_grad/Shapegradients/Sum_7_grad/Fill*
T0*-
_class#
!loc:@gradients/Sum_7_grad/Shape*
N*
_output_shapes
:
�
gradients/Sum_7_grad/Maximum/yConst*-
_class#
!loc:@gradients/Sum_7_grad/Shape*
value	B :*
dtype0*
_output_shapes
: 
�
gradients/Sum_7_grad/MaximumMaximum"gradients/Sum_7_grad/DynamicStitchgradients/Sum_7_grad/Maximum/y*
T0*-
_class#
!loc:@gradients/Sum_7_grad/Shape*
_output_shapes
:
�
gradients/Sum_7_grad/floordivFloorDivgradients/Sum_7_grad/Shapegradients/Sum_7_grad/Maximum*
_output_shapes
:*
T0*-
_class#
!loc:@gradients/Sum_7_grad/Shape
�
gradients/Sum_7_grad/ReshapeReshapegradients/Sqrt_5_grad/SqrtGrad"gradients/Sum_7_grad/DynamicStitch*
T0*
Tshape0*0
_output_shapes
:������������������
�
gradients/Sum_7_grad/TileTilegradients/Sum_7_grad/Reshapegradients/Sum_7_grad/floordiv*(
_output_shapes
:����������*

Tmultiples0*
T0
]
gradients/Pow_4_grad/ShapeShapesub*
T0*
out_type0*
_output_shapes
:
_
gradients/Pow_4_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
�
*gradients/Pow_4_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/Pow_4_grad/Shapegradients/Pow_4_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
v
gradients/Pow_4_grad/mulMulgradients/Sum_6_grad/TilePow_4/y*
T0*(
_output_shapes
:����������
_
gradients/Pow_4_grad/sub/yConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
e
gradients/Pow_4_grad/subSubPow_4/ygradients/Pow_4_grad/sub/y*
T0*
_output_shapes
: 
q
gradients/Pow_4_grad/PowPowsubgradients/Pow_4_grad/sub*
T0*(
_output_shapes
:����������
�
gradients/Pow_4_grad/mul_1Mulgradients/Pow_4_grad/mulgradients/Pow_4_grad/Pow*
T0*(
_output_shapes
:����������
�
gradients/Pow_4_grad/SumSumgradients/Pow_4_grad/mul_1*gradients/Pow_4_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
gradients/Pow_4_grad/ReshapeReshapegradients/Pow_4_grad/Sumgradients/Pow_4_grad/Shape*
T0*
Tshape0*(
_output_shapes
:����������
c
gradients/Pow_4_grad/Greater/yConst*
valueB
 *    *
dtype0*
_output_shapes
: 

gradients/Pow_4_grad/GreaterGreatersubgradients/Pow_4_grad/Greater/y*
T0*(
_output_shapes
:����������
g
$gradients/Pow_4_grad/ones_like/ShapeShapesub*
T0*
out_type0*
_output_shapes
:
i
$gradients/Pow_4_grad/ones_like/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *  �?
�
gradients/Pow_4_grad/ones_likeFill$gradients/Pow_4_grad/ones_like/Shape$gradients/Pow_4_grad/ones_like/Const*
T0*

index_type0*(
_output_shapes
:����������
�
gradients/Pow_4_grad/SelectSelectgradients/Pow_4_grad/Greatersubgradients/Pow_4_grad/ones_like*
T0*(
_output_shapes
:����������
o
gradients/Pow_4_grad/LogLoggradients/Pow_4_grad/Select*
T0*(
_output_shapes
:����������
d
gradients/Pow_4_grad/zeros_like	ZerosLikesub*
T0*(
_output_shapes
:����������
�
gradients/Pow_4_grad/Select_1Selectgradients/Pow_4_grad/Greatergradients/Pow_4_grad/Loggradients/Pow_4_grad/zeros_like*
T0*(
_output_shapes
:����������
v
gradients/Pow_4_grad/mul_2Mulgradients/Sum_6_grad/TilePow_4*
T0*(
_output_shapes
:����������
�
gradients/Pow_4_grad/mul_3Mulgradients/Pow_4_grad/mul_2gradients/Pow_4_grad/Select_1*
T0*(
_output_shapes
:����������
�
gradients/Pow_4_grad/Sum_1Sumgradients/Pow_4_grad/mul_3,gradients/Pow_4_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
gradients/Pow_4_grad/Reshape_1Reshapegradients/Pow_4_grad/Sum_1gradients/Pow_4_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
m
%gradients/Pow_4_grad/tuple/group_depsNoOp^gradients/Pow_4_grad/Reshape^gradients/Pow_4_grad/Reshape_1
�
-gradients/Pow_4_grad/tuple/control_dependencyIdentitygradients/Pow_4_grad/Reshape&^gradients/Pow_4_grad/tuple/group_deps*(
_output_shapes
:����������*
T0*/
_class%
#!loc:@gradients/Pow_4_grad/Reshape
�
/gradients/Pow_4_grad/tuple/control_dependency_1Identitygradients/Pow_4_grad/Reshape_1&^gradients/Pow_4_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/Pow_4_grad/Reshape_1*
_output_shapes
: 
_
gradients/Pow_5_grad/ShapeShapesub_1*
_output_shapes
:*
T0*
out_type0
_
gradients/Pow_5_grad/Shape_1Const*
dtype0*
_output_shapes
: *
valueB 
�
*gradients/Pow_5_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/Pow_5_grad/Shapegradients/Pow_5_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
v
gradients/Pow_5_grad/mulMulgradients/Sum_7_grad/TilePow_5/y*
T0*(
_output_shapes
:����������
_
gradients/Pow_5_grad/sub/yConst*
dtype0*
_output_shapes
: *
valueB
 *  �?
e
gradients/Pow_5_grad/subSubPow_5/ygradients/Pow_5_grad/sub/y*
T0*
_output_shapes
: 
s
gradients/Pow_5_grad/PowPowsub_1gradients/Pow_5_grad/sub*
T0*(
_output_shapes
:����������
�
gradients/Pow_5_grad/mul_1Mulgradients/Pow_5_grad/mulgradients/Pow_5_grad/Pow*
T0*(
_output_shapes
:����������
�
gradients/Pow_5_grad/SumSumgradients/Pow_5_grad/mul_1*gradients/Pow_5_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
�
gradients/Pow_5_grad/ReshapeReshapegradients/Pow_5_grad/Sumgradients/Pow_5_grad/Shape*
T0*
Tshape0*(
_output_shapes
:����������
c
gradients/Pow_5_grad/Greater/yConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
gradients/Pow_5_grad/GreaterGreatersub_1gradients/Pow_5_grad/Greater/y*
T0*(
_output_shapes
:����������
i
$gradients/Pow_5_grad/ones_like/ShapeShapesub_1*
T0*
out_type0*
_output_shapes
:
i
$gradients/Pow_5_grad/ones_like/ConstConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
gradients/Pow_5_grad/ones_likeFill$gradients/Pow_5_grad/ones_like/Shape$gradients/Pow_5_grad/ones_like/Const*
T0*

index_type0*(
_output_shapes
:����������
�
gradients/Pow_5_grad/SelectSelectgradients/Pow_5_grad/Greatersub_1gradients/Pow_5_grad/ones_like*(
_output_shapes
:����������*
T0
o
gradients/Pow_5_grad/LogLoggradients/Pow_5_grad/Select*
T0*(
_output_shapes
:����������
f
gradients/Pow_5_grad/zeros_like	ZerosLikesub_1*
T0*(
_output_shapes
:����������
�
gradients/Pow_5_grad/Select_1Selectgradients/Pow_5_grad/Greatergradients/Pow_5_grad/Loggradients/Pow_5_grad/zeros_like*(
_output_shapes
:����������*
T0
v
gradients/Pow_5_grad/mul_2Mulgradients/Sum_7_grad/TilePow_5*(
_output_shapes
:����������*
T0
�
gradients/Pow_5_grad/mul_3Mulgradients/Pow_5_grad/mul_2gradients/Pow_5_grad/Select_1*
T0*(
_output_shapes
:����������
�
gradients/Pow_5_grad/Sum_1Sumgradients/Pow_5_grad/mul_3,gradients/Pow_5_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
�
gradients/Pow_5_grad/Reshape_1Reshapegradients/Pow_5_grad/Sum_1gradients/Pow_5_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
m
%gradients/Pow_5_grad/tuple/group_depsNoOp^gradients/Pow_5_grad/Reshape^gradients/Pow_5_grad/Reshape_1
�
-gradients/Pow_5_grad/tuple/control_dependencyIdentitygradients/Pow_5_grad/Reshape&^gradients/Pow_5_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/Pow_5_grad/Reshape*(
_output_shapes
:����������
�
/gradients/Pow_5_grad/tuple/control_dependency_1Identitygradients/Pow_5_grad/Reshape_1&^gradients/Pow_5_grad/tuple/group_deps*
_output_shapes
: *
T0*1
_class'
%#loc:@gradients/Pow_5_grad/Reshape_1
u
gradients/sub_grad/ShapeShapemodel/Flatten/flatten/Reshape*
_output_shapes
:*
T0*
out_type0
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
gradients/sub_grad/SumSum-gradients/Pow_4_grad/tuple/control_dependency(gradients/sub_grad/BroadcastGradientArgs*
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
gradients/sub_grad/Sum_1Sum-gradients/Pow_4_grad/tuple/control_dependency*gradients/sub_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
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
-gradients/sub_grad/tuple/control_dependency_1Identitygradients/sub_grad/Reshape_1$^gradients/sub_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/sub_grad/Reshape_1*(
_output_shapes
:����������
w
gradients/sub_1_grad/ShapeShapemodel/Flatten/flatten/Reshape*
_output_shapes
:*
T0*
out_type0
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
gradients/sub_1_grad/SumSum-gradients/Pow_5_grad/tuple/control_dependency*gradients/sub_1_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
gradients/sub_1_grad/ReshapeReshapegradients/sub_1_grad/Sumgradients/sub_1_grad/Shape*
T0*
Tshape0*(
_output_shapes
:����������
�
gradients/sub_1_grad/Sum_1Sum-gradients/Pow_5_grad/tuple/control_dependency,gradients/sub_1_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
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
-gradients/sub_1_grad/tuple/control_dependencyIdentitygradients/sub_1_grad/Reshape&^gradients/sub_1_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/sub_1_grad/Reshape*(
_output_shapes
:����������
�
/gradients/sub_1_grad/tuple/control_dependency_1Identitygradients/sub_1_grad/Reshape_1&^gradients/sub_1_grad/tuple/group_deps*(
_output_shapes
:����������*
T0*1
_class'
%#loc:@gradients/sub_1_grad/Reshape_1
�
4gradients/model_1/Flatten/flatten/Reshape_grad/ShapeShapemodel_1/conv5/MaxPool2D/MaxPool*
out_type0*
_output_shapes
:*
T0
�
6gradients/model_1/Flatten/flatten/Reshape_grad/ReshapeReshape-gradients/sub_grad/tuple/control_dependency_14gradients/model_1/Flatten/flatten/Reshape_grad/Shape*
T0*
Tshape0*/
_output_shapes
:���������
�
gradients/AddNAddN+gradients/sub_grad/tuple/control_dependency-gradients/sub_1_grad/tuple/control_dependency*
N*(
_output_shapes
:����������*
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
:gradients/model_1/conv5/MaxPool2D/MaxPool_grad/MaxPoolGradMaxPoolGradmodel_1/conv5/BiasAddmodel_1/conv5/MaxPool2D/MaxPool6gradients/model_1/Flatten/flatten/Reshape_grad/Reshape*
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
8gradients/model/conv5/MaxPool2D/MaxPool_grad/MaxPoolGradMaxPoolGradmodel/conv5/BiasAddmodel/conv5/MaxPool2D/MaxPool4gradients/model/Flatten/flatten/Reshape_grad/Reshape*
T0*
strides
*
data_formatNHWC*
ksize
*
paddingSAME*/
_output_shapes
:���������

�
:gradients/model_2/conv5/MaxPool2D/MaxPool_grad/MaxPoolGradMaxPoolGradmodel_2/conv5/BiasAddmodel_2/conv5/MaxPool2D/MaxPool6gradients/model_2/Flatten/flatten/Reshape_grad/Reshape*
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
�
0gradients/model_1/conv5/BiasAdd_grad/BiasAddGradBiasAddGrad:gradients/model_1/conv5/MaxPool2D/MaxPool_grad/MaxPoolGrad*
T0*
data_formatNHWC*
_output_shapes
:
�
5gradients/model_1/conv5/BiasAdd_grad/tuple/group_depsNoOp1^gradients/model_1/conv5/BiasAdd_grad/BiasAddGrad;^gradients/model_1/conv5/MaxPool2D/MaxPool_grad/MaxPoolGrad
�
=gradients/model_1/conv5/BiasAdd_grad/tuple/control_dependencyIdentity:gradients/model_1/conv5/MaxPool2D/MaxPool_grad/MaxPoolGrad6^gradients/model_1/conv5/BiasAdd_grad/tuple/group_deps*
T0*M
_classC
A?loc:@gradients/model_1/conv5/MaxPool2D/MaxPool_grad/MaxPoolGrad*/
_output_shapes
:���������

�
?gradients/model_1/conv5/BiasAdd_grad/tuple/control_dependency_1Identity0gradients/model_1/conv5/BiasAdd_grad/BiasAddGrad6^gradients/model_1/conv5/BiasAdd_grad/tuple/group_deps*C
_class9
75loc:@gradients/model_1/conv5/BiasAdd_grad/BiasAddGrad*
_output_shapes
:*
T0
�
.gradients/model/conv5/BiasAdd_grad/BiasAddGradBiasAddGrad8gradients/model/conv5/MaxPool2D/MaxPool_grad/MaxPoolGrad*
data_formatNHWC*
_output_shapes
:*
T0
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
7gradients/model_1/conv5/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput*gradients/model_1/conv5/Conv2D_grad/ShapeNconv5/weights/read=gradients/model_1/conv5/BiasAdd_grad/tuple/control_dependency*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME*0
_output_shapes
:���������
�
�
8gradients/model_1/conv5/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFiltermodel_1/conv4/MaxPool2D/MaxPool,gradients/model_1/conv5/Conv2D_grad/ShapeN:1=gradients/model_1/conv5/BiasAdd_grad/tuple/control_dependency*
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
>gradients/model_1/conv5/Conv2D_grad/tuple/control_dependency_1Identity8gradients/model_1/conv5/Conv2D_grad/Conv2DBackpropFilter5^gradients/model_1/conv5/Conv2D_grad/tuple/group_deps*K
_classA
?=loc:@gradients/model_1/conv5/Conv2D_grad/Conv2DBackpropFilter*'
_output_shapes
:�*
T0
�
(gradients/model/conv5/Conv2D_grad/ShapeNShapeNmodel/conv4/MaxPool2D/MaxPoolconv5/weights/read*
T0*
out_type0*
N* 
_output_shapes
::
�
5gradients/model/conv5/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput(gradients/model/conv5/Conv2D_grad/ShapeNconv5/weights/read;gradients/model/conv5/BiasAdd_grad/tuple/control_dependency*
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
6gradients/model/conv5/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFiltermodel/conv4/MaxPool2D/MaxPool*gradients/model/conv5/Conv2D_grad/ShapeN:1;gradients/model/conv5/BiasAdd_grad/tuple/control_dependency*'
_output_shapes
:�*
	dilations
*
T0*
data_formatNHWC*
strides
*
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
<gradients/model/conv5/Conv2D_grad/tuple/control_dependency_1Identity6gradients/model/conv5/Conv2D_grad/Conv2DBackpropFilter3^gradients/model/conv5/Conv2D_grad/tuple/group_deps*I
_class?
=;loc:@gradients/model/conv5/Conv2D_grad/Conv2DBackpropFilter*'
_output_shapes
:�*
T0
�
*gradients/model_2/conv5/Conv2D_grad/ShapeNShapeNmodel_2/conv4/MaxPool2D/MaxPoolconv5/weights/read*
T0*
out_type0*
N* 
_output_shapes
::
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
8gradients/model_2/conv5/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFiltermodel_2/conv4/MaxPool2D/MaxPool,gradients/model_2/conv5/Conv2D_grad/ShapeN:1=gradients/model_2/conv5/BiasAdd_grad/tuple/control_dependency*
data_formatNHWC*
strides
*
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
8gradients/model/conv4/MaxPool2D/MaxPool_grad/MaxPoolGradMaxPoolGradmodel/conv4/conv4/Relumodel/conv4/MaxPool2D/MaxPool:gradients/model/conv5/Conv2D_grad/tuple/control_dependency*
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
:gradients/model_2/conv4/MaxPool2D/MaxPool_grad/MaxPoolGradMaxPoolGradmodel_2/conv4/conv4/Relumodel_2/conv4/MaxPool2D/MaxPool<gradients/model_2/conv5/Conv2D_grad/tuple/control_dependency*
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
:����������
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
;gradients/model/conv4/BiasAdd_grad/tuple/control_dependencyIdentity.gradients/model/conv4/conv4/Relu_grad/ReluGrad4^gradients/model/conv4/BiasAdd_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients/model/conv4/conv4/Relu_grad/ReluGrad*0
_output_shapes
:����������
�
=gradients/model/conv4/BiasAdd_grad/tuple/control_dependency_1Identity.gradients/model/conv4/BiasAdd_grad/BiasAddGrad4^gradients/model/conv4/BiasAdd_grad/tuple/group_deps*A
_class7
53loc:@gradients/model/conv4/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:�*
T0
�
0gradients/model_2/conv4/BiasAdd_grad/BiasAddGradBiasAddGrad0gradients/model_2/conv4/conv4/Relu_grad/ReluGrad*
_output_shapes	
:�*
T0*
data_formatNHWC
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
*gradients/model_1/conv4/Conv2D_grad/ShapeNShapeNmodel_1/conv3/MaxPool2D/MaxPoolconv4/weights/read*
T0*
out_type0*
N* 
_output_shapes
::
�
7gradients/model_1/conv4/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput*gradients/model_1/conv4/Conv2D_grad/ShapeNconv4/weights/read=gradients/model_1/conv4/BiasAdd_grad/tuple/control_dependency*
paddingSAME*0
_output_shapes
:����������*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
�
8gradients/model_1/conv4/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFiltermodel_1/conv3/MaxPool2D/MaxPool,gradients/model_1/conv4/Conv2D_grad/ShapeN:1=gradients/model_1/conv4/BiasAdd_grad/tuple/control_dependency*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME*(
_output_shapes
:��*
	dilations

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
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME*0
_output_shapes
:����������
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
:gradients/model/conv4/Conv2D_grad/tuple/control_dependencyIdentity5gradients/model/conv4/Conv2D_grad/Conv2DBackpropInput3^gradients/model/conv4/Conv2D_grad/tuple/group_deps*H
_class>
<:loc:@gradients/model/conv4/Conv2D_grad/Conv2DBackpropInput*0
_output_shapes
:����������*
T0
�
<gradients/model/conv4/Conv2D_grad/tuple/control_dependency_1Identity6gradients/model/conv4/Conv2D_grad/Conv2DBackpropFilter3^gradients/model/conv4/Conv2D_grad/tuple/group_deps*(
_output_shapes
:��*
T0*I
_class?
=;loc:@gradients/model/conv4/Conv2D_grad/Conv2DBackpropFilter
�
*gradients/model_2/conv4/Conv2D_grad/ShapeNShapeNmodel_2/conv3/MaxPool2D/MaxPoolconv4/weights/read*
N* 
_output_shapes
::*
T0*
out_type0
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
<gradients/model_2/conv4/Conv2D_grad/tuple/control_dependencyIdentity7gradients/model_2/conv4/Conv2D_grad/Conv2DBackpropInput5^gradients/model_2/conv4/Conv2D_grad/tuple/group_deps*
T0*J
_class@
><loc:@gradients/model_2/conv4/Conv2D_grad/Conv2DBackpropInput*0
_output_shapes
:����������
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
:gradients/model_1/conv3/MaxPool2D/MaxPool_grad/MaxPoolGradMaxPoolGradmodel_1/conv3/conv3/Relumodel_1/conv3/MaxPool2D/MaxPool<gradients/model_1/conv4/Conv2D_grad/tuple/control_dependency*
T0*
strides
*
data_formatNHWC*
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
=gradients/model_1/conv3/BiasAdd_grad/tuple/control_dependencyIdentity0gradients/model_1/conv3/conv3/Relu_grad/ReluGrad6^gradients/model_1/conv3/BiasAdd_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/model_1/conv3/conv3/Relu_grad/ReluGrad*0
_output_shapes
:���������&�
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
?gradients/model_2/conv3/BiasAdd_grad/tuple/control_dependency_1Identity0gradients/model_2/conv3/BiasAdd_grad/BiasAddGrad6^gradients/model_2/conv3/BiasAdd_grad/tuple/group_deps*
_output_shapes	
:�*
T0*C
_class9
75loc:@gradients/model_2/conv3/BiasAdd_grad/BiasAddGrad
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
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME
�
8gradients/model_1/conv3/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFiltermodel_1/conv2/MaxPool2D/MaxPool,gradients/model_1/conv3/Conv2D_grad/ShapeN:1=gradients/model_1/conv3/BiasAdd_grad/tuple/control_dependency*'
_output_shapes
:@�*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME
�
4gradients/model_1/conv3/Conv2D_grad/tuple/group_depsNoOp9^gradients/model_1/conv3/Conv2D_grad/Conv2DBackpropFilter8^gradients/model_1/conv3/Conv2D_grad/Conv2DBackpropInput
�
<gradients/model_1/conv3/Conv2D_grad/tuple/control_dependencyIdentity7gradients/model_1/conv3/Conv2D_grad/Conv2DBackpropInput5^gradients/model_1/conv3/Conv2D_grad/tuple/group_deps*
T0*J
_class@
><loc:@gradients/model_1/conv3/Conv2D_grad/Conv2DBackpropInput*/
_output_shapes
:���������&@
�
>gradients/model_1/conv3/Conv2D_grad/tuple/control_dependency_1Identity8gradients/model_1/conv3/Conv2D_grad/Conv2DBackpropFilter5^gradients/model_1/conv3/Conv2D_grad/tuple/group_deps*'
_output_shapes
:@�*
T0*K
_classA
?=loc:@gradients/model_1/conv3/Conv2D_grad/Conv2DBackpropFilter
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
7gradients/model_2/conv3/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput*gradients/model_2/conv3/Conv2D_grad/ShapeNconv3/weights/read=gradients/model_2/conv3/BiasAdd_grad/tuple/control_dependency*
paddingSAME*/
_output_shapes
:���������&@*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
�
8gradients/model_2/conv3/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFiltermodel_2/conv2/MaxPool2D/MaxPool,gradients/model_2/conv3/Conv2D_grad/ShapeN:1=gradients/model_2/conv3/BiasAdd_grad/tuple/control_dependency*'
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
gradients/AddN_5AddN?gradients/model_1/conv3/BiasAdd_grad/tuple/control_dependency_1=gradients/model/conv3/BiasAdd_grad/tuple/control_dependency_1?gradients/model_2/conv3/BiasAdd_grad/tuple/control_dependency_1*C
_class9
75loc:@gradients/model_1/conv3/BiasAdd_grad/BiasAddGrad*
N*
_output_shapes	
:�*
T0
�
:gradients/model_1/conv2/MaxPool2D/MaxPool_grad/MaxPoolGradMaxPoolGradmodel_1/conv2/conv2/Relumodel_1/conv2/MaxPool2D/MaxPool<gradients/model_1/conv3/Conv2D_grad/tuple/control_dependency*/
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
0gradients/model_1/conv2/conv2/Relu_grad/ReluGradReluGrad:gradients/model_1/conv2/MaxPool2D/MaxPool_grad/MaxPoolGradmodel_1/conv2/conv2/Relu*/
_output_shapes
:���������K@*
T0
�
.gradients/model/conv2/conv2/Relu_grad/ReluGradReluGrad8gradients/model/conv2/MaxPool2D/MaxPool_grad/MaxPoolGradmodel/conv2/conv2/Relu*
T0*/
_output_shapes
:���������K@
�
0gradients/model_2/conv2/conv2/Relu_grad/ReluGradReluGrad:gradients/model_2/conv2/MaxPool2D/MaxPool_grad/MaxPoolGradmodel_2/conv2/conv2/Relu*/
_output_shapes
:���������K@*
T0
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
?gradients/model_2/conv2/BiasAdd_grad/tuple/control_dependency_1Identity0gradients/model_2/conv2/BiasAdd_grad/BiasAddGrad6^gradients/model_2/conv2/BiasAdd_grad/tuple/group_deps*C
_class9
75loc:@gradients/model_2/conv2/BiasAdd_grad/BiasAddGrad*
_output_shapes
:@*
T0
�
*gradients/model_1/conv2/Conv2D_grad/ShapeNShapeNmodel_1/conv1/MaxPool2D/MaxPoolconv2/weights/read*
T0*
out_type0*
N* 
_output_shapes
::
�
7gradients/model_1/conv2/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput*gradients/model_1/conv2/Conv2D_grad/ShapeNconv2/weights/read=gradients/model_1/conv2/BiasAdd_grad/tuple/control_dependency*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME*/
_output_shapes
:���������K *
	dilations
*
T0
�
8gradients/model_1/conv2/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFiltermodel_1/conv1/MaxPool2D/MaxPool,gradients/model_1/conv2/Conv2D_grad/ShapeN:1=gradients/model_1/conv2/BiasAdd_grad/tuple/control_dependency*
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
4gradients/model_1/conv2/Conv2D_grad/tuple/group_depsNoOp9^gradients/model_1/conv2/Conv2D_grad/Conv2DBackpropFilter8^gradients/model_1/conv2/Conv2D_grad/Conv2DBackpropInput
�
<gradients/model_1/conv2/Conv2D_grad/tuple/control_dependencyIdentity7gradients/model_1/conv2/Conv2D_grad/Conv2DBackpropInput5^gradients/model_1/conv2/Conv2D_grad/tuple/group_deps*
T0*J
_class@
><loc:@gradients/model_1/conv2/Conv2D_grad/Conv2DBackpropInput*/
_output_shapes
:���������K 
�
>gradients/model_1/conv2/Conv2D_grad/tuple/control_dependency_1Identity8gradients/model_1/conv2/Conv2D_grad/Conv2DBackpropFilter5^gradients/model_1/conv2/Conv2D_grad/tuple/group_deps*K
_classA
?=loc:@gradients/model_1/conv2/Conv2D_grad/Conv2DBackpropFilter*&
_output_shapes
: @*
T0
�
(gradients/model/conv2/Conv2D_grad/ShapeNShapeNmodel/conv1/MaxPool2D/MaxPoolconv2/weights/read*
N* 
_output_shapes
::*
T0*
out_type0
�
5gradients/model/conv2/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput(gradients/model/conv2/Conv2D_grad/ShapeNconv2/weights/read;gradients/model/conv2/BiasAdd_grad/tuple/control_dependency*/
_output_shapes
:���������K *
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME
�
6gradients/model/conv2/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFiltermodel/conv1/MaxPool2D/MaxPool*gradients/model/conv2/Conv2D_grad/ShapeN:1;gradients/model/conv2/BiasAdd_grad/tuple/control_dependency*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME*&
_output_shapes
: @*
	dilations
*
T0
�
2gradients/model/conv2/Conv2D_grad/tuple/group_depsNoOp7^gradients/model/conv2/Conv2D_grad/Conv2DBackpropFilter6^gradients/model/conv2/Conv2D_grad/Conv2DBackpropInput
�
:gradients/model/conv2/Conv2D_grad/tuple/control_dependencyIdentity5gradients/model/conv2/Conv2D_grad/Conv2DBackpropInput3^gradients/model/conv2/Conv2D_grad/tuple/group_deps*H
_class>
<:loc:@gradients/model/conv2/Conv2D_grad/Conv2DBackpropInput*/
_output_shapes
:���������K *
T0
�
<gradients/model/conv2/Conv2D_grad/tuple/control_dependency_1Identity6gradients/model/conv2/Conv2D_grad/Conv2DBackpropFilter3^gradients/model/conv2/Conv2D_grad/tuple/group_deps*
T0*I
_class?
=;loc:@gradients/model/conv2/Conv2D_grad/Conv2DBackpropFilter*&
_output_shapes
: @
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
:���������K 
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
<gradients/model_2/conv2/Conv2D_grad/tuple/control_dependencyIdentity7gradients/model_2/conv2/Conv2D_grad/Conv2DBackpropInput5^gradients/model_2/conv2/Conv2D_grad/tuple/group_deps*J
_class@
><loc:@gradients/model_2/conv2/Conv2D_grad/Conv2DBackpropInput*/
_output_shapes
:���������K *
T0
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
:gradients/model_1/conv1/MaxPool2D/MaxPool_grad/MaxPoolGradMaxPoolGradmodel_1/conv1/conv1/Relumodel_1/conv1/MaxPool2D/MaxPool<gradients/model_1/conv2/Conv2D_grad/tuple/control_dependency*0
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
8gradients/model/conv1/MaxPool2D/MaxPool_grad/MaxPoolGradMaxPoolGradmodel/conv1/conv1/Relumodel/conv1/MaxPool2D/MaxPool:gradients/model/conv2/Conv2D_grad/tuple/control_dependency*
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
N*&
_output_shapes
: @*
T0*K
_classA
?=loc:@gradients/model_1/conv2/Conv2D_grad/Conv2DBackpropFilter
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
?gradients/model_1/conv1/BiasAdd_grad/tuple/control_dependency_1Identity0gradients/model_1/conv1/BiasAdd_grad/BiasAddGrad6^gradients/model_1/conv1/BiasAdd_grad/tuple/group_deps*
_output_shapes
: *
T0*C
_class9
75loc:@gradients/model_1/conv1/BiasAdd_grad/BiasAddGrad
�
.gradients/model/conv1/BiasAdd_grad/BiasAddGradBiasAddGrad.gradients/model/conv1/conv1/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes
: 
�
3gradients/model/conv1/BiasAdd_grad/tuple/group_depsNoOp/^gradients/model/conv1/BiasAdd_grad/BiasAddGrad/^gradients/model/conv1/conv1/Relu_grad/ReluGrad
�
;gradients/model/conv1/BiasAdd_grad/tuple/control_dependencyIdentity.gradients/model/conv1/conv1/Relu_grad/ReluGrad4^gradients/model/conv1/BiasAdd_grad/tuple/group_deps*0
_output_shapes
:���������2� *
T0*A
_class7
53loc:@gradients/model/conv1/conv1/Relu_grad/ReluGrad
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
7gradients/model_1/conv1/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput*gradients/model_1/conv1/Conv2D_grad/ShapeNconv1/weights/read=gradients/model_1/conv1/BiasAdd_grad/tuple/control_dependency*0
_output_shapes
:���������2�*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME
�
8gradients/model_1/conv1/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterpositive_input,gradients/model_1/conv1/Conv2D_grad/ShapeN:1=gradients/model_1/conv1/BiasAdd_grad/tuple/control_dependency*
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
5gradients/model/conv1/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput(gradients/model/conv1/Conv2D_grad/ShapeNconv1/weights/read;gradients/model/conv1/BiasAdd_grad/tuple/control_dependency*
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
6gradients/model/conv1/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilteranchor_input*gradients/model/conv1/Conv2D_grad/ShapeN:1;gradients/model/conv1/BiasAdd_grad/tuple/control_dependency*&
_output_shapes
: *
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME
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
7gradients/model_2/conv1/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput*gradients/model_2/conv1/Conv2D_grad/ShapeNconv1/weights/read=gradients/model_2/conv1/BiasAdd_grad/tuple/control_dependency*
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
8gradients/model_2/conv1/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilternegative_input,gradients/model_2/conv1/Conv2D_grad/ShapeN:1=gradients/model_2/conv1/BiasAdd_grad/tuple/control_dependency*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME*&
_output_shapes
: 
�
4gradients/model_2/conv1/Conv2D_grad/tuple/group_depsNoOp9^gradients/model_2/conv1/Conv2D_grad/Conv2DBackpropFilter8^gradients/model_2/conv1/Conv2D_grad/Conv2DBackpropInput
�
<gradients/model_2/conv1/Conv2D_grad/tuple/control_dependencyIdentity7gradients/model_2/conv1/Conv2D_grad/Conv2DBackpropInput5^gradients/model_2/conv1/Conv2D_grad/tuple/group_deps*0
_output_shapes
:���������2�*
T0*J
_class@
><loc:@gradients/model_2/conv1/Conv2D_grad/Conv2DBackpropInput
�
>gradients/model_2/conv1/Conv2D_grad/tuple/control_dependency_1Identity8gradients/model_2/conv1/Conv2D_grad/Conv2DBackpropFilter5^gradients/model_2/conv1/Conv2D_grad/tuple/group_deps*
T0*K
_classA
?=loc:@gradients/model_2/conv1/Conv2D_grad/Conv2DBackpropFilter*&
_output_shapes
: 
�
gradients/AddN_9AddN?gradients/model_1/conv1/BiasAdd_grad/tuple/control_dependency_1=gradients/model/conv1/BiasAdd_grad/tuple/control_dependency_1?gradients/model_2/conv1/BiasAdd_grad/tuple/control_dependency_1*
_output_shapes
: *
T0*C
_class9
75loc:@gradients/model_1/conv1/BiasAdd_grad/BiasAddGrad*
N
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
(conv1/weights/Momentum/Initializer/zerosFill8conv1/weights/Momentum/Initializer/zeros/shape_as_tensor.conv1/weights/Momentum/Initializer/zeros/Const*

index_type0* 
_class
loc:@conv1/weights*&
_output_shapes
: *
T0
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
conv1/weights/Momentum/AssignAssignconv1/weights/Momentum(conv1/weights/Momentum/Initializer/zeros* 
_class
loc:@conv1/weights*
validate_shape(*&
_output_shapes
: *
use_locking(*
T0
�
conv1/weights/Momentum/readIdentityconv1/weights/Momentum* 
_class
loc:@conv1/weights*&
_output_shapes
: *
T0
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
VariableV2*
	container *
shape: @*
dtype0*&
_output_shapes
: @*
shared_name * 
_class
loc:@conv2/weights
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
8conv3/weights/Momentum/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*%
valueB"      @   �   * 
_class
loc:@conv3/weights
�
.conv3/weights/Momentum/Initializer/zeros/ConstConst*
_output_shapes
: *
valueB
 *    * 
_class
loc:@conv3/weights*
dtype0
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
conv3/biases/Momentum/AssignAssignconv3/biases/Momentum'conv3/biases/Momentum/Initializer/zeros*
T0*
_class
loc:@conv3/biases*
validate_shape(*
_output_shapes	
:�*
use_locking(
�
conv3/biases/Momentum/readIdentityconv3/biases/Momentum*
_class
loc:@conv3/biases*
_output_shapes	
:�*
T0
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
(conv4/weights/Momentum/Initializer/zerosFill8conv4/weights/Momentum/Initializer/zeros/shape_as_tensor.conv4/weights/Momentum/Initializer/zeros/Const*
T0*

index_type0* 
_class
loc:@conv4/weights*(
_output_shapes
:��
�
conv4/weights/Momentum
VariableV2*
shared_name * 
_class
loc:@conv4/weights*
	container *
shape:��*
dtype0*(
_output_shapes
:��
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
conv4/weights/Momentum/readIdentityconv4/weights/Momentum* 
_class
loc:@conv4/weights*(
_output_shapes
:��*
T0
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
conv4/biases/Momentum/AssignAssignconv4/biases/Momentum'conv4/biases/Momentum/Initializer/zeros*
use_locking(*
T0*
_class
loc:@conv4/biases*
validate_shape(*
_output_shapes	
:�
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
shape:�*
dtype0*'
_output_shapes
:�*
shared_name * 
_class
loc:@conv5/weights*
	container 
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
conv5/weights/Momentum/readIdentityconv5/weights/Momentum*
T0* 
_class
loc:@conv5/weights*'
_output_shapes
:�
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
conv5/biases/Momentum/readIdentityconv5/biases/Momentum*
T0*
_class
loc:@conv5/biases*
_output_shapes
:
[
Momentum/learning_rateConst*
_output_shapes
: *
valueB
 *
�#<*
dtype0
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
+Momentum/update_conv2/weights/ApplyMomentumApplyMomentumconv2/weightsconv2/weights/MomentumMomentum/learning_rategradients/AddN_8Momentum/momentum*
T0* 
_class
loc:@conv2/weights*
use_nesterov(*&
_output_shapes
: @*
use_locking( 
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
shape: *
dtype0*
_output_shapes
: 
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
save/RestoreV2/tensor_namesConst"/device:CPU:0*
dtype0*
_output_shapes
:*�
value�B�BVariableBconv1/biasesBconv1/biases/MomentumBconv1/weightsBconv1/weights/MomentumBconv2/biasesBconv2/biases/MomentumBconv2/weightsBconv2/weights/MomentumBconv3/biasesBconv3/biases/MomentumBconv3/weightsBconv3/weights/MomentumBconv4/biasesBconv4/biases/MomentumBconv4/weightsBconv4/weights/MomentumBconv5/biasesBconv5/biases/MomentumBconv5/weightsBconv5/weights/Momentum
�
save/RestoreV2/shape_and_slicesConst"/device:CPU:0*
dtype0*
_output_shapes
:*=
value4B2B B B B B B B B B B B B B B B B B B B B B 
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
save/Assign_1Assignconv1/biasessave/RestoreV2:1*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*
_class
loc:@conv1/biases
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
save/Assign_3Assignconv1/weightssave/RestoreV2:3*
use_locking(*
T0* 
_class
loc:@conv1/weights*
validate_shape(*&
_output_shapes
: 
�
save/Assign_4Assignconv1/weights/Momentumsave/RestoreV2:4*
T0* 
_class
loc:@conv1/weights*
validate_shape(*&
_output_shapes
: *
use_locking(
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
save/Assign_6Assignconv2/biases/Momentumsave/RestoreV2:6*
use_locking(*
T0*
_class
loc:@conv2/biases*
validate_shape(*
_output_shapes
:@
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
save/Assign_11Assignconv3/weightssave/RestoreV2:11*
T0* 
_class
loc:@conv3/weights*
validate_shape(*'
_output_shapes
:@�*
use_locking(
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
save/Assign_13Assignconv4/biasessave/RestoreV2:13*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0*
_class
loc:@conv4/biases
�
save/Assign_14Assignconv4/biases/Momentumsave/RestoreV2:14*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0*
_class
loc:@conv4/biases
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
save/Assign_16Assignconv4/weights/Momentumsave/RestoreV2:16*
T0* 
_class
loc:@conv4/weights*
validate_shape(*(
_output_shapes
:��*
use_locking(
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
save/Assign_18Assignconv5/biases/Momentumsave/RestoreV2:18*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*
_class
loc:@conv5/biases
�
save/Assign_19Assignconv5/weightssave/RestoreV2:19*
validate_shape(*'
_output_shapes
:�*
use_locking(*
T0* 
_class
loc:@conv5/weights
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
	step/tagsConst*
dtype0*
_output_shapes
: *
valueB
 Bstep
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
valueB Bconv2/biases_1*
dtype0*
_output_shapes
: 
j
conv2/biases_1HistogramSummaryconv2/biases_1/tagconv2/biases/read*
_output_shapes
: *
T0
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
dtype0*
_output_shapes
: * 
valueB Bconv4/weights_1
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
conv5/biases/Momentum:0conv5/biases/Momentum/Assignconv5/biases/Momentum/read:02)conv5/biases/Momentum/Initializer/zeros:0Em��8      �Ӻ	�K�(��A*�q

step    

lossR�>
�
conv1/weights_1*�	    �A��    4?�?     `�@!  @p8�)\��i.9@2�����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6���82���bȬ�0���VlQ.��7Kaa+��[^:��"��S�F !��.����ڋ��vV�R9��T7���>h�'��f�ʜ�7
������6�]���O�ʗ�����Zr[v���h���`�8K�ߝ�>�?�s��>�FF�G ?����?f�ʜ�7
?�T7��?�vV�R9?ji6�9�?�S�F !?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��82?�u�w74?��%�V6?��%>��:?d�\D�X=?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?�������:�             �K@      i@      j@      j@      d@     ``@      `@     �`@     @V@     @Y@     @T@     @V@      Q@      L@      M@      K@      N@      F@     �A@      G@      A@      >@      7@      D@      ?@      .@      3@      0@      6@      .@      ,@      *@      &@      @      "@      "@      @      @      "@      @       @      @      @      @      @      @      @      @              @      �?               @               @               @              �?              @              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?               @      �?       @               @      �?              @              @      @      @      @       @       @      @              @      @      @       @       @      @      @      *@      *@      $@      &@      $@      &@      .@      &@      3@      1@      6@      >@     �C@     �B@      A@     �D@      H@      K@     �J@      I@      O@      O@     �Q@     �R@     �R@     �W@      [@      _@     �]@     �_@     �c@     �d@     �e@     �m@     �I@        
O
conv1/biases_1*=      @@2        �-���q=�������:              @@        
�
conv2/weights_1*�	   @_���    w��?      �@!  �0<<!�)���3`E@2����g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��})�l a��ߊ4F��h���`�8K�ߝ���(��澢f���侕XQ�þ��~��¾5�"�g���0�6�/n���u��gr��R%������
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>�f����>��(���>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?�������:�             �@     F�@     ��@     ��@     ��@     D�@     ��@     �@     ��@     �@     ��@     P�@     ��@     @�@     0�@     ��@     x�@     �{@     `|@      x@      x@     0u@     @s@     q@     �o@      i@     �j@     �d@      f@      d@     �^@     �_@     �^@     @Y@     @Z@     �T@     �U@      U@      M@     �O@      L@      E@     �D@     �H@      B@      9@      8@      <@      6@      >@      ,@      1@      &@       @      .@       @      0@       @      @      $@      "@       @      @      @      $@      @      @      @      �?      @      @      @      @      @      @       @      @      �?              �?      �?       @               @              �?               @              �?              �?              �?              �?       @      �?      �?              �?              �?      �?       @      �?              �?      �?              �?      @      @      �?      �?      @      @      @      @      @      @      @      @      (@      @       @      @      .@      (@      (@      *@      "@      ,@      5@      ,@      3@      7@      &@      ,@      3@     �D@      =@     �@@      A@     �J@     �D@      N@     �F@      O@     �Q@     �Q@     @R@      W@      V@     �\@     �[@      d@     �b@     `c@     �f@      g@     �h@     �h@     �q@     0p@     �s@     �t@     �u@     �x@     0{@     {@     �@     P�@     ؃@     X�@     �@     P�@     ��@     �@     X�@     ��@     �@     ��@     0�@     ̛@     D�@     ��@     x�@        
O
conv2/biases_1*=      P@2        �-���q=�������:              P@        
�
conv3/weights_1*�	   `"+��   `�*�?      �@!   ��{@))��h�IU@2�����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(龢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾K+�E��Ͼ['�?�;;�"�qʾ
�/eq
Ⱦ[#=�؏�>K���7�>��~]�[�>��>M|K�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?�������:�             ��@     ��@     Ȧ@     D�@     ¢@     d�@      �@     <�@     �@     �@     �@     ��@     �@     \�@     8�@     ��@     (�@     ��@     X�@     ��@     ��@     �@     `y@     z@     Pv@      t@     �p@     �p@     @q@     �l@     @l@      f@      e@     �b@     �a@     �^@      \@     �[@     �Y@      W@     @Q@      R@      P@     �O@     �G@     �H@      N@     �C@     �G@      <@      C@      8@      ;@      7@      .@      2@      7@      $@      @      (@      .@      &@      @      (@      @      @       @       @      @      @      "@      @      @      �?      @      @      �?       @      @      �?              �?      �?       @       @      �?               @       @      �?              @              �?      �?              �?              �?              �?              �?              �?              �?              �?      @      �?              �?              �?      @              �?      �?               @      �?      �?       @      @       @      @      @      �?      @      "@      @      @      @      @      @      @      @      @      &@      ,@      (@      0@      5@      1@      $@      9@      6@      =@      6@      ;@      B@      F@     �H@      E@     �L@     �L@     �P@     �S@      O@      T@     �T@     @X@     �X@      `@     �_@     �`@     `g@      e@     �h@     �k@     @l@     0p@     �q@      r@     @t@     @y@     �x@     �z@     �~@     8�@     (�@     �@     ��@     ��@      �@     h�@     ��@     `�@     ��@     L�@     �@     ̙@     Ԝ@     D�@     ��@     �@     N�@     Ԧ@     ܩ@     @�@        
O
conv3/biases_1*=      `@2        �-���q=�������:              `@        
�
conv4/weights_1*�	   �n���    ���?      �@!  ���M8@)3&P"�Ne@2�Ӗ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"�ji6�9���.����ڋ��vV�R9��5�i}1���d�r�x?�x��>h�'��f�ʜ�7
��������[���FF�G �I��P=��pz�w�7����(��澢f���侄iD*L��>E��a�W�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>��[�?1��a˲?6�]��?����?��d�r?�5�i}1?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�������:�             @^@     X�@     |�@     ��@     �@     @�@     ؋@     �@     ؆@     ��@     H�@     ��@     P~@     �|@     @{@      x@     Pw@     �s@     �o@      p@     p@      n@     �l@     �g@      g@      a@      a@     �\@     �^@     @X@     �V@     @T@     @T@     �P@      J@      O@      D@      I@      H@     �E@      B@      =@     �B@      :@      0@     �B@      6@      ,@      9@      0@      *@      &@      .@      .@      $@       @       @       @      @      $@      @      @      @      @       @      @       @      @       @              @       @              �?      �?      �?              �?       @       @      @      �?              �?              �?              @              �?              �?              �?              �?      �?      �?              �?              @       @       @       @      �?      @      �?       @       @       @      @       @      @       @      @      @      @      @      @      (@      @      *@      "@      $@      *@      @      0@      .@      (@      8@      6@      6@      6@      >@      =@      D@     �E@      J@      E@      L@     �O@     �P@      O@     @V@     �Z@     �W@     @\@      ]@     `a@     �a@     �b@     �f@     �d@     �i@     �h@     �n@     0q@     �r@     �t@     �u@     Px@     �x@     �{@      �@     ��@     �@     ��@     ؇@     x�@     H�@     �@     �@     �@     Ĕ@     ԗ@     @^@        
O
conv4/biases_1*=      p@2        �-���q=�������:              p@        
�
conv5/weights_1*�	   ���¿   `��?      �@!   �lb�?)T���I@2��?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8��u�w74���82�I�I�)�(�+A�F�&�U�4@@�$��[^:��"���ڋ��vV�R9�6�]���1��a˲��FF�G �>�?�s����7Kaa+?��VlQ.?�u�w74?��%�V6?d�\D�X=?���#@?�!�A?�T���C?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?�������:�              l@      q@     �q@     pp@      m@     `i@     �i@      d@     `a@     `c@     @a@     @`@     @]@     �Z@      U@     @P@     �Q@     �J@     �K@     �K@      M@     �F@     �A@     �D@     �A@      ?@     �C@      =@      8@      6@      1@      :@      2@      $@      *@      2@      "@      ,@      @      (@       @      @      @      "@      @      @      @      @      @      @       @      �?       @      �?              @      �?      �?       @      �?               @       @               @              �?              �?              �?              �?              �?              �?              �?              �?              �?               @      �?       @      @      �?      @      �?       @      @      "@      "@      @      &@      "@      $@      "@      @      &@      @      0@      1@      ,@      9@      5@      6@      7@      1@      :@      3@      <@      >@     �@@      :@     �D@     �D@      A@      K@      I@      M@     @V@      S@     �S@     �W@     �Z@     ``@      _@     @`@     �e@      g@     �e@     `n@     @k@     �p@     pp@     �r@     �k@        
O
conv5/biases_1*=      <@2        �-���q=�������:              <@        �ѐXV      �q�	s��(��A*ɬ

step  �?

loss���>
�
conv1/weights_1*�	   �5>��   �6i�?     `�@!  ���O�)�2Wa#;@2�����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.�I�I�)�(�+A�F�&�f�ʜ�7
������6�]���1��a˲�>�?�s��>�FF�G ?��d�r?�5�i}1?�T7��?ji6�9�?�S�F !?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?�������:�             �K@      i@      j@     �i@     `d@      `@     �_@     �`@     �U@     @X@      U@     @U@     �R@     �L@     �M@     �K@     �K@     �F@      C@      E@      A@      ?@      9@      C@      @@      0@      3@      *@      5@      0@      1@      (@      (@      @      @       @      @       @      "@      @      @       @      @      @      @      @      �?      @      �?       @               @       @       @       @      �?      �?      �?       @              �?              �?               @              �?              �?      �?              �?              �?               @              �?       @       @       @      �?       @              @      @      �?      @      @      @      �?      @      @      @       @      @       @      $@       @      ,@       @      $@      &@      *@      &@      .@      1@      3@      4@      >@      D@     �@@      C@      E@     �G@      K@      K@     �F@     @P@     �N@     @R@     �Q@     �S@      Y@     �Y@     �^@     �]@      `@     �c@     �d@     �e@     @m@      L@        
�
conv1/biases_1*�	    e�:�   ���F?      @@!  �W�k?)��,�q��>2�d�\D�X=���%>��:��7Kaa+�I�I�)�(�U�4@@�$��[^:��"��S�F !�ji6�9����ڋ��vV�R9���d�r�x?�x��6�]���1��a˲����%ᾙѩ�-߾E��a�Wܾ
�}���>X$�z�>G&�$�>�*��ڽ>�iD*L��>E��a�W�>�f����>��(���>1��a˲?6�]��?�T7��?�vV�R9?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?a�$��{E?
����G?�������:�              �?              �?              �?      �?      �?              �?              �?              �?               @      �?              �?              �?              �?              �?              �?              �?              �?      �?      @      �?      �?      �?              �?      �?              �?      �?      �?               @        
�
conv2/weights_1*�	    ����   `ʺ�?      �@! Dj9� �)
�]�q`E@2����g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G ���Zr[v��I��P=���ߊ4F��h���`�8K�ߝ�a�Ϭ(���(����uE���⾮��%���~]�[Ӿjqs&\�Ѿ;�"�qʾ
�/eq
Ⱦ�u��gr�>�MZ��K�>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�ߊ4F��>})�l a�>pz�w�7�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?�������:�             �@     :�@     ȟ@     ��@     ��@     0�@     ��@     ��@     ��@     �@     ��@      �@     ȉ@     H�@      �@     Ё@     ��@     �{@     �{@     px@     px@      u@     ps@     �p@      p@     �i@     �i@     �e@      e@     �d@     �]@      `@     �]@     @Z@     �Y@     @V@     �S@      W@     �M@     �M@      M@     �B@     �H@      J@      @@      8@      8@      :@      8@      <@      1@      0@      .@      $@      *@      @      $@       @      @       @      (@      $@      @      @      "@      @      @      @       @      @       @      @      @      @      �?      @      @      @               @              �?      �?              �?               @              �?              �?              �?              �?       @      �?      �?      �?              �?              �?              �?              �?      �?               @              �?      �?       @      �?       @       @       @      @      @      @      @      @      @      @      @      $@      "@      @      $@      .@      @      1@      (@      &@      0@      1@      1@      5@      4@      *@      2@     �B@     �@@     �A@     �C@     �H@     �D@     �K@     �F@     @P@     �R@      Q@     �Q@      X@     �U@      \@      \@     �c@     `c@     �b@      g@     �f@      i@     �h@      r@     p@     �s@     0t@     @v@     �x@     @{@     �z@      �@     8�@     �@     P�@     �@     P�@     ��@     �@     h�@     ��@     �@     |�@     �@     �@     `�@     ��@     ��@        
�
conv2/biases_1*�	   @�@�   @j�E?      P@! ���s�d?)�������>2����#@�d�\D�X=�uܬ�@8���%�V6��u�w74���82���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��5�i}1���d�r�x?�x��>h�'��6�]���1��a˲�>�?�s���O�ʗ�����Zr[v��I��P=��})�l a��ߊ4F��h���`�8K�ߝ��uE���⾮��%���~��¾�[�=�k��T�L<��u��6
��        �-���q=.��fc��>39W$:��>��|�~�>���]���>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>��~]�[�>��>M|K�>E��a�W�>�ѩ�-�>�uE����>�f����>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?�T���C?a�$��{E?
����G?�������:�               @              �?       @      �?              �?      �?              �?              �?              �?      �?      �?              �?              �?              �?              �?      �?      �?              �?              �?              �?              �?              �?               @              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?      �?              �?               @      �?      �?      �?      �?               @              �?      �?              �?       @      �?       @      �?       @      �?              �?       @      �?              �?      �?        
�
conv3/weights_1*�	   `i1��   �~2�?      �@! p`I�@)B5�b�IU@2�����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G ���Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%�E��a�Wܾ�iD*L�پ�_�T�l׾['�?�;;�"�qʾ
�/eq
Ⱦ5�"�g���0�6�/n����~]�[�>��>M|K�>�_�T�l�>�iD*L��>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?�������:�             ��@     ��@     Ҧ@     4�@     ֢@     Z�@     ��@     <�@     ��@     ԗ@     �@     ��@     ,�@     L�@      �@     ��@     8�@     ��@     p�@     ��@     ��@     �@     `y@     �z@      v@     �s@     �p@     �p@     @q@     �l@     �l@      e@      e@     �b@     @a@      `@     �[@      Z@     �Y@     @X@     �Q@     @R@     �N@      O@     �G@      K@     �J@     �G@     �B@      ?@     �C@      :@      :@      8@      ,@      7@      4@      @       @      *@      .@      "@      &@      $@      @      @      @      @      @       @      @      @      @      @      @       @       @      �?      @      �?              @      �?      @      �?       @       @              �?      �?       @       @              �?      �?              �?      �?              �?              �?               @              �?      �?               @       @              �?      @      �?       @      �?      @      �?      �?              �?      @       @      �?      @      @      @      @      @      "@      @      @      @      @      @       @      "@      $@      $@      4@      .@      7@      &@      :@      8@      :@      8@      :@      B@      E@      H@     �E@      M@     �I@     �Q@      T@     �O@      T@      T@     �W@      Y@     �`@     �_@     �`@     `g@     �d@     �h@      k@     @m@     0p@     �q@      r@     �s@     `y@     y@     �z@     �~@     p�@     Ё@     h�@     x�@     ��@     �@     x�@     ��@     X�@     ̓@     @�@     �@     ܙ@     Ȝ@     H�@     ��@     �@     V�@     ̦@     ک@     X�@        
�
conv3/biases_1*�	   ��C�   @��R?      `@! ��C\jp?)�߱(Hz�>2�a�$��{E��T���C��!�A�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲��FF�G �>�?�s���pz�w�7��})�l a�f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ��>M|Kվ��~]�[Ӿ;�"�qʾ
�/eq
ȾG&�$��5�"�g���0�6�/n����n�����豪}0ڰ�        �-���q=���m!#�>�4[_>��>jqs&\��>��~]�[�>��>M|K�>E��a�W�>�ѩ�-�>�uE����>�f����>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?a�$��{E?
����G?�qU���I?IcD���L?nK���LQ?�lDZrS?�������:�              �?      �?              �?      �?       @              �?              �?      �?      @      @      �?      @      �?      �?       @      �?       @      �?       @               @              �?      @      �?       @              @              �?              �?      �?      �?      �?              �?               @              �?      �?              �?               @              �?              �?       @              �?              �?              �?              �?      �?       @              �?              �?      �?      �?              @       @              �?      @      @      �?      @       @       @      @              @              @      @      �?       @       @       @       @      @      @      �?      �?              �?              �?              �?        
�
conv4/weights_1*�	    ����   �  �?      �@! �v�]8@)o�1��Ne@2�Ӗ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]�����[���FF�G �O�ʗ�����Zr[v��pz�w�7��})�l a�8K�ߝ�a�Ϭ(���(��澢f����E��a�Wܾ�iD*L�پ�XQ�þ��~��¾E��a�W�>�ѩ�-�>��Zr[v�>O�ʗ��>>�?�s��>1��a˲?6�]��?����?x?�x�?��d�r?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�������:�             �^@     P�@     |�@     ��@      �@     8�@     ��@     �@     ��@     ��@     H�@     ��@     `~@     �|@     P{@      x@      w@     �s@     p@     @p@     �o@      n@      l@     �g@     `g@     �`@     �a@     @\@     �^@      X@     �V@     �T@     @T@     �P@     �I@     �N@      D@      I@     �H@      E@     �B@      <@     �C@      9@      .@      C@      6@      0@      7@      .@      ,@      &@      .@      .@      $@       @      "@      @       @      "@      @      @       @      @              @      @              @              @      �?      �?       @      �?               @      �?      �?      @      �?      �?              �?              �?              �?              �?               @              �?              �?              �?              �?      �?              �?      �?              �?              �?              @      �?       @       @       @      �?      @      @      �?      @      �?      @      @       @      @      @      @      @      *@      @      &@      (@       @      ,@      @      &@      5@      (@      6@      7@      3@      :@      =@      =@      D@     �E@      I@      G@     �J@      O@     �P@      P@     @V@     @Z@     �W@     �\@      ]@     @a@     �a@     �b@     `f@     �d@     �i@     �h@      n@     @q@     �r@     �t@     �u@     Px@     �x@     |@     �@     ��@     0�@     ��@     ȇ@     `�@     X�@     �@     �@     �@     Ȕ@     З@     @^@        
�
conv4/biases_1*�	    �pF�   @6lN?      p@!��ť�I?)�K��p��>2�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
�6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ���(��澢f�����uE���⾮��%��_�T�l׾��>M|Kվjqs&\�ѾK+�E��Ͼ['�?�;;�"�qʾ
�/eq
Ⱦ����ž�XQ�þ5�"�g���0�6�/n���u`P+d����n�������������?�ګ�39W$:���.��fc���Z�TA[�����"�RT��+��y�+pm��mm7&c��`���nx6�X� ��f׽r����tO����f;H�\Q������%���9�e����K��󽉊-��J�'j��p���1���=��]���ݟ��uy�z�����i@4[���Qu�R"�PæҭUݽ�|86	Խ(�+y�6ҽ��.4Nν�!p/�^˽�b1�ĽK?�\��½�
6������Bb�!澽        �-���q=%�f*=\��$�=����z5�=���:�=K?�\���=�b1��=��.4N�=;3����=(�+y�6�=�|86	�=��
"
�=i@4[��=z�����=ݟ��uy�=�/�4��==��]���=��1���='j��p�=��-��J�=�K���=�9�e��=����%�=f;H�\Q�=�tO���=�f׽r��=nx6�X� >�`��>�mm7&c>y�+pm>RT��+�>���">Łt�=	>��f��p>.��fc��>39W$:��>R%�����>�u��gr�>�MZ��K�>����>豪}0ڰ>��n����>�u`P+d�>�XQ��>�����>
�/eq
�>K+�E���>jqs&\��>��~]�[�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?a�$��{E?
����G?IcD���L?k�1^�sO?�������:�              �?              �?               @              �?              @       @       @       @      @               @      @       @      �?               @              @      �?              @      �?      @              �?              �?       @              �?      �?       @      @      �?       @       @               @              �?               @               @              @               @      �?              �?       @       @              �?              �?              �?       @               @              �?      @              �?               @      �?       @      @       @       @              �?      �?              �?              �?              �?              �?      �?      �?              A@              �?              �?              �?              �?              �?      �?               @       @      �?       @       @      @       @               @      �?               @      �?               @       @       @              �?              �?              �?              �?      �?              �?      �?      �?              @      �?              �?      �?              �?      �?              @      �?      �?              �?      @      �?      �?      �?      @       @      @      @       @              �?       @      �?      �?      @       @       @      �?      �?      �?              �?      @       @      @      �?      @       @       @      �?               @      �?              �?               @      �?              �?              �?        
�
conv5/weights_1*�	   �#�¿   ख़�?      �@! `)�G	�?)��t�5I@2��?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9��1��a˲���[��.��fc���X$�z���7Kaa+?��VlQ.?��82?�u�w74?d�\D�X=?���#@?�!�A?�T���C?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?�������:�              l@      q@     pq@     �p@     @m@     @i@     `i@      d@     `a@     `c@     @a@     @`@     �]@     @Z@      U@     @P@     @Q@      K@      L@      J@      N@      F@      B@     �D@     �A@      ?@     �C@      =@      8@      6@      1@      :@      2@      &@      (@      2@      "@      ,@      @      (@       @      @       @      $@      @      @      @      @      @      @       @      �?       @              �?      @       @               @      �?              @      �?              �?      �?              �?              �?              �?              �?              �?              �?              �?              �?              �?               @       @      �?      @      �?      @      �?       @      @      "@      $@      @      (@      "@      "@      $@      @      (@      @      .@      2@      ,@      9@      5@      6@      7@      1@      :@      3@      <@      >@     �@@      ;@      D@      D@     �A@      K@     �H@     �M@      V@     @S@     �S@     �W@     �Z@     @`@      _@     ``@     �e@      g@     �e@     @n@     �k@     �p@     �p@     �r@     �k@        
�
conv5/biases_1*�	   ���   @�>      <@!   a]�4�)��S �E[<2�2!K�R���J��#���j�Z�TA[�����"�RT��+��y�+pm��f׽r����tO����f;H�\Q������%���9�e����K��󽉊-��J�'j��p���1���ݟ��uy�z�����i@4[��;3���н��.4Nν        �-���q=�!p/�^�=��.4N�=;3����=(�+y�6�=�|86	�=��
"
�=���X>�=H�����=PæҭU�=�Qu�R"�=i@4[��=z�����=�/�4��==��]���=��-��J�=�K���=�tO���=�f׽r��=���">Z�TA[�>�������:�              �?      �?      �?              �?      �?              �?      �?      �?              �?      �?               @               @      �?              �?              �?              �?      �?      �?              �?              �?              �?              �?              �?              �?              �?              �?        ^����V      	ѹ�	C%�(��A*��

step   @

lossR�>
�
conv1/weights_1*�	   �:��    G��?     `�@! �[=���)?h�>@2�����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0�U�4@@�$��[^:��"�ji6�9���.����ڋ���d�r�x?�x���FF�G �>�?�s���I��P=��pz�w�7��E��a�W�>�ѩ�-�>�.�?ji6�9�?�S�F !?U�4@@�$?+A�F�&?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?�������:�             �L@     �h@     �i@     `j@     �c@     @`@      `@     ``@      W@     @X@     �T@     �T@     @S@      L@      L@      L@     �J@     �F@      D@      E@      @@     �@@      :@      B@     �@@      5@      1@      *@      3@      .@      2@      .@      "@      @      "@      "@      @      @      $@      @      &@      @      @      @      @      �?       @      @      @      @               @      �?       @       @       @              �?              �?               @      @              �?              �?              �?              �?              �?       @              �?               @      @      �?      �?      �?      �?              @      �?      @      @      @       @      @      @      @      @      @      @      @       @      @      &@      &@      @      ,@      $@      *@      ,@       @      5@      0@      8@      <@     �C@      =@      E@      E@      H@     �K@      J@     �F@     �P@      P@     �Q@     @R@     �S@     �X@      Y@     �_@     �]@     �`@     �b@     �d@      f@      n@      I@        
�
conv1/biases_1*�	    R�D�   �˺X?      @@!  ���?)Ȑ���>2�a�$��{E��T���C��!�A�uܬ�@8���%�V6��u�w74���82���VlQ.��7Kaa+�I�I�)�(��FF�G �>�?�s���})�l a��ߊ4F���_�T�l׾��>M|Kվ5�"�g���0�6�/n���ѩ�-�>���%�>pz�w�7�>I��P=�>�FF�G ?��[�?�vV�R9?��ڋ?�S�F !?�[^:��"?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?d�\D�X=?���#@?�!�A?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?<DKc��T?ܗ�SsW?��bB�SY?�������:�              �?       @              �?              �?              �?      �?              �?              �?              �?              �?               @              �?              �?              �?              �?              �?              �?              �?              �?       @              @      �?      �?               @              �?      �?        
�
conv2/weights_1*�	   ��Ʃ�    �ǩ?      �@! >R{���)�$s��`E@2����g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ���I��P=��pz�w�7���ߊ4F��h���`�8K�ߝ���(��澢f���侮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پjqs&\�ѾK+�E��Ͼ�u`P+d�>0�6�/n�>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>�f����>��(���>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?�������:�             �@     6�@     ��@     ��@     ��@      �@     ��@     �@     ��@     ��@     ��@     (�@     ��@     (�@     0�@     �@     8�@     p|@     �{@     0x@      y@     �t@     �s@     0p@      p@     `j@     �h@      g@     �d@     �d@     �\@      a@     �]@     �Y@     @X@      V@     @U@     @V@     �L@     �P@     �K@     �D@     �G@     �G@      B@      =@      4@      7@      8@      8@      *@      6@      (@      ,@      1@      @      &@      &@      @      @      (@      $@      @      @      @       @      @      @      @      @      @      @       @      @               @       @      �?       @      �?              �?              �?      �?              �?              �?              �?              �?              �?              �?      �?              �?              �?              �?              @       @      �?      �?              @              �?       @       @      �?       @       @      @       @      &@              @      @       @      @      @      "@      "@      ,@       @      *@      $@      (@      4@      1@      *@      *@      2@      2@      ,@      8@     �D@     �@@     �B@     �A@     �N@      =@      L@     �E@     �Q@     �Q@     �P@      R@     @W@      W@     �\@     @\@     �b@      c@     �c@     �e@      g@     �i@     @h@     �q@     �p@     �r@      u@     v@     �x@     `{@     pz@      �@     h�@     ��@     @�@     (�@     8�@     ��@     �@     ��@     ��@     �@     ��@     �@     �@     \�@     ��@     ��@        
�

conv2/biases_1*�
	   �"�R�   @��T?      P@!  3Nvv?)�/�EZ�>2��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0�I�I�)�(�+A�F�&�U�4@@�$��[^:��"�ji6�9���.����ڋ��vV�R9��T7����5�i}1�f�ʜ�7
������6�]���})�l a��ߊ4F����(��澢f���侄iD*L�پ�_�T�l׾39W$:���.��fc���
�}�����4[_>���        �-���q=豪}0ڰ>��n����>�uE����>�f����>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?f�ʜ�7
?>h�'�?x?�x�?��d�r?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?�������:�              �?              �?      �?              �?      �?              �?      �?              �?      �?               @      �?               @              �?              �?       @              �?      �?               @      �?              �?              �?              �?              �?              �?               @              �?              �?              �?              �?              �?              �?              �?              �?      �?              �?       @              �?              �?      @              @              �?      �?       @      �?      �?      �?      �?      �?               @       @      �?        
�
conv3/weights_1*�	    D��    �O�?      �@! ��@)���*�IU@2�����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾;�"�qʾ
�/eq
Ⱦ����ž�XQ�þ��������?�ګ��*��ڽ>�[�=�k�>�����>
�/eq
�>;�"�q�>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?�������:�             h�@     ��@     Ц@     2�@     �@     H�@     ��@     L�@     �@     ��@     ��@     �@     <�@     T�@     ��@     ��@     h�@     ��@     8�@     �@     ��@     �@     `y@     �z@     �u@     �s@     �p@      q@     �p@     �m@     �k@     @e@      e@     �d@     �`@      `@     �[@     �Y@      X@     �X@     �R@     @P@     @P@      P@     �G@      G@      K@      L@     �C@      =@     �@@      ?@      8@      :@      0@      5@      3@       @      *@      ,@      .@      &@       @      &@       @       @      @      @      @      @      @      @      @       @      @      @      @              @               @       @               @      �?      �?      �?              �?              �?      �?      �?      �?      �?              �?      �?              �?              �?              �?              �?              �?      �?              �?              �?              �?              �?       @               @              �?       @               @      �?       @      �?      @      �?               @      @      @      @      @      @      @       @      @       @      @      @      @      @      $@      $@      (@      @      1@      5@      8@      &@      7@      ;@      7@      9@      <@      @@      E@      I@      B@      K@     �M@     �R@     �R@      Q@     �S@      T@     �V@     @Z@     @a@      ^@     @`@      g@     �d@      j@     �j@     �l@     �o@     �q@     `r@     �s@     �y@     �x@      z@     �~@     ��@     ��@     `�@     ��@     ��@     �@     (�@     ��@     T�@     �@     4�@     �@     ę@     ��@     N�@     v�@     &�@     N�@     Ԧ@     Ω@     ��@        
�
conv3/biases_1*�	    ��Z�    #`?      `@! @�����?)���\ǆ?2��m9�H�[���bB�SY�ܗ�SsW��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���T7����5�i}1�>h�'��f�ʜ�7
���[���FF�G �>�?�s���O�ʗ�����Zr[v��})�l a��ߊ4F��8K�ߝ�a�Ϭ(���(�����~]�[Ӿjqs&\�ѾK+�E��Ͼ        �-���q=R%�����>�u��gr�>['�?��>K+�E���>jqs&\��>��~]�[�>���%�>�uE����>8K�ߝ�>�h���`�>�ߊ4F��>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?�������:�              �?      �?              @      �?      �?       @      @       @               @      @       @       @       @      @       @      �?      @              �?              �?              �?               @              �?       @      �?      �?              �?              �?      �?              �?      �?               @              �?              �?              �?              �?              �?      �?              �?              �?               @               @              �?       @      �?      �?              �?              @      @       @      @       @              @      @      @              @      �?      �?      �?      �?      �?      @       @       @       @      @      �?      �?              �?              �?      �?      �?      �?        
�
conv4/weights_1*�	   `1 ��   �7�?      �@! uR��v8@)�f�y�Ne@2�Ӗ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"�ji6�9���.����ڋ��vV�R9��5�i}1���d�r�x?�x��>h�'��f�ʜ�7
��������[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=����(��澢f���侕XQ�þ��~��¾�u`P+d�>0�6�/n�>jqs&\��>��~]�[�>�ߊ4F��>})�l a�>O�ʗ��>>�?�s��>1��a˲?6�]��?����?f�ʜ�7
?>h�'�?�5�i}1?�T7��?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�������:�             �]@     d�@     p�@     ��@     �@     @�@     ؋@     �@     ��@     ��@     @�@     ��@     p~@     �|@     P{@     0x@      w@     �s@      p@     @p@     @o@     `n@     �l@     @g@     @g@      a@     �a@     �\@      ^@     @W@     �V@     @U@     @T@      P@      J@      O@      D@     �H@      I@     �D@      C@      ;@      D@      9@      0@     �B@      6@      1@      1@      2@      0@      (@      0@      &@      (@      $@       @      @      $@       @       @      @      @       @       @      @       @      �?      @      �?      �?      @              �?       @      �?               @       @      �?       @      �?              �?      �?      �?              �?               @              �?              �?              �?              �?              �?              �?       @              �?              �?              @      �?      @       @      �?      �?      @       @              @      �?      @      @      @       @      @      @      @      *@       @      $@      ,@      @      .@      @      ,@      3@      .@      3@      8@      1@      :@      >@      >@     �C@     �E@     �I@     �F@      K@     �O@     @P@      P@      V@      [@     @W@     �\@     �\@     @a@     �a@     �b@     �f@     `d@     @j@     �h@      n@     0q@      s@     �t@     �u@     Px@     �x@      |@     �@     ��@     0�@     ��@     ؇@     p�@     H�@     �@     ��@     �@     Д@     ė@      _@        
�
conv4/biases_1*�	   ��]�   @2�Y?      p@!�C��LS�)�]�	/g�>2�E��{��^��m9�H�[�<DKc��T��lDZrS�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��a�Ϭ(���(��澢f�����uE���⾙ѩ�-߾E��a�Wܾ�iD*L�پ��>M|Kվ��~]�[Ӿ;�"�qʾ
�/eq
Ⱦ����ž��~��¾�[�=�k���*��ڽ���|�~���MZ��K��.��fc���X$�z��T�L<��u��6
����f��p�Łt�=	���R����2!K�R���J�Z�TA[�����"�RT��+��y�+pm��mm7&c��`���nx6�X� ��f׽r����tO����f;H�\Q���9�e����K���'j��p���1���=��]���ݟ��uy�z������Qu�R"�PæҭUݽH����ڽ���X>ؽ��
"
ֽ��.4Nν�!p/�^˽5%����G�L���|_�@V5����M�eӧ���s�����:��        �-���q=V���Ұ�=y�訥=�Bb�!�=�
6����=�d7����=�!p/�^�=(�+y�6�=�|86	�=���X>�=H�����=PæҭU�=�Qu�R"�=i@4[��=�/�4��==��]���=��1���='j��p�=��-��J�=�K���=�9�e��=����%�=f;H�\Q�=�tO���=�f׽r��=nx6�X� >�`��>y�+pm>RT��+�>���">Z�TA[�>�#���j>�J>2!K�R�>��R���>4�e|�Z#>��o�kJ%>���]���>�5�L�>;9��R�>���?�ګ>5�"�g��>G&�$�>
�/eq
�>;�"�q�>['�?��>jqs&\��>��~]�[�>��>M|K�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�T���C?a�$��{E?
����G?�qU���I?k�1^�sO?nK���LQ?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?�������:�              �?              �?               @      �?      �?       @      �?              @       @       @       @       @       @       @              @      @       @              @      �?       @      �?      @      �?              @       @               @              �?       @      �?      �?      �?       @              �?       @      �?              �?              �?              �?       @              �?               @       @              �?      �?              �?              �?              �?              �?      �?      �?      �?              @              �?      �?      �?       @      �?      @      �?              �?              �?      �?               @              �?      �?              �?              �?              �?              �?              �?              =@              �?              �?              �?              �?              �?      �?              �?               @              �?      �?      �?      �?      �?      �?              �?      @      �?              �?       @       @      @               @      �?              �?               @              �?              �?               @       @              @       @              �?              �?       @              �?               @               @      �?      �?      �?              �?      �?              @      �?              �?      �?       @      �?      @       @      @      �?      �?      @      �?       @       @      �?      �?              @       @      @              �?              �?      @      �?               @              �?              �?        
�
conv5/weights_1*�	    r�¿   ����?      �@! ��x	F�?)��nդI@2��?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���bȬ�0���VlQ.��7Kaa+�I�I�)�(�U�4@@�$��[^:��"��FF�G �>�?�s���I��P=�>��Zr[v�>�7Kaa+?��VlQ.?��bȬ�0?d�\D�X=?���#@?�!�A?�T���C?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?�������:�             �k@     0q@     �q@     �p@     @m@     @i@     @i@     @d@      a@     �c@     �a@     �_@     �]@      Z@      U@     �P@      Q@     �K@      L@     �I@      O@     �E@      B@      D@     �A@      @@      C@      >@      8@      6@      .@      <@      1@      (@      (@      1@      $@      *@      @      $@      @      @       @      "@      @      @      @      @      @      @       @      �?      @              @       @      �?       @              �?       @      �?              �?              �?              �?               @              �?              �?              �?      �?              �?              �?               @       @      �?       @       @      @               @      @      "@      "@      @      (@      "@      $@      $@      @      *@      @      0@      1@      ,@      8@      4@      8@      7@      2@      7@      6@      ;@      >@     �@@      :@      E@      C@      B@      K@     �H@     �M@      V@     @S@     �S@     �W@     �Z@      `@      _@     ``@     �e@      g@     �e@     `n@     `k@     �p@     �p@     �r@     �k@        
�
conv5/biases_1*�	   ����   @c>      <@!  @@��@�)�F�4��t<2�%�����i
�k���f��p�Łt�=	���R����2!K�R���J��#���j����"�RT��+��y�+pm��mm7&c�nx6�X� ��f׽r����tO������1���=��]����/�4����
"
ֽ�|86	Խ�!p/�^˽�d7���Ƚ�
6������Bb�!澽G�L������6��(�+y�6�=�|86	�=��
"
�=���X>�=PæҭU�=�Qu�R"�==��]���=��1���=f;H�\Q�=�tO���=���">Z�TA[�>�#���j>2!K�R�>��R���>�������:�              �?              @              �?              �?              �?       @      �?              �?      �?              �?      �?              �?              �?              �?              �?              �?               @              �?              �?              �?               @      �?              �?        �kTU      ���	�<��(��A*��

step  @@

lossK�>
�
conv1/weights_1*�	    <���   `�?     `�@! �Q�ޠ �)�/�{E@2�����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&��[^:��"��S�F !���ڋ��vV�R9�x?�x��>h�'��f�ʜ�7
�pz�w�7��})�l a��ߊ4F��h���`h���`�>�ߊ4F��>��Zr[v�>O�ʗ��>��ڋ?�.�?ji6�9�?�[^:��"?U�4@@�$?�7Kaa+?��VlQ.?��bȬ�0?��82?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?�������:�             �L@     �i@     �h@     �j@     �c@     �`@      `@     �`@      V@      X@     @U@     �S@     �S@     �L@      H@     �O@     �J@     �G@     �C@      E@      B@      <@      :@     �D@      =@      5@      1@      .@      5@      *@      (@      2@      $@      @      (@      "@      @       @      @      @      @      @      @       @      @      @               @      �?      @      �?              �?              @      �?      �?       @      �?      �?      �?      �?              �?               @              �?      �?              �?      �?       @              �?              �?              �?      �?               @              �?               @              @       @      �?      @      @               @       @      �?      @      @      @      @      @      @      @      @      @      $@      *@      $@      "@      $@      (@      1@      $@      0@      1@      <@      >@      ?@      @@      D@      G@     �H@     �I@     �L@      H@     �N@      P@      R@     @P@     �T@     �X@      Z@      ^@     �^@     �_@     �c@     @d@      f@     @n@      J@        
�
conv1/biases_1*�	   @sU�    <�d?      @@!  .�e�?)R�V ?2�ܗ�SsW�<DKc��T�nK���LQ�k�1^�sO�
����G�a�$��{E�d�\D�X=���%>��:���82���bȬ�0�U�4@@�$��[^:��"�>h�'��f�ʜ�7
�>�?�s���O�ʗ���K+�E��Ͼ['�?�;�ѩ�-�>���%�>�f����>��(���>��[�?1��a˲?6�]��?>h�'�?x?�x�?��82?�u�w74?��%�V6?��%>��:?d�\D�X=?�!�A?�T���C?
����G?�qU���I?IcD���L?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?5Ucv0ed?Tw��Nof?�������:�              �?               @              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?      �?              �?              �?      �?               @              �?               @      �?              �?      @      �?              �?               @              �?        
�
conv2/weights_1*�	   ��ө�   ����?      �@! ��Z�)O����aE@2����g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`ѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾG&�$��5�"�g����XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?�������:�             �@     B�@     ��@     Н@     ��@     �@     ��@     �@     t�@     ؏@     ��@     X�@     ؉@      �@     X�@     Ё@     @�@     �|@     P{@     Px@     �x@     �t@     �s@     @p@     �o@     �k@      h@     �f@      e@     �b@      _@     �a@      [@     @[@     �Y@     @S@     @V@     �T@      S@     �O@     �H@     �H@      @@     �M@     �C@      5@      9@      8@      ;@      5@      .@      $@      &@      6@      0@      "@      "@       @      "@      @      .@      (@      @      @      @      @      @       @      @      @       @      @      �?      @      �?      @      @      �?      �?               @              �?      �?              �?              �?              �?              �?      �?              �?              �?       @      �?      �?              �?              �?              �?              �?              �?              �?       @       @       @       @              �?      �?      @      @      @      @      @      �?      @      @      @       @      "@      @      (@      $@      $@      &@      (@      0@      6@      *@      3@      :@       @      2@      8@      @@      @@     �A@      F@     �M@     �B@      I@      G@      P@     �S@      P@     �Q@     �X@     �U@     �[@     @\@     �b@     �d@     @c@     �e@     �f@     �i@      i@     �p@     0q@     �r@     0u@     v@     y@     �z@     �z@     �@     p�@     `�@     p�@     H�@     ��@     ؍@      �@     t�@     ��@     �@     ��@     ��@      �@     `�@     ��@     ��@        
�	
conv2/biases_1*�		    Ʀ[�   ��Wa?      P@!  �T(y�?)T��le�?2��m9�H�[���bB�SY�ܗ�SsW�<DKc��T�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�uܬ�@8���%�V6���82���bȬ�0�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1�x?�x��>h�'��f�ʜ�7
������I��P=��pz�w�7��})�l a���n�����豪}0ڰ�        �-���q=I��P=�>��Zr[v�>>h�'�?x?�x�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?�������:�               @              �?              �?      �?              �?      �?              @       @              �?               @              �?      �?      �?      �?      �?      �?       @              �?              �?              �?              �?      �?              �?               @              �?              �?              �?      �?      �?      �?       @              �?      �?              @               @              �?       @       @      �?       @      �?       @      �?               @      �?              �?               @        
�
conv3/weights_1*�	    �[��   @xy�?      �@! L*aO�@)ኣ~`JU@2�����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ뾢f�����uE���⾄iD*L�پ�_�T�l׾;�"�qʾ
�/eq
Ⱦ�*��ڽ�G&�$���
L�v�Q>H��'ϱS>['�?��>K+�E���>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>�uE����>�f����>��(���>a�Ϭ(�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?�������:�             8�@     ��@     Ҧ@     �@     ��@     D�@     ��@     8�@     �@     ��@     �@     �@     @�@     `�@     ��@     ��@     @�@     ؇@     P�@     ��@     ��@     0@      z@     `z@     0v@     �s@     �p@     `q@     0p@     `m@     �k@     �f@     �c@      e@     �a@      ]@     �]@      X@     @Y@     �V@     �Q@     �Q@     @Q@     @Q@      C@     �J@      K@      D@     �G@      C@      @@      8@      ;@     �@@      0@      0@      .@      ,@      @      ,@      &@      .@      ,@      (@      @      @      @      @      "@      @      @      @      @      @      @               @              @              @      @               @      @      �?       @       @      �?               @              �?               @              �?              �?              �?              �?              �?              �?              �?              �?      �?              @      @      @              �?      @       @      @      �?       @      @      �?      @       @      @      @      @      @      @      "@      @       @       @      ,@      "@      ,@      7@      2@      1@      7@      8@      3@      :@      8@      B@      I@     �C@      E@      J@      I@     �S@     �S@     @Q@     �R@     �T@     @W@     �Z@     @a@     �]@     �`@     �e@     @e@      j@     �j@     �l@     `o@     pr@     @r@     Pt@     �x@     @y@      z@     0~@     (�@     ��@     ��@     x�@     `�@     ȉ@     8�@     ��@     <�@     �@     4�@     ��@     ̙@     ��@     \�@     j�@     6�@     :�@     �@     ĩ@     ��@        
�
conv3/biases_1*�	    �d�   �Eh?      `@!  �����?)�Z��?2�Tw��Nof�5Ucv0ed����%��b���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !��.����ڋ��T7����5�i}1���d�r������6�]���1��a˲�>�?�s���O�ʗ�����Zr[v��I��P=��})�l a��ߊ4F��h���`�        �-���q=['�?��>K+�E���>�ߊ4F��>})�l a�>O�ʗ��>>�?�s��>�FF�G ?��[�?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?�������:�              �?      �?              �?      �?       @       @       @      �?      �?      @      @      @       @       @       @       @      �?      �?      �?      @      �?               @      �?       @              �?              �?              �?       @              �?       @              �?              �?              �?      �?               @              �?              �?              �?      �?      �?              �?      �?       @              �?      �?      @      �?               @      �?              �?               @              @      @      �?              @      @      �?      �?      @      @      @               @      @       @      �?       @      @      �?               @      �?               @        
�
conv4/weights_1*�	   `X��    V�?      �@! B�!��8@)��C�Ne@2�Ӗ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[��I��P=��pz�w�7��8K�ߝ�a�Ϭ(���(��澢f����豪}0ڰ������8K�ߝ�>�h���`�>�ߊ4F��>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�������:�             �\@     p�@     x�@     ��@      �@     P�@     ��@     8�@     ؆@     ��@     0�@     ��@     P~@     �|@     `{@     0x@     @w@     �s@      p@     �o@      p@     @n@     `l@     �g@      g@     @a@      a@     �]@      ^@      W@     �V@      U@     �T@     �N@     �K@      N@     �D@      G@     �H@     �F@      C@      9@     �D@      9@      3@      A@      8@      .@      1@      0@      4@      *@      *@      *@      (@       @      @       @       @       @      @      @      @      @       @      @      �?       @       @       @              �?       @               @      �?              �?              @      �?      �?      �?      �?              �?      �?              �?              �?               @              �?              �?      �?              �?              �?              �?       @              �?              �?               @              @       @              �?      @              @      �?      �?      @       @      @      @      @       @      @      @      @      $@      @      "@      ,@      (@      (@      @      (@      6@      .@      3@      5@      5@      5@      @@      ?@      C@      F@      J@      D@      N@     �M@     @Q@     �O@     �U@     �Z@     �W@     @]@     �[@     `a@     �a@     �b@     �f@     @d@     `j@     �h@     �m@     pq@     �r@      u@     Pu@     `x@     Px@     `|@     �@     p�@     H�@     ��@     Ї@     ��@     8�@     �@     ��@     �@     Ԕ@     ̗@     �^@        
�
conv4/biases_1*�	    	�f�   @��b?      p@!���K�)SO��?2�P}���h�Tw��Nof��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7���uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվjqs&\�ѾK+�E��Ͼ����ž�XQ�þG&�$��5�"�g�����-�z�!�%�����i
�k�Łt�=	���R����2!K�R���J��#���j�Z�TA[�����"�y�+pm��mm7&c��`���nx6�X� ��f׽r����tO����f;H�\Q������%����1���=��]���ݟ��uy�z������Qu�R"�PæҭUݽ��M�eӧ�y�訥�        �-���q=�EDPq�=����/�=��؜��=�d7����=PæҭU�=�Qu�R"�=i@4[��==��]���=��1���=��-��J�=�K���=�9�e��=����%�=f;H�\Q�=�f׽r��=nx6�X� >�mm7&c>y�+pm>RT��+�>���">�J>2!K�R�>��R���>Łt�=	>��f��p>�i
�k>���<�)>�'v�V,>cR�k�e>:�AC)8g>�u��gr�>�MZ��K�>���?�ګ>����>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>�uE����>�f����>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?�������:�              �?               @              �?      �?               @      �?       @      �?      @      @      @      @      �?       @      @      @      @      �?      @       @       @       @      �?      �?       @              �?      @               @      �?      �?      @      �?      �?      �?       @              �?               @              �?              �?              �?      �?      �?              �?       @              @              �?              �?               @      �?              @              �?       @      �?      �?              @              �?      �?              @       @              �?              �?               @              �?              :@              �?              �?              �?      �?              �?              �?      �?              @              �?              �?       @      @               @       @      @       @      �?              �?              �?              �?              �?              �?      �?      �?               @              �?      �?      �?              �?       @      �?              �?              �?               @      �?              �?       @      �?              �?              �?       @      �?      @              @              @              @      @       @      �?       @       @      @       @       @       @              �?      �?       @      �?      @       @      @      �?      �?      @              �?      �?       @      �?               @              �?        
�
conv5/weights_1*�	   `��¿   �+��?      �@! `mf�W�?)Tފ�[I@2��?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"�O�ʗ�����Zr[v��1��a˲?6�]��?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?d�\D�X=?���#@?�!�A?�T���C?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?�������:�              l@     q@     �q@     �p@     @m@     @i@     `i@     �c@     �a@     �c@     `a@      `@      ]@     �Z@      U@     �P@     �P@      L@      L@      J@      O@      F@      A@      E@      A@      ?@      D@      >@      4@      9@      ,@      =@      1@      &@      ,@      0@      "@      ,@      @      "@      @      @      @      "@       @      @       @      @      @      @       @              @              �?      @      @              @              �?      �?      �?      �?              �?       @              �?              �?              �?              �?              �?              �?              �?              @      �?      �?       @       @      @      �?      @      @       @      "@      @      (@       @      (@      "@      @      .@       @      0@      ,@      0@      7@      4@      :@      4@      4@      9@      4@      ;@      =@     �@@      ;@      E@      B@     �C@      J@     �H@      N@      V@     �R@     �S@     �W@     �Z@     �_@     �_@     ``@     �e@      g@     �e@     `n@     @k@     �p@     �p@     �r@     �k@        
�
conv5/biases_1*�	    հ!�    �7>      <@!   ��M�)⬲�nԁ<2�4�e|�Z#���-�z�!�%�����i
�k���f��p�Łt�=	���R����2!K�R���J��#���j�RT��+��y�+pm��mm7&c��`�������%���9�e����K��󽉊-��J�'j��p��/�4��ݟ��uy�z�����i@4[�����X>ؽ��
"
ֽ�d7����=�!p/�^�=ݟ��uy�=�/�4��==��]���=��1���=nx6�X� >�`��>�mm7&c>�#���j>�J>2!K�R�>Łt�=	>��f��p>�������:�              �?      �?      �?      @      �?      �?              �?      �?              �?               @              �?      �?               @              �?              �?              �?              �?              �?              �?              �?      �?              �?      �?              �?        ��C\8U      ���?	h@�(��A*��

step  �@

loss�L�>
�
conv1/weights_1*�	   �V���   �-�?     `�@!  �͈���)�����O@2�����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+��T7����5�i}1�6�]���1��a˲���[���FF�G �>�?�s���})�l a��ߊ4F��})�l a�>pz�w�7�>��Zr[v�>O�ʗ��>�vV�R9?��ڋ?�.�?ji6�9�?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?�������:�             �M@     `i@     `h@     �j@      d@     �`@      `@     �_@     �W@     �W@     �U@      T@      T@     �J@      I@      N@      J@     �H@      C@     �E@     �B@      >@      4@     �E@      ;@      6@      6@      .@      1@      0@      $@      5@      $@      "@      @      3@      @      @      @      @      @      @      @      �?      @       @              @              @      �?      @       @      @              �?      �?              @              �?               @      �?              �?              �?              �?              �?              �?       @      �?              �?              �?              �?              �?      �?      �?      �?      �?      @      �?      �?      �?      @       @      @       @      @      @      @      @      @      @       @      @      $@      (@      @      @      *@      0@      @      ,@      *@      3@      5@      1@     �A@     �B@      6@      G@      F@     �G@      J@     �M@      F@     �N@      Q@     @Q@      Q@     �T@     �V@      ]@     @[@     �^@      `@      d@     �c@     �e@      n@     �M@        
�
conv1/biases_1*�	   ��4a�   ���k?      @@!  H�ћ�?)�_�?2����%��b��l�P�`���bB�SY�ܗ�SsW�<DKc��T�IcD���L��qU���I�
����G��vV�R9��T7���O�ʗ�����Zr[v���iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿ�XQ�þ��~��¾})�l a�>pz�w�7�>��Zr[v�>O�ʗ��>ji6�9�?�S�F !?�[^:��"?�7Kaa+?��VlQ.?�T���C?a�$��{E?�qU���I?IcD���L?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?ߤ�(g%k?�N�W�m?�������:�              �?              �?      �?              �?      �?              �?              �?              �?              �?              �?              �?              �?               @      �?              �?               @              �?              �?      �?              �?      �?       @      �?       @      �?       @              �?        
�
conv2/weights_1*�	   @���   �U'�?      �@! ]	ם��)��O��bE@2����g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s�����Zr[v��I��P=��})�l a��ߊ4F��8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%�E��a�Wܾ�iD*L�پ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;;�"�qʾ39W$:���.��fc���.��fc��>39W$:��>�XQ��>�����>;�"�q�>['�?��>�f����>��(���>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?�������:�             ��@     P�@     l�@     ��@     ��@     $�@     x�@     4�@     `�@     �@     x�@     0�@     x�@     ��@     ��@     ��@     ��@     �|@     �z@     �x@     Px@     Pt@     t@     �p@     �m@     �m@     @h@     �e@     �d@      c@      `@     @a@     @Z@     @\@      Y@     @S@     �U@     @V@     �Q@     �O@      L@      I@     �A@      I@     �D@      8@      4@      ;@      9@      9@      $@      4@      ,@      (@      .@      "@      *@      "@      (@      @      $@      @      @      @       @      @      @       @      @      @      @      @       @      @      @               @       @      �?              �?               @              �?      �?               @      �?               @              �?      �?              �?              �?              �?              �?              �?              �?              �?               @      �?       @              �?      @      �?       @      @      �?      �?      �?      �?      @      @      @       @      @       @       @      @      @      @      (@      1@      $@      &@      ,@      ,@      0@      1@      0@      >@      .@      2@      6@      @@      <@     �A@     �A@      M@      H@     �H@      G@     �N@     �R@     �P@     �R@      V@     �W@     �Z@     @\@     �b@     @d@     �d@     �e@     �e@     �j@     �h@     0p@     r@     �r@      t@     `w@     �x@      z@     P{@     `@     ��@     ��@     8�@     8�@     P�@     ��@     ��@     ��@     ��@      �@     ��@     �@     ��@     H�@     ��@     �@        
�	
conv2/biases_1*�		   �1ud�    � h?      P@!  ��Ê?)�g;��q?2�Tw��Nof�5Ucv0ed����%��b��l�P�`�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82��7Kaa+�I�I�)�(�U�4@@�$��[^:��"��S�F !�ji6�9����ڋ��vV�R9������6�]���>�?�s���O�ʗ���a�Ϭ(���(����uE���⾮��%�        �-���q=['�?��>K+�E���>O�ʗ��>>�?�s��>>h�'�?x?�x�?+A�F�&?I�I�)�(?�7Kaa+?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?�������:�              �?      �?      �?              �?      �?      �?      �?      �?              @      �?              �?              �?      �?              �?       @              �?              �?      �?      �?              �?              �?              �?              �?              �?               @              �?              �?              �?              �?      �?              �?      �?              �?      �?      @       @              @               @      �?              @      �?      �?      �?      �?               @      �?              �?               @        
�
conv3/weights_1*�	   �Os��   �MĮ?      �@! ����@)�[j�JU@2�����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ;�"�qʾ
�/eq
Ⱦ����ž�XQ�þ�MZ��K���u��gr�����?�ګ>����>;�"�q�>['�?��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>�uE����>�f����>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?�������:�             ��@     �@     ��@     2�@     �@     L�@     ��@     8�@     �@     ��@      �@     �@     H�@     \�@     ��@     ��@     @�@     ��@     H�@     ��@     X�@     p@     0{@     �y@     0w@     ps@     �p@     @q@      p@     �l@     �k@     �f@      d@      e@     `a@     �_@     @Z@     @Y@     @Y@     �R@     �T@     �Q@     �Q@     �N@     �K@     �E@      I@      A@     �G@      A@     �@@      ;@      @@      <@      6@      6@      ,@      (@      $@      .@      "@      (@      *@      *@      @      @       @      @       @      @      @      @      @      @      @      �?      @              �?      @              �?       @      @               @      �?              �?      �?      �?              �?               @              �?       @              �?              �?              �?              �?              �?              �?      �?      �?              �?              �?      �?      �?      @      @      �?      @      �?       @      @       @      @      @      @      @      @      @      @      @      @      @      @      @      @      &@      "@      @      1@      2@      1@      3@      8@      3@      :@      6@      :@     �D@     �D@      D@      F@      K@      J@     �O@      U@     �Q@     �N@     �V@      Z@     @X@      b@     �^@     �`@     �d@     `f@      i@     �j@     �l@     �n@     �r@     �r@     0t@     �x@     y@      z@     �~@     0�@     x�@     X�@     X�@     ��@     �@     ��@     ��@     L�@     �@     8�@     ��@     ��@     �@     L�@     \�@     ,�@     F�@      �@     ��@     �@        
�
conv3/biases_1*�	   ��Pq�    ��t?      `@!  x�{|�?)~�v�!�)?2�uWy��r�;8�clp�ߤ�(g%k�P}���h��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.���vV�R9��T7���x?�x��>h�'��f�ʜ�7
�������FF�G �>�?�s���I��P=��pz�w�7��        �-���q=
�/eq
�>;�"�q�>E��a�W�>�ѩ�-�>})�l a�>pz�w�7�>1��a˲?6�]��?��d�r?�5�i}1?�T7��?��ڋ?�.�?ji6�9�?�S�F !?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?�N�W�m?;8�clp?hyO�s?&b՞
�u?�������:�              �?              �?               @      �?      @      @      @       @      �?      @      �?      �?      �?       @       @       @      @      �?      �?              �?       @      �?       @      �?              �?      �?              �?              �?              @              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?       @              �?       @      @              �?      �?              �?       @      �?      �?       @      @      �?      �?       @      @      @              �?      @       @       @      @              @               @      @      �?       @              �?              @              �?              �?        
�
conv4/weights_1*�	   ����   �q�?      �@! ��:��8@)�� 'Oe@2�Ӗ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������1��a˲���[����(��澢f���侮��%�>�uE����>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>1��a˲?6�]��?����?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�������:�             �\@     h�@     ��@     ��@     $�@     P�@     ��@     8�@     ��@     ��@     ��@     Ђ@     ~@     @}@     �z@     Px@      w@     �s@     p@     p@     �o@     �n@     @l@     �g@     `g@      a@     �`@     @_@     @\@     �X@      W@     �S@      U@     �N@     �K@     �M@      E@     �G@     �H@      E@     �C@      <@      C@      =@      1@     �@@      ;@      .@      1@      .@      .@      *@      ,@      *@      *@       @       @      @      @      "@      @      @      @      @              @       @               @      �?       @      �?      �?       @      �?      @              �?      �?       @              �?      �?       @               @               @              �?               @              �?               @       @              �?      �?      �?              �?      @      �?               @      �?               @       @       @      @      @      @      @      @      @       @      @      "@      $@      @      @      $@      .@      &@      "@      (@      3@      0@      2@      9@      3@      7@      :@      A@     �C@      E@      I@     �E@      M@      O@      Q@     �O@      U@      [@     @X@     �]@     @Z@     �a@     `a@      c@      g@     �c@     �j@      i@     @m@     @q@      s@      u@     �u@      x@     �x@     |@      �@     ��@     @�@     ��@     Ї@     ��@     8�@     ��@     �@     �@     Ԕ@     ��@     �_@        
�
conv4/biases_1*�	   �do�   `4�g?      p@! y�'�\�)�U;Pr$?2�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[����Zr[v��I��P=���h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ��>M|Kվ��~]�[Ӿ['�?�;;�"�qʾ
�/eq
Ⱦ��|�~���MZ��K��4��evk'���o�kJ%�4�e|�Z#���-�z�!�%�����i
�k���f��p�Łt�=	��J��#���j�Z�TA[�����"�RT��+��y�+pm�nx6�X� ��f׽r���f;H�\Q������%���9�e����K���EDPq���8�4L���        �-���q=�/k��ڂ=̴�L���=�Į#��=���6�=�d7����=�!p/�^�=�/�4��==��]���='j��p�=��-��J�=�K���=�9�e��=f;H�\Q�=�tO���=�`��>�mm7&c>y�+pm>RT��+�>���">Z�TA[�>�#���j>�J>2!K�R�>��R���>Łt�=	>��f��p>�i
�k>%���>��-�z�!>4�e|�Z#>��o�kJ%>���m!#�>�4[_>��>�u��gr�>�MZ��K�>���]���>�5�L�>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>��~���>�XQ��>jqs&\��>��~]�[�>�iD*L��>E��a�W�>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?��d�r?�5�i}1?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?5Ucv0ed?Tw��Nof?P}���h?�������:�              �?              �?              �?      �?              �?      �?      �?      �?       @      �?      @      @      @      �?      �?      @      @      @       @              @      �?      �?      @      @              �?      @      �?      �?       @              �?       @      �?      �?      �?      �?              �?      �?       @              �?              �?              �?              �?       @      �?              �?              �?      �?              �?               @       @              �?               @              �?      �?      �?       @       @              �?      �?               @      @              �?              �?       @      �?              �?              9@              �?              �?              �?               @               @              �?              �?              @      �?      �?               @      �?      �?       @               @      �?      �?      �?              �?      �?              �?              �?              �?              �?               @              �?              �?              @              �?      �?              �?              �?       @      �?      �?      �?              �?      @               @              �?              �?              @       @              �?      @      @       @               @       @      @      �?      �?      �?      �?              @      �?      �?       @      @       @      @      @       @      �?       @      �?       @               @      @        
�
conv5/weights_1*�	    O�¿   @���?      �@! Ы"�1@)�"�.vI@2��?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8�+A�F�&�U�4@@�$��[^:��"��S�F !��vV�R9��T7���O�ʗ�����Zr[v��x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?�7Kaa+?��VlQ.?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?�������:�              l@     q@     pq@     �p@     �l@     `i@      i@      d@     �a@     @c@     `a@     @`@     �\@     �Z@      U@     �P@     �P@     �L@      L@      I@     �O@     �F@      A@     �D@      B@      ;@     �D@      @@      4@      6@      0@      =@      2@      &@      *@      ,@      (@      ,@      @       @      @       @      "@       @      @       @       @      @      "@      @      @              @      �?      �?      