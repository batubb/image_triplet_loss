       �K"	   (*��Abrain.Event:2f~S�     �:E	��4(*��A"��
�
anchor_inputPlaceholder*
dtype0*0
_output_shapes
:���������2�*%
shape:���������2�
�
positive_inputPlaceholder*
dtype0*0
_output_shapes
:���������2�*%
shape:���������2�
�
negative_inputPlaceholder*
dtype0*0
_output_shapes
:���������2�*%
shape:���������2�
�
.conv1/weights/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*%
valueB"             * 
_class
loc:@conv1/weights
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
,conv1/weights/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *
valueB
 *�Er=* 
_class
loc:@conv1/weights
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
(conv1/weights/Initializer/random_uniformAdd,conv1/weights/Initializer/random_uniform/mul,conv1/weights/Initializer/random_uniform/min*
T0* 
_class
loc:@conv1/weights*&
_output_shapes
: 
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
conv1/weights/AssignAssignconv1/weights(conv1/weights/Initializer/random_uniform*
use_locking(*
T0* 
_class
loc:@conv1/weights*
validate_shape(*&
_output_shapes
: 
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
conv1/biases/AssignAssignconv1/biasesconv1/biases/Initializer/zeros*
T0*
_class
loc:@conv1/biases*
validate_shape(*
_output_shapes
: *
use_locking(
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
.conv2/weights/Initializer/random_uniform/shapeConst*%
valueB"          @   * 
_class
loc:@conv2/weights*
dtype0*
_output_shapes
:
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
dtype0*&
_output_shapes
: @*

seed *
T0* 
_class
loc:@conv2/weights*
seed2 
�
,conv2/weights/Initializer/random_uniform/subSub,conv2/weights/Initializer/random_uniform/max,conv2/weights/Initializer/random_uniform/min*
_output_shapes
: *
T0* 
_class
loc:@conv2/weights
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
model/conv2/Conv2DConv2Dmodel/conv1/MaxPool2D/MaxPoolconv2/weights/read*
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
dtype0*'
_output_shapes
:@�*
shared_name * 
_class
loc:@conv3/weights*
	container *
shape:@�
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
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME*0
_output_shapes
:���������&�*
	dilations
*
T0
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
,conv4/weights/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *
valueB
 *   >* 
_class
loc:@conv4/weights
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
conv4/biases/Initializer/zerosConst*
valueB�*    *
_class
loc:@conv4/biases*
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
conv4/biases/AssignAssignconv4/biasesconv4/biases/Initializer/zeros*
T0*
_class
loc:@conv4/biases*
validate_shape(*
_output_shapes	
:�*
use_locking(
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
:����������*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(
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
conv5/weights/AssignAssignconv5/weights(conv5/weights/Initializer/random_uniform*
use_locking(*
T0* 
_class
loc:@conv5/weights*
validate_shape(*'
_output_shapes
:�
�
conv5/weights/readIdentityconv5/weights*'
_output_shapes
:�*
T0* 
_class
loc:@conv5/weights
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
conv5/biases/AssignAssignconv5/biasesconv5/biases/Initializer/zeros*
use_locking(*
T0*
_class
loc:@conv5/biases*
validate_shape(*
_output_shapes
:
q
conv5/biases/readIdentityconv5/biases*
T0*
_class
loc:@conv5/biases*
_output_shapes
:
j
model/conv5/dilation_rateConst*
dtype0*
_output_shapes
:*
valueB"      
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
model/conv5/MaxPool2D/MaxPoolMaxPoolmodel/conv5/BiasAdd*
T0*
data_formatNHWC*
strides
*
ksize
*
paddingSAME*/
_output_shapes
:���������
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
valueB"      *
dtype0*
_output_shapes
:
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
model_1/conv2/Conv2DConv2Dmodel_1/conv1/MaxPool2D/MaxPoolconv2/weights/read*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME*/
_output_shapes
:���������K@*
	dilations

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
model_1/conv3/Conv2DConv2Dmodel_1/conv2/MaxPool2D/MaxPoolconv3/weights/read*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME*0
_output_shapes
:���������&�*
	dilations
*
T0
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
model_1/conv3/MaxPool2D/MaxPoolMaxPoolmodel_1/conv3/conv3/Relu*0
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
model_1/conv4/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
�
model_1/conv4/Conv2DConv2Dmodel_1/conv3/MaxPool2D/MaxPoolconv4/weights/read*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME*0
_output_shapes
:����������*
	dilations

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
model_1/conv4/MaxPool2D/MaxPoolMaxPoolmodel_1/conv4/conv4/Relu*
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
valueB"      *
dtype0*
_output_shapes
:
�
model_2/conv1/Conv2DConv2Dnegative_inputconv1/weights/read*
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
model_2/conv1/BiasAddBiasAddmodel_2/conv1/Conv2Dconv1/biases/read*
T0*
data_formatNHWC*0
_output_shapes
:���������2� 
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
model_2/conv2/Conv2DConv2Dmodel_2/conv1/MaxPool2D/MaxPoolconv2/weights/read*
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
model_2/conv2/MaxPool2D/MaxPoolMaxPoolmodel_2/conv2/conv2/Relu*
ksize
*
paddingSAME*/
_output_shapes
:���������&@*
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
model_2/conv3/Conv2DConv2Dmodel_2/conv2/MaxPool2D/MaxPoolconv3/weights/read*
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
model_2/conv4/Conv2DConv2Dmodel_2/conv3/MaxPool2D/MaxPoolconv4/weights/read*
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
model_2/conv4/BiasAddBiasAddmodel_2/conv4/Conv2Dconv4/biases/read*
data_formatNHWC*0
_output_shapes
:����������*
T0
r
model_2/conv4/conv4/ReluRelumodel_2/conv4/BiasAdd*0
_output_shapes
:����������*
T0
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
model_2/conv5/MaxPool2D/MaxPoolMaxPoolmodel_2/conv5/BiasAdd*
ksize
*
paddingSAME*/
_output_shapes
:���������*
T0*
data_formatNHWC*
strides

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
'model_2/Flatten/flatten/Reshape/shape/1Const*
dtype0*
_output_shapes
: *
valueB :
���������
�
%model_2/Flatten/flatten/Reshape/shapePack%model_2/Flatten/flatten/strided_slice'model_2/Flatten/flatten/Reshape/shape/1*

axis *
N*
_output_shapes
:*
T0
�
model_2/Flatten/flatten/ReshapeReshapemodel_2/conv5/MaxPool2D/MaxPool%model_2/Flatten/flatten/Reshape/shape*
Tshape0*(
_output_shapes
:����������*
T0
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
SumSummulSum/reduction_indices*
	keep_dims( *

Tidx0*
T0*#
_output_shapes
:���������
J
Pow/yConst*
dtype0*
_output_shapes
: *
valueB
 *   @
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
divRealDivSummul_1*#
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
:���������*
	keep_dims( *

Tidx0*
T0
L
Pow_2/yConst*
valueB
 *   @*
dtype0*
_output_shapes
: 
i
Pow_2Powmodel_1/Flatten/flatten/ReshapePow_2/y*(
_output_shapes
:����������*
T0
Y
Sum_4/reduction_indicesConst*
value	B :*
dtype0*
_output_shapes
: 
w
Sum_4SumPow_2Sum_4/reduction_indices*#
_output_shapes
:���������*
	keep_dims( *

Tidx0*
T0
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
Sqrt_3SqrtSum_5*
T0*#
_output_shapes
:���������
J
mul_3MulSqrt_2Sqrt_3*#
_output_shapes
:���������*
T0
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
Sum_6SumPow_4Sum_6/reduction_indices*'
_output_shapes
:���������*
	keep_dims(*

Tidx0*
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
sub_2SubSqrt_4Sqrt_5*'
_output_shapes
:���������*
T0
J
add/yConst*
valueB
 *  HC*
dtype0*
_output_shapes
: 
J
addAddsub_2add/y*'
_output_shapes
:���������*
T0
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
VariableV2*
shared_name *
dtype0*
_output_shapes
: *
	container *
shape: 
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
gradients/ShapeConst*
_output_shapes
: *
valueB *
dtype0
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
gradients/Mean_grad/ConstConst*
valueB: *
dtype0*
_output_shapes
:
�
gradients/Mean_grad/ProdProdgradients/Mean_grad/Shape_1gradients/Mean_grad/Const*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
e
gradients/Mean_grad/Const_1Const*
_output_shapes
:*
valueB: *
dtype0
�
gradients/Mean_grad/Prod_1Prodgradients/Mean_grad/Shape_2gradients/Mean_grad/Const_1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
_
gradients/Mean_grad/Maximum/yConst*
_output_shapes
: *
value	B :*
dtype0
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
Truncate( *
_output_shapes
: *

DstT0*

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
"gradients/Maximum_grad/zeros/ConstConst*
_output_shapes
: *
valueB
 *    *
dtype0
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
,gradients/Maximum_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/Maximum_grad/Shapegradients/Maximum_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
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
gradients/Maximum_grad/ReshapeReshapegradients/Maximum_grad/Sumgradients/Maximum_grad/Shape*
Tshape0*'
_output_shapes
:���������*
T0
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
1gradients/Maximum_grad/tuple/control_dependency_1Identity gradients/Maximum_grad/Reshape_1(^gradients/Maximum_grad/tuple/group_deps*3
_class)
'%loc:@gradients/Maximum_grad/Reshape_1*
_output_shapes
: *
T0
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
gradients/add_grad/Sum_1Sum/gradients/Maximum_grad/tuple/control_dependency*gradients/add_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
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
gradients/sub_2_grad/ShapeShapeSqrt_4*
_output_shapes
:*
T0*
out_type0
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
gradients/sub_2_grad/SumSum+gradients/add_grad/tuple/control_dependency*gradients/sub_2_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
gradients/sub_2_grad/ReshapeReshapegradients/sub_2_grad/Sumgradients/sub_2_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
gradients/sub_2_grad/Sum_1Sum+gradients/add_grad/tuple/control_dependency,gradients/sub_2_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
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
gradients/Sqrt_4_grad/SqrtGradSqrtGradSqrt_4-gradients/sub_2_grad/tuple/control_dependency*'
_output_shapes
:���������*
T0
�
gradients/Sqrt_5_grad/SqrtGradSqrtGradSqrt_5/gradients/sub_2_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:���������
_
gradients/Sum_6_grad/ShapeShapePow_4*
_output_shapes
:*
T0*
out_type0
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
gradients/Sum_6_grad/Shape_1Const*
dtype0*
_output_shapes
: *
valueB *-
_class#
!loc:@gradients/Sum_6_grad/Shape
�
 gradients/Sum_6_grad/range/startConst*
dtype0*
_output_shapes
: *
value	B : *-
_class#
!loc:@gradients/Sum_6_grad/Shape
�
 gradients/Sum_6_grad/range/deltaConst*
value	B :*-
_class#
!loc:@gradients/Sum_6_grad/Shape*
dtype0*
_output_shapes
: 
�
gradients/Sum_6_grad/rangeRange gradients/Sum_6_grad/range/startgradients/Sum_6_grad/Size gradients/Sum_6_grad/range/delta*

Tidx0*-
_class#
!loc:@gradients/Sum_6_grad/Shape*
_output_shapes
:
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
gradients/Sum_7_grad/ShapeShapePow_5*
_output_shapes
:*
T0*
out_type0
�
gradients/Sum_7_grad/SizeConst*
value	B :*-
_class#
!loc:@gradients/Sum_7_grad/Shape*
dtype0*
_output_shapes
: 
�
gradients/Sum_7_grad/addAddSum_7/reduction_indicesgradients/Sum_7_grad/Size*
_output_shapes
: *
T0*-
_class#
!loc:@gradients/Sum_7_grad/Shape
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
 gradients/Sum_7_grad/range/startConst*
value	B : *-
_class#
!loc:@gradients/Sum_7_grad/Shape*
dtype0*
_output_shapes
: 
�
 gradients/Sum_7_grad/range/deltaConst*
value	B :*-
_class#
!loc:@gradients/Sum_7_grad/Shape*
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
gradients/Sum_7_grad/Fill/valueConst*
value	B :*-
_class#
!loc:@gradients/Sum_7_grad/Shape*
dtype0*
_output_shapes
: 
�
gradients/Sum_7_grad/FillFillgradients/Sum_7_grad/Shape_1gradients/Sum_7_grad/Fill/value*
_output_shapes
: *
T0*

index_type0*-
_class#
!loc:@gradients/Sum_7_grad/Shape
�
"gradients/Sum_7_grad/DynamicStitchDynamicStitchgradients/Sum_7_grad/rangegradients/Sum_7_grad/modgradients/Sum_7_grad/Shapegradients/Sum_7_grad/Fill*
T0*-
_class#
!loc:@gradients/Sum_7_grad/Shape*
N*
_output_shapes
:
�
gradients/Sum_7_grad/Maximum/yConst*
dtype0*
_output_shapes
: *
value	B :*-
_class#
!loc:@gradients/Sum_7_grad/Shape
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
gradients/Sum_7_grad/ReshapeReshapegradients/Sqrt_5_grad/SqrtGrad"gradients/Sum_7_grad/DynamicStitch*0
_output_shapes
:������������������*
T0*
Tshape0
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
gradients/Pow_4_grad/Shape_1Const*
dtype0*
_output_shapes
: *
valueB 
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
gradients/Pow_4_grad/sub/yConst*
dtype0*
_output_shapes
: *
valueB
 *  �?
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
gradients/Pow_4_grad/SumSumgradients/Pow_4_grad/mul_1*gradients/Pow_4_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
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
gradients/Pow_4_grad/LogLoggradients/Pow_4_grad/Select*(
_output_shapes
:����������*
T0
d
gradients/Pow_4_grad/zeros_like	ZerosLikesub*
T0*(
_output_shapes
:����������
�
gradients/Pow_4_grad/Select_1Selectgradients/Pow_4_grad/Greatergradients/Pow_4_grad/Loggradients/Pow_4_grad/zeros_like*(
_output_shapes
:����������*
T0
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
-gradients/Pow_4_grad/tuple/control_dependencyIdentitygradients/Pow_4_grad/Reshape&^gradients/Pow_4_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/Pow_4_grad/Reshape*(
_output_shapes
:����������
�
/gradients/Pow_4_grad/tuple/control_dependency_1Identitygradients/Pow_4_grad/Reshape_1&^gradients/Pow_4_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/Pow_4_grad/Reshape_1*
_output_shapes
: 
_
gradients/Pow_5_grad/ShapeShapesub_1*
T0*
out_type0*
_output_shapes
:
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
gradients/Pow_5_grad/sub/yConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
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
gradients/Pow_5_grad/SumSumgradients/Pow_5_grad/mul_1*gradients/Pow_5_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
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
gradients/Pow_5_grad/GreaterGreatersub_1gradients/Pow_5_grad/Greater/y*
T0*(
_output_shapes
:����������
i
$gradients/Pow_5_grad/ones_like/ShapeShapesub_1*
_output_shapes
:*
T0*
out_type0
i
$gradients/Pow_5_grad/ones_like/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *  �?
�
gradients/Pow_5_grad/ones_likeFill$gradients/Pow_5_grad/ones_like/Shape$gradients/Pow_5_grad/ones_like/Const*
T0*

index_type0*(
_output_shapes
:����������
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
gradients/Pow_5_grad/Sum_1Sumgradients/Pow_5_grad/mul_3,gradients/Pow_5_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
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
/gradients/Pow_5_grad/tuple/control_dependency_1Identitygradients/Pow_5_grad/Reshape_1&^gradients/Pow_5_grad/tuple/group_deps*1
_class'
%#loc:@gradients/Pow_5_grad/Reshape_1*
_output_shapes
: *
T0
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
(gradients/sub_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/sub_grad/Shapegradients/sub_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
gradients/sub_grad/SumSum-gradients/Pow_4_grad/tuple/control_dependency(gradients/sub_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
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
:*
	keep_dims( *

Tidx0
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
gradients/sub_1_grad/SumSum-gradients/Pow_5_grad/tuple/control_dependency*gradients/sub_1_grad/BroadcastGradientArgs*
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
gradients/sub_1_grad/Sum_1Sum-gradients/Pow_5_grad/tuple/control_dependency,gradients/sub_1_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
^
gradients/sub_1_grad/NegNeggradients/sub_1_grad/Sum_1*
T0*
_output_shapes
:
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
6gradients/model_1/Flatten/flatten/Reshape_grad/ReshapeReshape-gradients/sub_grad/tuple/control_dependency_14gradients/model_1/Flatten/flatten/Reshape_grad/Shape*/
_output_shapes
:���������*
T0*
Tshape0
�
gradients/AddNAddN+gradients/sub_grad/tuple/control_dependency-gradients/sub_1_grad/tuple/control_dependency*
N*(
_output_shapes
:����������*
T0*-
_class#
!loc:@gradients/sub_grad/Reshape
�
2gradients/model/Flatten/flatten/Reshape_grad/ShapeShapemodel/conv5/MaxPool2D/MaxPool*
_output_shapes
:*
T0*
out_type0
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
7gradients/model_1/conv5/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput*gradients/model_1/conv5/Conv2D_grad/ShapeNconv5/weights/read=gradients/model_1/conv5/BiasAdd_grad/tuple/control_dependency*0
_output_shapes
:���������
�*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME
�
8gradients/model_1/conv5/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFiltermodel_1/conv4/MaxPool2D/MaxPool,gradients/model_1/conv5/Conv2D_grad/ShapeN:1=gradients/model_1/conv5/BiasAdd_grad/tuple/control_dependency*'
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
>gradients/model_1/conv5/Conv2D_grad/tuple/control_dependency_1Identity8gradients/model_1/conv5/Conv2D_grad/Conv2DBackpropFilter5^gradients/model_1/conv5/Conv2D_grad/tuple/group_deps*'
_output_shapes
:�*
T0*K
_classA
?=loc:@gradients/model_1/conv5/Conv2D_grad/Conv2DBackpropFilter
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
6gradients/model/conv5/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFiltermodel/conv4/MaxPool2D/MaxPool*gradients/model/conv5/Conv2D_grad/ShapeN:1;gradients/model/conv5/BiasAdd_grad/tuple/control_dependency*
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
7gradients/model_2/conv5/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput*gradients/model_2/conv5/Conv2D_grad/ShapeNconv5/weights/read=gradients/model_2/conv5/BiasAdd_grad/tuple/control_dependency*
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
8gradients/model_2/conv5/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFiltermodel_2/conv4/MaxPool2D/MaxPool,gradients/model_2/conv5/Conv2D_grad/ShapeN:1=gradients/model_2/conv5/BiasAdd_grad/tuple/control_dependency*'
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
>gradients/model_2/conv5/Conv2D_grad/tuple/control_dependency_1Identity8gradients/model_2/conv5/Conv2D_grad/Conv2DBackpropFilter5^gradients/model_2/conv5/Conv2D_grad/tuple/group_deps*K
_classA
?=loc:@gradients/model_2/conv5/Conv2D_grad/Conv2DBackpropFilter*'
_output_shapes
:�*
T0
�
gradients/AddN_1AddN?gradients/model_1/conv5/BiasAdd_grad/tuple/control_dependency_1=gradients/model/conv5/BiasAdd_grad/tuple/control_dependency_1?gradients/model_2/conv5/BiasAdd_grad/tuple/control_dependency_1*
T0*C
_class9
75loc:@gradients/model_1/conv5/BiasAdd_grad/BiasAddGrad*
N*
_output_shapes
:
�
:gradients/model_1/conv4/MaxPool2D/MaxPool_grad/MaxPoolGradMaxPoolGradmodel_1/conv4/conv4/Relumodel_1/conv4/MaxPool2D/MaxPool<gradients/model_1/conv5/Conv2D_grad/tuple/control_dependency*
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
=gradients/model/conv4/BiasAdd_grad/tuple/control_dependency_1Identity.gradients/model/conv4/BiasAdd_grad/BiasAddGrad4^gradients/model/conv4/BiasAdd_grad/tuple/group_deps*
_output_shapes	
:�*
T0*A
_class7
53loc:@gradients/model/conv4/BiasAdd_grad/BiasAddGrad
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
7gradients/model_1/conv4/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput*gradients/model_1/conv4/Conv2D_grad/ShapeNconv4/weights/read=gradients/model_1/conv4/BiasAdd_grad/tuple/control_dependency*
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
8gradients/model_1/conv4/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFiltermodel_1/conv3/MaxPool2D/MaxPool,gradients/model_1/conv4/Conv2D_grad/ShapeN:1=gradients/model_1/conv4/BiasAdd_grad/tuple/control_dependency*
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
4gradients/model_1/conv4/Conv2D_grad/tuple/group_depsNoOp9^gradients/model_1/conv4/Conv2D_grad/Conv2DBackpropFilter8^gradients/model_1/conv4/Conv2D_grad/Conv2DBackpropInput
�
<gradients/model_1/conv4/Conv2D_grad/tuple/control_dependencyIdentity7gradients/model_1/conv4/Conv2D_grad/Conv2DBackpropInput5^gradients/model_1/conv4/Conv2D_grad/tuple/group_deps*
T0*J
_class@
><loc:@gradients/model_1/conv4/Conv2D_grad/Conv2DBackpropInput*0
_output_shapes
:����������
�
>gradients/model_1/conv4/Conv2D_grad/tuple/control_dependency_1Identity8gradients/model_1/conv4/Conv2D_grad/Conv2DBackpropFilter5^gradients/model_1/conv4/Conv2D_grad/tuple/group_deps*
T0*K
_classA
?=loc:@gradients/model_1/conv4/Conv2D_grad/Conv2DBackpropFilter*(
_output_shapes
:��
�
(gradients/model/conv4/Conv2D_grad/ShapeNShapeNmodel/conv3/MaxPool2D/MaxPoolconv4/weights/read* 
_output_shapes
::*
T0*
out_type0*
N
�
5gradients/model/conv4/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput(gradients/model/conv4/Conv2D_grad/ShapeNconv4/weights/read;gradients/model/conv4/BiasAdd_grad/tuple/control_dependency*0
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
6gradients/model/conv4/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFiltermodel/conv3/MaxPool2D/MaxPool*gradients/model/conv4/Conv2D_grad/ShapeN:1;gradients/model/conv4/BiasAdd_grad/tuple/control_dependency*
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
*gradients/model_2/conv4/Conv2D_grad/ShapeNShapeNmodel_2/conv3/MaxPool2D/MaxPoolconv4/weights/read* 
_output_shapes
::*
T0*
out_type0*
N
�
7gradients/model_2/conv4/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput*gradients/model_2/conv4/Conv2D_grad/ShapeNconv4/weights/read=gradients/model_2/conv4/BiasAdd_grad/tuple/control_dependency*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME*0
_output_shapes
:����������*
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
8gradients/model/conv3/MaxPool2D/MaxPool_grad/MaxPoolGradMaxPoolGradmodel/conv3/conv3/Relumodel/conv3/MaxPool2D/MaxPool:gradients/model/conv4/Conv2D_grad/tuple/control_dependency*0
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
=gradients/model/conv3/BiasAdd_grad/tuple/control_dependency_1Identity.gradients/model/conv3/BiasAdd_grad/BiasAddGrad4^gradients/model/conv3/BiasAdd_grad/tuple/group_deps*
_output_shapes	
:�*
T0*A
_class7
53loc:@gradients/model/conv3/BiasAdd_grad/BiasAddGrad
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
?gradients/model_2/conv3/BiasAdd_grad/tuple/control_dependency_1Identity0gradients/model_2/conv3/BiasAdd_grad/BiasAddGrad6^gradients/model_2/conv3/BiasAdd_grad/tuple/group_deps*C
_class9
75loc:@gradients/model_2/conv3/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:�*
T0
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
T0*
data_formatNHWC*
strides
*
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
T0*
data_formatNHWC*
strides
*
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
5gradients/model/conv3/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput(gradients/model/conv3/Conv2D_grad/ShapeNconv3/weights/read;gradients/model/conv3/BiasAdd_grad/tuple/control_dependency*/
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
:���������&@
�
<gradients/model/conv3/Conv2D_grad/tuple/control_dependency_1Identity6gradients/model/conv3/Conv2D_grad/Conv2DBackpropFilter3^gradients/model/conv3/Conv2D_grad/tuple/group_deps*'
_output_shapes
:@�*
T0*I
_class?
=;loc:@gradients/model/conv3/Conv2D_grad/Conv2DBackpropFilter
�
*gradients/model_2/conv3/Conv2D_grad/ShapeNShapeNmodel_2/conv2/MaxPool2D/MaxPoolconv3/weights/read* 
_output_shapes
::*
T0*
out_type0*
N
�
7gradients/model_2/conv3/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput*gradients/model_2/conv3/Conv2D_grad/ShapeNconv3/weights/read=gradients/model_2/conv3/BiasAdd_grad/tuple/control_dependency*/
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
8gradients/model_2/conv3/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFiltermodel_2/conv2/MaxPool2D/MaxPool,gradients/model_2/conv3/Conv2D_grad/ShapeN:1=gradients/model_2/conv3/BiasAdd_grad/tuple/control_dependency*
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
paddingSAME*/
_output_shapes
:���������K@*
T0*
data_formatNHWC*
strides
*
ksize
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
0gradients/model_1/conv2/conv2/Relu_grad/ReluGradReluGrad:gradients/model_1/conv2/MaxPool2D/MaxPool_grad/MaxPoolGradmodel_1/conv2/conv2/Relu*
T0*/
_output_shapes
:���������K@
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
=gradients/model_1/conv2/BiasAdd_grad/tuple/control_dependencyIdentity0gradients/model_1/conv2/conv2/Relu_grad/ReluGrad6^gradients/model_1/conv2/BiasAdd_grad/tuple/group_deps*C
_class9
75loc:@gradients/model_1/conv2/conv2/Relu_grad/ReluGrad*/
_output_shapes
:���������K@*
T0
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
T0*
out_type0*
N* 
_output_shapes
::
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
<gradients/model_2/conv2/Conv2D_grad/tuple/control_dependencyIdentity7gradients/model_2/conv2/Conv2D_grad/Conv2DBackpropInput5^gradients/model_2/conv2/Conv2D_grad/tuple/group_deps*
T0*J
_class@
><loc:@gradients/model_2/conv2/Conv2D_grad/Conv2DBackpropInput*/
_output_shapes
:���������K 
�
>gradients/model_2/conv2/Conv2D_grad/tuple/control_dependency_1Identity8gradients/model_2/conv2/Conv2D_grad/Conv2DBackpropFilter5^gradients/model_2/conv2/Conv2D_grad/tuple/group_deps*
T0*K
_classA
?=loc:@gradients/model_2/conv2/Conv2D_grad/Conv2DBackpropFilter*&
_output_shapes
: @
�
gradients/AddN_7AddN?gradients/model_1/conv2/BiasAdd_grad/tuple/control_dependency_1=gradients/model/conv2/BiasAdd_grad/tuple/control_dependency_1?gradients/model_2/conv2/BiasAdd_grad/tuple/control_dependency_1*C
_class9
75loc:@gradients/model_1/conv2/BiasAdd_grad/BiasAddGrad*
N*
_output_shapes
:@*
T0
�
:gradients/model_1/conv1/MaxPool2D/MaxPool_grad/MaxPoolGradMaxPoolGradmodel_1/conv1/conv1/Relumodel_1/conv1/MaxPool2D/MaxPool<gradients/model_1/conv2/Conv2D_grad/tuple/control_dependency*
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
=gradients/model_1/conv1/BiasAdd_grad/tuple/control_dependencyIdentity0gradients/model_1/conv1/conv1/Relu_grad/ReluGrad6^gradients/model_1/conv1/BiasAdd_grad/tuple/group_deps*0
_output_shapes
:���������2� *
T0*C
_class9
75loc:@gradients/model_1/conv1/conv1/Relu_grad/ReluGrad
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
;gradients/model/conv1/BiasAdd_grad/tuple/control_dependencyIdentity.gradients/model/conv1/conv1/Relu_grad/ReluGrad4^gradients/model/conv1/BiasAdd_grad/tuple/group_deps*A
_class7
53loc:@gradients/model/conv1/conv1/Relu_grad/ReluGrad*0
_output_shapes
:���������2� *
T0
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
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(
�
8gradients/model_1/conv1/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterpositive_input,gradients/model_1/conv1/Conv2D_grad/ShapeN:1=gradients/model_1/conv1/BiasAdd_grad/tuple/control_dependency*&
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
(gradients/model/conv1/Conv2D_grad/ShapeNShapeNanchor_inputconv1/weights/read*
out_type0*
N* 
_output_shapes
::*
T0
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
6gradients/model/conv1/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilteranchor_input*gradients/model/conv1/Conv2D_grad/ShapeN:1;gradients/model/conv1/BiasAdd_grad/tuple/control_dependency*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME*&
_output_shapes
: *
	dilations
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
*gradients/model_2/conv1/Conv2D_grad/ShapeNShapeNnegative_inputconv1/weights/read*
out_type0*
N* 
_output_shapes
::*
T0
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
conv1/weights/Momentum/AssignAssignconv1/weights/Momentum(conv1/weights/Momentum/Initializer/zeros*
use_locking(*
T0* 
_class
loc:@conv1/weights*
validate_shape(*&
_output_shapes
: 
�
conv1/weights/Momentum/readIdentityconv1/weights/Momentum*&
_output_shapes
: *
T0* 
_class
loc:@conv1/weights
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
conv1/biases/Momentum/AssignAssignconv1/biases/Momentum'conv1/biases/Momentum/Initializer/zeros*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*
_class
loc:@conv1/biases
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
conv2/weights/Momentum/AssignAssignconv2/weights/Momentum(conv2/weights/Momentum/Initializer/zeros*
T0* 
_class
loc:@conv2/weights*
validate_shape(*&
_output_shapes
: @*
use_locking(
�
conv2/weights/Momentum/readIdentityconv2/weights/Momentum*&
_output_shapes
: @*
T0* 
_class
loc:@conv2/weights
�
'conv2/biases/Momentum/Initializer/zerosConst*
dtype0*
_output_shapes
:@*
_class
loc:@conv2/biases*
valueB@*    
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
'conv3/biases/Momentum/Initializer/zerosConst*
_class
loc:@conv3/biases*
valueB�*    *
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
conv3/biases/Momentum/readIdentityconv3/biases/Momentum*
T0*
_class
loc:@conv3/biases*
_output_shapes	
:�
�
8conv4/weights/Momentum/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:* 
_class
loc:@conv4/weights*%
valueB"      �      
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
dtype0*
_output_shapes
:*
_class
loc:@conv5/biases*
valueB*    
�
conv5/biases/Momentum
VariableV2*
_class
loc:@conv5/biases*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name 
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
+Momentum/update_conv2/weights/ApplyMomentumApplyMomentumconv2/weightsconv2/weights/MomentumMomentum/learning_rategradients/AddN_8Momentum/momentum*
use_nesterov(*&
_output_shapes
: @*
use_locking( *
T0* 
_class
loc:@conv2/weights
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
+Momentum/update_conv3/weights/ApplyMomentumApplyMomentumconv3/weightsconv3/weights/MomentumMomentum/learning_rategradients/AddN_6Momentum/momentum*
use_nesterov(*'
_output_shapes
:@�*
use_locking( *
T0* 
_class
loc:@conv3/weights
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
+Momentum/update_conv4/weights/ApplyMomentumApplyMomentumconv4/weightsconv4/weights/MomentumMomentum/learning_rategradients/AddN_4Momentum/momentum*
use_nesterov(*(
_output_shapes
:��*
use_locking( *
T0* 
_class
loc:@conv4/weights
�
*Momentum/update_conv4/biases/ApplyMomentumApplyMomentumconv4/biasesconv4/biases/MomentumMomentum/learning_rategradients/AddN_3Momentum/momentum*
T0*
_class
loc:@conv4/biases*
use_nesterov(*
_output_shapes	
:�*
use_locking( 
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
save/SaveV2/tensor_namesConst*
dtype0*
_output_shapes
:*�
value�B�BVariableBconv1/biasesBconv1/biases/MomentumBconv1/weightsBconv1/weights/MomentumBconv2/biasesBconv2/biases/MomentumBconv2/weightsBconv2/weights/MomentumBconv3/biasesBconv3/biases/MomentumBconv3/weightsBconv3/weights/MomentumBconv4/biasesBconv4/biases/MomentumBconv4/weightsBconv4/weights/MomentumBconv5/biasesBconv5/biases/MomentumBconv5/weightsBconv5/weights/Momentum
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
save/Assign_3Assignconv1/weightssave/RestoreV2:3*
use_locking(*
T0* 
_class
loc:@conv1/weights*
validate_shape(*&
_output_shapes
: 
�
save/Assign_4Assignconv1/weights/Momentumsave/RestoreV2:4*
validate_shape(*&
_output_shapes
: *
use_locking(*
T0* 
_class
loc:@conv1/weights
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
save/Assign_6Assignconv2/biases/Momentumsave/RestoreV2:6*
use_locking(*
T0*
_class
loc:@conv2/biases*
validate_shape(*
_output_shapes
:@
�
save/Assign_7Assignconv2/weightssave/RestoreV2:7*
validate_shape(*&
_output_shapes
: @*
use_locking(*
T0* 
_class
loc:@conv2/weights
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
save/Assign_11Assignconv3/weightssave/RestoreV2:11*
validate_shape(*'
_output_shapes
:@�*
use_locking(*
T0* 
_class
loc:@conv3/weights
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
save/Assign_17Assignconv5/biasessave/RestoreV2:17*
T0*
_class
loc:@conv5/biases*
validate_shape(*
_output_shapes
:*
use_locking(
�
save/Assign_18Assignconv5/biases/Momentumsave/RestoreV2:18*
T0*
_class
loc:@conv5/biases*
validate_shape(*
_output_shapes
:*
use_locking(
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
conv3/biases_1HistogramSummaryconv3/biases_1/tagconv3/biases/read*
T0*
_output_shapes
: 
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
conv4/biases_1HistogramSummaryconv4/biases_1/tagconv4/biases/read*
T0*
_output_shapes
: 
c
conv5/weights_1/tagConst*
dtype0*
_output_shapes
: * 
valueB Bconv5/weights_1
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
conv5/biases_1HistogramSummaryconv5/biases_1/tagconv5/biases/read*
_output_shapes
: *
T0
�
Merge/MergeSummaryMergeSummarysteplossconv1/weights_1conv1/biases_1conv2/weights_1conv2/biases_1conv3/weights_1conv3/biases_1conv4/weights_1conv4/biases_1conv5/weights_1conv5/biases_1*
N*
_output_shapes
: "_|�0
5     ��B�	�7(*��AJ��
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
anchor_inputPlaceholder*
dtype0*0
_output_shapes
:���������2�*%
shape:���������2�
�
positive_inputPlaceholder*
dtype0*0
_output_shapes
:���������2�*%
shape:���������2�
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
,conv1/weights/Initializer/random_uniform/mulMul6conv1/weights/Initializer/random_uniform/RandomUniform,conv1/weights/Initializer/random_uniform/sub* 
_class
loc:@conv1/weights*&
_output_shapes
: *
T0
�
(conv1/weights/Initializer/random_uniformAdd,conv1/weights/Initializer/random_uniform/mul,conv1/weights/Initializer/random_uniform/min* 
_class
loc:@conv1/weights*&
_output_shapes
: *
T0
�
conv1/weights
VariableV2* 
_class
loc:@conv1/weights*
	container *
shape: *
dtype0*&
_output_shapes
: *
shared_name 
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
conv1/weights/readIdentityconv1/weights* 
_class
loc:@conv1/weights*&
_output_shapes
: *
T0
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
conv1/biases/AssignAssignconv1/biasesconv1/biases/Initializer/zeros*
_output_shapes
: *
use_locking(*
T0*
_class
loc:@conv1/biases*
validate_shape(
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
model/conv1/Conv2DConv2Danchor_inputconv1/weights/read*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*0
_output_shapes
:���������2� *
	dilations

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
conv2/weights/AssignAssignconv2/weights(conv2/weights/Initializer/random_uniform*
use_locking(*
T0* 
_class
loc:@conv2/weights*
validate_shape(*&
_output_shapes
: @
�
conv2/weights/readIdentityconv2/weights*&
_output_shapes
: @*
T0* 
_class
loc:@conv2/weights
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
model/conv2/Conv2DConv2Dmodel/conv1/MaxPool2D/MaxPoolconv2/weights/read*
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
model/conv2/BiasAddBiasAddmodel/conv2/Conv2Dconv2/biases/read*
data_formatNHWC*/
_output_shapes
:���������K@*
T0
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
dtype0*
_output_shapes
: * 
_class
loc:@conv3/weights*
valueB
 *�[q=
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
conv3/weights/AssignAssignconv3/weights(conv3/weights/Initializer/random_uniform*
T0* 
_class
loc:@conv3/weights*
validate_shape(*'
_output_shapes
:@�*
use_locking(
�
conv3/weights/readIdentityconv3/weights*
T0* 
_class
loc:@conv3/weights*'
_output_shapes
:@�
�
conv3/biases/Initializer/zerosConst*
dtype0*
_output_shapes	
:�*
_class
loc:@conv3/biases*
valueB�*    
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
model/conv3/Conv2DConv2Dmodel/conv2/MaxPool2D/MaxPoolconv3/weights/read*
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
model/conv3/MaxPool2D/MaxPoolMaxPoolmodel/conv3/conv3/Relu*0
_output_shapes
:����������*
T0*
data_formatNHWC*
strides
*
ksize
*
paddingSAME
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
conv4/biases/Initializer/zerosConst*
_class
loc:@conv4/biases*
valueB�*    *
dtype0*
_output_shapes	
:�
�
conv4/biases
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
conv4/biases/AssignAssignconv4/biasesconv4/biases/Initializer/zeros*
T0*
_class
loc:@conv4/biases*
validate_shape(*
_output_shapes	
:�*
use_locking(
r
conv4/biases/readIdentityconv4/biases*
T0*
_class
loc:@conv4/biases*
_output_shapes	
:�
j
model/conv4/dilation_rateConst*
dtype0*
_output_shapes
:*
valueB"      
�
model/conv4/Conv2DConv2Dmodel/conv3/MaxPool2D/MaxPoolconv4/weights/read*
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
dtype0*
_output_shapes
: * 
_class
loc:@conv5/weights*
valueB
 *��>
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
model/conv5/Conv2DConv2Dmodel/conv4/MaxPool2D/MaxPoolconv5/weights/read*
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
model/conv5/BiasAddBiasAddmodel/conv5/Conv2Dconv5/biases/read*
T0*
data_formatNHWC*/
_output_shapes
:���������

�
model/conv5/MaxPool2D/MaxPoolMaxPoolmodel/conv5/BiasAdd*
T0*
data_formatNHWC*
strides
*
ksize
*
paddingSAME*/
_output_shapes
:���������
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
#model/Flatten/flatten/strided_sliceStridedSlicemodel/Flatten/flatten/Shape)model/Flatten/flatten/strided_slice/stack+model/Flatten/flatten/strided_slice/stack_1+model/Flatten/flatten/strided_slice/stack_2*
end_mask *
_output_shapes
: *
T0*
Index0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask 
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
model_1/conv1/MaxPool2D/MaxPoolMaxPoolmodel_1/conv1/conv1/Relu*
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
model_1/conv2/BiasAddBiasAddmodel_1/conv2/Conv2Dconv2/biases/read*
data_formatNHWC*/
_output_shapes
:���������K@*
T0
q
model_1/conv2/conv2/ReluRelumodel_1/conv2/BiasAdd*/
_output_shapes
:���������K@*
T0
�
model_1/conv2/MaxPool2D/MaxPoolMaxPoolmodel_1/conv2/conv2/Relu*
ksize
*
paddingSAME*/
_output_shapes
:���������&@*
T0*
strides
*
data_formatNHWC
l
model_1/conv3/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
�
model_1/conv3/Conv2DConv2Dmodel_1/conv2/MaxPool2D/MaxPoolconv3/weights/read*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME*0
_output_shapes
:���������&�*
	dilations
*
T0
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
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME*0
_output_shapes
:����������
�
model_1/conv4/BiasAddBiasAddmodel_1/conv4/Conv2Dconv4/biases/read*
data_formatNHWC*0
_output_shapes
:����������*
T0
r
model_1/conv4/conv4/ReluRelumodel_1/conv4/BiasAdd*
T0*0
_output_shapes
:����������
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
dtype0*
_output_shapes
:*
valueB"      
�
model_1/conv5/Conv2DConv2Dmodel_1/conv4/MaxPool2D/MaxPoolconv5/weights/read*/
_output_shapes
:���������
*
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
%model_1/Flatten/flatten/strided_sliceStridedSlicemodel_1/Flatten/flatten/Shape+model_1/Flatten/flatten/strided_slice/stack-model_1/Flatten/flatten/strided_slice/stack_1-model_1/Flatten/flatten/strided_slice/stack_2*
T0*
Index0*
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
%model_1/Flatten/flatten/Reshape/shapePack%model_1/Flatten/flatten/strided_slice'model_1/Flatten/flatten/Reshape/shape/1*
T0*

axis *
N*
_output_shapes
:
�
model_1/Flatten/flatten/ReshapeReshapemodel_1/conv5/MaxPool2D/MaxPool%model_1/Flatten/flatten/Reshape/shape*
T0*
Tshape0*(
_output_shapes
:����������
l
model_2/conv1/dilation_rateConst*
dtype0*
_output_shapes
:*
valueB"      
�
model_2/conv1/Conv2DConv2Dnegative_inputconv1/weights/read*
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
model_2/conv1/MaxPool2D/MaxPoolMaxPoolmodel_2/conv1/conv1/Relu*/
_output_shapes
:���������K *
T0*
strides
*
data_formatNHWC*
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
model_2/conv2/Conv2DConv2Dmodel_2/conv1/MaxPool2D/MaxPoolconv2/weights/read*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME*/
_output_shapes
:���������K@*
	dilations

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
model_2/conv3/dilation_rateConst*
_output_shapes
:*
valueB"      *
dtype0
�
model_2/conv3/Conv2DConv2Dmodel_2/conv2/MaxPool2D/MaxPoolconv3/weights/read*
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
T0*
strides
*
data_formatNHWC*
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
valueB"      *
dtype0*
_output_shapes
:
�
model_2/conv5/Conv2DConv2Dmodel_2/conv4/MaxPool2D/MaxPoolconv5/weights/read*
paddingSAME*/
_output_shapes
:���������
*
	dilations
*
T0*
strides
*
data_formatNHWC*
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
-model_2/Flatten/flatten/strided_slice/stack_2Const*
_output_shapes
:*
valueB:*
dtype0
�
%model_2/Flatten/flatten/strided_sliceStridedSlicemodel_2/Flatten/flatten/Shape+model_2/Flatten/flatten/strided_slice/stack-model_2/Flatten/flatten/strided_slice/stack_1-model_2/Flatten/flatten/strided_slice/stack_2*
new_axis_mask *
end_mask *
_output_shapes
: *
Index0*
T0*
shrink_axis_mask*

begin_mask *
ellipsis_mask 
r
'model_2/Flatten/flatten/Reshape/shape/1Const*
_output_shapes
: *
valueB :
���������*
dtype0
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
Sum/reduction_indicesConst*
dtype0*
_output_shapes
: *
value	B :
q
SumSummulSum/reduction_indices*#
_output_shapes
:���������*

Tidx0*
	keep_dims( *
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
:���������*

Tidx0*
	keep_dims( *
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
Pow_1Powmodel_2/Flatten/flatten/ReshapePow_1/y*(
_output_shapes
:����������*
T0
Y
Sum_2/reduction_indicesConst*
value	B :*
dtype0*
_output_shapes
: 
w
Sum_2SumPow_1Sum_2/reduction_indices*#
_output_shapes
:���������*

Tidx0*
	keep_dims( *
T0
C
Sqrt_1SqrtSum_2*#
_output_shapes
:���������*
T0
H
mul_1MulSqrtSqrt_1*#
_output_shapes
:���������*
T0
H
divRealDivSummul_1*
T0*#
_output_shapes
:���������

mul_2Mulmodel_1/Flatten/flatten/Reshapemodel/Flatten/flatten/Reshape*(
_output_shapes
:����������*
T0
Y
Sum_3/reduction_indicesConst*
_output_shapes
: *
value	B :*
dtype0
w
Sum_3Summul_2Sum_3/reduction_indices*
T0*#
_output_shapes
:���������*

Tidx0*
	keep_dims( 
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
Sum_4SumPow_2Sum_4/reduction_indices*

Tidx0*
	keep_dims( *
T0*#
_output_shapes
:���������
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
Sum_5/reduction_indicesConst*
dtype0*
_output_shapes
: *
value	B :
w
Sum_5SumPow_3Sum_5/reduction_indices*
T0*#
_output_shapes
:���������*

Tidx0*
	keep_dims( 
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
subSubmodel/Flatten/flatten/Reshapemodel_1/Flatten/flatten/Reshape*(
_output_shapes
:����������*
T0
L
Pow_4/yConst*
valueB
 *   @*
dtype0*
_output_shapes
: 
M
Pow_4PowsubPow_4/y*(
_output_shapes
:����������*
T0
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
:���������*

Tidx0*
	keep_dims(
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
Sum_7/reduction_indicesConst*
value	B :*
dtype0*
_output_shapes
: 
{
Sum_7SumPow_5Sum_7/reduction_indices*

Tidx0*
	keep_dims(*
T0*'
_output_shapes
:���������
G
Sqrt_5SqrtSum_7*'
_output_shapes
:���������*
T0
N
sub_2SubSqrt_4Sqrt_5*
T0*'
_output_shapes
:���������
J
add/yConst*
valueB
 *  HC*
dtype0*
_output_shapes
: 
J
addAddsub_2add/y*
T0*'
_output_shapes
:���������
N
	Maximum/yConst*
_output_shapes
: *
valueB
 *    *
dtype0
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
VariableV2*
shared_name *
dtype0*
	container *
_output_shapes
: *
shape: 
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
gradients/Mean_grad/ReshapeReshapegradients/Fill!gradients/Mean_grad/Reshape/shape*
Tshape0*
_output_shapes

:*
T0
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
gradients/Mean_grad/Shape_2Const*
valueB *
dtype0*
_output_shapes
: 
c
gradients/Mean_grad/ConstConst*
_output_shapes
:*
valueB: *
dtype0
�
gradients/Mean_grad/ProdProdgradients/Mean_grad/Shape_1gradients/Mean_grad/Const*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
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
"gradients/Maximum_grad/zeros/ConstConst*
_output_shapes
: *
valueB
 *    *
dtype0
�
gradients/Maximum_grad/zerosFillgradients/Maximum_grad/Shape_2"gradients/Maximum_grad/zeros/Const*
T0*

index_type0*'
_output_shapes
:���������
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
gradients/add_grad/ShapeShapesub_2*
out_type0*
_output_shapes
:*
T0
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
gradients/add_grad/Sum_1Sum/gradients/Maximum_grad/tuple/control_dependency*gradients/add_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
�
gradients/add_grad/Reshape_1Reshapegradients/add_grad/Sum_1gradients/add_grad/Shape_1*
Tshape0*
_output_shapes
: *
T0
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
gradients/sub_2_grad/Shape_1ShapeSqrt_5*
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
:*

Tidx0*
	keep_dims( *
T0
�
gradients/sub_2_grad/ReshapeReshapegradients/sub_2_grad/Sumgradients/sub_2_grad/Shape*'
_output_shapes
:���������*
T0*
Tshape0
�
gradients/sub_2_grad/Sum_1Sum+gradients/add_grad/tuple/control_dependency,gradients/sub_2_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
^
gradients/sub_2_grad/NegNeggradients/sub_2_grad/Sum_1*
_output_shapes
:*
T0
�
gradients/sub_2_grad/Reshape_1Reshapegradients/sub_2_grad/Neggradients/sub_2_grad/Shape_1*
Tshape0*'
_output_shapes
:���������*
T0
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
gradients/Sqrt_4_grad/SqrtGradSqrtGradSqrt_4-gradients/sub_2_grad/tuple/control_dependency*'
_output_shapes
:���������*
T0
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
gradients/Sum_6_grad/modFloorModgradients/Sum_6_grad/addgradients/Sum_6_grad/Size*
_output_shapes
: *
T0*-
_class#
!loc:@gradients/Sum_6_grad/Shape
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
 gradients/Sum_6_grad/range/deltaConst*-
_class#
!loc:@gradients/Sum_6_grad/Shape*
value	B :*
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
gradients/Sum_6_grad/Fill/valueConst*-
_class#
!loc:@gradients/Sum_6_grad/Shape*
value	B :*
dtype0*
_output_shapes
: 
�
gradients/Sum_6_grad/FillFillgradients/Sum_6_grad/Shape_1gradients/Sum_6_grad/Fill/value*
T0*-
_class#
!loc:@gradients/Sum_6_grad/Shape*

index_type0*
_output_shapes
: 
�
"gradients/Sum_6_grad/DynamicStitchDynamicStitchgradients/Sum_6_grad/rangegradients/Sum_6_grad/modgradients/Sum_6_grad/Shapegradients/Sum_6_grad/Fill*-
_class#
!loc:@gradients/Sum_6_grad/Shape*
N*
_output_shapes
:*
T0
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
gradients/Sum_7_grad/SizeConst*
_output_shapes
: *-
_class#
!loc:@gradients/Sum_7_grad/Shape*
value	B :*
dtype0
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
gradients/Sum_7_grad/rangeRange gradients/Sum_7_grad/range/startgradients/Sum_7_grad/Size gradients/Sum_7_grad/range/delta*
_output_shapes
:*

Tidx0*-
_class#
!loc:@gradients/Sum_7_grad/Shape
�
gradients/Sum_7_grad/Fill/valueConst*-
_class#
!loc:@gradients/Sum_7_grad/Shape*
value	B :*
dtype0*
_output_shapes
: 
�
gradients/Sum_7_grad/FillFillgradients/Sum_7_grad/Shape_1gradients/Sum_7_grad/Fill/value*
_output_shapes
: *
T0*-
_class#
!loc:@gradients/Sum_7_grad/Shape*

index_type0
�
"gradients/Sum_7_grad/DynamicStitchDynamicStitchgradients/Sum_7_grad/rangegradients/Sum_7_grad/modgradients/Sum_7_grad/Shapegradients/Sum_7_grad/Fill*
N*
_output_shapes
:*
T0*-
_class#
!loc:@gradients/Sum_7_grad/Shape
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
gradients/Pow_4_grad/mul_1Mulgradients/Pow_4_grad/mulgradients/Pow_4_grad/Pow*(
_output_shapes
:����������*
T0
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
$gradients/Pow_4_grad/ones_like/ShapeShapesub*
_output_shapes
:*
T0*
out_type0
i
$gradients/Pow_4_grad/ones_like/ConstConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
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
gradients/Pow_4_grad/Sum_1Sumgradients/Pow_4_grad/mul_3,gradients/Pow_4_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
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
gradients/Pow_5_grad/mulMulgradients/Sum_7_grad/TilePow_5/y*(
_output_shapes
:����������*
T0
_
gradients/Pow_5_grad/sub/yConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
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
gradients/Pow_5_grad/SumSumgradients/Pow_5_grad/mul_1*gradients/Pow_5_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
gradients/Pow_5_grad/ReshapeReshapegradients/Pow_5_grad/Sumgradients/Pow_5_grad/Shape*(
_output_shapes
:����������*
T0*
Tshape0
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
gradients/Pow_5_grad/Select_1Selectgradients/Pow_5_grad/Greatergradients/Pow_5_grad/Loggradients/Pow_5_grad/zeros_like*(
_output_shapes
:����������*
T0
v
gradients/Pow_5_grad/mul_2Mulgradients/Sum_7_grad/TilePow_5*
T0*(
_output_shapes
:����������
�
gradients/Pow_5_grad/mul_3Mulgradients/Pow_5_grad/mul_2gradients/Pow_5_grad/Select_1*
T0*(
_output_shapes
:����������
�
gradients/Pow_5_grad/Sum_1Sumgradients/Pow_5_grad/mul_3,gradients/Pow_5_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
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
/gradients/Pow_5_grad/tuple/control_dependency_1Identitygradients/Pow_5_grad/Reshape_1&^gradients/Pow_5_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/Pow_5_grad/Reshape_1*
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
gradients/sub_grad/SumSum-gradients/Pow_4_grad/tuple/control_dependency(gradients/sub_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
gradients/sub_grad/ReshapeReshapegradients/sub_grad/Sumgradients/sub_grad/Shape*
T0*
Tshape0*(
_output_shapes
:����������
�
gradients/sub_grad/Sum_1Sum-gradients/Pow_4_grad/tuple/control_dependency*gradients/sub_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
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
gradients/sub_1_grad/SumSum-gradients/Pow_5_grad/tuple/control_dependency*gradients/sub_1_grad/BroadcastGradientArgs*
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
gradients/sub_1_grad/Sum_1Sum-gradients/Pow_5_grad/tuple/control_dependency,gradients/sub_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
^
gradients/sub_1_grad/NegNeggradients/sub_1_grad/Sum_1*
T0*
_output_shapes
:
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
6gradients/model_2/Flatten/flatten/Reshape_grad/ReshapeReshape/gradients/sub_1_grad/tuple/control_dependency_14gradients/model_2/Flatten/flatten/Reshape_grad/Shape*/
_output_shapes
:���������*
T0*
Tshape0
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
8gradients/model/conv5/MaxPool2D/MaxPool_grad/MaxPoolGradMaxPoolGradmodel/conv5/BiasAddmodel/conv5/MaxPool2D/MaxPool4gradients/model/Flatten/flatten/Reshape_grad/Reshape*/
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
:gradients/model_2/conv5/MaxPool2D/MaxPool_grad/MaxPoolGradMaxPoolGradmodel_2/conv5/BiasAddmodel_2/conv5/MaxPool2D/MaxPool6gradients/model_2/Flatten/flatten/Reshape_grad/Reshape*
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
5gradients/model/conv5/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput(gradients/model/conv5/Conv2D_grad/ShapeNconv5/weights/read;gradients/model/conv5/BiasAdd_grad/tuple/control_dependency*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*0
_output_shapes
:���������
�*
	dilations

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
7gradients/model_2/conv5/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput*gradients/model_2/conv5/Conv2D_grad/ShapeNconv5/weights/read=gradients/model_2/conv5/BiasAdd_grad/tuple/control_dependency*0
_output_shapes
:���������
�*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME
�
8gradients/model_2/conv5/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFiltermodel_2/conv4/MaxPool2D/MaxPool,gradients/model_2/conv5/Conv2D_grad/ShapeN:1=gradients/model_2/conv5/BiasAdd_grad/tuple/control_dependency*
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
8gradients/model/conv4/MaxPool2D/MaxPool_grad/MaxPoolGradMaxPoolGradmodel/conv4/conv4/Relumodel/conv4/MaxPool2D/MaxPool:gradients/model/conv5/Conv2D_grad/tuple/control_dependency*0
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
:gradients/model_2/conv4/MaxPool2D/MaxPool_grad/MaxPoolGradMaxPoolGradmodel_2/conv4/conv4/Relumodel_2/conv4/MaxPool2D/MaxPool<gradients/model_2/conv5/Conv2D_grad/tuple/control_dependency*0
_output_shapes
:����������*
T0*
strides
*
data_formatNHWC*
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
0gradients/model_1/conv4/BiasAdd_grad/BiasAddGradBiasAddGrad0gradients/model_1/conv4/conv4/Relu_grad/ReluGrad*
data_formatNHWC*
_output_shapes	
:�*
T0
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
=gradients/model/conv4/BiasAdd_grad/tuple/control_dependency_1Identity.gradients/model/conv4/BiasAdd_grad/BiasAddGrad4^gradients/model/conv4/BiasAdd_grad/tuple/group_deps*
_output_shapes	
:�*
T0*A
_class7
53loc:@gradients/model/conv4/BiasAdd_grad/BiasAddGrad
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
7gradients/model_1/conv4/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput*gradients/model_1/conv4/Conv2D_grad/ShapeNconv4/weights/read=gradients/model_1/conv4/BiasAdd_grad/tuple/control_dependency*
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
8gradients/model_1/conv4/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFiltermodel_1/conv3/MaxPool2D/MaxPool,gradients/model_1/conv4/Conv2D_grad/ShapeN:1=gradients/model_1/conv4/BiasAdd_grad/tuple/control_dependency*(
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
4gradients/model_1/conv4/Conv2D_grad/tuple/group_depsNoOp9^gradients/model_1/conv4/Conv2D_grad/Conv2DBackpropFilter8^gradients/model_1/conv4/Conv2D_grad/Conv2DBackpropInput
�
<gradients/model_1/conv4/Conv2D_grad/tuple/control_dependencyIdentity7gradients/model_1/conv4/Conv2D_grad/Conv2DBackpropInput5^gradients/model_1/conv4/Conv2D_grad/tuple/group_deps*
T0*J
_class@
><loc:@gradients/model_1/conv4/Conv2D_grad/Conv2DBackpropInput*0
_output_shapes
:����������
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
N* 
_output_shapes
::*
T0*
out_type0
�
7gradients/model_2/conv4/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput*gradients/model_2/conv4/Conv2D_grad/ShapeNconv4/weights/read=gradients/model_2/conv4/BiasAdd_grad/tuple/control_dependency*
data_formatNHWC*
strides
*
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
:gradients/model_1/conv3/MaxPool2D/MaxPool_grad/MaxPoolGradMaxPoolGradmodel_1/conv3/conv3/Relumodel_1/conv3/MaxPool2D/MaxPool<gradients/model_1/conv4/Conv2D_grad/tuple/control_dependency*
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
8gradients/model/conv3/MaxPool2D/MaxPool_grad/MaxPoolGradMaxPoolGradmodel/conv3/conv3/Relumodel/conv3/MaxPool2D/MaxPool:gradients/model/conv4/Conv2D_grad/tuple/control_dependency*
strides
*
data_formatNHWC*
ksize
*
paddingSAME*0
_output_shapes
:���������&�*
T0
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
?gradients/model_1/conv3/BiasAdd_grad/tuple/control_dependency_1Identity0gradients/model_1/conv3/BiasAdd_grad/BiasAddGrad6^gradients/model_1/conv3/BiasAdd_grad/tuple/group_deps*
_output_shapes	
:�*
T0*C
_class9
75loc:@gradients/model_1/conv3/BiasAdd_grad/BiasAddGrad
�
.gradients/model/conv3/BiasAdd_grad/BiasAddGradBiasAddGrad.gradients/model/conv3/conv3/Relu_grad/ReluGrad*
data_formatNHWC*
_output_shapes	
:�*
T0
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
7gradients/model_1/conv3/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput*gradients/model_1/conv3/Conv2D_grad/ShapeNconv3/weights/read=gradients/model_1/conv3/BiasAdd_grad/tuple/control_dependency*
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
5gradients/model/conv3/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput(gradients/model/conv3/Conv2D_grad/ShapeNconv3/weights/read;gradients/model/conv3/BiasAdd_grad/tuple/control_dependency*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME*/
_output_shapes
:���������&@
�
6gradients/model/conv3/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFiltermodel/conv2/MaxPool2D/MaxPool*gradients/model/conv3/Conv2D_grad/ShapeN:1;gradients/model/conv3/BiasAdd_grad/tuple/control_dependency*
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
7gradients/model_2/conv3/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput*gradients/model_2/conv3/Conv2D_grad/ShapeNconv3/weights/read=gradients/model_2/conv3/BiasAdd_grad/tuple/control_dependency*
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
8gradients/model_2/conv3/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFiltermodel_2/conv2/MaxPool2D/MaxPool,gradients/model_2/conv3/Conv2D_grad/ShapeN:1=gradients/model_2/conv3/BiasAdd_grad/tuple/control_dependency*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME*'
_output_shapes
:@�*
	dilations

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
:gradients/model_1/conv2/MaxPool2D/MaxPool_grad/MaxPoolGradMaxPoolGradmodel_1/conv2/conv2/Relumodel_1/conv2/MaxPool2D/MaxPool<gradients/model_1/conv3/Conv2D_grad/tuple/control_dependency*/
_output_shapes
:���������K@*
T0*
strides
*
data_formatNHWC*
ksize
*
paddingSAME
�
8gradients/model/conv2/MaxPool2D/MaxPool_grad/MaxPoolGradMaxPoolGradmodel/conv2/conv2/Relumodel/conv2/MaxPool2D/MaxPool:gradients/model/conv3/Conv2D_grad/tuple/control_dependency*
strides
*
data_formatNHWC*
ksize
*
paddingSAME*/
_output_shapes
:���������K@*
T0
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
=gradients/model_2/conv2/BiasAdd_grad/tuple/control_dependencyIdentity0gradients/model_2/conv2/conv2/Relu_grad/ReluGrad6^gradients/model_2/conv2/BiasAdd_grad/tuple/group_deps*/
_output_shapes
:���������K@*
T0*C
_class9
75loc:@gradients/model_2/conv2/conv2/Relu_grad/ReluGrad
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
7gradients/model_1/conv2/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput*gradients/model_1/conv2/Conv2D_grad/ShapeNconv2/weights/read=gradients/model_1/conv2/BiasAdd_grad/tuple/control_dependency*/
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
<gradients/model_1/conv2/Conv2D_grad/tuple/control_dependencyIdentity7gradients/model_1/conv2/Conv2D_grad/Conv2DBackpropInput5^gradients/model_1/conv2/Conv2D_grad/tuple/group_deps*/
_output_shapes
:���������K *
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
5gradients/model/conv2/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput(gradients/model/conv2/Conv2D_grad/ShapeNconv2/weights/read;gradients/model/conv2/BiasAdd_grad/tuple/control_dependency*
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
6gradients/model/conv2/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFiltermodel/conv1/MaxPool2D/MaxPool*gradients/model/conv2/Conv2D_grad/ShapeN:1;gradients/model/conv2/BiasAdd_grad/tuple/control_dependency*
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
*gradients/model_2/conv2/Conv2D_grad/ShapeNShapeNmodel_2/conv1/MaxPool2D/MaxPoolconv2/weights/read*
T0*
out_type0*
N* 
_output_shapes
::
�
7gradients/model_2/conv2/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput*gradients/model_2/conv2/Conv2D_grad/ShapeNconv2/weights/read=gradients/model_2/conv2/BiasAdd_grad/tuple/control_dependency*/
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
8gradients/model_2/conv2/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFiltermodel_2/conv1/MaxPool2D/MaxPool,gradients/model_2/conv2/Conv2D_grad/ShapeN:1=gradients/model_2/conv2/BiasAdd_grad/tuple/control_dependency*
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
4gradients/model_2/conv2/Conv2D_grad/tuple/group_depsNoOp9^gradients/model_2/conv2/Conv2D_grad/Conv2DBackpropFilter8^gradients/model_2/conv2/Conv2D_grad/Conv2DBackpropInput
�
<gradients/model_2/conv2/Conv2D_grad/tuple/control_dependencyIdentity7gradients/model_2/conv2/Conv2D_grad/Conv2DBackpropInput5^gradients/model_2/conv2/Conv2D_grad/tuple/group_deps*
T0*J
_class@
><loc:@gradients/model_2/conv2/Conv2D_grad/Conv2DBackpropInput*/
_output_shapes
:���������K 
�
>gradients/model_2/conv2/Conv2D_grad/tuple/control_dependency_1Identity8gradients/model_2/conv2/Conv2D_grad/Conv2DBackpropFilter5^gradients/model_2/conv2/Conv2D_grad/tuple/group_deps*
T0*K
_classA
?=loc:@gradients/model_2/conv2/Conv2D_grad/Conv2DBackpropFilter*&
_output_shapes
: @
�
gradients/AddN_7AddN?gradients/model_1/conv2/BiasAdd_grad/tuple/control_dependency_1=gradients/model/conv2/BiasAdd_grad/tuple/control_dependency_1?gradients/model_2/conv2/BiasAdd_grad/tuple/control_dependency_1*
_output_shapes
:@*
T0*C
_class9
75loc:@gradients/model_1/conv2/BiasAdd_grad/BiasAddGrad*
N
�
:gradients/model_1/conv1/MaxPool2D/MaxPool_grad/MaxPoolGradMaxPoolGradmodel_1/conv1/conv1/Relumodel_1/conv1/MaxPool2D/MaxPool<gradients/model_1/conv2/Conv2D_grad/tuple/control_dependency*
T0*
strides
*
data_formatNHWC*
ksize
*
paddingSAME*0
_output_shapes
:���������2� 
�
8gradients/model/conv1/MaxPool2D/MaxPool_grad/MaxPoolGradMaxPoolGradmodel/conv1/conv1/Relumodel/conv1/MaxPool2D/MaxPool:gradients/model/conv2/Conv2D_grad/tuple/control_dependency*0
_output_shapes
:���������2� *
T0*
strides
*
data_formatNHWC*
ksize
*
paddingSAME
�
:gradients/model_2/conv1/MaxPool2D/MaxPool_grad/MaxPoolGradMaxPoolGradmodel_2/conv1/conv1/Relumodel_2/conv1/MaxPool2D/MaxPool<gradients/model_2/conv2/Conv2D_grad/tuple/control_dependency*0
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
;gradients/model/conv1/BiasAdd_grad/tuple/control_dependencyIdentity.gradients/model/conv1/conv1/Relu_grad/ReluGrad4^gradients/model/conv1/BiasAdd_grad/tuple/group_deps*0
_output_shapes
:���������2� *
T0*A
_class7
53loc:@gradients/model/conv1/conv1/Relu_grad/ReluGrad
�
=gradients/model/conv1/BiasAdd_grad/tuple/control_dependency_1Identity.gradients/model/conv1/BiasAdd_grad/BiasAddGrad4^gradients/model/conv1/BiasAdd_grad/tuple/group_deps*
_output_shapes
: *
T0*A
_class7
53loc:@gradients/model/conv1/BiasAdd_grad/BiasAddGrad
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
8gradients/model_1/conv1/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterpositive_input,gradients/model_1/conv1/Conv2D_grad/ShapeN:1=gradients/model_1/conv1/BiasAdd_grad/tuple/control_dependency*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME*&
_output_shapes
: *
	dilations

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
>gradients/model_1/conv1/Conv2D_grad/tuple/control_dependency_1Identity8gradients/model_1/conv1/Conv2D_grad/Conv2DBackpropFilter5^gradients/model_1/conv1/Conv2D_grad/tuple/group_deps*
T0*K
_classA
?=loc:@gradients/model_1/conv1/Conv2D_grad/Conv2DBackpropFilter*&
_output_shapes
: 
�
(gradients/model/conv1/Conv2D_grad/ShapeNShapeNanchor_inputconv1/weights/read* 
_output_shapes
::*
T0*
out_type0*
N
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
6gradients/model/conv1/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilteranchor_input*gradients/model/conv1/Conv2D_grad/ShapeN:1;gradients/model/conv1/BiasAdd_grad/tuple/control_dependency*
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
gradients/AddN_9AddN?gradients/model_1/conv1/BiasAdd_grad/tuple/control_dependency_1=gradients/model/conv1/BiasAdd_grad/tuple/control_dependency_1?gradients/model_2/conv1/BiasAdd_grad/tuple/control_dependency_1*C
_class9
75loc:@gradients/model_1/conv1/BiasAdd_grad/BiasAddGrad*
N*
_output_shapes
: *
T0
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
(conv1/weights/Momentum/Initializer/zerosFill8conv1/weights/Momentum/Initializer/zeros/shape_as_tensor.conv1/weights/Momentum/Initializer/zeros/Const*
T0*

index_type0* 
_class
loc:@conv1/weights*&
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
conv1/weights/Momentum/AssignAssignconv1/weights/Momentum(conv1/weights/Momentum/Initializer/zeros* 
_class
loc:@conv1/weights*
validate_shape(*&
_output_shapes
: *
use_locking(*
T0
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
(conv2/weights/Momentum/Initializer/zerosFill8conv2/weights/Momentum/Initializer/zeros/shape_as_tensor.conv2/weights/Momentum/Initializer/zeros/Const*

index_type0* 
_class
loc:@conv2/weights*&
_output_shapes
: @*
T0
�
conv2/weights/Momentum
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
'conv2/biases/Momentum/Initializer/zerosConst*
valueB@*    *
_class
loc:@conv2/biases*
dtype0*
_output_shapes
:@
�
conv2/biases/Momentum
VariableV2*
shape:@*
dtype0*
_output_shapes
:@*
shared_name *
_class
loc:@conv2/biases*
	container 
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
8conv3/weights/Momentum/Initializer/zeros/shape_as_tensorConst*
_output_shapes
:*%
valueB"      @   �   * 
_class
loc:@conv3/weights*
dtype0
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
(conv3/weights/Momentum/Initializer/zerosFill8conv3/weights/Momentum/Initializer/zeros/shape_as_tensor.conv3/weights/Momentum/Initializer/zeros/Const*'
_output_shapes
:@�*
T0*

index_type0* 
_class
loc:@conv3/weights
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
conv3/biases/Momentum/AssignAssignconv3/biases/Momentum'conv3/biases/Momentum/Initializer/zeros*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0*
_class
loc:@conv3/biases
�
conv3/biases/Momentum/readIdentityconv3/biases/Momentum*
T0*
_class
loc:@conv3/biases*
_output_shapes	
:�
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
conv4/weights/Momentum/AssignAssignconv4/weights/Momentum(conv4/weights/Momentum/Initializer/zeros* 
_class
loc:@conv4/weights*
validate_shape(*(
_output_shapes
:��*
use_locking(*
T0
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
VariableV2*
_output_shapes	
:�*
shared_name *
_class
loc:@conv4/biases*
	container *
shape:�*
dtype0
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
8conv5/weights/Momentum/Initializer/zeros/shape_as_tensorConst*
_output_shapes
:*%
valueB"            * 
_class
loc:@conv5/weights*
dtype0
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
VariableV2*
_class
loc:@conv5/biases*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name 
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
+Momentum/update_conv1/weights/ApplyMomentumApplyMomentumconv1/weightsconv1/weights/MomentumMomentum/learning_rategradients/AddN_10Momentum/momentum*
T0* 
_class
loc:@conv1/weights*
use_nesterov(*&
_output_shapes
: *
use_locking( 
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
+Momentum/update_conv2/weights/ApplyMomentumApplyMomentumconv2/weightsconv2/weights/MomentumMomentum/learning_rategradients/AddN_8Momentum/momentum* 
_class
loc:@conv2/weights*
use_nesterov(*&
_output_shapes
: @*
use_locking( *
T0
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
+Momentum/update_conv3/weights/ApplyMomentumApplyMomentumconv3/weightsconv3/weights/MomentumMomentum/learning_rategradients/AddN_6Momentum/momentum*
use_nesterov(*'
_output_shapes
:@�*
use_locking( *
T0* 
_class
loc:@conv3/weights
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
+Momentum/update_conv4/weights/ApplyMomentumApplyMomentumconv4/weightsconv4/weights/MomentumMomentum/learning_rategradients/AddN_4Momentum/momentum*
use_locking( *
T0* 
_class
loc:@conv4/weights*
use_nesterov(*(
_output_shapes
:��
�
*Momentum/update_conv4/biases/ApplyMomentumApplyMomentumconv4/biasesconv4/biases/MomentumMomentum/learning_rategradients/AddN_3Momentum/momentum*
T0*
_class
loc:@conv4/biases*
use_nesterov(*
_output_shapes	
:�*
use_locking( 
�
+Momentum/update_conv5/weights/ApplyMomentumApplyMomentumconv5/weightsconv5/weights/MomentumMomentum/learning_rategradients/AddN_2Momentum/momentum*
T0* 
_class
loc:@conv5/weights*
use_nesterov(*'
_output_shapes
:�*
use_locking( 
�
*Momentum/update_conv5/biases/ApplyMomentumApplyMomentumconv5/biasesconv5/biases/MomentumMomentum/learning_rategradients/AddN_1Momentum/momentum*
_output_shapes
:*
use_locking( *
T0*
_class
loc:@conv5/biases*
use_nesterov(
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
save/filename/inputConst*
dtype0*
_output_shapes
: *
valueB Bmodel
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
save/SaveV2/tensor_namesConst*
_output_shapes
:*�
value�B�BVariableBconv1/biasesBconv1/biases/MomentumBconv1/weightsBconv1/weights/MomentumBconv2/biasesBconv2/biases/MomentumBconv2/weightsBconv2/weights/MomentumBconv3/biasesBconv3/biases/MomentumBconv3/weightsBconv3/weights/MomentumBconv4/biasesBconv4/biases/MomentumBconv4/weightsBconv4/weights/MomentumBconv5/biasesBconv5/biases/MomentumBconv5/weightsBconv5/weights/Momentum*
dtype0
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
save/Assign_1Assignconv1/biasessave/RestoreV2:1*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*
_class
loc:@conv1/biases
�
save/Assign_2Assignconv1/biases/Momentumsave/RestoreV2:2*
T0*
_class
loc:@conv1/biases*
validate_shape(*
_output_shapes
: *
use_locking(
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
save/Assign_6Assignconv2/biases/Momentumsave/RestoreV2:6*
_output_shapes
:@*
use_locking(*
T0*
_class
loc:@conv2/biases*
validate_shape(
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
save/Assign_8Assignconv2/weights/Momentumsave/RestoreV2:8*
T0* 
_class
loc:@conv2/weights*
validate_shape(*&
_output_shapes
: @*
use_locking(
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
save/Assign_13Assignconv4/biasessave/RestoreV2:13*
use_locking(*
T0*
_class
loc:@conv4/biases*
validate_shape(*
_output_shapes	
:�
�
save/Assign_14Assignconv4/biases/Momentumsave/RestoreV2:14*
_output_shapes	
:�*
use_locking(*
T0*
_class
loc:@conv4/biases*
validate_shape(
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
conv1/biases_1/tagConst*
_output_shapes
: *
valueB Bconv1/biases_1*
dtype0
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
: ""
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
conv5/biases/Momentum:0conv5/biases/Momentum/Assignconv5/biases/Momentum/read:02)conv5/biases/Momentum/Initializer/zeros:0"�
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
conv5/biases:0conv5/biases/Assignconv5/biases/read:02 conv5/biases/Initializer/zeros:08���]:      ��
O	~�/*��A*�t

step    

loss��GC
�
conv1/weights_1*�	   �H��   ��H�?     `�@!   :���)Gs]�&�@2�����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��vV�R9��T7����5�i}1�>h�'��f�ʜ�7
��������Zr[v��I��P=���T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?�������:�              K@     @h@     �i@     �e@     �c@     �e@     �_@     �Y@     �^@     @Y@      X@     �R@     �U@     �N@     �P@      G@     �H@     �F@      F@     �D@      >@      <@      >@      0@      :@      7@      3@      4@      *@      0@      .@      (@      (@      $@      "@      @      @      @       @      @       @      @      @      @       @      @      @       @       @      @      @      @       @       @               @      �?      �?              �?              �?              �?       @              �?      �?              �?              �?      �?              �?       @              �?               @       @       @      �?       @      @       @      �?       @       @      @       @      @      @       @      @      @      �?      @       @      @      @       @       @       @      $@      "@      "@      @      *@      .@      .@      ,@      :@      ,@      3@      :@      6@      @@     �@@      G@      9@      C@     �F@     �N@      L@      K@     �L@     �Q@     @V@     �X@     �S@     �Z@      `@     �^@     �_@     �b@     �d@      h@     �h@      G@        
O
conv1/biases_1*=      @@2        �-���q=�������:              @@        
�
conv2/weights_1*�	    ���   �q��?      �@!  �� � @)��iW�aE@2����g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(龢f�����uE���⾮��%ᾙѩ�-߾jqs&\�ѾK+�E��Ͼ�XQ�þ��~��¾�XQ��>�����>
�/eq
�>;�"�q�>K+�E���>jqs&\��>��>M|K�>�_�T�l�>�iD*L��>�uE����>�f����>a�Ϭ(�>8K�ߝ�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?�������:�             $�@     n�@     ��@     t�@     �@     l�@     $�@     �@     \�@     ��@     ��@     H�@     �@      �@     H�@     @�@     ؁@     p}@     �z@      y@      x@     u@     q@     Pr@     `p@     @l@     �g@     `f@     �f@     @e@     @`@     @]@     �[@     �U@     @U@     �T@     �Q@     �T@     @T@      J@      J@     �F@      G@      @@     �C@     �D@     �A@      7@      ?@      9@      8@      3@      3@      0@      0@      "@      (@      *@      "@       @      (@      @      @       @      @      �?      @      @      @      @      @      �?       @       @       @      �?      �?      @       @      @       @      �?              �?       @      �?              �?              �?      �?      �?              @              �?              �?              �?              �?              �?      �?              �?               @              �?              �?       @      �?      @       @      �?               @      @       @      @      @              @      @      @       @      @      @      @      @      @      @      @      &@      $@       @      *@      &@      1@      1@      6@      :@      3@      7@      8@      6@     �C@      A@      D@      C@      N@     �J@      Q@     �R@      Q@     @P@     �U@     @V@     �\@      [@     �`@     @`@     `d@     `f@     `i@      j@      k@     @m@     �p@     �s@     �t@     �w@     Py@     �|@     }@     @@     ��@      �@     ؆@     h�@     �@     ��@     ��@     ܑ@     ��@     x�@     ��@     ��@     X�@     ��@     �@     В@        
O
conv2/biases_1*=      P@2        �-���q=�������:              P@        
�
conv3/weights_1*�	    t+��   �r+�?      �@!  �}5@)�/)|�[U@2�����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7���ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f����E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;;�"�qʾ�MZ��K���u��gr��豪}0ڰ>��n����>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>���%�>�uE����>�f����>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?�������:�             ��@     T�@     H�@     ��@     ^�@     Π@     0�@     ؜@     ��@     $�@     4�@     ��@     ��@     ��@     ؎@     ��@     X�@     h�@     x�@     Ё@     �@      ~@     �{@     �z@     �v@     pt@     �r@      p@     �l@     �k@      h@      g@     �e@     �a@     �_@      `@     @_@     �X@     �V@     �U@      P@     �Q@     �Q@      L@      K@      M@      E@      E@      B@      C@      ?@      5@      <@      ;@      ;@      6@      4@      2@      $@      (@      @      @      .@       @      @      @      @      &@      �?      @      @      "@       @      @       @       @       @      @              �?       @       @       @      @      �?              �?              �?      �?      �?              �?      �?              �?              �?              �?              �?              �?              �?              �?      �?              �?      �?      �?              �?              �?      �?               @      �?      @      �?       @              �?       @      �?      �?      �?       @       @      @      �?      @      @      @       @      @      @      &@      @      @       @       @      @      ,@      &@      1@      4@      "@      $@      3@      9@      ?@      A@      =@     �@@     �@@      :@     �J@      E@     �E@     �O@     �M@     �I@     @Q@      T@      V@     �V@     �W@     �\@     @_@     �c@      d@     `g@     �h@     `g@     p@     �o@     Pp@     `q@     �u@     �x@     @{@     �z@     0@     Ѐ@     ��@     ؄@     ��@     ȋ@     ��@     ��@     ��@     h�@     ��@     ��@     l�@     T�@     L�@     П@     V�@     ��@     �@     l�@     ��@     0�@        
O
conv3/biases_1*=      `@2        �-���q=�������:              `@        
�
conv4/weights_1*�	   �m���    ]��?      �@!   ؞�@) �tn\e@2�Ӗ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7�����d�r�x?�x��6�]���1��a˲�>�?�s���O�ʗ���I��P=��pz�w�7���uE���⾮��%������>
�/eq
�>jqs&\��>��~]�[�>��(���>a�Ϭ(�>�h���`�>�ߊ4F��>I��P=�>��Zr[v�>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�������:�              a@     <�@      �@     ܒ@     ��@     ��@     �@     (�@     �@     ��@     p�@     P�@     ��@     |@      |@     �w@     `v@     pr@     �q@     �o@     �l@     �k@      j@     �f@     �a@      f@     �b@     �_@     �[@      Y@     �X@     �R@      Y@     �Q@     �O@      O@      D@      E@     �H@      D@      B@      8@      B@      9@      8@      2@      $@      3@      2@      4@      (@      &@      0@       @      $@       @      @      "@      @      @      @       @      �?      @      @      @      @      @      �?      @               @              �?       @      �?              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?      �?      �?      �?       @              �?              �?      �?      �?       @       @      �?      @      �?      @       @       @               @      @      @      @      @      @      @      @      @       @      "@      @      &@      ,@      (@      0@      ,@      .@      $@      ,@      2@      9@      6@      7@     �D@      :@      =@      J@     �J@      F@     �N@     �J@      N@     �Q@     �W@      T@      U@      `@      ^@      `@     �_@     �d@     �c@     �j@     �j@     �h@     @n@     �q@     �s@     pt@      x@     �v@     �w@     �@     �@     x�@     h�@     ��@     ��@     h�@     ��@     �@     ��@     �@     0�@     ��@     �a@        
O
conv4/biases_1*=      p@2        �-���q=�������:              p@        
�
conv5/weights_1*�	    �¿   ����?      �@!   ���+�)z�ϸI@2�	�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.�U�4@@�$��[^:��"��S�F !���Zr[v��I��P=��6�]��?����?ji6�9�?�S�F !?�[^:��"?U�4@@�$?��82?�u�w74?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?�������:�	             �i@     `t@     �p@     �q@     �n@      k@     `f@     �g@     �d@      b@      ]@     �\@     �Z@     �X@      V@     @W@      O@      O@      L@      K@     �L@      L@      @@      >@      F@      D@      :@      7@      7@      ?@      ,@      6@      2@      (@      4@      *@      &@      *@       @      "@      (@      @      @       @      @      @      @      @      @              @      �?      @      @      �?      �?      �?      @      @      @       @      �?               @      �?       @      �?      �?              �?      �?              �?              �?              �?              �?              @               @      �?       @              �?      �?      @      �?      @      �?       @      @      @       @      @      @      $@      "@       @      @      (@      ,@      $@      &@      "@      "@      *@      1@      8@      *@      ,@      :@      3@      "@      :@      A@      <@      @@      ;@     �E@      M@      I@     �P@      E@     �N@     @T@     @R@     �S@     �X@     @]@     �]@      a@     @`@      b@      e@      e@     @h@     �k@      n@     �o@      t@     �j@        
O
conv5/biases_1*=      <@2        �-���q=�������:              <@        c��3(W      [��	���5*��A*��

step  �?

loss^�GC
�
conv1/weights_1*�	    `e��    h^�?     `�@! �7�8�)K<{N��@2�����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8��u�w74���82���bȬ�0���VlQ.�ji6�9���.����ڋ��vV�R9��5�i}1���d�r�x?�x��>h�'��O�ʗ�����Zr[v��
�/eq
Ⱦ����žf�ʜ�7
?>h�'�?�5�i}1?�T7��?�vV�R9?��ڋ?ji6�9�?�S�F !?I�I�)�(?�7Kaa+?��VlQ.?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?�������:�              K@      h@      j@     @e@     �c@      e@      `@      Z@      _@     �X@     �X@     @R@     @W@      K@     �Q@      H@      G@      F@     �H@     �B@      ?@      <@      9@      5@      8@      7@      4@      3@      0@      ,@      (@      1@      (@      @       @      $@      @      @      @      "@      @      @      @      @       @      @       @      @      @       @       @      @      @       @              @       @      �?              �?      �?      �?              �?      �?      �?              �?              �?              �?               @              �?               @               @      �?              �?      �?      �?      @      @       @      @      @      @      @      @      @      @      @      @       @      $@      @      @      (@      @      $@       @      &@      @      ,@      (@      0@      1@      8@      *@      0@      9@      <@      ;@     �C@     �C@      =@      A@     �H@      L@      O@     �J@     �K@      R@     �V@     �X@      T@     @Z@      `@     @^@     �_@     `b@     @e@     @g@      i@     �G@        
�
conv1/biases_1*�	     �3�   @?�<?      @@!  v�]?)�|��9�>2��u�w74���82���bȬ�0��7Kaa+�I�I�)�(�+A�F�&��[^:��"��S�F !�f�ʜ�7
������>�?�s���O�ʗ���;�"�qʾ
�/eq
Ⱦ��~��¾�[�=�k���
L�v�Q�28���FP�������M>28���FP>�*��ڽ>�[�=�k�>jqs&\��>��~]�[�>��>M|K�>I��P=�>��Zr[v�>1��a˲?6�]��?�T7��?�vV�R9?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��bȬ�0?��82?��%�V6?uܬ�@8?��%>��:?d�\D�X=?�������:�              �?       @              �?       @              �?              �?              �?              �?              �?              �?              �?              �?              �?      �?              �?              �?              �?              �?              �?      �?      �?      �?      @              �?              �?              @        
�
conv2/weights_1*�	   `����   @���?      �@! �w�}@)ё���aE@2����g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پjqs&\�ѾK+�E��Ͼ['�?�;;�"�qʾ
�/eq
Ⱦ�u`P+d�>0�6�/n�>;�"�q�>['�?��>K+�E���>jqs&\��>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>a�Ϭ(�>8K�ߝ�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?�������:�             �@     t�@     ��@     ��@     �@     T�@     D�@     �@     X�@     ��@     ��@     p�@      �@     �@     �@     8�@     ؁@     �}@      {@      y@     �w@     �u@     pp@     Pr@     �p@     �l@      g@     `g@     �e@     @e@     �`@     �^@     �Z@      U@     @V@     �S@      S@      U@     �R@      K@     �K@     �C@      I@      C@      ?@      E@     �A@      <@      6@      =@      9@      *@      2@      3@      3@       @      0@      &@      $@      $@       @      @      @      "@      @      �?      @      @      @      @      @       @       @      �?       @      @      �?      @      @       @       @      �?              �?      �?      �?       @              �?              �?              �?              �?              �?              �?      �?              �?              �?              �?              �?              �?       @              �?              �?              �?      �?      �?       @      �?      �?      @      �?      �?      @      �?      �?      @      $@      @      @       @      @      @      @      "@      @      @      $@      @      "@      0@      ,@      2@      ;@      1@      ;@      ,@      3@      9@      7@      D@      ?@      B@      C@     �P@     �H@     @R@      R@      Q@     @P@     @V@     �X@     �Z@     �Y@     @a@     @`@     @d@     �f@     �h@     �j@     �j@      m@      q@      s@     �t@     x@      y@     }@     `}@     �~@     p�@      �@     �@     X�@     ؊@     ȍ@     ��@     �@     p�@     ��@     ��@     ��@     @�@     ��@     �@     �@        
�
conv2/biases_1*�
	   �`kA�   �DiD?      P@!  >2Noe?)d�F�g(�>2��!�A����#@�d�\D�X=��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9���d�r�x?�x��>h�'��f�ʜ�7
������6�]����FF�G �>�?�s���pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(龙ѩ�-߾E��a�Wܾ['�?�;;�"�qʾ����ž�XQ�þ        �-���q=f^��`{>�����~>['�?��>K+�E���>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>a�Ϭ(�>8K�ߝ�>�h���`�>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�T���C?a�$��{E?�������:�              �?      �?               @              �?              �?              �?      �?              �?              �?       @              �?      �?              �?      �?              �?              �?      �?              �?      �?              �?              �?              �?              @              �?              �?              �?              �?      �?      �?              �?      �?              �?              �?       @      �?      �?               @              �?      �?              �?      �?               @               @               @      �?      @      �?      �?      �?      �?               @              �?        
�
conv3/weights_1*�	   �0��    �4�?      �@! g+���@)E?
A\U@2�����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f���侮��%ᾙѩ�-߾�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�Ѿ
�/eq
Ⱦ����ž�MZ��K���u��gr����|�~�>���]���>�*��ڽ>�[�=�k�>�����>
�/eq
�>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?�������:�             ��@     T�@     V�@     �@     Z�@     ʠ@     :�@     Ĝ@     ��@     4�@     0�@     ��@     ܑ@     ��@     ��@     ��@     @�@     p�@     H�@      �@      �@      ~@     �{@     �z@     v@      u@     �r@     �o@     �l@     �k@     @h@     �f@     �e@     �a@     �`@     @^@     �_@     @Y@     �U@     @U@      S@      Q@     �P@      M@      K@      J@      G@      C@      D@     �@@      D@      9@      2@      @@      4@      8@      5@      3@      $@      $@      "@      "@      ,@      @      @      @      @      "@      @      @      @      @      �?      �?      @      @      �?      @      �?      �?       @       @              @       @      �?               @              �?      �?      �?              �?              �?               @      �?              �?              �?              �?              �?               @               @              �?              @              �?              �?              �?               @       @               @      �?      �?      �?      @              �?      @       @      @      @      @      @      @      "@      @      @       @      .@      "@      0@       @      1@      2@      &@      *@      0@      8@      8@     �C@      <@      >@      A@      ;@      H@     �G@     �B@      P@     @P@     �J@      P@      U@     @V@     �T@      Z@     @]@     �]@     @c@     �d@     `g@     �h@      g@     �o@     pp@     `p@     �p@     v@     �x@     �z@     0{@      @     ��@     ��@     ��@     ��@     ��@     ��@     ��@     ��@     l�@     ��@     ��@     ��@     H�@     D�@     �@     8�@     ƣ@     �@     r�@     ��@     8�@        
�
conv3/biases_1*�	    �>�    �MK?      `@! �^"jMi?)үI:k��>2����#@�d�\D�X=���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�>h�'��f�ʜ�7
������6�]�����[���FF�G �>�?�s���O�ʗ���I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%���~]�[Ӿjqs&\�Ѿ['�?�;;�"�qʾ����ž�XQ�þ���m!#���
�%W��        �-���q=�[�=�k�>��~���>�XQ��>�����>
�/eq
�>E��a�W�>�ѩ�-�>�uE����>�f����>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?�qU���I?IcD���L?�������:�              �?               @      @      �?       @      @       @               @      @              �?       @      �?       @      @      �?      @              �?       @      �?              �?       @      �?               @              �?       @       @      �?               @              �?              �?              �?              �?              �?              @              �?      �?              �?              �?              �?              �?              �?              �?      �?               @      �?      @              �?               @              @      �?      @       @      �?      �?      @      @      �?       @      @      �?       @      @       @      �?              �?      @      �?               @              �?        
�
conv4/weights_1*�	    & ��    ]��?      �@! ܽ��j@):'}�\e@2�Ӗ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ���d�r�x?�x��6�]���1��a˲�I��P=��pz�w�7���uE���⾮��%ᾙѩ�-߾E��a�Wܾ�����>
�/eq
�>�iD*L��>E��a�W�>a�Ϭ(�>8K�ߝ�>�ߊ4F��>})�l a�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�������:�              a@     @�@      �@     ؒ@     ��@     ��@      �@     8�@     �@     ��@     p�@     P�@     ��@      |@     �{@     �w@     �v@     �r@     �q@     �o@     `l@     `k@     @j@     �f@      b@      f@     �b@     @_@     @\@      Y@     �X@     �R@      X@      S@     �N@      O@     �D@     �D@      J@      C@     �A@      9@     �@@      <@      7@      3@      $@      3@      3@      2@      (@      (@      *@      (@      $@      @      @      @      @      @      @      �?      �?      @      @      @      @      @      �?      @               @      �?      �?       @       @              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?               @              �?               @              �?              �?      �?      �?              �?      @      @       @      �?      @      @      �?      �?       @      @      @      @       @      @      @      @      @       @      "@       @      (@      ,@      (@      .@      *@      .@      &@      *@      2@      :@      6@      8@      D@      :@      <@     �K@      I@      G@     �M@     �J@      N@     @R@     �V@     �T@     �T@     @`@     �]@     ``@     �_@      d@     @d@      k@     �j@     �h@     `n@     �q@     �s@     `t@      x@     �v@     pw@     �@     �@     ��@     p�@     ��@     p�@     p�@     x�@     $�@     Đ@     �@     (�@     ��@     �a@        
�
conv4/biases_1*�	   ���G�   ��OS?      p@!���~b?)��WkP?�>2��qU���I�
����G��!�A����#@�uܬ�@8���%�V6���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`f�����uE���⾮��%ᾙѩ�-߾�iD*L�پ�_�T�l׾��>M|Kվ��~]�[ӾK+�E��Ͼ['�?�;;�"�qʾ
�/eq
Ⱦ�XQ�þ��~��¾�[�=�k���*��ڽ�0�6�/n���u`P+d����������?�ګ����]������|�~���u��gr��R%������.��fc���X$�z���
�%W����ӤP��������~�f^��`{�ہkVl�p�w`f���n��#���j�Z�TA[�����"�RT��+���`���nx6�X� ��f׽r����tO����f;H�\Q������%���9�e����K��󽉊-��J�'j��p���1���=��]����/�4��ݟ��uy�z�����i@4[���Qu�R"�H����ڽ���X>ؽ��
"
ֽ(�+y�6ҽ;3���н�b1�ĽK?�\��½y�訥�V���Ұ��        �-���q=�Į#��=���6�=��؜��=�d7����=�!p/�^�=��.4N�=;3����=�|86	�=��
"
�=���X>�=H�����=PæҭU�=�Qu�R"�=i@4[��=z�����=ݟ��uy�=�/�4��='j��p�=��-��J�=�9�e��=����%�=nx6�X� >�`��>�mm7&c>y�+pm>�
�%W�>���m!#�>�MZ��K�>��|�~�>���]���>�5�L�>;9��R�>豪}0ڰ>��n����>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>;�"�q�>['�?��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?�!�A?�T���C?a�$��{E?
����G?�qU���I?�lDZrS?<DKc��T?�������:�              �?              �?              �?              �?       @       @               @      �?      �?      �?              @      @       @       @      @      @      @       @      �?       @      @      @              @      �?       @      @              �?      �?      @               @              �?               @               @              �?      �?      �?              �?              �?              �?               @              �?              �?              �?              �?               @              �?              �?              �?               @      �?       @      @       @       @      �?      �?       @       @      �?               @              @       @              �?      �?              @              �?              �?             �E@              �?              �?              �?       @              �?               @       @              �?      �?              �?              �?              @              �?       @      �?              �?              �?              �?      �?              �?               @              �?      @      �?      �?              �?               @              �?      �?       @              �?      �?      �?      �?       @              �?              �?              �?               @      �?      @      �?       @      �?       @       @      �?      �?      @              @      �?      �?      �?              @              �?      �?       @      �?      �?       @              �?       @              �?              �?        
�
conv5/weights_1*�	    j�¿   ����?      �@! ,#�+�)��*I@2�	�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.�U�4@@�$��[^:��"��T7����5�i}1���Zr[v��I��P=��6�]��?����?ji6�9�?�S�F !?�[^:��"?U�4@@�$?��82?�u�w74?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?�������:�	             �i@     `t@     �p@     �q@      o@     �j@     �f@     �g@     �d@      b@      ]@     �\@     �Z@     �X@      V@      W@      O@     �O@      L@      K@     �L@      L@     �@@      =@     �E@      E@      9@      8@      6@      ?@      ,@      6@      2@      (@      4@      *@      &@      *@      "@       @      &@       @      @      @      @      @      @       @      @              @      �?       @      @      �?      �?      �?      @       @      @      �?      �?      �?               @      �?      �?      �?               @              �?              �?              �?              �?              �?               @               @      �?       @      �?      �?      �?      @      �?      @      �?       @      @      @      @       @      @      &@       @      "@      @      &@      ,@      $@      &@      "@      "@      ,@      0@      9@      &@      .@      :@      3@      "@      :@     �@@      <@     �@@      :@      F@      M@      I@     �P@      E@     �N@     �T@     �Q@     �S@     �X@     @]@     �]@     �`@     �`@     �a@      e@      e@      h@     �k@      n@     �o@      t@     �j@        
�
conv5/biases_1*�	   @�X
�   �4>      <@!   oe�>)�Ú��S<2����"�RT��+���mm7&c��`���nx6�X� ��f׽r����9�e����K��󽉊-��J�'j��p���1���=��]����/�4��ݟ��uy�PæҭUݽH����ڽK?�\��½�
6������>�i�E��_�H�}������X>�=H�����=i@4[��=z�����=ݟ��uy�=��-��J�=�K���=f;H�\Q�=�tO���=�f׽r��=nx6�X� >�`��>�mm7&c>y�+pm>Z�TA[�>�#���j>�������:�              �?               @              �?              �?      �?      �?               @      �?       @              �?              �?              �?              �?               @      �?              �?              @       @              �?              �?              �?        ����V      	燀	�b�<*��A*٭

step   @

loss��GC
�
conv1/weights_1*�	   �΄��   `d��?     `�@!  ,`�)��w���@2�	����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !���ڋ��vV�R9��T7����5�i}1���d�r�8K�ߝ�a�Ϭ(��_�T�l�>�iD*L��>>�?�s��>�FF�G ?�5�i}1?�T7��?�vV�R9?�.�?ji6�9�?�S�F !?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?�������:�	              M@     �g@     @j@     �e@     �c@     @e@     �_@     @Y@     �_@     @X@     �X@     @R@     @W@      M@      Q@      I@      G@     �D@     �K@     �@@      <@      ?@      =@      3@      5@      3@      9@      3@      2@      0@      &@      3@      @      @      *@      @      @      @       @      @      @      @      @      �?       @      @      @      @      @       @      @      �?      �?      @       @               @              �?      �?              �?              �?      �?              �?      �?              �?              �?              �?              �?              �?      �?              �?      @               @       @      �?      �?               @       @      @      �?       @      @      @      @      @       @       @      @      @      @      @      @      @      @      *@      (@      @       @      *@      "@      ,@      (@      .@      3@      6@      $@      1@      2@      @@      ;@     �C@     �D@      ;@     �D@     �F@     �J@     �N@      M@     �M@      Q@     �V@     �W@     @T@      [@     �_@     �]@     �_@     �b@     �d@     �g@     �h@     �H@        
�
conv1/biases_1*�	   �}�J�   @��U?      @@! `L�.;p?)pHӧ�1�>2�IcD���L��qU���I��!�A����#@�d�\D�X=���82���bȬ�0��7Kaa+�I�I�)�(�U�4@@�$��[^:��"��S�F !�ji6�9����ڋ��vV�R9�jqs&\�ѾK+�E��Ͼ���m!#���
�%W����ӤP�����z!�?�������0c�w&���qa��iD*L��>E��a�W�>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>>h�'�?x?�x�?�vV�R9?��ڋ?+A�F�&?I�I�)�(?��%�V6?uܬ�@8?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?�qU���I?IcD���L?<DKc��T?ܗ�SsW?�������:�              �?              �?       @              �?               @              �?               @              �?              �?              �?              �?              �?              �?              �?              �?      �?              �?              �?              �?              �?              @               @      �?              �?               @        
�
conv2/weights_1*�	   �櫩�   ��ީ?      �@! ����@)CF/}{bE@2����g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾K+�E��Ͼ['�?�;�u`P+d����n�����.��fc��>39W$:��>�[�=�k�>��~���>�XQ��>�����>jqs&\��>��~]�[�>�_�T�l�>�iD*L��>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?�������:�             ��@     ��@     ��@     ��@     ��@     D�@     p�@     Ԓ@     p�@     x�@     ؍@     ��@     �@     P�@     Ђ@     h�@     ��@     �}@     P{@      y@     �w@      u@     q@     pr@      o@     �m@     �g@     �e@     �f@     �e@      a@     �_@     �W@     �W@      V@     @V@     �Q@     �Q@     @S@      K@     �J@     �G@     �K@     �B@      <@     �D@     �B@      5@     �@@      <@      7@      .@      3@      6@      ,@      (@       @      *@      @      @       @       @      &@      @      @      @      @      @       @      @      @       @       @              @      @      �?      �?      @      �?       @      �?      �?       @              �?      �?              �?               @              �?              �?              �?              �?              �?              �?              �?              �?              �?      �?      �?              �?      �?              �?      �?               @              �?              �?      @              �?      �?      �?      @      @       @      @      @       @      @      @      @      @      @      @       @      (@      &@      @      $@      ,@      6@      6@      4@      5@      2@      7@      6@      =@      @@      <@     �E@     �C@      N@     �K@     �R@     �P@      S@     �P@     �U@     �W@      \@     @Y@     @`@      a@      e@     �e@     �i@     �i@      k@     @m@     0q@     0r@     �u@     �w@      y@     P}@     `}@     �~@     8�@     ��@     h�@     `�@     ��@     ȍ@     ��@     �@     |�@     |�@     ��@     ؙ@     D�@     ��@     �@     �@        
�	
conv2/biases_1*�		   �O4O�   �W�K?      P@!  8�Eu?)��L%'�>2�k�1^�sO�IcD���L��qU���I�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.�I�I�)�(�+A�F�&��[^:��"��S�F !��5�i}1���d�r�f�ʜ�7
�������FF�G �>�?�s���O�ʗ�����Zr[v���iD*L�پ�_�T�l׾R%������39W$:���        �-���q=�[�=�k�>��~���>
�/eq
�>;�"�q�>['�?��>K+�E���>�ѩ�-�>���%�>a�Ϭ(�>8K�ߝ�>��Zr[v�>O�ʗ��>��[�?1��a˲?6�]��?f�ʜ�7
?>h�'�?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?I�I�)�(?�7Kaa+?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?�������:�              �?      �?              �?      �?      �?              �?       @      �?      �?       @       @      �?              �?              �?              �?              �?              �?              �?              �?              �?              @              �?              �?              �?              �?              �?              �?              �?      �?               @              �?               @      �?      �?              @               @      �?      �?      �?              �?       @       @              @       @       @       @        
�
conv3/weights_1*�	   �6��   �kV�?      �@! ]�Ϫ@)vi�b\U@2�����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f����E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿ
�/eq
Ⱦ����ž�*��ڽ�G&�$���MZ��K���u��gr��jqs&\��>��~]�[�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?�������:�             ��@     B�@     d�@     �@     b�@      @     *�@     �@     ��@     �@     X�@     ��@     ̑@     ��@     ��@     �@     h�@      �@     ��@     ��@      �@     0~@     �{@     {@     v@     �t@     �r@     0p@     @l@     @l@      g@      h@     �e@     `b@     ``@     �\@      _@     �Z@      W@     �R@     �R@     @S@      N@      O@      H@     �K@     �G@     �E@      @@      C@     �C@      ;@      2@      <@      ;@      2@      2@      1@      ,@      (@       @      @      (@      @      @      @      @      "@      @      @      @      @      @      @       @       @      @      @      �?      �?      �?       @              @      @       @      �?      @       @       @       @      �?               @      �?      �?      �?              �?              �?              �?              �?              �?       @      �?              �?               @      �?      �?      �?      �?      �?      �?      @              �?       @      @      @      �?      �?      @      @      @       @      @      @      "@      @      &@       @       @      *@      (@      .@      2@      (@      1@      .@      7@      9@      ?@     �A@      A@      ;@      ;@      I@      F@      B@     @P@      P@     �L@     @P@      V@     �T@      W@     @W@      ]@     �^@     �a@     �e@      g@     @i@     �g@     �o@      p@     �p@     �p@     �v@     0x@     �z@     p{@     `@     ��@     �@     ��@     �@     ��@     ؉@     ��@     �@     h�@     l�@     ĕ@     ��@     T�@      �@     �@     P�@     ƣ@     �@     ��@     ��@     ��@        
�
conv3/biases_1*�	    �|R�   �DzX?      `@!  vH��y?)2h����>2��lDZrS�nK���LQ�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��T7����5�i}1���d�r�x?�x��>h�'��6�]���1��a˲���[��>�?�s���O�ʗ�����Zr[v��I��P=��})�l a��ߊ4F��h���`��uE���⾮��%ᾙѩ�-߾jqs&\�ѾK+�E��Ͼ;�"�qʾ
�/eq
Ⱦ;9��R���5�L�����m!#���
�%W��        �-���q=�u��gr�>�MZ��K�>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>�f����>��(���>})�l a�>pz�w�7�>I��P=�>��Zr[v�>��[�?1��a˲?����?f�ʜ�7
?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�������:�              �?              �?      �?      �?       @       @      @              @      �?       @      �?      �?       @              �?      @      @       @               @               @      �?              �?      �?              �?               @      �?              �?      �?      �?              �?      �?              �?      �?              �?              �?              �?              �?              @              �?              �?      �?      �?              �?              �?      �?       @              �?              �?              @               @       @      �?              @      �?      �?       @      �?       @      @      @               @      @      @      @      @       @      �?      �?      �?              �?              �?       @       @              �?              �?        
�
conv4/weights_1*�	   ����    ��?      �@! ,N��p@)�=.�\e@2�Ӗ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����d�r�x?�x��6�]���1��a˲�I��P=��pz�w�7���uE���⾮��%��u`P+d����n����������>
�/eq
�>E��a�W�>�ѩ�-�>a�Ϭ(�>8K�ߝ�>pz�w�7�>I��P=�>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�������:�              a@     <�@     �@     Ԓ@     ��@     ��@     ��@     @�@     �@     ��@     x�@     P�@     ��@     P|@     �{@     �w@     �v@     �r@     �q@     �o@     `l@     `k@     `j@     �f@     �a@      f@     �b@      _@      \@     �X@      Y@     @R@     �W@     �S@     �P@      L@      D@     �E@      J@     �B@     �A@      =@      A@      5@      8@      4@      (@      2@      0@      4@      ,@      $@      *@      $@      *@      @      @      "@      @      @       @       @      �?      @      @      @      �?      @      @      @      �?      @      �?      @      @              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?      �?              �?       @      �?      �?      �?      @      @      �?              @       @      �?              @      @      @      @      @      @       @      @      @      $@       @      @      *@      0@      &@      .@      (@      *@      *@      0@      .@      :@      6@      8@     �D@      <@      8@     �M@     �G@      F@     @P@      I@      O@     �Q@     �V@     @U@     �S@     ``@     �]@      `@      `@     �c@     �d@     �j@     `k@     `h@     `n@     �q@      t@     �t@     �w@     �v@     Pw@     �@     �@     ��@     x�@     ��@     ��@     h�@     h�@     $�@     Đ@     $�@     $�@     ��@     �a@        
�
conv4/biases_1*�	   ���S�   �f�\?      p@!B_'�t?)q'1�jM�>2�<DKc��T��lDZrS�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a�h���`�8K�ߝ���(��澢f���侙ѩ�-߾E��a�Wܾ��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;;�"�qʾ
�/eq
Ⱦ����ž5�"�g���0�6�/n���u`P+d����|�~���MZ��K��39W$:���.��fc�����z!�?��T�L<��u��6
��K���7��[#=�؏����-�z�!�%����Łt�=	���R�����J��#���j����"�RT��+��y�+pm��mm7&c��`���nx6�X� ��f׽r����tO����f;H�\Q������%���9�e����K��󽉊-��J�'j��p���1���=��]����/�4��PæҭUݽH����ڽ���X>ؽG�L������6���Į#�������/���-���q�        �-���q=;3����=(�+y�6�=H�����=PæҭU�=i@4[��=z�����==��]���=��1���='j��p�=��-��J�=�K���=�9�e��=����%�=�tO���=�f׽r��=nx6�X� >�mm7&c>y�+pm>���">Z�TA[�>2!K�R�>��R���>���?�ګ>����>豪}0ڰ>��n����>�u`P+d�>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�������:�              �?               @      �?      �?      �?      �?      �?              �?              @       @      @       @      �?      �?      �?      @      @       @      @       @       @               @       @      @      �?      �?      @              �?              @               @              �?               @              �?              �?              @              �?      �?       @              �?      �?       @              �?      �?              �?              �?              �?      �?              �?              �?              �?              �?               @               @      �?      �?       @      @      @      �?              �?      �?      @       @      �?      �?               @       @              �?              �?              �?      C@               @              �?               @               @      �?               @      �?      �?              �?      �?               @               @               @              �?      �?               @              �?      �?      �?      �?              �?              �?      �?              �?      �?      @              �?              �?              �?              �?      �?              �?      �?              @      �?      �?      �?       @      �?      �?      @      @       @      @               @       @      �?       @       @      �?      �?      �?      �?      @              �?               @      �?      �?       @              �?              �?               @      �?        
�
conv5/weights_1*�	   ���¿   ����?      �@! �8��+�)\Q�I@2�	�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.�U�4@@�$��[^:��"���Zr[v��I��P=��jqs&\�ѾK+�E��Ͼf�ʜ�7
?>h�'�?x?�x�?�S�F !?�[^:��"?U�4@@�$?��82?�u�w74?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?�������:�	             �i@     Pt@     �p@     �q@      o@      k@     �f@     �g@     �d@     �a@     �]@     @\@     �Z@      Y@     �U@      W@     �O@     �N@     �L@      L@     �K@      K@      A@      >@     �E@     �E@      8@      9@      6@      ?@      ,@      5@      2@      *@      4@      *@      &@      (@      $@       @      $@       @      @      @      @      @      @       @      @       @      �?      @      @       @      �?      �?      �?      @      @      @      �?      �?      �?              �?       @               @              �?              �?              �?              �?      �?              �?      �?               @              �?      �?      �?      �?      @      @      �?      �?      @      �?       @      @      @      @       @      @      (@       @      "@      @      "@      .@      $@      &@       @      $@      ,@      .@      :@      $@      0@      :@      3@      "@      :@      A@      ;@      @@      :@      G@      L@      J@     @P@     �D@      O@      U@     �Q@      S@      Y@      ]@     �]@     �`@     �`@     �a@      e@     @e@     �g@      l@      n@     �o@      t@     �j@        
�
conv5/biases_1*�	   ���    W�>      <@!  �EE$+>)8<�n'n<2���R����2!K�R���J��#���j�Z�TA[��nx6�X� ��f׽r����tO����f;H�\Q������%���9�e���'j��p���1����/�4��ݟ��uy�z�����i@4[��PæҭUݽH����ڽ(�+y�6ҽ;3���н�EDPq���8�4L����>�i�E��_�H�}���(�+y�6�=�|86	�=�K���=�9�e��=����%�=�tO���=�f׽r��=nx6�X� >RT��+�>���">Z�TA[�>�#���j>�J>2!K�R�>��R���>Łt�=	>��f��p>�������:�              �?              �?       @              �?              �?              �?              �?              �?      �?      �?              �?              �?              �?              �?              �?              �?      �?               @      �?               @      �?      �?              �?              �?      �?        �cI5hV      
O�	g��C*��A*٬

step  @@

loss��GC
�
conv1/weights_1*�	    �ή�   ��Ʈ?     `�@! @��b�)`2��m�@2�����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���VlQ.��7Kaa+�U�4@@�$��[^:��"�ji6�9���.��>h�'��f�ʜ�7
���[���FF�G ��h���`�8K�ߝ���Zr[v�>O�ʗ��>6�]��?����?�T7��?�vV�R9?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?�7Kaa+?��VlQ.?��bȬ�0?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?�������:�              N@     �g@      j@     @e@      d@      e@     �`@     �Y@     �]@     �X@     �X@     �R@     �U@     �L@     �Q@      J@      H@     �C@      K@      A@      =@      @@      <@      2@      2@      >@      0@      3@      3@      1@      .@      &@      (@      @      &@      @      "@      @      @      @      @      @       @       @      @       @      @       @      �?      �?       @       @      �?      @       @      @      @              @               @              �?              �?              �?              �?              �?               @              �?              �?       @              �?              �?      @              @              @      @      @      @      @      �?      �?       @      �?      @      �?      @      @      "@      @       @      "@      @      ,@      &@      &@       @      *@      *@      1@      *@      2@      ,@      ,@      :@      <@      A@      <@      E@      ?@     �E@     �D@      I@     �P@     �L@     �L@     @R@      V@     @W@     @V@     �Y@      _@     �\@     @`@      c@     �d@     �f@     �i@      I@        
�
conv1/biases_1*�	   ���[�   �1�_?      @@! hI��?)��g�>2��m9�H�[���bB�SY�ܗ�SsW�<DKc��T�IcD���L��qU���I�a�$��{E��T���C��!�A����#@�d�\D�X=���bȬ�0���VlQ.��7Kaa+�I�I�)�(���d�r�x?�x��8K�ߝ�a�Ϭ(��_�T�l׾��>M|Kվ��������?�ګ���Ő�;F��`�}6D���|�~�>���]���>��~���>�XQ��>�f����>��(���>pz�w�7�>I��P=�>1��a˲?6�]��?�T7��?�vV�R9?uܬ�@8?��%>��:?d�\D�X=?�!�A?�T���C?a�$��{E?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?E��{��^?�l�P�`?�������:�              �?              �?              �?              �?              �?      �?              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?      �?              �?      �?              �?       @               @       @               @        
�
conv2/weights_1*�	    �˩�   `? �?      �@! �ƈ�@)�EscE@2����g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ뾢f�����uE���⾮��%ᾙѩ�-߾K+�E��Ͼ['�?�;����ž�XQ�þ豪}0ڰ�������MZ��K�>��|�~�>jqs&\��>��~]�[�>��(���>a�Ϭ(�>8K�ߝ�>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?�������:�             ��@     ��@     ��@     ��@     ��@     ,�@     d�@     Ȓ@     x�@     p�@     �@     ��@     ��@     ��@     ��@     ��@     x�@     �}@      {@      y@     �w@     �t@     �q@     �q@     @o@      m@     @i@      f@     �e@      f@     �`@     �_@      Z@     @S@     �Y@     �W@      R@     �P@     �N@     �P@     �I@     �D@      K@      G@     �A@      D@     �B@      :@      ;@      9@      6@      2@      0@      7@      0@      0@      $@      @      $@      @      @      @      $@      @       @      @       @      @      "@      @       @              @      �?               @              �?       @      @       @      �?      �?      �?       @      �?      �?              �?              �?               @              �?              �?              �?              �?              �?      @              �?              �?              @              @       @      �?      @      �?       @      �?      @       @      @      @       @      @      @       @      @      @      @      "@      (@      @      "@      0@       @      3@      3@      8@      <@      0@      3@      6@      8@     �@@     �@@     �D@      G@     �I@     �L@     �R@     �R@     @T@      Q@     �T@     @V@     �Z@      [@     �_@     ``@     �e@     �e@      i@      j@     �j@     �m@     �p@     s@      u@     �w@     z@     P|@     �}@      @     H�@     ��@     @�@     x�@     ��@     ��@     T�@     �@     l�@     ��@     ��@     ��@     `�@     ğ@     �@     �@        
�	
conv2/biases_1*�	   @Z�    ��V?      P@!  �Շp�?)i~�K4�>2��m9�H�[���bB�SY�ܗ�SsW�nK���LQ�k�1^�sO�
����G�a�$��{E��T���C����#@�d�\D�X=���%>��:�uܬ�@8��u�w74���82���bȬ�0��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��vV�R9��T7���O�ʗ�����Zr[v���uE���⾮��%�5�"�g���0�6�/n��        �-���q=�uE����>�f����>��(���>8K�ߝ�>�h���`�>pz�w�7�>I��P=�>>�?�s��>�FF�G ?��[�?6�]��?����?�.�?ji6�9�?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?�������:�              �?      �?               @              �?       @              @      �?      �?              �?       @              �?              �?      �?              �?               @               @              �?               @              �?      �?              �?              �?              �?      �?              �?              �?              @      �?              �?      �?               @      �?      �?              �?               @       @      @      �?       @       @       @      �?      �?       @        
�
conv3/weights_1*�	    �Q��   �
��?      �@! "�8��@)RU��\U@2�����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾�iD*L�پ�_�T�l׾
�/eq
Ⱦ����ž�MZ��K���u��gr��jqs&\��>��~]�[�>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?�������:�             h�@     D�@     l�@     �@     h�@     Ơ@     �@     ��@     ��@     ,�@     4�@     ��@     ��@     ̐@     ��@     ��@     x�@      �@     ��@     �@     @@     0~@     �{@     �z@     �v@      t@     �r@     �p@     �k@     �k@     @f@     �h@     �f@     �a@      `@      _@     �\@     �[@     �W@     �S@     @Q@      R@     �P@     �M@     �L@      F@     �G@      E@     �D@      A@      :@     �C@      >@      8@      7@      .@      2@      .@      *@      &@      "@      (@      .@      @      @      @      @      $@      @      @      @      @      @      @              @      @      @              �?      @       @      @      @       @      �?              �?       @       @               @              �?      �?               @              �?              �?              �?              �?              �?       @              �?               @               @      �?              @      �?              �?      �?      �?      @      �?      @      @      @      @      @      @       @       @      @      @      "@      $@      .@      &@      (@      1@      .@      2@      3@      7@      9@     �@@     �A@      ;@      ;@      ?@     �I@      B@      E@      N@     �P@     @P@      L@     @X@     �R@      X@     �W@     �]@     �_@      a@      e@     @h@      h@     �i@     `n@     @p@     �p@     �p@     �v@     Px@     pz@     �{@     �@     `�@     ��@     h�@     �@     `�@     �@     `�@     �@     l�@     l�@     ��@     ��@     <�@     �@     ԟ@     V�@     ��@      �@     x�@     ��@     ��@        
�
conv3/biases_1*�	    ��Y�    ��]?      `@!  �\4?)�|1�e,?2��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�U�4@@�$��[^:��"��S�F !�ji6�9����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��f�ʜ�7
������6�]���1��a˲���[��})�l a��ߊ4F���uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ豪}0ڰ������        �-���q=�����>
�/eq
�>���%�>�uE����>�h���`�>�ߊ4F��>O�ʗ��>>�?�s��>6�]��?����?f�ʜ�7
?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�������:�              �?              �?       @      �?      �?               @       @               @      @      @       @      @      @      �?       @      @      �?              �?       @              �?      @       @              �?      �?       @              �?               @      �?      �?       @               @              �?      �?               @              �?              @              �?              �?              �?              �?               @      �?              �?              �?       @      �?               @      �?      �?      �?               @       @      �?      @       @      @      @      �?      �?       @      @      @      @      �?      �?      �?               @      �?      �?       @      �?        
�
conv4/weights_1*�	   @���   @M�?      �@! 0�)�	@)ՈTG�\e@2�Ӗ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ���d�r�x?�x��>h�'��6�]���1��a˲���[��I��P=��pz�w�7���f�����uE���⾮��%������>
�/eq
�>�h���`�>�ߊ4F��>�FF�G ?��[�?1��a˲?6�]��?f�ʜ�7
?>h�'�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�������:�             �`@     H�@     ��@     ؒ@     ��@     ��@     ��@     P�@     Ї@     ��@     ��@     h�@     ��@     @|@     �{@     �w@     pv@     �r@     �q@      p@     @l@     �k@     �i@     �f@      b@     �e@     �b@      _@     �\@      X@     �X@     @S@     @W@      T@      P@     �I@      H@     �E@      G@     �D@     �@@      >@      @@      6@      8@      3@      1@      .@      0@      3@      .@      $@      "@      .@      "@       @       @      $@      @      @       @       @       @      @      @      @      @      @       @      @      �?      @      �?      �?      �?       @              �?      �?              �?      �?              �?               @      �?              �?              �?              �?              �?              �?              �?              @              �?       @      �?       @       @      @      @       @       @       @      @       @      @      @      @      @      @      @       @      "@      $@      &@      ,@      (@      ,@      ,@      $@      1@      ,@      1@      9@      8@      6@      D@      <@      9@     �J@     �J@      E@     @P@      J@      O@      R@     �V@     @U@     @T@      `@     �]@      `@      `@     �c@     �d@     @j@     �k@     �h@     @n@     �q@      t@     Pt@     �w@     �v@     �w@     �@     �@     x�@     ��@     p�@     ��@     p�@     p�@     �@     А@     ,�@     (�@     ��@     �a@        
�
conv4/biases_1*�	   �ǎ\�   �Ld?      p@!� XϊV�?)��\�p�?2�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
�6�]���1��a˲���[���FF�G �>�?�s���O�ʗ���pz�w�7��})�l a��ߊ4F��8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%���>M|Kվ��~]�[ӾK+�E��Ͼ['�?�;��~��¾�[�=�k���*��ڽ�G&�$���u`P+d����n�����豪}0ڰ������;9��R���5�L����|�~���MZ��K�����m!#���
�%W����ӤP����z��6��so쩾4���o�kJ%�4�e|�Z#�%�����i
�k���R����2!K�R���J��#���j�Z�TA[�����"�RT��+��y�+pm��mm7&c�nx6�X� ��f׽r����tO����f;H�\Q������%���9�e����K��󽉊-��J�ݟ��uy�z�����i@4[���Qu�R"�H����ڽ���X>ؽ        �-���q=�
6����=K?�\���=�b1��=i@4[��=z�����=ݟ��uy�=�/�4��==��]���=��1���='j��p�=��-��J�=�K���=�9�e��=����%�=�tO���=�f׽r��=nx6�X� >�`��>�mm7&c>y�+pm>RT��+�>���">Z�TA[�>��f��p>�i
�k>%���>��-�z�!>4�e|�Z#>��o�kJ%>��8"uH>6��>?�J>��x��U>Fixі�W>�H5�8�t>�i����v>�5�L�>;9��R�>�u`P+d�>0�6�/n�>5�"�g��>�[�=�k�>��~���>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��>M|K�>�_�T�l�>�iD*L��>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�ߊ4F��>})�l a�>pz�w�7�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?�������:�              �?      �?       @               @              �?               @      �?      @      @               @      �?      @              �?       @       @       @       @      @      @      @      �?      �?      �?      �?              �?      �?      �?              �?      �?              �?              �?      �?      �?      �?      �?              �?      �?              �?      �?      @              �?              �?              �?               @              �?              �?              �?              �?              �?              �?      �?              �?              �?              �?               @      �?       @       @      �?      @      @      @               @               @      �?      �?              �?              �?              �?              �?              B@              �?      �?              �?       @              �?      �?      �?              �?      �?      �?              �?               @      �?      �?              �?      �?              �?              �?              �?              �?              �?              �?              �?              �?       @              �?              �?               @              �?              �?      �?              �?              @       @      �?               @      @              �?      �?      �?              �?      �?               @      �?              �?       @              �?      �?              �?       @               @               @      @       @      @      �?       @       @       @      @               @              �?      �?      �?      �?      �?       @      @       @       @      �?      �?      �?      �?              �?      @        
�
conv5/weights_1*�	    ��¿   ����?      �@! D�6$�+�)�7��[I@2�	�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�U�4@@�$��[^:��"���Zr[v��I��P=���vV�R9?��ڋ?ji6�9�?�S�F !?�[^:��"?U�4@@�$?�7Kaa+?��VlQ.?��82?�u�w74?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?�������:�	             �i@     Pt@     �p@     �q@      o@      k@     �f@     �g@     �d@     �a@     �]@     @\@     �Z@      Y@     �U@     @W@      P@      N@     �L@      L@      K@     �K@     �@@      @@     �C@     �F@      8@      ;@      6@      =@      .@      4@      2@      ,@      4@      ,@      &@      &@      "@      @      (@      @      @      @      @      @      @       @      @      �?       @       @      @      @              �?       @      �?      @      @      �?       @              �?      �?               @      �?              �?              �?              �?              �?              �?              �?               @              �?      �?      �?      �?      �?       @       @      �?      @      @              @       @      @      @      @      @      &@      "@       @       @      @      0@       @      *@      @      &@      .@      ,@      9@      &@      0@      :@      2@      $@      :@      A@      9@      A@      9@     �G@      L@     �I@     @P@      D@      P@      U@     �Q@     �R@     @Y@      ]@     �]@     �`@     �`@     �a@      e@     `e@     �g@     �k@     `n@     �o@     t@     �j@        
�
conv5/biases_1*�	   `�9�   �_�!>      <@!   �=�4>)�Q_��<2��i
�k���f��p�Łt�=	���R����2!K�R��RT��+��y�+pm��mm7&c��f׽r����tO�����K��󽉊-��J�'j��p���1���;3���н��.4Nν(�+y�6�=�|86	�=H�����=PæҭU�=�Qu�R"�=i@4[��==��]���=��1���=����%�=f;H�\Q�=�tO���=�`��>�mm7&c>y�+pm>���">Z�TA[�>�#���j>�J>2!K�R�>��R���>��f��p>�i
�k>%���>��-�z�!>4�e|�Z#>�������:�              �?      �?      �?      �?              �?      �?              �?              �?       @      �?              �?              �?              �?      �?      �?              �?              �?      �?              �?      �?              �?              �?              �?              �?               @      �?        ���I�U      �
�	J@�K*��A*��

step  �@

lossB�GC
�
conv1/weights_1*�	   `ᮿ   `��?     `�@! �1�o��)b_p5�@2�����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8��u�w74���82���bȬ�0���VlQ.��7Kaa+�+A�F�&�U�4@@�$��[^:��"��S�F !���ڋ��vV�R9�f�ʜ�7
���������]������|�~��E��a�W�>�ѩ�-�>��[�?1��a˲?�.�?ji6�9�?�S�F !?�[^:��"?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?�������:�             �P@     �f@     �j@     `d@      d@      e@      `@     �Z@     �\@      Y@     @Y@     �R@     �U@      M@     @P@     �L@     �G@      D@      K@     �A@      >@      :@      >@      6@      3@      5@      5@      4@      5@      0@      *@      "@       @      "@      @      "@      $@      @       @      "@      @      @      @      @      �?      �?      @      @      �?      �?      @      @      �?      @              @      �?              �?              @               @              �?              �?              �?              �?              �?              �?       @      �?              �?              �?       @      �?              @              @       @      @              @      @      @      @              @      @      @      @      "@      @       @      @      $@       @      &@      &@      &@      (@      ,@      0@      ,@      1@      (@      ,@      =@      <@      =@     �A@      B@     �A@     �C@      G@     �G@     @Q@      M@      L@     @Q@     �U@     �X@      V@      [@      ^@     �\@     @`@      c@      d@     `g@      i@     �K@        
�
conv1/biases_1*�	    �e�   @U�d?      @@! �/c0�?)����&�?2�Tw��Nof�5Ucv0ed�<DKc��T��lDZrS�nK���LQ��qU���I�
����G�uܬ�@8���%�V6��u�w74���82���bȬ�0��S�F !�ji6�9���ѩ�-߾E��a�Wܾf^��`{�E'�/��x���n����>�u`P+d�>��>M|K�>�_�T�l�>pz�w�7�>I��P=�>>�?�s��>�FF�G ?x?�x�?��d�r?��ڋ?�.�?�S�F !?�[^:��"?�u�w74?��%�V6?a�$��{E?
����G?nK���LQ?�lDZrS?<DKc��T?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?�������:�              �?              �?       @              �?               @              �?      �?              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?      �?              �?      �?              @      @      �?        
�
conv2/weights_1*�	   ����    ɑ�?      �@! 	#��@)��m��dE@2����g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��})�l a��ߊ4F��h���`�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾K+�E��Ͼ['�?�;��Ő�;F��`�}6D���n����>�u`P+d�>jqs&\��>��~]�[�>�iD*L��>E��a�W�>a�Ϭ(�>8K�ߝ�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?�������:�             ��@     ��@     Ȟ@     ��@     �@     8�@     @�@     �@     \�@     ��@     �@     ��@     h�@     ��@     ��@     ��@     8�@     0~@     �z@      y@     Px@     t@     �q@     pr@     �m@      m@     �i@      f@     �d@     `f@     �`@     �]@     @\@      W@      W@     �U@      R@     �M@     @P@      N@     �J@      J@     �F@     �I@      C@      E@     �D@      6@      7@      7@      8@      4@      6@      8@      ,@      "@      *@      &@       @      "@      @      @      "@      @      @      @       @      @      &@      @      @      @       @      �?              �?      @              �?      @      �?       @              �?      �?              �?              �?              �?              �?              �?              �?               @              �?              �?              �?              �?      �?               @               @       @       @      �?      �?      �?              @       @       @      @      @      @      @       @      �?      @      $@      @      "@      $@      @      3@      ,@      1@      3@      8@      9@      3@      6@      4@      8@      A@      @@     �A@      G@     �L@      H@      T@     @U@     �Q@     @R@     �U@     �T@     @Y@     @^@     @_@      _@     @e@     �g@      h@      k@     `j@     �m@     �p@     �r@     �t@     0w@     `z@     p|@     �}@     `@     @�@     `�@     8�@     ��@     ��@     ��@     ��@     �@     \�@     ��@     t�@     �@     P�@     ��@     Ԡ@     P�@        
�	
conv2/biases_1*�		   �'>c�   @V7`?      P@!  l:4�?)�h��
?2�5Ucv0ed����%��b�E��{��^��m9�H�[���bB�SY�ܗ�SsW�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���bȬ�0���VlQ.�I�I�)�(�+A�F�&��.����ڋ��vV�R9��T7����5�i}1���[���FF�G �8K�ߝ�a�Ϭ(龢f�����uE����        �-���q=�ѩ�-�>���%�>I��P=�>��Zr[v�>O�ʗ��>>h�'�?x?�x�?��d�r?�5�i}1?+A�F�&?I�I�)�(?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?�������:�              �?              �?      �?      �?              �?              �?      �?      �?       @              �?               @      �?      �?               @              �?              �?              �?       @              �?              �?              �?               @              �?               @      �?              �?               @              �?              �?              �?       @              �?      �?      �?              @      �?              @       @      @      @      �?              �?      �?       @       @        
�
conv3/weights_1*�	   �"w��   ����?      �@! ���Zu@)0��9�]U@2�����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f���侮��%ᾙѩ�-߾�iD*L�پ�_�T�l׾K+�E��Ͼ['�?�;;�"�qʾ
�/eq
ȾG&�$��5�"�g����MZ��K���u��gr�����]���>�5�L�>K+�E���>jqs&\��>��~]�[�>�_�T�l�>�iD*L��>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?�������:�             h�@     D�@     h�@     Ҥ@     ��@     ��@     (�@     ܜ@     ̙@     �@     8�@     ��@     ȑ@     Ȑ@     ��@     �@     P�@     (�@     h�@     ��@     �@      }@     �{@     �z@     pv@     0t@     @s@     �p@     `l@     �j@     �f@     `i@     �d@      b@     @a@     @]@     �]@     �[@      V@      U@     �R@     �R@     �M@     �L@      M@      E@     �G@     �G@     �A@      B@      ;@      C@     �@@      <@      1@      0@      *@      6@      "@      &@      ,@      &@      &@      @      "@      @      @      @      @       @       @      @      @       @      @       @       @      @              @      �?      @       @      @      @      �?               @              �?       @      �?              �?               @              �?              �?              �?              �?              �?              �?      �?              �?               @      @       @      �?              �?              �?               @      �?              �?      @       @       @       @               @       @      @      @       @      @      @      @      @      @      @      "@       @      @      (@      $@      *@      0@      2@      2@      3@      ;@      7@      ;@     �D@      7@      B@      :@     �E@     �H@     �F@      O@     �O@     @P@     �L@     @V@      U@     �V@      U@     �^@      a@      b@      c@     �g@     `h@     �i@     �n@     �p@     Pp@     �p@     Pv@     �w@     �z@     p{@     @@     ��@     ��@     ��@     ��@     p�@     ؉@     ��@     $�@     D�@     h�@     ��@     �@     �@     �@     ܟ@     F�@     ڣ@     ��@     ��@     ��@     �@        
�
conv3/biases_1*�	   ��p`�   @2�b?      `@!  P(*C�?)�I��*�?2��l�P�`�E��{��^��m9�H�[���bB�SY��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9�>h�'��f�ʜ�7
������6�]���1��a˲���[��O�ʗ�����Zr[v���ߊ4F��h���`iD*L�پ�_�T�l׾        �-���q=G&�$�>�*��ڽ>;�"�q�>['�?��>pz�w�7�>I��P=�>��[�?1��a˲?�5�i}1?�T7��?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?�������:�              �?      @      �?               @       @      @      @      @      �?      @      @      �?      �?      @       @      �?      �?      @      �?              �?               @      �?      �?              @      �?               @      @      �?              �?              �?              �?              �?              @              �?              �?              �?              �?              �?              �?      �?       @              �?               @               @               @       @      @      �?      �?      �?      @      �?      @      @      �?      @       @      �?      @      @      �?       @      �?      �?      �?      �?        
�
conv4/weights_1*�	   @���   �#�?      �@! �ؕ^T@)�,��%]e@2�Ӗ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.���vV�R9��T7�����d�r�x?�x��1��a˲���[��I��P=��pz�w�7���uE���⾮��%������>
�/eq
�>��Zr[v�>O�ʗ��>�FF�G ?��[�?1��a˲?6�]��?f�ʜ�7
?>h�'�?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�������:�             �`@     L�@     �@     �@     ��@     ��@     ��@     H�@     Ї@     ��@     ��@     `�@     ��@     |@     �{@     �w@     �v@     �r@     �q@     0p@     `l@     �k@     �i@     @g@     �a@     `e@      c@     �_@     �\@     �W@      Y@     �S@      W@     �R@     @Q@     �I@      G@     �F@      G@     �D@     �A@      ;@      ?@      5@      9@      2@      5@      ,@      .@      2@      *@      *@      &@      &@      $@       @      @      $@      @      @      @      �?      @      @      @      @      �?      @      �?      @      @       @      �?      �?       @               @               @              �?               @              �?              �?              �?              �?      �?      �?              �?              @              @      �?               @       @      @      @      @               @      @      @      @      @       @      @      @      @       @      &@      @      0@       @      0@      0@      (@      *@      .@      *@      3@      8@      8@      9@      B@      <@     �@@     �F@     �J@     �E@     �O@      L@     �M@     �R@      V@     �U@     @T@     �_@      ^@     @_@      `@     �c@     �d@      j@     @l@      i@      n@     Pq@      t@     `t@     �w@     �v@     Pw@     �@     �@     P�@     ��@     p�@     ��@     `�@     x�@     �@     ̐@      �@     0�@     ��@      b@        
�
conv4/biases_1*�	   ��e�   `3An?      p@!��\>��?)�CG���#?2�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.���vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���>�?�s���O�ʗ���I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ���(��澢f�����uE���⾮��%���~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;;�"�qʾ�XQ�þ��~��¾�*��ڽ�G&�$����������?�ګ��5�L�����]����.��fc���X$�z��BvŐ�r�ہkVl�p�/�p`B�p��Dp�@�_"s�$1�7'_��+/���o�kJ%�4�e|�Z#���-�z�!�%�����i
�k���f��p�Łt�=	���R����2!K�R���J��#���j�Z�TA[�����"�RT��+��y�+pm��mm7&c��`����f׽r����tO�����K��󽉊-��J��/�4��ݟ��uy�        �-���q=_�H�}��=�>�i�E�=��.4N�=;3����=z�����=ݟ��uy�=�K���=�9�e��=����%�=f;H�\Q�=�f׽r��=nx6�X� >�`��>RT��+�>���">�J>2!K�R�>��R���>Łt�=	>��f��p>�i
�k>��-�z�!>4�e|�Z#>���<�)>�'v�V,>7'_��+/>_"s�$1>�
�%W�>���m!#�>39W$:��>R%�����>�MZ��K�>��|�~�>�u`P+d�>0�6�/n�>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>���%�>�uE����>�f����>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?�������:�              �?      �?      �?      �?       @              @      �?      �?      �?      �?       @      @      �?      @      @       @      �?      @       @      �?       @      @       @       @       @              �?      �?      �?              @       @               @               @      �?              �?              �?              �?              �?      �?              �?       @              �?              �?              �?      �?       @      �?              �?              �?               @              �?              �?              �?              �?              �?              �?              �?      �?              @      �?      �?       @       @       @       @       @       @      @      @               @              �?              �?              A@              �?               @              �?              �?              �?              �?      �?               @              �?      �?      �?              �?              �?              �?              �?              �?              �?              �?               @              �?      �?              �?              �?              �?       @              �?      �?              �?              �?      �?              �?               @              �?       @      �?      @              �?       @              �?              �?      �?      �?              @      �?      �?      �?       @       @       @      @      �?      �?      @       @      @       @      @      �?      �?      �?      �?      @              �?      �?      �?      �?       @      �?      �?      @              �?       @      �?      �?      �?        
�
conv5/weights_1*�	   ��¿   ����?      �@! �>+�)�ڪ� I@2�	�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.�I�I�)�(�+A�F�&�U�4@@�$��[^:��"�pz�w�7��})�l a��.�?ji6�9�?U�4@@�$?+A�F�&?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?�������:�	             �i@     pt@     �p@     �q@      o@     �j@     �f@     �g@      e@     �a@     @]@     �\@     �Z@      Y@     �U@     �V@     @P@      N@      M@      L@     �J@      M@      @@      ?@      D@     �E@      9@      <@      5@      <@      ,@      5@      1@      .@      2@      2@      $@      &@      "@      @      &@      "@      @      @      @      @      @      �?      @       @      �?       @      @      @               @      �?      �?      @      @      @      �?              �?              �?              �?              �?      �?      �?              �?              �?              �?              @              �?              �?      �?       @      �?      �?       @       @       @      @       @       @      @      @      @      @      @      $@      @      &@      @       @      .@      @      ,@      @      (@      ,@      .@      :@      &@      ,@      <@      1@      (@      9@     �A@      7@     �A@      :@     �E@     �L@      K@      P@      C@     �P@     @T@     �R@     �R@     @Y@      ]@     @]@     �`@      a@     �a@      e@      e@      h@     �k@      n@     p@     �s@     @k@        
�
conv5/biases_1*�	   @U�!�   ��@.>      <@!  ��@>)o�G���<2�4�e|�Z#���-�z�!�%�����i
�k���f��p�2!K�R���J��#���j�Z�TA[�����"�RT��+���f׽r����tO�����Qu�R"�PæҭUݽ;3���н��.4NνPæҭU�=�Qu�R"�=i@4[��=ݟ��uy�=�/�4��=�9�e��=����%�=�tO���=�f׽r��=nx6�X� >�`��>�mm7&c>y�+pm>RT��+�>���">Z�TA[�>�#���j>�J>2!K�R�>Łt�=	>��f��p>�i
�k>%���>4�e|�Z#>��o�kJ%>�'v�V,>7'_��+/>�������:�              �?      �?               @              �?              �?               @              �?              �?              �?              �?      �?              �?              �?              �?               @              �?      �?              �?      �?              �?              �?              �?              �?               @        �M�U      �
�	�a�Q*��A*��

step  �@

loss�GC
�
conv1/weights_1*�	   @�s��   @�̯?     `�@! �ą��)��i2�@2�	����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��5�i}1���d�r�O�ʗ�����Zr[v�����%ᾙѩ�-߾�u`P+d�>0�6�/n�>����?f�ʜ�7
?�T7��?�vV�R9?��ڋ?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?�������:�	              R@     �f@     `j@     �e@     �b@     �e@     �_@     �Z@     �]@     �X@      X@      T@      U@      L@     @Q@      J@     �F@     �G@     �I@      A@      ?@      ;@      9@      7@      5@      8@      2@      3@      3@      2@      $@      @      (@      $@      (@      (@      @      @      (@      $@       @      @       @      @      @      @       @      �?       @      �?               @       @      @      @              �?      �?       @              �?              �?      �?      �?      �?              �?              �?              �?              �?              �?              �?              �?      �?              �?      �?       @               @      �?               @              �?      �?      @       @       @      @      �?      �?      @      �?      @       @       @      @      @      @      @      @       @      @      *@      "@      0@      (@      ,@      0@      $@      (@      ,@      0@      .@      @@      <@      =@      7@     �E@      D@      B@     �H@     �G@     @P@     �P@     �I@      Q@     �U@     @X@      U@     �\@     �]@     @\@     @_@     `c@     �d@     `f@     �h@     �M@        
�
conv1/biases_1*�	    �l�    	�o?      @@! �{��?)���`M�?2��N�W�m�ߤ�(g%k��m9�H�[���bB�SY�ܗ�SsW��lDZrS�nK���LQ�k�1^�sO�IcD���L��!�A����#@�d�\D�X=���%>��:���VlQ.��7Kaa+�I�I�)�(��uE���⾮��%ᾔ4[_>������m!#����n����>�u`P+d�>�ߊ4F��>})�l a�>I��P=�>��Zr[v�>�FF�G ?��[�?�vV�R9?��ڋ?�.�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?�u�w74?��%�V6?��%>��:?d�\D�X=?nK���LQ?�lDZrS?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?�������:�              �?              �?      �?              �?              �?              �?              �?              �?      �?              �?              �?              �?              �?              �?              �?              �?      �?              �?              �?              �?              �?              �?              �?      �?               @      �?      �?       @       @        
�
conv2/weights_1*�	   ��/��   ���?      �@! ��`~0@)����fE@2����g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲��FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(龢f�����uE���⾮��%ᾙѩ�-߾��~]�[Ӿjqs&\�ѾK+�E��Ͼjqs&\��>��~]�[�>��>M|K�>�f����>��(���>a�Ϭ(�>8K�ߝ�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?�������:�             ��@     f�@     ̞@     ̝@     ��@     X�@     4�@     �@     X�@     x�@     X�@     h�@     ��@     Ѕ@     Ȃ@     x�@     �@     �~@     �y@     �y@     Px@     `t@     Pr@     �q@     �m@     �m@     @i@     �f@     �c@     @h@     �`@      \@     �\@     �T@      X@     @U@     �R@      O@     �O@     �I@     �N@      G@     �I@     �K@      C@      A@      D@      :@      4@      9@      >@      2@      4@      7@      7@      (@      "@       @      *@      @      @      @      @       @      @      @       @              $@      @      @       @      @       @       @      @              �?      �?      @       @       @      �?      �?      �?              �?              �?      �?       @              �?      �?              �?      �?              @               @              @              �?              �?      �?       @       @       @      @               @       @      @      @      @      @      @      @      @      @      @      @      (@      $@      @      ,@      2@      &@      6@      0@      <@      ,@      ;@      .@      8@     �@@      A@     �C@     �H@     �K@      K@     �S@     �R@     �R@     @S@     �S@      U@     �Y@     @]@     �^@     �a@     `d@     �g@      h@     �j@     @j@      m@     q@      r@     pu@     �w@     �y@      }@      }@     p@     ��@     ��@     h�@     h�@     h�@      �@     ��@     ԑ@     \�@     ��@     l�@     ��@     X�@     ��@     ֠@     p�@      �?        
�	
conv2/biases_1*�		   ��vi�   �:e?      P@!  	�?)�<LJwA?2�ߤ�(g%k�P}���h�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[�nK���LQ�k�1^�sO�IcD���L�
����G�a�$��{E��T���C��!�A����#@�uܬ�@8���%�V6��u�w74���82��[^:��"��S�F !��.����ڋ�>h�'��f�ʜ�7
���(��澢f����;�"�qʾ
�/eq
Ⱦ        �-���q=��ӤP��>�
�%W�>>�?�s��>�FF�G ?��d�r?�5�i}1?�T7��?�vV�R9?ji6�9�?�S�F !?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?�������:�              �?              �?      �?              �?               @      �?               @      �?              @              �?      �?       @              �?              �?              �?              �?              �?               @              �?              �?              �?              �?              �?              �?      �?              �?      �?       @      �?               @               @              �?      �?      �?              �?       @       @      @      @              �?      �?       @      �?       @      �?        
�
conv3/weights_1*�	   �B���   ���?      �@! �6�-"@)��2��^U@2�����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾄiD*L�پ�_�T�l׾��>M|Kվ�MZ��K���u��gr��jqs&\��>��~]�[�>��>M|K�>�iD*L��>E��a�W�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?�������:�             P�@     L�@     B�@     �@     ��@     ��@     *�@     Ȝ@     ��@     (�@     8�@     ��@     ȑ@     ��@      �@     ��@     h�@     `�@     0�@     h�@     0�@     `}@     �{@     �z@     �v@     �s@     �s@     0p@     `n@     `i@     `h@      i@      d@      b@      a@     �^@     @\@     �\@     �T@     @T@      S@      V@      Q@     �G@     �I@     �D@      H@      I@      A@     �B@      A@      A@      ;@      :@      .@      1@      *@      2@      *@      $@      *@      ,@      .@       @       @      "@      @      @      @      @      @      @      @       @      @      @       @      @       @      @      @      �?      �?      @       @      �?       @      �?       @       @              �?              �?               @      �?              �?              �?      �?              �?              �?       @      �?      �?      @              �?      �?      @              �?      �?      �?              @      @      @      @      �?      @      @      @      @      @      @      @      @      @      "@      "@      "@      .@      (@       @      (@      1@      4@      1@      8@      ;@      ;@      D@      6@      9@     �B@     �F@      I@     �F@     �P@      N@     �P@      I@     �U@     @R@      \@     �S@     �`@     �_@     �a@     `c@     `g@     `h@     �h@      p@     �p@     �o@     0q@     �v@     w@     p{@     �{@     �~@     ��@     ��@     ��@     ȅ@     ��@     ��@     Ѝ@      �@     P�@     h�@     ��@     ̘@     $�@     P�@     ��@     2�@     ޣ@     �@     ~�@     ��@     ��@        
�
conv3/biases_1*�	   @�5c�   @�fh?      `@!   @�0�?)�~��/#?2�5Ucv0ed����%��b��l�P�`��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.���vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲�pz�w�7��})�l a��*��ڽ�G&�$��        �-���q=G&�$�>�*��ڽ>K+�E���>jqs&\��>>�?�s��>�FF�G ?��[�?1��a˲?����?f�ʜ�7
?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?�������:�              �?      @              �?      �?      @       @      @      @       @      @       @       @       @      �?       @      @      �?              �?      �?               @              @      �?              �?              �?       @              �?      �?              �?      �?              �?              �?              �?              �?              @              �?              �?              �?              �?              �?              �?               @              �?               @              �?              �?      �?      �?      �?      �?              �?              �?       @      @      @      �?       @      @       @      @      @      @      @      @       @      @              �?               @        
�
conv4/weights_1*�	   �N��    ��?      �@! `E�2@)&��_�]e@2�Ӗ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����d�r�x?�x����[���FF�G �>�?�s���O�ʗ���I��P=��pz�w�7����(��澢f�����uE���⾮��%������>
�/eq
�>pz�w�7�>I��P=�>�FF�G ?��[�?1��a˲?6�]��?����?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�������:�             �`@     X�@     �@     ��@     �@     x�@     �@      �@     Ї@     ��@     ��@     `�@     ��@     |@     �{@     �w@     �v@     �r@     �q@      p@     �l@      l@      i@     @g@     �a@     �e@     �b@     �^@     @^@     �V@     �Y@      T@     �W@     �R@     �P@     �J@     �E@      G@     �H@     �C@     �@@      =@      <@      4@      :@      2@      2@      .@      4@      1@      (@      *@      ,@      "@      (@       @      @      @      @       @      @      @      @      �?       @      @       @      @       @      @       @      �?      �?       @      �?               @              �?      �?      �?              �?              �?              �?              �?              �?              �?              �?      �?              �?              �?               @      �?      �?       @       @       @      �?      @       @       @       @       @      �?      @       @      @      @      @      @      @      @      &@      @      &@      1@      &@      ,@      .@      *@      &@      1@      6@      7@      :@      6@      B@      >@     �@@      E@      K@      F@     �N@     �L@      O@      S@     @T@     �V@     �T@     @_@     �^@     @^@      `@      d@      d@     �j@      l@     �h@     @n@      q@     �s@     �t@     �w@     �v@     �w@     @@     �@     0�@     ��@     x�@     ��@     `�@     p�@     (�@     ��@     @�@      �@     ��@     �b@        
�
conv4/biases_1*�	   `��k�   �"�t?      p@!����LY�?)����gM2?2��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !���ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
�6�]���1��a˲��FF�G �>�?�s���O�ʗ�����Zr[v��pz�w�7��})�l a��ߊ4F��h���`��%ᾙѩ�-߾�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ����ž�XQ�þ��~��¾��n�����豪}0ڰ���������?�ګ�;9��R���5�L��39W$:���.��fc�����8"uH���Ő�;F��z��6��so쩾4�6NK��2�_"s�$1�4�e|�Z#���-�z�!�%�����i
�k���f��p�Łt�=	���R����2!K�R���J�Z�TA[�����"�RT��+��y�+pm��mm7&c��`����tO����f;H�\Q��'j��p���1����Qu�R"�PæҭUݽ        �-���q=�Qu�R"�=i@4[��='j��p�=��-��J�=�tO���=�f׽r��=nx6�X� >�`��>RT��+�>���">Z�TA[�>�#���j>�J>Łt�=	>��f��p>�i
�k>%���>��-�z�!>��o�kJ%>4��evk'>�'v�V,>7'_��+/>K���7�>u��6
�>X$�z�>.��fc��>;9��R�>���?�ګ>��n����>�u`P+d�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�vV�R9?��ڋ?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?�������:�              �?               @      �?       @       @              �?       @              �?      @      �?      �?      �?      @      @      @      @               @      @              @      �?       @      �?      @       @      �?               @       @      �?       @              �?       @       @              �?      �?       @               @              �?              �?              �?              �?              �?               @              �?       @              �?      �?              �?              �?              �?              �?              �?              �?              �?              �?       @               @      @              @       @              �?              @      �?      �?              @              �?              �?             �@@              �?              �?               @              �?              �?       @      �?      �?              �?      �?      �?      �?               @              �?              �?              �?              �?              �?               @      �?      �?              �?              �?      �?      �?              �?              �?              �?      �?              �?              �?      �?              �?              �?              �?      �?              @              �?      @      �?      �?              @              �?      @       @       @       @       @      @      @       @      @      �?      �?       @       @       @       @       @      �?       @      �?      �?      �?       @      �?       @              @       @       @              �?      �?        
�
conv5/weights_1*�	   ��¿    ڥ�?      �@! ���*�)���I"I@2�	�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.�U�4@@�$��[^:��"��S�F !�ji6�9���f�����uE���⾔S�F !?�[^:��"?I�I�)�(?�7Kaa+?��82?�u�w74?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?�������:�	             �i@     `t@     �p@     �q@     �n@      k@     `f@     �g@      e@     �a@     �]@     @\@     �Z@      Y@     �U@     �V@     �P@      M@      N@      L@     �J@      M@      ?@      A@      C@      D@      <@      =@      4@      <@      *@      6@      3@      &@      5@      1@      $@      *@      @      @      "@      (@      @      @      @      "@      @       @      @       @      �?      @       @       @      �?       @              @      �?      @      �?      @              �?               @              �?              �?               @              �?              �?              �?               @              �?      @      �?      �?      @              �?       @      @      @               @      @      @      @      @      (@      @      $@      @       @      ,@       @      ,@      @      (@      ,@      .@      6@      .@      .@      8@      2@      .@      8@     �A@      8@      A@      9@      E@      L@      L@     �N@      E@      P@     �T@     �R@     @S@      Y@     �\@     �]@     �`@      a@     �a@     `e@     �d@      h@     �k@     �m@      p@     �s@     `k@        
�
conv5/biases_1*�	   �h�(�   ��d6>      <@!   ��F>)Z�r��<2����<�)�4��evk'���o�kJ%�4�e|�Z#���-�z�!���f��p�Łt�=	���R����2!K�R���J��#���j�Z�TA[���f׽r����tO����f;H�\Q���b1��=��؜��=��1���='j��p�=�tO���=�f׽r��=nx6�X� >�`��>�mm7&c>y�+pm>RT��+�>���">Z�TA[�>�#���j>�J>2!K�R�>��R���>Łt�=	>��f��p>�i
�k>���<�)>�'v�V,>�so쩾4>�z��6>�������:�              �?      �?      �?      �?              �?              �?              �?      �?              �?      �?              �?              �?              �?      �?      �?      �?              �?              �?      �?      �?              @              �?      �?              �?               @        @�z�(V      �+	�VSW*��A*��

step  �@

lossb�GC
�
conv1/weights_1*�	   �~:��    7�?     `�@!  ��.�)��\q��@2�����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+��T7����5�i}1���d�r�x?�x��f�ʜ�7
������1��a˲���[��>�?�s���O�ʗ�����(��澢f����x?�x�?��d�r?�T7��?�vV�R9?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?�������:�             �S@     `g@      j@     �e@     �b@     �d@      a@      Z@     �]@     �Y@     �V@      S@     �T@      N@     �Q@      E@     �L@     �F@      J@      A@     �@@      7@      ?@      6@      6@      *@      6@      3@      .@      5@      .@      @      (@      @      1@      $@       @      @      @      @      @      @      "@      @      @       @      @      @      @              @       @       @       @      �?      �?      @              �?      @              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?      �?      �?               @      @      �?       @      @       @       @       @       @      @      �?      �?      �?       @      @      @      @      @      @      @      @      "@      &@      (@      1@      &@      ,@      &@      1@      $@      ;@      4@      7@      :@      8@      <@     �E@     �D@      B@      K@      F@      N@     �P@      K@     �Q@     �U@     @X@     �S@     �[@     �^@     �]@     �^@      c@     �d@     `f@     �h@     �N@        
�
conv1/biases_1*�	   @�q�   �_�t?      @@!  z�g�?)w�p(?2�uWy��r�;8�clp��l�P�`�E��{��^��m9�H�[���bB�SY�
����G�a�$��{E��T���C��!�A�I�I�)�(�+A�F�&��uE���⾮��%�.��fc���X$�z���*��ڽ>�[�=�k�>��Zr[v�>O�ʗ��>�FF�G ?��[�?1��a˲?�S�F !?�[^:��"?I�I�)�(?�7Kaa+?��82?�u�w74?��%>��:?d�\D�X=?���#@?�!�A?k�1^�sO?nK���LQ?�lDZrS?��bB�SY?�m9�H�[?5Ucv0ed?Tw��Nof?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?�������:�              �?              �?      �?      �?              �?      �?      �?              �?              �?              �?              �?              �?              �?      �?              �?              �?              �?              �?              �?              �?      �?              �?               @              @              @      �?      �?        
�
conv2/weights_1*�	   �����    )ī?      �@! �{|@)R����iE@2����g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲��FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7���ߊ4F��h���`�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�Ѿ豪}0ڰ������jqs&\��>��~]�[�>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?�������:�             t�@     d�@     ��@     ��@     �@     T�@     $�@     �@     �@     �@     P�@     ��@     X�@     0�@     (�@     ��@     ��@     �~@     Pz@     �y@     �w@     �t@     Pq@     �r@     �o@      j@     �k@      e@     �c@     �g@     �a@     @^@      X@     @Y@      T@      X@     �Q@      P@     �N@     �I@     �K@     �K@     �I@     �H@      C@      C@      A@      7@      7@      >@      :@      ,@      8@      7@      2@      ,@      *@       @      $@      (@       @      @      @      @       @      "@      @      @      "@      @       @      @      @       @       @      @              �?       @      �?              �?              @              �?              �?      �?      �?              �?              �?      �?              �?              @              �?      �?      �?       @              @      �?      �?              @              �?      �?       @       @      @              @      @      @      @      &@      @      @       @      @      @      @      $@      @      "@      (@       @      0@      9@      .@      ;@      2@      ;@      1@      9@      @@      >@      G@     �F@     �P@      I@     �R@     @R@      S@     @R@     @U@     �U@     �Y@     @[@     �^@     @`@      e@     �g@     �h@     �j@     @j@     �m@     �q@     `q@     Pu@     �w@     �y@     �|@     �}@     �@      �@     ��@      �@     X�@     ��@     �@     ��@     ȑ@     X�@     ��@     x�@     �@     \�@     ��@      @     ̓@       @        
�	
conv2/biases_1*�		   ��n�    �l?      P@!  ʻ��?)���p� ?2�;8�clp��N�W�m�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�<DKc��T��lDZrS�nK���LQ�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���bȬ�0���VlQ.�I�I�)�(�+A�F�&��T7����5�i}1���d�r�x?�x�������6�]�����[���FF�G ��f�����uE���⾞[�=�k���*��ڽ�        �-���q=8K�ߝ�>�h���`�>1��a˲?6�]��?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?�7Kaa+?��VlQ.?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?a�$��{E?
����G?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?�������:�              �?              �?              �?      �?              �?       @              �?      �?              �?              �?      �?      �?              �?      �?              �?              �?              �?              �?              �?              �?              �?              �?               @              �?              �?              �?      �?      �?      �?              �?               @      �?      @      �?              �?              �?      �?       @      @       @       @       @      �?       @      �?      �?       @      �?      �?      �?        
�
conv3/weights_1*�	   ��ݮ�   ����?      �@! ��w7�$@)]� `U@2�����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE�����_�T�l׾��>M|Kվ�u`P+d����n������MZ��K���u��gr����~���>�XQ��>�����>;�"�q�>['�?��>jqs&\��>��~]�[�>�ѩ�-�>���%�>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?�������:�             X�@     <�@     H�@     �@     v�@     ��@     �@     �@     ��@     D�@     4�@     ��@     ��@     ��@      �@     ��@     X�@     ��@     (�@     P�@     H�@     }@     �{@     @z@     `v@      t@     �s@     Pp@      m@      j@     �i@      g@     `d@     �c@     �`@     �\@      ]@     @\@      V@     @U@     @T@      T@      P@      H@     �F@      F@     �J@     �H@     �@@      D@      ;@      ?@      ;@      7@      7@      1@      4@      0@      .@      .@      ,@      "@      .@      @       @      $@      @       @       @      @      @      @      @      @      �?      �?      @      @       @      �?       @       @              @       @      @              �?      �?      �?      �?      �?      �?              �?              �?              �?              �?      �?              �?              �?              �?               @      �?      �?      �?              �?      �?       @       @      �?       @       @      �?      @      @       @       @       @              @       @      @      �?      @      &@      @      @      (@      "@      @      (@      (@      $@      &@      2@      2@      4@      9@      8@      ?@      =@      7@      <@     �A@     �F@      C@     �L@      N@     �P@      O@      K@     �S@     �S@     �[@      W@     �^@     �`@      `@     �c@     �g@     �g@      j@      p@     �p@     �n@     �q@     �v@     pw@     �z@     �{@     �~@     ��@     ��@     ��@     X�@     X�@     ��@     ��@     P�@     D�@     `�@     |�@     И@     (�@     <�@     ��@     *�@     �@     ��@     p�@     ��@      �@        
�
conv3/biases_1*�	   ���f�   @��n?      `@! @ru���?)�� g޴.?2�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C����#@�d�\D�X=���%>��:���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.���vV�R9��T7����5�i}1���d�r������6�]���1��a˲���[��        �-���q=���?�ګ>����>�XQ��>�����>�_�T�l�>�iD*L��>})�l a�>pz�w�7�>I��P=�>��d�r?�5�i}1?�T7��?�vV�R9?�.�?ji6�9�?�S�F !?I�I�)�(?�7Kaa+?��bȬ�0?��82?�u�w74?��%>��:?d�\D�X=?���#@?�!�A?�T���C?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?�������:�              @       @      �?              �?       @      @      @       @      @      @      @      @               @      �?      �?              �?      �?              �?       @       @      @       @              �?      �?      @               @              �?              @              �?              �?               @              �?              �?              �?              �?      �?              �?              �?              �?      �?              �?              �?      �?              �?      �?      �?      �?              @      @      @      @      �?      @              @      @      �?      @       @       @      �?       @      �?              �?        
�
conv4/weights_1*�	   ��%��   ��&�?      �@! H��!
@)��H6^e@2�Ӗ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.���5�i}1���d�r�x?�x��1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��8K�ߝ�a�Ϭ(��uE���⾮��%ᾄiD*L�پ�_�T�l׾�����>
�/eq
�>})�l a�>pz�w�7�>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?>h�'�?x?�x�?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�������:�              `@     h�@     �@     �@     ��@     @�@      �@     (�@     ��@     x�@     h�@     ��@     ��@     �|@     �{@     pw@     0v@      s@     �q@     0p@     �l@     `k@     �i@     �f@     �a@     �e@     �b@     �^@      ^@     �V@     �Y@      S@     �Y@     @R@     �N@     �K@     �F@     �C@      L@     �C@     �A@      <@      :@      0@      8@      7@      6@      *@      (@      4@      (@      (@      ,@      ,@      &@      "@      @       @      @      @      @       @      @      �?       @      @       @      @              @      @      @               @      �?               @      �?              �?              �?      �?              �?      �?              �?              �?              �?              �?              �?              �?      �?               @      �?              �?              @              �?      �?      �?               @      @      @      �?      �?       @      @      @       @      @      @       @      @      @      "@      &@      @      (@      &@      ,@      .@      1@       @      .@      0@      3@      9@      ;@      7@      A@      <@      A@     �E@      J@     �G@      P@      L@     �N@     @R@      U@     �U@     @V@     �]@      _@      ^@     �`@     �c@      d@      k@     �k@     �h@      o@     �p@     Pt@     �t@     `w@     0w@     �w@     �~@      �@     �@     �@     ��@     ��@     @�@     ��@     (�@     ��@     (�@      �@     ��@     �b@        
�
conv4/biases_1*�	   �
_s�   @N�z?      p@!���2�?)��aٛ�@?2�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��S�F !�ji6�9���.����ڋ��vV�R9��5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������1��a˲���[���FF�G �I��P=��pz�w�7��})�l a�a�Ϭ(���(��澢f�����uE���⾙ѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾;�"�qʾ
�/eq
Ⱦ����ž0�6�/n���u`P+d����n�����豪}0ڰ�������5�L�����]������z!�?��T�L<��������M�6��>?�J�p
T~�;�u 5�9��z��6��so쩾4����<�)�4��evk'���o�kJ%���-�z�!�%�����i
�k���f��p�Łt�=	���R����2!K�R���J��#���j�Z�TA[�����"�RT��+��y�+pm��mm7&c��`���nx6�X� ��9�e����K��󽉊-��J�'j��p�ݟ��uy�z�����x�_��y�'1˅Jjw�        �-���q=��-��J�=�K���=����%�=f;H�\Q�=�tO���=nx6�X� >�`��>���">Z�TA[�>�J>2!K�R�>��R���>Łt�=	>��f��p>�i
�k>%���>��-�z�!>4��evk'>���<�)>�'v�V,>7'_��+/>�����0c>cR�k�e>�
�%W�>���m!#�>
�}���>X$�z�>R%�����>�u��gr�>����>豪}0ڰ>��n����>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>8K�ߝ�>�h���`�>})�l a�>pz�w�7�>I��P=�>O�ʗ��>>�?�s��>�FF�G ?��[�?�5�i}1?�T7��?�vV�R9?��ڋ?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?�������:�              �?              �?      @      �?       @               @              @       @      �?      �?      �?              @      @      �?      @       @      �?       @      �?      @      @      �?      �?      �?       @              @      �?      �?      �?              �?      �?              �?              �?              �?              @              �?      �?              �?      �?              �?              �?              �?      �?      �?              �?      �?              �?      �?              �?              �?              �?              �?              �?              �?              �?      �?              @       @      @      �?               @       @              �?              �?              �?      �?      �?              �?              �?              �?              �?              =@              �?              �?      �?              �?               @               @      �?              �?      �?      �?      �?               @      �?      �?              �?              �?              �?              �?              �?      �?              �?       @      �?              @               @              �?              �?              �?              �?      �?              �?              �?              �?              �?              �?              �?      �?       @       @      @      �?      �?      @               @      �?       @      �?      @       @      @      @      @      �?      @       @       @       @               @      �?              �?       @      @       @              @       @              @      �?      �?               @      �?        
�
conv5/weights_1*�	   @�¿    *��?      �@! ��S�/*�)w>�$I@2�	�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.�U�4@@�$��[^:��"��5�i}1���d�r�6�]���1��a˲��uE���⾮��%�K+�E���>jqs&\��>U�4@@�$?+A�F�&?�7Kaa+?��VlQ.?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?�������:�	             �i@     �t@     �p@     `q@     `o@     �j@     �f@     �g@     �d@      b@     �]@     �[@     @Z@     @Z@     �U@      V@      P@      N@      O@      K@     �J@     �L@      A@     �@@     �D@     �B@      :@      A@      0@      =@      .@      3@      4@      (@      3@      .@      ,@      ,@      @      @       @      $@      @      @       @      @      �?       @      @      �?      @      @      �?      @               @      �?       @       @       @       @      �?      �?      �?      �?      �?              �?              �?              �?              �?              �?              �?              �?              �?               @              �?      �?      �?      �?              �?      @              �?      @      @       @       @      �?      @      @      @      @      $@      @      &@      "@      @      *@      @      .@      $@      "@      .@      ,@      8@      &@      2@      9@      0@      2@      8@      B@      4@      >@      =@      E@     �K@     �L@      N@     �F@      P@     �S@     @S@     @S@     �X@      ]@     @]@     �`@      a@     �a@     �e@     �d@     �h@     �k@     �m@      p@     �s@     �k@        
�
conv5/biases_1*�	   ࣯0�   ���9>      <@!   �vK>)�=)ӧ<2�_"s�$1�7'_��+/��'v�V,����<�)�4��evk'���o�kJ%���R����2!K�R���J��`���nx6�X� �����%���9�e����
6������Bb�!澽�tO���=�f׽r��=nx6�X� >�`��>y�+pm>RT��+�>���">Z�TA[�>�#���j>�J>2!K�R�>��R���>%���>��-�z�!>4�e|�Z#>��o�kJ%>���<�)>�'v�V,>6NK��2>�so쩾4>�z��6>u 5�9>p
T~�;>�������:�              �?      �?      �?               @              �?       @              �?              �?              �?               @              �?               @      �?      �?               @              �?              �?      �?      �?              �?              �?      �?              �?        ���jV      ��v_	j��\*��A*��

step  �@

loss&�GC
�
conv1/weights_1*�	   ��W��   `2�?     `�@! ��n��)C�@8
@2���]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8��u�w74���82���VlQ.��7Kaa+�I�I�)�(�+A�F�&�ji6�9���.���5�i}1?�T7��?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?�������:�              �?     �T@     `g@     �i@     �e@     �b@     �d@     �`@     �Y@     �]@     �Y@      W@      Q@     �U@     �P@      N@     �J@     �I@     �I@     �F@      =@      B@      :@      @@      5@      4@      7@      4@      0@      ,@      0@      0@      ,@      "@      .@      &@      "@      @      @      @       @      $@      @      @      @      @      @      @      @      @       @      �?      @      @      �?              @              �?      �?       @              �?              @               @              �?              �?              �?              �?       @      @       @      �?      @      @      @      �?      �?              �?      �?       @      @      @      @      @      @      "@      @      @      (@      ,@      .@      &@      &@      $@      6@      ,@      4@      2@      :@      8@      :@      :@     �C@     �E@      A@     �L@      G@      P@      P@      K@     �Q@      U@     �W@     �R@     �]@     �^@     @\@     �_@     �c@      d@     �f@     `h@     @P@        
�
conv1/biases_1*�	   �w'u�   `�9?      @@!  ��rǪ?){�!Î:?2�&b՞
�u�hyO�s�ߤ�(g%k�P}���h��l�P�`�E��{��^��m9�H�[���bB�SY�k�1^�sO�IcD���L��qU���I�
����G���%>��:�uܬ�@8�U�4@@�$��[^:��"��vV�R9��T7�����~]�[Ӿjqs&\�Ѿ��|�~���MZ��K����~���>�XQ��>�FF�G ?��[�?1��a˲?f�ʜ�7
?>h�'�?I�I�)�(?�7Kaa+?d�\D�X=?���#@?�T���C?a�$��{E?�lDZrS?<DKc��T?�m9�H�[?E��{��^?���%��b?5Ucv0ed?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?�������:�              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?      �?              �?              �?              �?              �?              �?               @              �?              �?      �?      �?               @      �?      @      �?        
�
conv2/weights_1*�	   �����   �aV�?      �@! �1�h$@)+�k=nE@2�I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=���ߊ4F��h���`�8K�ߝ뾢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ
�/eq
Ⱦ����ž�XQ�þ;�"�q�>['�?��>jqs&\��>��~]�[�>�_�T�l�>�iD*L��>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?�������:�              �?     �@     t�@      �@     d�@     4�@     �@     ,�@     �@     �@     �@     (�@     Њ@     ��@     ��@     p�@     ��@     h�@     @     �z@     py@     �x@     �t@     0p@     �r@     �o@     �l@     `j@     �d@     @d@     @f@      a@      _@     @Z@     �U@     @V@     �W@     �Q@     �P@     @P@      L@     �F@      L@      H@      I@     �E@     �B@     �@@      6@      6@      8@      =@      ,@      2@      5@      5@      1@      (@      *@      *@       @      @      @      @      "@      �?      @       @      �?      @      �?      @      @      @      @      @       @       @      �?      �?       @      �?       @              �?      �?              �?       @      �?               @              �?      �?              �?               @              �?              �?      �?      �?              �?      �?               @      �?              �?               @      @      @              @      �?      "@      @      @      "@       @       @      @      @      @      (@      $@      *@      (@      &@      (@      0@      4@      3@      8@      (@     �@@      1@      =@     �@@     �A@     �E@     �B@     �K@      F@     �S@     @S@     @S@     �R@     �U@      V@     �\@     �Z@     �\@     �`@     �c@     �g@     @i@     �k@     �h@      n@     �q@     �r@     t@     @w@      z@     @}@     �}@     0@     ��@     0�@     ��@     ��@     ��@     ��@     ��@     ܑ@     �@     ��@     ��@     �@     ��@     |�@     ܠ@     0�@      @        
�	
conv2/biases_1*�		   ��Ap�   �q0s?      P@!  �2�?)G���a+?2�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed���bB�SY�ܗ�SsW�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G��!�A����#@�d�\D�X=���%>��:���82���bȬ�0��7Kaa+�I�I�)�(�+A�F�&�ji6�9���.����Zr[v��I��P=���ߊ4F��h���`�8K�ߝ�E��a�Wܾ�iD*L�پ        �-���q=>�?�s��>�FF�G ?1��a˲?6�]��?f�ʜ�7
?>h�'�?I�I�)�(?�7Kaa+?��bȬ�0?��82?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?�������:�              �?      �?      �?              �?              �?               @      �?               @              �?      �?      �?              �?              �?      �?              �?              �?              �?      �?              �?               @              �?              �?              �?              �?              �?              @      �?              @       @              �?      �?              �?               @      �?      �?      �?      @      �?      @       @      �?               @       @      �?      �?      �?        
�
conv3/weights_1*�	    v��    !�?      �@!�	��r(@)��bU@2�����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾��>M|Kվ��~]�[Ӿjqs&\�Ѿ
�/eq
Ⱦ����ž�MZ��K���u��gr��jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?�������:�             p�@     .�@     "�@     �@     ��@     ��@     �@     8�@     ę@      �@     ,�@     ̓@     t�@     �@     ��@     Њ@     ��@     �@     �@     p�@     �@     �|@      |@     �z@     �u@     �s@     t@     p@     @n@     `j@     @i@      g@      d@      c@     `b@     �[@     �^@     @[@     �W@     �W@      R@     @S@      L@     �H@     �N@      C@      D@      G@      E@      @@      <@     �@@      9@      >@      6@      1@      *@      4@      .@      .@      (@      $@      0@      &@      @       @      @       @       @       @      @      @      @      @              @      @      @              @      @      �?      @      �?       @              �?      @       @       @              �?      �?              �?              �?      �?              �?              �?              �?              �?              �?              @              �?      �?              �?       @       @               @      �?              �?       @      �?      �?      @       @       @      @      @      @       @      @      @      "@      &@      @      @       @      "@      $@      (@      @      $@      *@      .@      5@      2@      5@      5@      :@      A@      9@     �@@     �C@      E@      G@      E@     �J@     �P@      P@      M@     �U@     �R@     �Z@      W@     �\@      a@     �a@     �b@     �g@     `g@     �j@     �n@     Pq@     �n@     �r@     �v@     �w@     �y@     Pz@     �@     ��@     ��@     P�@     (�@     ��@     Љ@     ��@     <�@     @�@     ��@     ��@     ��@     \�@     P�@     �@     ,�@     ܣ@     �@     ^�@     ��@      �@        
�
conv3/biases_1*�	    ��m�   �3*s?      `@!  ��@��?)���~��7?2��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.���vV�R9��T7����5�i}1���d�r���Zr[v��I��P=��})�l a��ߊ4F��T�L<��u��6
��豪}0ڰ>��n����>��~]�[�>��>M|K�>E��a�W�>�ѩ�-�>1��a˲?6�]��?����?f�ʜ�7
?�T7��?�vV�R9?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��bȬ�0?��82?uܬ�@8?��%>��:?���#@?�!�A?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?�������:�              �?      @       @               @       @      @      @      @       @              �?      @      @       @      �?       @       @               @              @      �?      @      �?       @      �?              �?      �?      �?      �?      �?              �?              �?      �?      �?              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?       @              �?              �?              �?              �?               @              �?      �?      �?       @      @      @      @      @      @              @      @      @      �?       @      @       @      �?      �?      �?      �?        
�
conv4/weights_1*�	   ��4��     E�?      �@! ��"F@)��_e@2�Ӗ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��f�ʜ�7
��������Zr[v��I��P=��pz�w�7���uE���⾮��%������>
�/eq
�>8K�ߝ�>�h���`�>�FF�G ?��[�?1��a˲?6�]��?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�������:�             �_@     l�@     ܕ@     �@     �@     X�@     ،@     H�@     �@     ��@     h�@     ��@     x�@     p|@     �{@     @w@      v@      s@     �q@     `p@      l@     �k@     �i@     �f@     �b@     �d@     `c@      ^@      ]@      X@     �Y@     �R@     �Z@     @Q@     �N@     �K@      F@     �D@      K@     �B@      D@      7@      =@      2@      4@      7@      7@      .@      0@      (@      $@      3@      "@      ,@      $@      "@      @      $@      @      @      @      @      @       @      @      @       @      @       @      @      �?      @      �?      @      �?      �?              �?      �?              �?               @              �?      �?              �?              �?              �?              �?      �?      �?              �?              �?      �?      �?       @               @               @      @      �?      �?               @       @      @       @      @      @      @      @      @      $@      &@       @      (@      $@      ,@      1@      .@      &@      &@      2@      7@      6@      9@      7@      A@      ;@     �B@      E@     �H@      G@      P@     �M@     �L@     @S@     �U@     �T@     �W@     �[@     ``@     @]@      a@     �c@     �c@     `k@     �k@     �h@     @o@     �p@     @t@     �t@     @w@     w@     �w@     �~@     8�@     �@     �@     ��@     ��@      �@     ��@     �@     ��@     �@     ,�@     ��@     @c@        
�
conv4/biases_1*�	   @Bw�    W��?      p@!j]��Qޮ?)��Q�5�L?2�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&��[^:��"��S�F !��5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲�>�?�s���O�ʗ����f�����uE���⾙ѩ�-߾E��a�Wܾ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;�[�=�k���*��ڽ�5�"�g���0�6�/n���u`P+d����n��������?�ګ�;9��R��39W$:���.��fc���
�}�����4[_>���p
T~�;�u 5�9�7'_��+/��'v�V,����<�)�4��evk'���o�kJ%�4�e|�Z#���-�z�!�%�����i
�k���f��p�Łt�=	�2!K�R���J��#���j�Z�TA[�����"�RT��+��y�+pm��mm7&c�f;H�\Q������%���9�e���ݟ��uy�z������|86	Խ(�+y�6ҽ��.4Nν�!p/�^˽        �-���q=��؜��=�d7����=��.4N�=;3����=�K���=�9�e��=�f׽r��=nx6�X� >���">Z�TA[�>�J>2!K�R�>��R���>Łt�=	>��f��p>�i
�k>%���>��-�z�!>���<�)>�'v�V,>7'_��+/>_"s�$1>6NK��2>�so쩾4>X$�z�>.��fc��>�MZ��K�>��|�~�>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>�ѩ�-�>���%�>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>})�l a�>pz�w�7�>I��P=�>��Zr[v�>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?x?�x�?��d�r?�5�i}1?�T7��?��ڋ?�.�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?�������:�              �?       @       @              �?      �?       @      �?      �?      �?       @       @      �?      �?      @      @      @       @      �?              �?      @      �?       @       @              @              �?      @       @       @      �?      �?       @              �?              �?      �?      �?      �?              �?      �?              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?               @               @              �?              �?      �?       @              �?       @       @      �?              �?      �?      �?               @              �?              �?      �?              �?              �?              �?              ;@              �?              �?              �?              �?               @              �?              �?       @      �?       @      �?              �?              �?              �?              �?              �?              �?              �?              �?              �?               @      �?              �?              �?              �?      �?              �?              �?              �?              �?              �?      �?              @              �?              �?              �?              �?      @       @               @      @       @      �?      �?              @       @      @      �?       @      �?       @      @      @       @      @      @      @       @      �?              @       @               @      @              @      @      �?      @      �?      �?      @              �?       @        
�
conv5/weights_1*�	   ��¿   ����?      �@! !�L)�)�_�J"(I@2�	�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�U�4@@�$��[^:��"���[���FF�G �>h�'�?x?�x�?�.�?ji6�9�?�S�F !?�[^:��"?I�I�)�(?�7Kaa+?��bȬ�0?��82?�u�w74?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?�������:�	             �i@     �t@     �p@     0q@     �o@      j@      g@      h@      d@     `b@      ]@     �[@     �Z@     �Y@      V@     �T@     @Q@      N@      N@      N@     �I@     �L@      >@     �A@     �B@     �B@      ;@      A@      3@      =@      0@      2@      4@      &@      3@      .@      $@      4@      @       @      @       @      @      @      &@      @      �?       @      @              @      @       @       @      @       @               @      @       @               @               @              �?      �?              �?              �?              �?              �?              �?              �?              �?              �?      @              �?      �?      �?      �?      @      �?      �?      @       @      �?      @       @      @       @      @      @      $@      @      "@       @       @      ,@       @      ,@      "@      $@      0@      ,@      7@      ,@      ,@      ;@      1@      ,@      :@      B@      5@      >@      =@      F@      J@      L@     �M@      G@     �O@     @T@      S@     �S@     �X@     @\@      ^@     �`@      a@     �a@     �e@      d@     �h@     @k@     �n@      p@     �s@      l@        
�
conv5/biases_1*�	   `��6�    �C>>      <@!  ��zL>)#�	���<2�u 5�9��z��6��so쩾4�6NK��2�7'_��+/��'v�V,����<�)�Łt�=	���R����2!K�R���J��#���j��tO����f;H�\Q��;3���н��.4NνV���Ұ�=y�訥=�9�e��=����%�=f;H�\Q�=�tO���=�f׽r��=nx6�X� >�`��>�mm7&c>y�+pm>Z�TA[�>�#���j>�J>2!K�R�>��R���>��-�z�!>4�e|�Z#>4��evk'>���<�)>7'_��+/>_"s�$1>6NK��2>�so쩾4>�z��6>u 5�9>p
T~�;>����W_>>�������:�              �?              �?              �?       @              �?              �?      �?              �?              �?              �?              �?              �?              �?      �?              �?              �?      �?      �?       @              �?               @              �?              �?              �?              �?        h3E{�V      ���	���b*��A*ɭ

step   A

loss̓GC
�
conv1/weights_1*�	   �{3��   @(�?     `�@! ���S��)E_��9@2�	��]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��S�F !�ji6�9��6�]���1��a˲���Zr[v��I��P=���FF�G ?��[�?1��a˲?f�ʜ�7
?>h�'�?�5�i}1?�T7��?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?�������:�	              @     �X@     @h@      j@     �d@     `b@     �e@     �_@     @\@      [@     @Y@     @X@     �N@     @V@     �M@      Q@     �H@      J@     �H@      G@      C@      =@      :@      ?@      5@      4@      4@      8@      2@      3@      (@      ,@      ,@      1@      *@      &@      @       @      @      @      $@      @      �?      @      @      @      �?       @      �?               @       @      @              �?       @               @      �?      �?       @       @              �?               @              �?              �?              �?      �?              �?              �?               @       @      �?      �?      �?               @              �?               @      �?      �?       @       @      @       @      �?       @       @      @       @      @      �?      @      @      @       @       @      "@      "@      @      $@      &@       @      (@      "@      $@      6@      *@      6@      8@      3@      4@      ?@      :@      @@     �A@      G@      M@      I@     �L@     �N@      M@     �R@     �S@     @X@     �U@     �[@     �\@     @]@     �`@     �a@     @d@     �h@     �e@      O@        
�
conv1/biases_1*�	   `ӭx�   �'�?      @@! �Bt(#�?)��u ��A?2�o��5sz�*QH�x�Tw��Nof�5Ucv0ed��m9�H�[���bB�SY�<DKc��T��lDZrS�IcD���L��qU���I��.����ڋ��vV�R9���������?�ګ��5�L�����]���������>
�/eq
�>8K�ߝ�>�h���`�>6�]��?����?f�ʜ�7
?>h�'�?�T7��?�vV�R9?��bȬ�0?��82?
����G?�qU���I?��bB�SY?�m9�H�[?E��{��^?�l�P�`?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?�������:�              �?              �?               @              �?               @              �?      �?              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?      �?              @              �?      @              �?        
�
conv2/weights_1*�	    ����   `Zc�?      �@! ��P8�)@)�S[`tE@2�I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ5�"�g���0�6�/n���u`P+d����n�����;9��R���5�L��u��6
��K���7��K+�E���>jqs&\��>��~]�[�>a�Ϭ(�>8K�ߝ�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?�������:�              @     ��@     ��@     ��@     $�@     X�@     �@     �@     ̒@     �@     X�@     h�@     P�@     ��@     `�@     ��@     �@     H�@     �~@     �{@      y@      w@     �u@     �p@     �q@     @p@      k@      l@     �d@     �c@     �e@     �b@     �]@      Y@     �T@     @W@     �U@      R@     �P@      R@      N@     �I@     �J@     �D@      H@     �C@     �B@      ?@      7@      9@      =@      6@      7@      2@      >@      *@      .@      &@      "@      @      &@      @       @      (@      @      @      @      @      @      @      @      @      @       @       @      @      @      �?              @      @       @      �?              �?      �?              �?              �?      �?              �?              �?              �?      �?              �?              �?              �?              �?              �?      �?              �?              �?      @               @      �?              @       @      �?              @      @       @      @      @      @      @      *@      @      @      @      @      *@       @      &@      ,@      (@      &@      5@      4@      0@      ,@      9@      8@      6@      A@     �D@     �F@      G@      L@     �E@     �R@      R@     �R@     �V@     �S@     �W@      [@     �[@      ^@     �_@     `c@     �f@     �i@     `k@      i@     �n@     pq@      r@     @t@     �w@     �y@     @}@     }@     @@     `�@     ��@      �@      �@     X�@     ��@     ��@     �@     0�@     ��@     ��@     ԙ@     l�@     ��@     ܠ@     \�@      7@        
�

conv2/biases_1*�
	    ��p�   ��v?      P@!  (Y���?)@��ٟ|1?2�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�5Ucv0ed����%��b��l�P�`�E��{��^�nK���LQ�k�1^�sO�IcD���L��qU���I�a�$��{E��T���C��!�A����#@�uܬ�@8���%�V6���VlQ.��7Kaa+���ڋ��vV�R9������6�]���1��a˲���[��>�?�s���O�ʗ���        �-���q=��Zr[v�>O�ʗ��>�FF�G ?��[�?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?�7Kaa+?��VlQ.?��bȬ�0?��82?��%>��:?d�\D�X=?���#@?�!�A?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?�������:�              �?      �?              �?              �?              �?               @      �?      �?              �?              �?              �?              �?              �?              �?              �?              �?               @              �?              �?              �?              �?              �?              �?      �?              �?      �?      �?               @               @               @      @              �?              �?              �?      @               @       @      �?      �?       @       @      @       @      �?               @              �?        
�
conv3/weights_1*�	   �T��   �,��?      �@!`X�H�-@)�\3VeU@2�����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(龢f�����uE����E��a�Wܾ�iD*L�پ��������?�ګ��MZ��K���u��gr��4�j�6Z�Fixі�W�����>豪}0ڰ>jqs&\��>��~]�[�>��>M|K�>�ѩ�-�>���%�>��(���>a�Ϭ(�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�������:�             X�@     �@     �@     8�@     X�@     ��@     $�@     �@     ę@     �@     @�@     �@     4�@     Đ@     X�@     ��@     ��@      �@     0�@     `�@     �@     }@     �{@     �z@     v@     @t@      s@     �p@     �k@     �k@     @k@     `f@     �e@     `b@     �`@     �_@     @^@      \@     �X@     �T@     �S@     �T@      I@      J@     �L@      D@     �D@      F@     �D@      B@      =@     �@@      :@      =@      5@      4@      .@      ,@      2@      &@      *@      @      3@      &@      @      "@      @      @      @      @      @      @      @       @       @              �?      @      �?      @      �?       @       @       @      �?      �?      �?       @      �?      @               @              �?              �?              �?              �?              �?              �?      �?              �?              @               @      �?              @      �?       @       @      @       @      @       @      �?       @       @      @      @      @      @      @       @      @      @       @      @       @      "@      @       @      @      1@      0@      1@      1@      7@      4@      =@      =@      <@      @@      D@      I@      G@      G@      I@     �O@      K@      N@     @T@      R@     @Z@     �V@      \@     �`@     �b@     �d@     @e@     �g@     @k@     �p@      p@     �o@     �r@     �v@     �w@     �x@      {@      @     ��@     ��@     H�@     ��@     ��@     �@     P�@     L�@     $�@     ��@     ��@     ��@     l�@     \�@     ԟ@     <�@     ܣ@     ��@     f�@     ��@     �@      @        
�
conv3/biases_1*�	    �ws�   ���y?      `@! ����ȣ?)��Q��A?2�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���vV�R9��T7����5�i}1���d�r�x?�x��>h�'��O�ʗ�����Zr[v�����m!#���
�%W��R%�����>�u��gr�>�_�T�l�>�iD*L��>�uE����>�f����>�FF�G ?��[�?1��a˲?��d�r?�5�i}1?ji6�9�?�S�F !?�[^:��"?+A�F�&?I�I�)�(?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?�������:�               @               @              @      �?       @      @      �?       @      �?              �?       @       @      @      �?      �?       @      @              �?       @              �?      �?       @       @       @              �?       @      �?              �?      �?              �?              �?      �?              �?      �?              �?              �?              �?              �?              �?              �?      �?              �?              �?      �?               @              @              �?               @      @       @      @      �?      �?      @       @      @      @       @      @       @      @      @       @              �?       @              �?        
�
conv4/weights_1*�	   �E��   �%m�?      �@! �Ңt�@)0�U`e@2�Ӗ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��>�?�s���O�ʗ�����Zr[v��pz�w�7��})�l a��ߊ4F��h���`��uE���⾮��%������>
�/eq
�>�_�T�l�>�iD*L��>�FF�G ?��[�?1��a˲?6�]��?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�������:�              _@     X�@     ��@     �@     �@     h�@     Ќ@     8�@     �@     `�@     ��@     X�@     ��@     P|@     �{@     `w@     �u@     ps@     �q@     @p@     �l@      k@      i@     �f@     �b@     �d@      d@     �\@     �]@     �V@     @Z@     �T@     @X@      S@      N@     �K@      F@     �F@     �E@     �D@     �B@      6@      @@      6@      4@      9@      3@      ,@      .@      .@      &@      .@      *@      (@      "@      $@      @      &@      @      @      @      @      @      @       @      @       @      @       @      @               @      �?       @              �?       @              �?      �?              �?              �?      �?              �?              �?              �?              �?              �?              �?              �?              �?      �?      �?              �?       @      �?              �?      �?      �?      @      �?       @               @      �?      @      @      @      @      @      @      @       @      .@      $@      &@      $@       @      ,@      6@      &@      .@      ,@      4@      8@      8@      ;@     �@@      A@     �A@      B@      H@      E@     @P@     �M@     �Q@      Q@     �T@     �W@     @V@     �[@      `@      ^@     @`@      d@     �c@     `k@     @k@      i@     �n@      q@     @t@     �t@     �w@     �v@      x@     �~@     �@      �@     ��@     X�@     Ј@     (�@     ��@     (�@     ��@      �@     4�@     t�@     �d@        
�
conv4/biases_1*�	   �G�}�    ���?      p@!j�����?)Q�4bfxW?2�>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��T7����5�i}1���d�r�x?�x��f�ʜ�7
��������[���FF�G �>�?�s������%ᾙѩ�-߾��>M|Kվ��~]�[ӾG&�$��5�"�g������?�ګ�;9��R��R%������39W$:����
�%W����ӤP���/�p`B�p��Dp�@�����W_>�7'_��+/��'v�V,����<�)�4��evk'���o�kJ%���-�z�!�%�����i
�k���f��p�Łt�=	���R����2!K�R���J����"�RT��+��y�+pm��mm7&c��f׽r����tO������-��J�'j��p�=��]����/�4���
6������Bb�!澽        �-���q=�d7����=�!p/�^�=H�����=PæҭU�='j��p�=��-��J�=�f׽r��=nx6�X� >�`��>�mm7&c>Z�TA[�>�#���j>�J>%���>��-�z�!>4�e|�Z#>��o�kJ%>4��evk'>�'v�V,>7'_��+/>_"s�$1>�so쩾4>�z��6>u 5�9>p
T~�;>��z!�?�>��ӤP��>R%�����>�u��gr�>���]���>�5�L�>5�"�g��>G&�$�>�����>
�/eq
�>['�?��>K+�E���>jqs&\��>��~]�[�>�_�T�l�>�iD*L��>���%�>�uE����>��(���>a�Ϭ(�>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?6�]��?����?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?eiS�m�?#�+(�ŉ?�������:�              �?       @      �?      �?              �?               @      @               @      �?       @      @      @       @      @      �?      @               @       @      @               @       @              �?      @      �?       @       @               @      �?      �?      �?      �?       @      �?      �?               @      �?      �?               @              �?      �?              �?              �?              �?              �?              �?              �?              �?      �?               @       @       @      �?              �?              �?       @      �?       @      �?              �?              �?              �?              �?              �?              �?              ;@              �?              �?              �?              �?              �?               @      �?               @      �?      �?      �?              �?      �?              �?              �?              �?              �?               @              �?              �?               @               @              �?              �?               @              �?              �?      �?              �?      �?              �?              �?      �?      �?              �?              �?              �?              �?      �?      @       @               @      �?      @       @      �?       @      @              �?       @      @              @      @      @      �?      �?      @      �?      @      �?              @      @       @      �?       @               @       @      @       @      @       @               @      �?      @              �?        
�
conv5/weights_1*�	   �<�¿    ��?      �@! ��� (�)��z�,I@2�	�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��ߊ4F��h���`��[^:��"?U�4@@�$?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?�������:�	             �i@     �t@      q@     �p@      p@     �i@     �g@     �g@     �d@      b@     �\@      \@     @[@      Y@     �V@     @T@     �Q@     �L@     �N@      O@     �H@     �L@      >@     �C@     �A@      @@     �@@      ?@      4@      9@      4@      0@      5@      *@      3@      (@      (@      .@      &@      @      "@      @      @       @      @      @      @      �?       @       @      @      @      @       @       @       @      �?      @       @       @               @      �?              �?       @      �?              �?      �?       @              �?              �?              �?              �?      �?      �?       @      �?      �?      �?      �?              �?      @              @       @      @      @              @      @              @      @      *@       @      @       @      "@      (@       @      1@      $@       @      0@      .@      6@      .@      ,@      :@      3@      *@      =@     �@@      7@      ;@      >@     �F@      J@      K@      N@     �G@     �N@     @T@      S@     �T@     @X@     @\@     �^@      `@     �`@      b@     @e@      d@     �h@     �j@     �n@     `o@     �s@     �l@        
�
conv5/biases_1*�	   ��<�    ieD>      <@!  �쿵S>)�YZP���<2�����W_>�p
T~�;�u 5�9��z��6�_"s�$1�7'_��+/����<�)�4��evk'���o�kJ%�4�e|�Z#��i
�k���f��p�Łt�=	���R����2!K�R��Z�TA[�����"��`���nx6�X� ��f׽r����tO����G�L��=5%���=�9�e��=����%�=f;H�\Q�=�tO���=��R���>Łt�=	>��f��p>�i
�k>%���>���<�)>�'v�V,>7'_��+/>6NK��2>�so쩾4>u 5�9>p
T~�;>����W_>>�`�}6D>��Ő�;F>�������:�              �?              �?              �?              �?              �?              �?              �?      �?              �?              �?               @              �?              �?              �?               @      @              �?              �?       @              �?              �?      �?              �?        v�=�HW      I�	6�>h*��A	*��

step  A

loss�-GC
�
conv1/weights_1*�	   �b��   � ��?     `�@! �^����)ے��6u@2�	��]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���VlQ.��7Kaa+�I�I�)�(��S�F !�ji6�9���vV�R9��T7�����d�r�x?�x��>h�'��f�ʜ�7
�6�]���1��a˲�I��P=��pz�w�7���h���`�8K�ߝ�>�?�s��>�FF�G ?>h�'�?x?�x�?��d�r?��ڋ?�.�?ji6�9�?�S�F !?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�������:�	              &@     �X@     @g@     @j@     �d@     �b@     �e@      _@     �Z@     �]@     @Y@     �V@      P@     �T@      M@      R@     �I@     �L@     �I@      D@      B@      ;@      A@      =@      5@      3@      7@      6@      ,@      4@      2@      *@      0@      *@      @       @       @      @      @       @       @       @      �?      @      @       @      @      @      @               @      @      �?      @              @      �?              �?      �?              �?              �?              �?      �?      �?              �?              �?              �?              �?              �?      �?              �?               @              �?              �?      �?      �?              �?       @      @       @      @      �?      @      @       @       @               @      @       @      @      @      @      @       @      @      "@      "@      *@      $@      @      *@      (@      1@      4@      .@      "@      8@     �@@      ?@      =@      C@      8@     �D@      M@      H@      O@     @Q@      L@     �Q@     �T@     @W@     �T@     �]@     @\@     �Z@     �`@      b@     �c@      i@     @f@     �R@      @        
�
conv1/biases_1*�	   �-|�    ?q�?      @@! @�C��?)~��i%�V?2����T}�o��5sz�*QH�x�&b՞
�u�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T�k�1^�sO�IcD���L�a�$��{E��T���C����#@�d�\D�X=�uܬ�@8���%�V6����?�ګ�;9��R�������>
�/eq
�>['�?��>K+�E���>x?�x�?��d�r?�T7��?�vV�R9?ji6�9�?�S�F !?��%�V6?uܬ�@8?�qU���I?IcD���L?�lDZrS?<DKc��T?Tw��Nof?P}���h?uWy��r?hyO�s?&b՞
�u?*QH�x?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�������:�              �?              �?              �?              �?      �?              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?               @              �?              �?      �?       @      �?              @      �?              �?        
�
conv2/weights_1*�	   � �    ���?      �@! �:�n�1@)�އr|E@2�I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾�_�T�l׾��>M|KվK+�E��Ͼ['�?�;��~��¾�[�=�k���*��ڽ���n�����豪}0ڰ��5�L�>;9��R�>��~]�[�>��>M|K�>�uE����>�f����>a�Ϭ(�>8K�ߝ�>�h���`�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?�������:�              @     l�@     h�@     ؞@     (�@     8�@     ,�@     ��@     ��@     ��@     h�@     ��@     0�@     ��@     (�@     ��@      �@     �@      ~@     �z@     Pz@     �w@     �u@     �p@     �q@      p@      l@     �j@     �f@     @c@     @e@     @b@      _@      Y@     �T@     �T@      S@     �S@      S@     @P@     �H@      N@      I@     �F@     �H@     �C@      C@      <@      8@      <@      ;@      4@      2@      6@      2@      4@      1@      2@      ,@      (@      (@      @       @      @      @      @      @      @      @      @      @      @      @      �?              @      @              @       @      �?      �?      �?       @       @               @      �?              �?              �?      �?       @              �?              �?              �?      �?              �?              �?              �?               @              �?      �?              �?      �?      @       @              @       @      �?      @       @      �?      @      @      @       @      @      @      @      @      @      @      $@      "@      (@      @      .@      *@      *@      (@      0@      2@      8@      8@      :@      4@      5@      =@      @@     �G@     �J@      I@     �E@     �Q@     �T@     @R@      U@     @U@     @T@     �^@     �Z@     �_@     �a@     �c@      e@     �h@      k@     @j@      m@     �q@     pr@     �s@     Pw@      {@     P|@     �}@     0~@     �@     Ѓ@      �@     x�@     ��@     `�@     ��@     �@     �@     ��@     h�@     ��@     ��@     �@     ��@     ��@      I@      �?        
�	
conv2/biases_1*�		    I�v�    ��}?      P@! ���pA�?) ���JA?2�*QH�x�&b՞
�u�uWy��r�;8�clp�ߤ�(g%k�P}���h��l�P�`�E��{��^��m9�H�[�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I��T���C��!�A�d�\D�X=���%>��:�uܬ�@8���%�V6���82���bȬ�0��vV�R9��T7���x?�x��>h�'��6�]���1��a˲���~��¾�[�=�k��        �-���q=>h�'�?x?�x�?�5�i}1?�T7��?�vV�R9?I�I�)�(?�7Kaa+?��VlQ.?��82?�u�w74?��%�V6?uܬ�@8?�!�A?�T���C?a�$��{E?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?�������:�              �?               @              �?              �?      �?              �?              �?              �?              �?              �?      �?      �?              �?              �?              �?               @              �?              �?              �?              �?      �?              �?      �?              �?      @      �?              �?      �?               @              �?      �?      �?              @      �?       @               @      @      �?      �?       @      @       @      �?      �?      �?      �?      �?        
�
conv3/weights_1*�	   ����    �8�?      �@!�>��B2@)D���iU@2�����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾄiD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿ�u��gr��R%�������
�%W�>���m!#�>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>�iD*L��>E��a�W�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�������:�             h�@     �@     ��@     V�@     N�@     z�@      �@     D�@     ̙@     ��@     4�@     �@     P�@     ��@     0�@     ��@     �@     ��@     @�@     0�@     ��@     �|@     �z@     `y@     �v@     `t@     s@     �o@     �m@     `k@     �h@     �g@     `f@      c@     �a@     @]@     �]@     �[@     �[@     @T@     @R@     �T@      M@     �L@     �K@      F@     �K@     �F@     �A@      ?@      :@      A@      4@      >@      5@      2@      2@      *@      $@      .@      ,@      ,@      *@      $@      @      @      $@      @      @      @       @      @      @      @       @       @       @      @      @       @       @      @               @      �?       @      �?      �?              �?      �?       @      �?      �?              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?      @      �?       @      �?              �?      �?               @       @       @      �?      @      �?      @      �?      �?      @      @      @      @      @       @      @      �?      @      &@      ,@      "@      &@      $@      ,@      ,@      2@      2@      =@      ;@      8@      9@      =@      ;@      B@      G@      F@      C@      N@     @P@     �F@      L@     �S@      V@     @W@     �S@      _@      a@     �`@     `c@     �g@      h@      i@     �p@     �q@     �n@     �r@     �v@     �w@     �y@     pz@     p@     0�@     �@     �@     p�@     ��@     P�@     ��@     �@     t�@     \�@     l�@     ܘ@     t�@     <�@     �@     ��@     ��@     ��@     r�@     ��@     ��@      (@        
�
conv3/biases_1*�	    ��w�   ��4�?      `@! `�r��?)F
��7[K?2�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��S�F !�ji6�9����ڋ��vV�R9��5�i}1���d�r�>�?�s���O�ʗ����u��gr�>�MZ��K�>���%�>�uE����>�f����>��(���>1��a˲?6�]��?f�ʜ�7
?>h�'�?�[^:��"?U�4@@�$?�7Kaa+?��VlQ.?��%�V6?uܬ�@8?��%>��:?d�\D�X=?�T���C?a�$��{E?
����G?�qU���I?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?>	� �?����=��?�������:�              �?      �?               @      �?      �?      @       @      @      @       @      �?       @      @      @      �?       @              �?       @       @              �?      @      @               @      �?      �?       @      �?      @      �?               @              �?              �?               @              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?      @      �?              �?      �?      @       @      @      @      �?      �?      @       @      @      @      @      @      @       @       @      �?              @              �?        
�
conv4/weights_1*�	   �0\��   �؞�?      �@! H�_�[@)�f���ae@2�Ӗ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��T7����5�i}1���d�r�x?�x��>h�'�������6�]���1��a˲�O�ʗ�����Zr[v���ߊ4F��h���`��uE���⾮��%������>
�/eq
�>�iD*L��>E��a�W�>�FF�G ?��[�?1��a˲?6�]��?x?�x�?��d�r?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�������:�             �_@     @�@     ؕ@     �@     �@     ��@     Ќ@     (�@     P�@     0�@     ��@     ��@     ��@     P|@     �{@     Pw@     Pu@     �s@     �q@     p@      n@     �i@     `i@     �f@     `c@     @e@     @c@     �^@     @\@     �W@     �X@     @S@     �Y@      S@     �O@      M@      E@      G@     �D@      E@      >@      8@     �A@      2@      7@      4@      5@      3@      ,@      1@      $@      *@      0@       @       @      (@       @      "@      @      @      @      @      @      �?       @      @      @      @       @      @              �?              @              �?              �?      �?              �?              �?      �?              �?               @              �?              �?              �?              �?               @               @              �?       @      �?              �?       @              @      �?       @      @      �?      @      @       @      @      @      @      @      "@      "@      &@      @      $@      &@      "@      ,@      2@      2@      ,@      (@      .@      =@      4@      ;@      A@     �B@     �B@     �E@      I@      ?@      O@      N@     �Q@     �Q@      T@     @Y@      V@      [@     ``@     �]@     �_@      d@     �c@     �k@     �k@     `h@     �n@     pq@     �s@     �t@     @w@     pv@     �x@     �~@     �@     8�@     ��@     X�@     �@     ��@     ��@     4�@     ��@     �@     `�@     d�@     `e@        
�
conv4/biases_1*�	    n���   �v��?      p@!�
��޽?)IVb���b?2����J�\������=���>	� �����T}�o��5sz�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1�x?�x��>h�'��f�ʜ�7
��������[���FF�G �O�ʗ�����Zr[v���uE���⾮��%ᾙѩ�-߾��~]�[Ӿjqs&\�Ѿ�u`P+d����n��������]������|�~��R%������39W$:�����ӤP�����z!�?��u��6
��K���7����Ő�;F��`�}6D�/�p`B�p��Dp�@�6NK��2�_"s�$1�7'_��+/��'v�V,����<�)�4��evk'�%�����i
�k���f��p�Łt�=	�Z�TA[�����"��f׽r����tO����H����ڽ���X>ؽ        �-���q=�|86	�=��
"
�=f;H�\Q�=�tO���=�mm7&c>y�+pm>RT��+�>���">Z�TA[�>�#���j>�J>2!K�R�>��f��p>�i
�k>%���>��o�kJ%>4��evk'>���<�)>�'v�V,>�so쩾4>�z��6>u 5�9>p
T~�;>����W_>>p��Dp�@>
�}���>X$�z�>�MZ��K�>��|�~�>���]���>�5�L�>;9��R�>�*��ڽ>�[�=�k�>
�/eq
�>;�"�q�>['�?��>K+�E���>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>�uE����>�f����>a�Ϭ(�>8K�ߝ�>O�ʗ��>>�?�s��>��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?�������:�               @      �?      �?      �?              �?      @              @      �?      �?      @      @       @      @      @              �?      �?      @      @      @      �?      �?      �?              @      @      �?              �?       @      @               @              �?              �?              �?       @               @      �?      �?      �?              �?       @      �?              �?              �?              �?      �?              �?               @              �?              �?              �?              �?              �?              �?               @      �?       @      �?      �?              �?       @      �?              �?              �?              �?              6@              �?              �?              �?              �?              �?      �?       @              �?      �?              �?              �?              �?              �?              �?              �?              �?      �?              �?              �?               @      �?      �?              �?              �?              �?              �?              �?               @              �?       @      �?      �?              �?      �?              �?              �?              �?       @      �?              �?              �?       @      �?      @              @      @       @               @      @      �?      @              �?      @      @       @      @      �?      @      @               @      �?              @       @      @      @      �?               @      @      �?       @       @      �?      �?      @       @      �?      �?      �?      �?        
�
conv5/weights_1*�	    ��¿   `�j�?      �@! �<�#�&�)m��3I@2�	�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.�U�4@@�$��[^:��"��T7����5�i}1�6�]���1��a˲�R%������39W$:����ߊ4F��>})�l a�>�5�i}1?�T7��?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�������:�	             �i@     �t@     �p@      q@      p@     @i@     `g@     �g@     �d@     @b@      ]@      \@     �[@     �W@     �V@      U@     �P@      O@     �K@     �O@      I@      L@      >@     �C@     �A@     �A@      <@      @@      7@      :@      1@      1@      5@      *@      5@      $@      *@      .@       @      @       @      $@      @      @      @      @      @      �?      @      �?      @      @       @       @              @      �?      @      @       @              @              �?      �?      �?              �?              �?              �?              �?              �?              �?              �?              �?      �?      �?      �?      �?       @      �?       @      �?      �?              @      �?      �?       @      @      �?      �?       @      �?       @      @      @      "@      @      @      @      @      ,@      "@      3@       @      "@      .@      .@      6@      0@      *@      :@      0@      0@      ?@      A@      5@      9@      <@     �F@     �L@      I@     �N@     �G@      O@     @T@     @S@     �T@     @X@      [@      `@     �_@      a@      b@     `e@     @c@      i@     �j@     `n@      p@     �s@     �l@       @        
�
conv5/biases_1*�	   �`@�   `�H>      <@!  ��y�R>).\W���<2�p��Dp�@�����W_>�p
T~�;��so쩾4�6NK��2��'v�V,����<�)�4��evk'���-�z�!�%������R����2!K�R���J��#���j��EDPq���8�4L���<QGEԬ���M�eӧ�y�訥�f;H�\Q�=�tO���=�J>2!K�R�>Łt�=	>��f��p>�i
�k>%���>��-�z�!>4�e|�Z#>7'_��+/>_"s�$1>6NK��2>�so쩾4>�z��6>p��Dp�@>/�p`B>��Ő�;F>��8"uH>�������:�              �?      �?              �?               @      �?              �?              �?       @      �?              �?      �?              �?              �?              �?              �?      �?       @              �?              �?      �?      �?       @              �?              �?        B���V      	燀	���m*��A
*٭

step   A

losspGC
�
conv1/weights_1*�	   ��\��   ઞ�?     `�@! �\D��)�e���@2��{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9��I��P=�>��Zr[v�>�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?+A�F�&?I�I�)�(?�7Kaa+?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�������:�              @      8@     @Y@      i@      j@     `b@      e@      b@      a@     �[@     �\@     �W@     @V@      T@     �P@     �O@     �O@      K@     �I@      J@      J@     �@@      ?@      :@      =@      4@      ?@      7@      3@      ,@      6@      $@      &@      (@      &@       @      @      @      @       @      "@       @      $@      @      @      @       @      @      @      @      @      @      �?       @       @       @      �?               @              �?      �?              �?              �?              �?      �?              �?               @      �?              �?      @               @       @      �?      �?       @      @              @      �?              �?      @      @      @      @      @      @      @      @       @      &@      $@      ,@      (@      @      *@      3@      2@      (@      3@      9@      9@      <@      ?@      F@      @@     �E@      L@      K@      I@     �L@     �O@     @P@     �W@     @T@      W@      \@      ]@     �Y@     �`@     @b@     @d@     `g@     `e@     �R@      &@        
�
conv1/biases_1*�	   �¥�   @�ܓ?      @@!  쓂B�?)�d_^�7a?2�>	� �����T}�hyO�s�uWy��r��l�P�`�E��{��^��m9�H�[���bB�SY��qU���I�
����G�a�$��{E��T���C��!�A���%>��:�uܬ�@8���%�V6��5�L�����]����['�?��>K+�E���>�ѩ�-�>���%�>f�ʜ�7
?>h�'�?ji6�9�?�S�F !?�[^:��"?���#@?�!�A?E��{��^?�l�P�`?���%��b?P}���h?ߤ�(g%k?&b՞
�u?*QH�x?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�������:�              �?              �?              �?              �?              �?              �?      �?              �?      �?              �?              �?              �?              �?              �?      �?              �?              �?      �?              �?               @               @              �?      @       @      �?      �?              �?        
�
conv2/weights_1*�	    ,���   @R2�?      �@! H�C�(6@)�
�g�E@2�I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7���ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾�4[_>��>
�}���>['�?��>K+�E���>��~]�[�>��>M|K�>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?�������:�              2@     �@     F�@     ��@     X�@     4�@      �@     8�@     ��@     �@     �@     �@     H�@     ��@     (�@      �@     ��@     ��@      ~@      {@     �y@      x@      u@     Pq@     �p@      p@     �j@     �l@     `d@     �d@     @e@     �`@     �`@     @Z@      S@     @U@     �X@     �S@      O@     @R@     �L@     �K@      H@      G@      G@      A@     �D@     �@@      1@      7@      >@      8@      *@      2@      6@      5@      .@      (@      ,@      $@      @      @      @      @      @      @      @       @      @      @       @      @      @       @      �?       @      �?      @      �?      �?              �?              �?              @       @               @              �?              �?               @      �?              �?              �?              �?              �?       @       @       @      �?      �?      �?       @      �?      �?       @      @       @      @      @      @               @       @      @      @      @       @      @      @      $@      "@      &@      @       @      @      (@      $@      2@      1@      5@      5@      6@      :@      3@      9@      6@     �A@      B@     �D@      I@     �J@      H@     �P@     �P@     �Q@      V@     @S@     �T@     �]@     @]@     @`@      b@     �c@      g@     �h@     �j@     �i@      n@     `r@     pr@     Pr@     �v@     �z@     �}@     @}@     �~@     ��@     �@      �@     X�@     x�@     ȍ@     �@     ��@     ��@     ��@     ��@     �@     $�@     �@     ��@     ��@      X@      @        
�

conv2/biases_1*�
	   `��r�   ��K�?      P@! @y��?)Jr���E?2�hyO�s�uWy��r�;8�clp�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�a�$��{E��T���C�d�\D�X=���%>��:���82���bȬ�0���VlQ.��7Kaa+�U�4@@�$��[^:��"�ji6�9���.���5�i}1���d�r�x?�x��>h�'��f�ʜ�7
���ӤP�����z!�?��        �-���q=6�]��?����?>h�'�?x?�x�?��d�r?�5�i}1?��ڋ?�.�?�[^:��"?U�4@@�$?�u�w74?��%�V6?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?�������:�               @      �?              �?      �?              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?      �?              �?              �?              �?              �?              �?              �?              �?              �?              �?               @               @               @              �?      �?              �?              �?               @              �?       @              �?      �?      @      @      �?      �?       @      @       @      �?      �?      �?      �?        
�
conv3/weights_1*�	   ��鯿   `�ٱ?      �@!@���`�6@)D�NoU@2�����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%�
�/eq
Ⱦ����ž�XQ�þ5�"�g��>G&�$�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>��~]�[�>��>M|K�>�ѩ�-�>���%�>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�������:�             h�@     ĩ@     ̦@     R�@     X�@     ��@     �@     ��@     ��@     ��@     (�@     �@     P�@     ��@     �@     p�@     (�@     ��@      �@     ؁@      �@     �}@     �y@     z@     @v@     �s@     �s@      p@      k@      l@      i@     @g@     �e@      e@     @`@     �]@     �]@     �_@     �Z@     �U@     �T@     @T@      I@     �K@      M@     �B@     �I@     �F@     �F@      @@      @@      @@      B@      :@      7@      (@      (@       @      3@      "@      2@      "@      (@      ,@      @      &@      "@      @      @      �?      @      @      @      @      �?              �?       @       @      @       @      �?      �?      @              �?      �?      �?              �?      �?      �?              �?              �?      �?              �?              �?      �?      �?              �?              �?              �?              �?      @      �?       @       @       @      �?       @              @      �?       @      @       @       @               @      @      @      @      @       @       @      @      @      @      @      $@      &@      0@      (@      @      0@      (@      6@      ,@      6@      *@      A@      A@      @@      A@      @@     �G@     �C@     �G@      P@     �K@      L@     �K@     �S@     @V@     �U@     �X@     �Y@     �`@     �`@      c@     �c@      j@     @h@     �q@     �q@     �n@     �r@     0v@     �w@     @z@     0{@     @~@      �@     ��@     (�@     ��@     ��@     ��@     �@     Ԑ@     ��@     �@     ĕ@     ��@     l�@     �@     �@     �@     �@     ��@     V�@     ��@     0�@      :@        
�
conv3/biases_1*�	   �kn|�   �z�?      `@! @���Ӳ?)Nê��;T?2����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�uܬ�@8���%�V6���82���bȬ�0���VlQ.�I�I�)�(�+A�F�&��S�F !�ji6�9���.����ڋ���d�r�x?�x����[���FF�G �;9��R�>���?�ګ>�_�T�l�>�iD*L��>8K�ߝ�>�h���`�>O�ʗ��>>�?�s��>x?�x�?��d�r?�5�i}1?�T7��?��bȬ�0?��82?�u�w74?��%�V6?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?�������:�              �?      �?      �?              �?      �?       @      @      @      @      �?       @      �?      @      @       @              �?      �?       @      �?      �?      @      @      �?              @              �?              �?       @              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?      �?       @              �?       @      �?       @      �?      �?       @       @      @      @      @      �?       @       @       @      @      @      @      @      @      @              �?               @              �?        
�
conv4/weights_1*�	   `
c��   @=��?      �@! ����O @)��Fcde@2�Ӗ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��5�i}1���d�r�x?�x��>h�'��I��P=��pz�w�7��a�Ϭ(���(����uE���⾮��%�
�/eq
Ⱦ����ž��~���>�XQ��>�����>
�/eq
�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�������:�             �_@     0�@     ܕ@     �@     ��@     @�@     ��@     (�@     `�@     0�@     h�@     ��@     ��@     @|@     �{@     �w@     �u@     0s@     �q@     �o@      n@     �i@     `i@     `g@      d@     @d@     �c@     �^@     �[@      X@      Y@      T@     �X@     �R@     �O@      P@      E@     �D@      F@      E@     �@@      7@      =@      2@      7@      <@      0@      3@      (@      2@      *@      &@      0@       @      &@      @      @      "@       @      &@      @      @      @      @      @      @       @      @              @              �?      �?       @      �?               @              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?              @      �?       @       @       @      @      �?       @      @      �?              �?      @      @       @      �?      @      @       @      @      $@      (@      @      @      @      0@      &@      ,@      .@      2@      &@      .@      <@      8@      9@      C@     �A@     �A@     �D@      I@      C@     �M@     �O@      Q@      P@      W@     @W@      W@     �[@      ^@      `@     �`@     `c@     �b@     @k@     �l@     �g@     `o@     �p@     @t@     �t@     �v@     �v@     �x@     p~@      �@     8�@     x�@     p�@     �@     ��@      �@     ,�@     ��@     �@     h�@     D�@      g@        
�
conv4/biases_1*�	   @�؅�   @�:�?      p@!��eW��?)�ȼ��m?2�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1�>h�'��f�ʜ�7
�6�]���1��a˲���[��8K�ߝ�a�Ϭ(���(��澢f�����uE����E��a�Wܾ�iD*L�پ�*��ڽ�G&�$��5�"�g����5�L�����]����
�}�����4[_>����
�%W����ӤP�����8"uH���Ő�;F�/�p`B�p��Dp�@��z��6��so쩾4�6NK��2�_"s�$1�7'_��+/��'v�V,����<�)�4��evk'�4�e|�Z#���-�z�!�%�����tO����f;H�\Q����-��J�'j��p�        �-���q=��-��J�=�K���=�9�e��=����%�=�#���j>�J>2!K�R�>��R���>Łt�=	>��f��p>��o�kJ%>4��evk'>�'v�V,>7'_��+/>_"s�$1>6NK��2>u 5�9>p
T~�;>����W_>>�`�}6D>��Ő�;F>R%�����>�u��gr�>�5�L�>;9��R�>���?�ګ>����>G&�$�>�*��ڽ>�[�=�k�>��~���>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>pz�w�7�>I��P=�>>�?�s��>�FF�G ?1��a˲?6�]��?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�������:�               @      �?      �?              �?              �?       @      �?      �?      @      @       @      @       @      �?       @       @              @      �?       @      @       @       @              �?      �?       @      @       @               @      �?               @      �?               @              �?      �?              �?      �?              �?      �?               @      @      �?              �?              �?      �?              �?      �?              �?              �?              �?      �?              �?              �?              �?              �?              �?               @       @      �?              �?      �?      �?               @       @              �?              �?              6@              �?              �?              @      �?              �?      �?              �?              �?              �?              �?      �?              �?              �?              �?              �?              �?              �?              �?               @              �?              �?      �?              �?              �?              �?              �?              �?               @              �?              �?      �?              �?       @       @      �?       @              �?      �?      �?      �?      �?      @      @      @      �?       @      @      @               @      @       @      @       @      @       @       @      �?       @       @      �?       @      @      @       @      @      �?       @      �?      �?       @       @      �?       @      @      �?      �?      �?              �?        
�
conv5/weights_1*�	   �$�¿   �M��?      �@! h��Q�$�)�Y�É<I@2�	�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�U�4@@�$��[^:��"��.����ڋ��5�i}1���d�r�8K�ߝ�>�h���`�>�FF�G ?��[�?f�ʜ�7
?>h�'�?x?�x�?��d�r?U�4@@�$?+A�F�&?I�I�)�(?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�������:�	             `i@     �t@     �p@      q@     p@     �i@     �g@      g@     �d@     �a@     @]@     @\@     �[@     �W@     �V@     �U@     �O@     @P@     �K@      O@      J@      K@      ?@      D@      A@     �A@      <@      ;@      :@      >@      .@      4@      .@      2@      3@      (@      "@      2@      @       @      @      "@      "@      @      @      @      @      @      @      �?      @      @      @      @       @       @      �?      �?      @       @               @      �?               @              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?      �?               @      �?              �?              @              �?      @              �?       @      @              @      �?      @       @      @       @      "@      @      @       @      @      (@      $@      0@      &@      &@      *@      *@      7@      &@      0@      ;@      4@      .@      ?@      A@      7@      8@      7@      G@      K@     �K@     �M@      I@      L@      V@     �R@     �T@     �Y@     �Y@      `@      `@     `a@     �a@     `e@     @c@     �h@      k@     @n@     0p@     �s@     �l@      @        
�
conv5/biases_1*�	    �E�   �M�K>      <@!   ��P>)�ܩC��<2���Ő�;F��`�}6D�p��Dp�@�����W_>�p
T~�;�u 5�9��z��6��so쩾4�_"s�$1�7'_��+/�4��evk'���o�kJ%�4�e|�Z#���f��p�Łt�=	���R����2!K�R���J�RT��+��y�+pm���1���=��]���H�����=PæҭU�=RT��+�>���">2!K�R�>��R���>Łt�=	>��f��p>�i
�k>%���>��-�z�!>4�e|�Z#>6NK��2>�so쩾4>�z��6>u 5�9>p
T~�;>/�p`B>�`�}6D>6��>?�J>������M>�������:�              �?              �?              �?              �?              �?              �?      �?               @              �?      �?              �?              �?              �?              �?              �?      �?      �?              �?      �?      �?               @               @      �?              �?              �?        ���XV      �q�	��hs*��A*ɬ

step  0A

lossfKFC
�
conv1/weights_1*�	   �~��   �ү�?     `�@!  �Wo�)_D�hPg@2�	� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !��.����ڋ��T7����5�i}1�f�ʜ�7
������6�]���O�ʗ�����Zr[v��E��a�Wܾ�iD*L�پ�T7��?�vV�R9?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?�������:�	               @      0@      <@     �Z@     `g@     �i@     �a@     �d@     `c@     �`@     �Y@      ]@     �X@      T@     @R@     @S@     �O@      P@      I@     �G@      K@     �C@      F@     �@@      <@      :@      8@      8@      4@      :@      2@      (@      &@      .@      "@      (@      @      *@      �?       @      @      @      "@      "@      @      @      @      @      �?      @      @      �?      @      @      @      �?      @              �?      �?      �?       @      �?      �?       @      �?      �?               @              �?              �?      �?              �?              �?              �?              �?      �?      �?      �?              �?              @       @       @              �?       @              �?      @              �?       @      �?      @      �?      @      @      @      @      &@      @      $@      @      0@       @       @       @      ,@      2@      0@      4@      7@      9@      4@      ;@      <@     �B@     �@@      K@     �G@     �G@     �L@      N@      M@     @U@     �S@     @T@     �U@     @]@     �\@      Z@     �`@      c@      c@     �f@     �d@     @V@      ;@       @        
�
conv1/biases_1*�	   `ʔ��   @p��?      @@! �Ҥ�$�?)�U�j�q?2�����=���>	� ��5Ucv0ed����%��b���bB�SY�ܗ�SsW�
����G�a�$��{E�x?�x��>h�'��.��fc���X$�z���u`P+d�>0�6�/n�>�h���`�>�ߊ4F��>��d�r?�5�i}1?I�I�)�(?�7Kaa+?��VlQ.?
����G?�qU���I?IcD���L?k�1^�sO?���%��b?5Ucv0ed?�N�W�m?;8�clp?hyO�s?&b՞
�u?o��5sz?���T}?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�������:�               @              �?              �?              �?              �?              �?              �?              �?              �?              �?      �?               @              �?              �?              �?              �?              �?              �?      �?      �?      �?      �?              �?      �?       @       @      �?              �?        
�
conv2/weights_1*�	   �gˮ�   @���?      �@! �bk{=@)�z®�E@2�����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=���ߊ4F��h���`���(��澢f�����uE����E��a�Wܾ�iD*L�پ��>M|Kվ��~]�[ӾK+�E��Ͼ['�?�;;�"�qʾ�[�=�k���*��ڽ����?�ګ�;9��R����~]�[�>��>M|K�>�_�T�l�>�iD*L��>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?�������:�               @      =@     h�@     �@     �@     �@     �@      �@     <�@     <�@     �@     ��@     ��@     Ȋ@     @�@     X�@      �@     �@     �@     P~@     {@     @z@     �w@     �t@      r@     �p@     p@     `k@      k@     `h@     @b@     �d@     `a@     �[@     �^@     �U@     @U@     �V@      P@     @R@     �R@     �L@      K@     �F@     �F@     �D@      B@     �C@      =@      8@      2@     �@@      ;@      .@      9@      0@      4@      2@      $@      (@      &@      &@      @      $@      $@      @      @      @       @      @      @      @      @      @      �?      �?      �?       @      �?      �?       @      @      �?      �?              �?              �?      @               @               @              �?      �?              �?              �?               @      �?      �?               @      �?              �?      �?               @      �?      @       @       @      @       @      �?      @      @      @      @       @      @       @       @      @      @      @      @      @      $@      @      @      "@      (@      1@      3@      2@      1@      *@      4@      1@      1@      5@      B@      <@     �A@      F@      G@     �M@      I@     �O@     @S@      T@     �P@     �S@      W@     �\@     ``@     �`@     �`@     �a@     �h@      i@     �j@     @j@     �m@     Pq@     �s@     s@     �u@     `y@     �}@     �}@      ~@     `�@     �@     ��@     ��@     ��@     ��@     ,�@     p�@     ̓@     ��@     ȗ@     �@     L�@     �@     b�@     ��@     @b@      .@        
�	
conv2/biases_1*�		   �mww�    ���?      P@! �X�?);|y5S?2�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�P}���h�Tw��Nof���bB�SY�ܗ�SsW��lDZrS�nK���LQ�
����G�a�$��{E��T���C��u�w74���82���bȬ�0��[^:��"��S�F !�f�ʜ�7
������6�]���>�?�s���O�ʗ������m!#���
�%W���.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?��VlQ.?��bȬ�0?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�T���C?a�$��{E?
����G?�qU���I?IcD���L?nK���LQ?�lDZrS?<DKc��T?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�������:�              �?      �?              �?      �?               @              �?              �?              �?      �?              �?       @              �?              �?      �?              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?       @              �?              @      �?              @       @      �?      �?              �?      @              �?       @      @      @       @      @      �?      �?              �?      �?        
�
conv3/weights_1*�	    TE��    y��?      �@!`/"�^|;@)Ơd0wU@2�����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F����(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ����>豪}0ڰ>5�"�g��>G&�$�>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>��~]�[�>��>M|K�>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?�������:�             ��@     ~�@     ̦@     $�@     ��@     h�@     &�@     T�@     Й@     ��@     ,�@     <�@     \�@     А@     Ѝ@     ��@     X�@     ؆@     �@     ؁@     ؀@      @     @z@     @y@     �v@     �s@     �r@      p@      m@     �i@     �i@     �f@     `d@     @c@      a@     �a@      _@     �^@     �Z@     @U@     @T@      T@      J@      M@      K@      F@      H@     �I@     �@@      E@      ;@      ?@      =@      :@      3@      7@      (@      ,@      0@      *@      *@      1@      @      @      @       @      @      @      @      @      @      "@      �?      @      @       @       @      @      @       @      @      @               @      �?      �?       @              @       @      �?      �?              �?      �?      �?              �?              �?               @      �?              �?              �?               @              �?      �?      �?               @      �?               @      �?      @      �?       @      @      �?               @      @      @      @      @      @      @      (@      �?       @      "@      (@       @      "@      &@      ,@      ,@      2@      7@      (@      6@      8@      =@      ;@      A@      C@      =@     �G@      C@     �E@      N@      S@     �F@      L@      S@     @U@     �V@     @W@     �\@     @`@     �b@     �b@     �c@     `g@      j@     �p@     `q@     �m@     @s@     �v@     �w@     �y@     �z@     �~@      �@     ؃@     X�@     ��@     ��@     8�@     P�@     p�@     ��@     �@     \�@     Ԙ@     (�@     8�@     4�@     �@     �@     �@     �@     ��@     ��@      H@      @        
�
conv3/biases_1*�	   �lC��   �u��?      `@! ��<�r�?)T����\?2�����=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:���%�V6��u�w74���82���bȬ�0��7Kaa+�I�I�)�(��[^:��"��S�F !��T7����5�i}1�6�]���1��a˲���[���FF�G �>�?�s����u`P+d�>0�6�/n�>1��a˲?6�]��?����?f�ʜ�7
?�vV�R9?��ڋ?��82?�u�w74?d�\D�X=?���#@?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?���J�\�?-Ա�L�?eiS�m�?�������:�              �?      �?              �?              �?      @      �?       @       @              @      @      @      �?      �?               @              �?              �?       @      @      @      �?      �?      �?              �?       @               @              �?               @              �?              �?              �?              �?      �?              �?              �?              �?              �?              �?              �?              �?      �?      �?      @      �?               @       @       @              @      @       @      @      �?      @       @      @       @      @       @      �?      @      @      �?               @      �?        
�
conv4/weights_1*�	   `�|��   ��?�?      �@! �~���#@)�஦�ge@2�Ӗ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�>h�'��f�ʜ�7
������6�]����ߊ4F��h���`��uE���⾮��%������>
�/eq
�>�FF�G ?��[�?1��a˲?6�]��?f�ʜ�7
?>h�'�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�������:�              `@     ,�@     ��@     ,�@     �@     �@     �@     8�@     X�@     8�@     P�@     ��@     H�@     �|@     �{@     0w@     Pv@     0r@     `r@     �o@      n@     �i@     �h@      h@      d@     �c@     @b@     �_@     @^@     �X@     �X@     �Q@      Y@     @T@     @P@      N@     �E@      D@      B@      F@      D@      9@      >@      4@      5@      2@      7@      2@      2@      .@      *@      "@      *@      2@       @      @      @      "@      �?      "@      @      @      �?       @      @      �?       @      @      �?       @      @      @      @       @      �?       @              �?              �?              �?              �?              �?               @              �?              �?      �?      �?              �?              �?      �?      �?              �?              �?      �?      �?       @      @      �?      @      @      @      @      @       @       @      @      @      $@       @      ,@      @      @      @      ,@      ,@      .@      2@      &@      (@      .@      :@      6@      :@     �C@     �A@     �A@      D@     �J@      D@     �O@      M@     �O@     �P@     �T@     @Z@     �V@     �Z@     �_@     ``@     �_@     �b@     `c@     �k@      l@     �g@     �n@     �q@     �s@     �t@     �v@     �v@     y@     �~@     �@     `�@     @�@     0�@     �@     �@     8�@     �@     Đ@     ��@     ��@      �@     @i@        
�
conv4/biases_1*�	    �0��   ��s�?      p@!�����?)�$��Ryu?2��7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=������T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:���%�V6��u�w74���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ�f�ʜ�7
������6�]�����[���FF�G ��ߊ4F��h���`�8K�ߝ���(��澢f�����uE���⾮��%���~]�[Ӿjqs&\�Ѿ
�/eq
Ⱦ����ž���?�ګ�;9��R��39W$:���.��fc���
�}�����4[_>���������M�6��>?�J�/�p`B�p��Dp�@�p
T~�;�u 5�9��z��6��so쩾4�6NK��2�_"s�$1�7'_��+/��'v�V,����<�)�4��evk'���o�kJ%�4�e|�Z#���-�z�!����"�RT��+���tO����f;H�\Q��'j��p���1���        �-���q=�tO���=�f׽r��=�`��>�mm7&c>y�+pm>RT��+�>�#���j>�J>��R���>Łt�=	>�i
�k>%���>4��evk'>���<�)>7'_��+/>_"s�$1>6NK��2>�so쩾4>����W_>>p��Dp�@>��8"uH>6��>?�J>��|�~�>���]���>;9��R�>���?�ګ>����>豪}0ڰ>��~���>�XQ��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>I��P=�>��Zr[v�>�FF�G ?��[�?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?�������:�              �?       @              �?      �?              @               @       @      @      @      �?       @       @      @       @       @              @      �?      @       @      �?      �?       @       @       @       @      �?               @      �?               @      @               @              �?      �?      �?      �?      �?      �?              �?              �?              �?      �?              �?              �?      �?              �?              �?              �?              �?              �?              �?              �?              �?              �?               @      �?               @      �?               @      �?      �?              �?      �?              �?              �?              �?              5@              �?              �?              �?              �?              �?              �?               @              �?              �?               @              �?              �?              �?              �?               @              �?               @              �?               @              �?              �?       @              �?              �?              �?              �?      �?              �?              �?              �?              �?      �?       @      �?       @               @      �?              �?      @       @      @      @       @      @      �?              �?      @      @      @      @       @      �?       @      @      �?              @      �?      @      �?      @      @      @      @       @              �?              @      @      @              �?       @      �?              �?        
�
conv5/weights_1*�	   ���¿    %��?      �@! ��7�!�)iw���HI@2�	�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�U�4@@�$��[^:��"�����?f�ʜ�7
?>h�'�?U�4@@�$?+A�F�&?��VlQ.?��bȬ�0?��82?�u�w74?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�������:�	              i@     �t@     @p@     �p@     Pp@     �i@     �g@     �f@     �c@     `b@      ^@     @\@      [@     �X@     �V@      V@     �N@     @P@     �J@      M@     �L@     �J@      ?@      F@      >@     �@@      =@      :@      =@      :@      3@      4@      1@      3@      4@       @      (@      0@      @      @      @       @       @      @       @       @      @       @      @      @       @      @      @      @               @      �?       @      �?      @      �?      @               @              �?              @      @              �?              �?      �?              �?              �?      �?       @              @              @       @      @       @      �?      �?      @      @       @      �?      @              @      @       @       @      @      @      @      (@      &@      (@      (@      $@      *@      .@      4@      ,@      ,@      ;@      4@      .@     �A@      B@      6@      7@      8@     �E@     �K@     �H@     @P@     �G@     �M@     �T@     �S@     �S@     �Y@     �Z@     �^@      `@     �a@     �a@     @e@      c@     �h@      k@      n@     0p@     �s@     �m@      "@        
�
conv5/biases_1*�	   `1(H�   ���O>      <@!  ��B�P>)�{�8���<2���8"uH���Ő�;F�/�p`B�p��Dp�@�p
T~�;�u 5�9�6NK��2�_"s�$1����<�)�4��evk'���o�kJ%�%�����i
�k���f��p�Łt�=	���R����2!K�R�����"�RT��+���f׽r����tO����'j��p�=��-��J�=�tO���=�f׽r��=Z�TA[�>�#���j>�J>2!K�R�>%���>��-�z�!>4�e|�Z#>��o�kJ%>6NK��2>�so쩾4>�z��6>u 5�9>p
T~�;>����W_>>��Ő�;F>��8"uH>������M>28���FP>�������:�              �?               @              �?              �?              �?      �?              �?              �?      �?      �?              �?              �?              �?              �?              �?              �?              �?              @              �?      �?      �?               @              �?              �?        q溇xW      �7�Z	���x*��A*�

step  @A

lossA�EC
�
conv1/weights_1*�	   `R+��   ����?     `�@!  E���)�*QT@2�	8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.�I�I�)�(�+A�F�&�U�4@@�$�x?�x��>h�'��8K�ߝ�a�Ϭ(龄iD*L�پ�_�T�l׾f�ʜ�7
?>h�'�?��d�r?�5�i}1?�T7��?�vV�R9?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?�������:�	              �?       @      7@     �H@     @Z@     �h@     @i@     �a@     �c@     @c@     @`@     �W@     �[@     �Y@     �U@      T@     �R@     �L@     �Q@     �M@      G@     �D@     �F@      =@      @@      8@      @@      8@      4@      :@      4@      1@      0@      *@      &@      (@      (@      @      @      @      "@      @      @       @       @      @       @       @      @       @      @      @              �?              �?      @       @      �?              �?              �?       @              @              �?              �?              �?              �?              �?               @       @               @              @       @      �?      �?      @               @      @       @      @       @      @       @              @      @      @      �?      $@       @      @      "@      @       @      @      0@      ,@      "@      6@      3@      1@      .@      5@      5@      @@      :@      ;@     �A@     �A@      :@      G@      J@     �K@     @P@     �N@      S@      T@     �W@      W@     �Y@     @\@      X@     �^@     `d@     �c@     �e@     �c@      W@     �A@      ,@        
�
conv1/biases_1*�	   ��D��   ��f�?      @@! j���
�?)p����?2����J�\������=���>	� ��P}���h�Tw��Nof���ڋ��vV�R9��T7��������6�]���BvŐ�r>�H5�8�t>���]���>�5�L�>I��P=�>��Zr[v�>��ڋ?�.�?��82?�u�w74?��%�V6?uܬ�@8?nK���LQ?�lDZrS?ܗ�SsW?��bB�SY?�N�W�m?;8�clp?uWy��r?*QH�x?o��5sz?���T}?>	� �?����=��?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?�v��ab�?�/��?�uS��a�?�������:�              �?      �?              �?              �?      �?              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?      �?              �?      �?              �?              �?      �?              �?      �?               @      @               @       @        
�
conv2/weights_1*�	   ��,��   �'�?      �@!��K	�jB@)��e�E@2�����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�a�Ϭ(���(��澢f�����uE����
�/eq
Ⱦ����ž0�6�/n���u`P+d����|�~�>���]���>['�?��>K+�E���>��>M|K�>�_�T�l�>�iD*L��>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�������:�              "@      E@     X�@     ��@     ܞ@     �@     \�@     �@     �@     l�@     ��@     ��@     @�@     8�@     �@     ��@     @�@     x�@     @     P}@     �{@     px@     �x@     �t@     Pr@     �p@     Pp@      n@     �h@     �f@      d@     �d@      b@     �Z@      W@     �Y@     �W@      V@     �P@      P@     �R@     �P@     �G@      K@     �A@     �E@     �C@      C@      B@      6@      8@      9@      7@      7@      9@      *@      2@      0@      ,@      .@      @      "@      @      "@      @      @       @      @      @      @      @      @       @       @      @       @       @      �?      @       @       @              @      �?               @              @               @               @              �?              �?              �?              �?               @      �?              �?              �?      �?      �?              �?      @      @       @              �?      @      @      @       @      @      @       @      @      @       @      @      @      &@      @      �?      @      @      &@      @      .@      @       @      1@      4@      4@      5@      4@      5@      ;@      7@      >@     �A@      D@      G@     �P@     �K@     �M@      T@     �P@      S@      W@     �V@     @]@     @_@      `@     `a@     �b@     @h@     �g@      k@     �i@      n@      q@     �r@     �r@      v@     �y@     `}@     @@     0~@     ��@     (�@     Ї@     ��@     (�@     ��@     �@     X�@     �@     ��@     ��@     0�@     0�@     �@     L�@     ��@     �j@      <@       @        
�	
conv2/biases_1*�		    juv�   @�ފ?      P@! ���#��?)���
>KZ?2�*QH�x�&b՞
�u�uWy��r�;8�clp��N�W�m�ߤ�(g%k��m9�H�[���bB�SY�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��!�A����#@���%>��:�uܬ�@8��7Kaa+�I�I�)�(��T7����5�i}1�6�]���1��a˲�=�.^ol�ڿ�ɓ�i�1��a˲?6�]��?x?�x�?��d�r?�5�i}1?�T7��?�[^:��"?U�4@@�$?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�������:�              �?              �?      �?      @              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?      �?      �?              �?      �?      �?      �?      �?              �?      �?      �?      �?              �?              �?      @              �?      �?      @               @      @      �?      @       @      @      @      �?              �?      �?        
�
conv3/weights_1*�	   �=���   �V��?      �@! #?v=NA@)�k����U@2���]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾jqs&\�ѾK+�E��Ͼ�[�=�k���*��ڽ�R%������39W$:���豪}0ڰ>��n����>�����>
�/eq
�>['�?��>K+�E���>��~]�[�>��>M|K�>�uE����>�f����>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?�������:�              @     X�@     8�@     ��@     �@     h�@     ��@     "�@      �@     ��@     Ė@     |�@     l�@     ��@     t�@     (�@     (�@     p�@     X�@      �@     Ё@     ��@     p}@     �{@     z@     �u@     `t@     0r@     �o@     `l@     `k@     �h@     �f@     �c@     �c@      b@     �]@     �^@     �^@     �Z@     @Z@     �S@     �Q@     �J@     �K@      K@      D@      L@     �I@     �C@     �E@      >@      @@      3@      :@      8@      &@      2@      *@      5@      4@      1@      0@      "@      0@      @      @      @       @      @      @      @      @       @      @      @      @      �?      @              @      �?       @      @       @      @              �?      �?      �?      �?      �?              @               @      �?      �?              �?              �?              �?              �?              �?              �?              @               @              �?      �?      �?       @      �?       @      �?       @       @      @       @      �?              @      @       @      �?      @      @      @      @      @      @      "@      $@      ,@      $@      &@      @      "@      *@      *@      1@      0@      2@      <@      ;@      <@      <@      B@     �B@      D@     �E@      F@     �K@     �P@      P@     �P@     @T@     �T@     �V@     �Y@     �Y@     �[@     `b@     @b@     `e@     �f@     @j@     �p@     �q@      o@     �r@     @v@     �w@      z@     pz@     �~@     ��@     ��@     ��@     ��@     h�@     ��@     ��@     ��@     В@     L�@     d�@     ��@     ��@     �@     6�@     *�@     ƣ@     ��@     ��@     j�@     ��@     �S@      ,@      �?        
�
conv3/biases_1*�	   ���   ��W�?      `@! �﻽\�?)���n7ic?2�-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b�E��{��^��m9�H�[�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO��qU���I�
����G�a�$��{E��T���C��!�A����#@���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0�I�I�)�(�+A�F�&��T7����5�i}1���d�r�x?�x�������6�]�����[���FF�G �})�l a��ߊ4F��5�"�g��>G&�$�>6�]��?����?��d�r?�5�i}1?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?eiS�m�?#�+(�ŉ?�7c_XY�?�������:�              �?              �?              �?      �?      @       @      �?      @       @       @      @      @      @      �?              �?              @       @      �?      @               @       @      �?              �?              �?      �?              �?      �?              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?      �?       @               @               @              �?              �?      �?              @      �?      @      @      @       @       @      @      @       @      �?      @      @      @      �?      @      @      @               @      �?        
�
conv4/weights_1*�	   ����   ����?      �@! ����0)@)
�3I�le@2�Ӗ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>�?�s���O�ʗ����uE���⾮��%���~]�[Ӿjqs&\�Ѿ�����>
�/eq
�>E��a�W�>�ѩ�-�>})�l a�>pz�w�7�>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?�������:�             ``@      �@     ��@     �@     ,�@     �@     �@     ��@     ��@     �@     X�@     X�@     P�@     �|@     �z@      x@     �v@     �q@      r@     `o@     @n@     �i@     `i@     `h@      c@     �d@     @c@      ^@      ^@     @Y@     �V@     �T@     �V@      U@     @R@     �H@      F@      F@     �@@     �D@      A@     �A@      <@      6@      7@      4@      1@      1@      4@      ,@      (@      &@      &@      &@      @      @      $@       @      @       @      @      @      @      @      @      �?      @      @               @      �?      @       @       @       @              �?      �?      �?              �?              �?              �?              �?              �?              �?              �?              �?               @              �?              �?      @      �?      @       @      �?              @              @      @      �?      �?      �?      @      @      @      @      @      @      @      �?       @      (@       @      $@      "@      ,@      (@      .@      ,@      *@      &@      (@      ?@      8@      8@     �A@     �D@     �@@     �E@     �F@      H@      L@     �K@     �P@     �P@      U@     �W@     @W@     �[@      `@      a@     �_@      c@     �b@      k@     `l@     `h@     `n@     �q@     0s@     `u@     �v@     �v@      y@      ~@     �@     p�@     H�@     8�@      �@     �@     @�@      �@     ̐@     ��@     x�@     <�@     �j@      @        
�
conv4/biases_1*�	   `U%��   @���?      p@!�xڂ��?)T.ǂ�~?2��#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������O�ʗ�����Zr[v��I��P=��pz�w�7��a�Ϭ(���(��澢f����K+�E��Ͼ['�?�;��n�����豪}0ڰ�������u��gr��R%������.��fc���X$�z��28���FP�������M�/�p`B�p��Dp�@�����W_>�p
T~�;�u 5�9��z��6��so쩾4�6NK��2�_"s�$1�7'_��+/��'v�V,����<�)�4��evk'���o�kJ%�4�e|�Z#�2!K�R���J�RT��+��y�+pm�����%���9�e���        �-���q=�Į#��=���6�=����%�=f;H�\Q�=y�+pm>RT��+�>�J>2!K�R�>�i
�k>%���>��-�z�!>4�e|�Z#>7'_��+/>_"s�$1>6NK��2>�so쩾4>�z��6>u 5�9>p
T~�;>p��Dp�@>/�p`B>�`�}6D>6��>?�J>������M>u��6
�>T�L<�>�5�L�>;9��R�>���?�ګ>����>豪}0ڰ>��n����>5�"�g��>G&�$�>�XQ��>�����>jqs&\��>��~]�[�>�iD*L��>E��a�W�>�ѩ�-�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>��[�?1��a˲?6�]��?����?x?�x�?��d�r?�vV�R9?��ڋ?�S�F !?�[^:��"?U�4@@�$?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�������:�              �?       @              �?      �?              �?       @      @      @      �?      @       @              @      @       @      �?      �?      �?      @      �?      @      @       @      �?      �?               @       @              @               @       @               @      �?               @              @              �?      �?              �?      �?      �?      �?              �?              �?              �?               @              �?              �?              �?      �?               @              �?      �?              �?              �?              �?              �?               @      �?       @      �?      �?              �?      �?              �?              �?              �?              �?              �?              4@              �?              �?              �?              �?              �?              �?              �?      �?              �?              �?              �?      �?              �?              �?              �?              �?              �?              �?              �?              �?              �?       @              �?              �?      �?              �?              �?              �?              �?              �?      �?               @       @      �?       @              �?       @       @              @      �?       @      @       @      @      @      �?       @               @       @      @      @      @      @              �?      @               @       @       @      @      @      @       @      @      @       @       @      �?      �?      �?       @      @       @              �?       @      �?              �?        
�
conv5/weights_1*�	   �k�¿    ǥ�?      �@! ��r�5�)��=��\I@2�	�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���VlQ.��7Kaa+�U�4@@�$��[^:��"��S�F !�ji6�9����d�r?�5�i}1?�T7��?�vV�R9?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�������:�	             @i@     `t@     �p@     �p@     `o@     `j@     �g@     @g@     `c@      c@     �Z@     �^@     �Z@     �X@     @W@      U@     �O@     �O@     �I@     �O@      N@     �G@      A@      E@      ;@      B@      ;@      ;@      8@      <@      .@      3@      ;@      2@      3@      ,@      *@      &@      @      @      @      @       @      @      @      @      @      @      @      �?       @      �?      @      @      @      @      @      �?      �?      @              @               @              �?              �?              �?              �?               @              �?              �?      �?      �?              �?       @       @       @       @       @       @               @      @      @      �?      @      @       @      @      @      @       @      "@      "@      @      "@       @      "@      *@      &@      *@      *@      &@      5@      "@      2@      9@      8@      5@      >@      A@      :@      5@      8@     �F@      L@      E@     �L@      M@     �K@     �T@      T@      U@     �Y@     @Z@     @_@     @`@     �a@     �a@     `e@     �a@     @i@     `k@     @n@     0p@     �s@     �m@      .@      �?        
�
conv5/biases_1*�	   ���I�    HBQ>      <@!   4G�F>)��=s�<2�6��>?�J���8"uH��`�}6D�/�p`B�����W_>�p
T~�;�6NK��2�_"s�$1�7'_��+/��'v�V,�4��evk'���o�kJ%���-�z�!�%������f��p�Łt�=	�y�+pm��mm7&c��f׽r����tO�����Qu�R"�PæҭUݽ�8ŜU|=%�f*=��
"
�=���X>�==��]���=��1���=��-�z�!>4�e|�Z#>��o�kJ%>4��evk'>7'_��+/>_"s�$1>6NK��2>�so쩾4>�z��6>u 5�9>����W_>>p��Dp�@>/�p`B>��8"uH>6��>?�J>28���FP>�
L�v�Q>�������:�               @              �?              �?              �?              �?              �?              @              �?              �?              �?              �?              �?              �?              �?              �?      �?      �?              �?               @              �?              �?      �?              �?              �?        W%�*�V      �ž2	��x~*��A*��

step  PA

loss�BEC
�
conv1/weights_1*�	   ��Z��   @i��?     `�@! @gΑ] @).F$#Xh@2�8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8��u�w74���82��7Kaa+�I�I�)�(������6�]�������?f�ʜ�7
?ji6�9�?�S�F !?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?�������:�               @      $@      6@     �E@      S@     @d@     �f@     �b@     �a@     �b@     �`@      Q@     �[@     �\@     @W@     �P@     �R@     �P@      R@      O@      I@      I@      B@      ?@      6@      9@      >@      2@      ?@      9@      6@      2@      *@      (@      "@      *@      (@      $@      &@      @      @      "@       @      @      @       @      @      @      @      �?       @      @       @       @       @      @      �?      @               @              �?              �?              �?               @              �?       @      �?              @              �?      �?      �?      @              �?       @      �?      �?      @      @      @      @      @      @      @      @      @      @       @      @      ,@      "@      "@      $@       @      2@      6@      ,@      3@      &@      8@      ;@      8@      ;@     �D@      A@      D@      E@     �I@      I@      I@     �T@     @S@     �R@     �U@     �V@     �X@     �[@     @Y@     �]@     �d@      b@     �a@     �f@     �^@     �X@      C@      3@      2@      @       @        
�
conv1/biases_1*�	    �ȣ�   �Љ�?      @@! &�I/�?)`�DN��?2�`��a�8���uS��a��-Ա�L�����J�\���N�W�m�ߤ�(g%k��m9�H�[���bB�SY����m!#���
�%W���u��gr�>�MZ��K�>>�?�s��>�FF�G ?+A�F�&?I�I�)�(?��%>��:?d�\D�X=?�!�A?�T���C?a�$��{E?
����G?�m9�H�[?E��{��^?;8�clp?uWy��r?hyO�s?&b՞
�u?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�������:�              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?      �?      @      �?              �?      �?              �?      @      �?              �?      �?      �?       @        
�
conv2/weights_1*�	   `����    5��?      �@! �)�N@)j>֓��E@2�����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7���ߊ4F��h���`�8K�ߝ���(��澢f�����uE���⾮��%ᾙѩ�-߾��>M|Kվ��~]�[Ӿjqs&\�Ѿ['�?�;;�"�qʾ['�?��>K+�E���>jqs&\��>��>M|K�>�_�T�l�>�iD*L��>���%�>�uE����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?�������:�              @      G@     ��@     ğ@     ��@     l�@     \�@     ��@     �@      �@     8�@     �@     ��@     8�@     �@     �@     ��@     ؀@     X�@     �|@     �z@     Px@     pw@     �t@     �q@     �q@     �p@      k@     �h@     `e@      f@     �c@     @b@     �]@     @]@      V@     �S@     �V@     �P@     �M@     �Q@      Q@      Q@     �F@      C@     �E@     �H@     �A@      B@      :@      5@      7@      5@      6@      3@      0@      4@      3@      0@      *@      "@      @      @      @      "@      @      @      @      @      �?      @       @      �?      @       @      �?       @       @      �?      @      @               @       @      �?              �?      �?              �?       @              �?              �?      �?              �?              �?       @              �?      �?              �?               @      �?      �?              @       @      �?      �?      �?      �?      �?       @      @       @       @      @      @       @      "@      @      @      @       @      @      @       @       @       @       @      $@      @      &@      (@      *@      *@      1@      ,@      ;@      3@      7@      4@      B@      A@     �B@     �D@     �C@      R@     �D@     @P@      R@     @R@     @S@     �T@     @V@      ]@     �^@      b@     �`@     �c@     `e@      i@      k@     �k@      l@     r@     0r@     Ps@      v@     0y@     �{@     0~@     �~@     Ѐ@     ��@     ��@     Ї@     (�@     @�@     8�@     8�@     ��@     Ĕ@     ��@     ԙ@     ��@     T�@     ~�@     ��@     �t@     �[@      *@      @        
�

conv2/biases_1*�
	   `��   ����?      P@!  ҁ���?)C�^i�u?2�eiS�m��-Ա�L��o��5sz�*QH�x�hyO�s�uWy��r�ߤ�(g%k�P}���h�5Ucv0ed����%��b�E��{��^��m9�H�[���bB�SY�ܗ�SsW��lDZrS�nK���LQ�IcD���L��qU���I��!�A����#@�d�\D�X=���%>��:�uܬ�@8��u�w74���82���VlQ.��7Kaa+�I�I�)�(�+A�F�&��[^:��"��S�F !��5�i}1���d�r�x?�x��>h�'���[^:��"?U�4@@�$?��82?�u�w74?��%�V6?uܬ�@8?���#@?�!�A?�T���C?a�$��{E?�qU���I?IcD���L?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?�������:�              �?              �?               @              �?              �?              �?              �?              �?              �?              �?              �?      �?              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?      �?               @              @              �?      �?              @               @      �?              @      �?      �?      �?      @              �?      @       @       @      �?      @       @      �?        
�
conv3/weights_1*�	   ��:��    =�?      �@! f};�G@)�BČܖU@2���]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%�E��a�Wܾ�iD*L�پ��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ
�/eq
Ⱦ����ž�XQ�þ��~��¾ہkVl�p>BvŐ�r>��n����>�u`P+d�>�����>
�/eq
�>['�?��>K+�E���>��~]�[�>��>M|K�>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?�������:�              @      �@     h�@     Ȧ@     �@     $�@     ��@     �@     L�@     t�@     �@     h�@     |�@     ��@     <�@     ��@     ��@     �@     ��@     �@     ؂@     0�@     �}@     �z@     �x@      x@     Pu@     �r@     �l@     �m@      l@     `i@     `f@     �c@      c@      b@     @]@     @[@     �[@     �]@     �V@     �Q@     @T@     �N@     �O@     �I@     �C@      E@      M@      A@     �@@      9@      ?@      :@      <@      5@      4@      6@      ,@      "@      $@      *@      @      $@      &@      $@      @      @      "@      $@      $@      @      @      @      �?       @       @       @      @       @              �?      �?              @               @      �?      �?      �?      �?              �?      �?              �?               @              �?              �?              �?              �?              �?               @              �?              �?              �?              �?      �?              �?              �?      �?       @       @       @      �?       @      �?      @       @       @       @      @      @      @      @      @      @      @      @      "@      &@      (@      ,@      *@      0@      "@      0@      *@      (@      3@      ,@      ;@      =@      :@      >@      =@      =@     �J@      K@      M@      G@     �L@     �K@     @Q@     �V@     @S@      [@     @Y@     �^@      \@     �`@     `c@     �e@     �f@     �j@     �n@      q@     �p@     �r@     �u@     �v@     �z@     �z@     �@     0�@     ȃ@     ��@     ��@     ��@     ��@     ��@     �@     H�@     ��@     ��@     �@     0�@     ��@     .�@     f�@     ��@     ��@     ��@     4�@     x�@      `@     �C@      @       @        
�
conv3/biases_1*�	    !툿   �嬏?      `@!  g ��?)lW7��k?2�#�+(�ŉ�eiS�m��-Ա�L�����J�\�����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`���bB�SY�ܗ�SsW��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��[^:��"��S�F !�ji6�9���.���5�i}1���d�r�>h�'��f�ʜ�7
������6�]�����(���>a�Ϭ(�>�T7��?�vV�R9?+A�F�&?I�I�)�(?uܬ�@8?��%>��:?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�������:�              �?              �?              @       @              @      @      @       @      @      @      @              �?      �?              @               @      �?              �?      �?      �?              @      �?      �?              �?      �?              �?              �?              �?              �?               @              �?               @               @               @              �?      �?               @              �?              @       @              �?      �?      �?      �?       @      �?       @               @       @      @      @      @       @      @      @      �?      @      @      @      @              �?               @        
�
conv4/weights_1*�	    ����   @v��?      �@! �>Qk�0@)�3h?te@2�Ӗ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���uE���⾮��%ᾟMZ��K���u��gr�������>
�/eq
�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?�������:�             �`@     Е@     ��@     ��@     �@     ��@     Ȍ@      �@     `�@     0�@     `�@     h�@     X�@     `|@      {@      x@     �v@     �r@     �q@     �n@     `n@     �j@     `i@     �f@     @c@     �d@     �c@     @]@     �_@     @[@     �T@     �T@      W@     @T@      P@     �N@      H@     �C@      D@      C@      A@      ;@     �@@      6@      3@      9@      4@      .@      .@      ,@      0@      &@      "@      $@       @      @      $@       @      @      "@      @      @      @       @      @      �?      @      @              �?      �?       @      @      @              �?              �?      �?       @      �?      �?              �?              �?              �?              �?              �?              �?              �?               @              �?              �?      �?      �?       @      �?              �?               @      @      @       @       @       @      @      @      @       @      @      @      @      @      @      (@      $@      "@      *@      2@      &@      *@      (@      "@      1@      ,@      8@      7@      7@      =@      C@      G@      D@     �E@      G@     �O@      J@      O@     @S@     �S@      W@     �U@     �Y@     @a@     �`@     @`@     `c@     �d@     �h@      m@     @h@      m@     �r@     pr@     `u@     �w@     �v@     �y@     p}@     �@     X�@     �@     `�@      �@     ��@     `�@     ��@     �@     ��@     t�@     (�@     �l@      *@        
�
conv4/biases_1*�	   @xБ�    �A�?      p@!�;���p�?)��?�C�?2��Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"�ji6�9���.���vV�R9��T7����5�i}1���d�r�f�ʜ�7
������a�Ϭ(���(����_�T�l׾��>M|Kվ��|�~���MZ��K��R%������39W$:����
L�v�Q�28���FP�/�p`B�p��Dp�@�����W_>�p
T~�;�u 5�9��z��6��so쩾4�6NK��2��'v�V,����<�)�4��evk'���o�kJ%�4�e|�Z#�%�����i
�k�2!K�R���J�f;H�\Q������%��'j��p���1���        �-���q=�mm7&c>y�+pm>Łt�=	>��f��p>%���>��-�z�!>4�e|�Z#>6NK��2>�so쩾4>�z��6>u 5�9>/�p`B>�`�}6D>������M>28���FP>�u��gr�>�MZ��K�>豪}0ڰ>��n����>�u`P+d�>��~���>�XQ��>�����>
�/eq
�>jqs&\��>��~]�[�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>��[�?1��a˲?6�]��?��d�r?�5�i}1?�vV�R9?��ڋ?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?�������:�              �?      �?      �?               @              �?      �?      @      �?       @       @      @       @      @       @      @               @       @              @      �?      �?       @      @      �?      @      �?      �?      �?       @              @      �?      �?              �?      �?       @              �?      @               @      �?               @              �?      �?              @              �?      �?      �?              �?              �?              �?              �?              �?              �?              �?       @       @              �?      �?      �?              �?              �?      �?              �?              �?              �?              �?              3@              �?              �?              �?      �?              �?               @              �?              �?              �?              �?      �?              �?      �?      �?              �?              �?      �?      �?      �?              �?       @              �?      �?              �?               @              �?       @      �?      �?      �?              �?              �?      @      �?              @              @      �?       @       @       @       @       @      �?      @      @      @      @      @      �?      �?       @       @      @       @      �?      �?       @      @      @      @      @              @       @      @      �?      �?      �?      @       @      @      �?               @      �?              �?        
�
conv5/weights_1*�	   �ÿ   �&�?      �@!  �z�)g~9yI@2�	�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82�I�I�)�(�+A�F�&�U�4@@�$��[^:��"�f�ʜ�7
������6�]���1��a˲���[���FF�G ��T7��?�vV�R9?��ڋ?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�������:�	             �h@     Pt@     �p@     �p@     �o@      k@     @g@     @g@     �c@     �b@     �Z@      ]@     �\@     �X@      X@      U@      L@     �N@     �K@      O@      N@     �J@      <@      D@      =@      A@      ;@      =@      2@      <@      4@      3@      9@      0@      0@      1@      ,@      &@      $@      @      (@       @      @      @      @      @       @      @       @      @       @      @       @      @       @      @              �?      @      @      �?       @              �?      �?      �?               @              �?              �?              �?              �?              �?      �?              �?      �?       @              �?       @      �?      �?       @      @       @               @      @      @      �?      �?      @      @       @      @      @       @      $@      "@      @      @      @      .@      &@      ,@      "@      (@      3@      &@      (@      8@      8@      8@     �@@     �B@      :@      9@      3@      F@      L@      F@      P@      I@     �I@     �U@     @R@     @V@      Z@     @Z@     �]@     ``@      b@     `a@      f@     �a@     �g@     `l@     @n@      p@     �r@      n@      A@      @        
�
conv5/biases_1*�	    ܴL�   ���Q>      <@!  ���%B>)�[��\F�<2�������M�6��>?�J���8"uH���Ő�;F��`�}6D�/�p`B�p��Dp�@��z��6��so쩾4�_"s�$1�7'_��+/�4��evk'���o�kJ%�4�e|�Z#���-�z�!��i
�k���f��p�Łt�=	���R�������"�RT��+��z�����=ݟ��uy�=nx6�X� >�`��>�mm7&c>��-�z�!>4�e|�Z#>��o�kJ%>4��evk'>���<�)>_"s�$1>6NK��2>�so쩾4>u 5�9>p
T~�;>/�p`B>�`�}6D>��8"uH>6��>?�J>�
L�v�Q>H��'ϱS>�������:�              �?              �?      �?              �?              �?               @              �?      �?      �?              �?               @              �?              �?              �?      �?              �?      �?              �?              �?      �?               @               @              �?              �?        ��a��V      ���	;Q�*��A*ɭ

step  `A

losspAC
�
conv1/weights_1*�	   �E$��   ����?     `�@! ���y1@)���)Y@2�	��(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0�+A�F�&�U�4@@�$��S�F !�ji6�9����ڋ��vV�R9�>h�'��f�ʜ�7
�pz�w�7��})�l a����?f�ʜ�7
?ji6�9�?�S�F !?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?�������:�	              �?       @      &@      2@      6@      C@     @U@     �a@     �d@     �b@      _@      c@     @Z@     �X@     @Z@     �Y@      U@      Q@     �O@     �R@      O@      N@      F@      L@      <@      <@      :@      5@     �@@     �@@      4@      2@      7@      6@      &@      2@      &@      2@      @      &@      @      "@       @      @      @      @      $@      @      @      @      @      @       @       @      @      �?       @       @      �?      �?      @               @      �?              �?              �?              �?              �?              �?              �?               @              �?      @              �?              �?      @      �?       @       @      �?      �?       @      @      @      �?      @      @       @      �?       @      @      @      @      @      @      $@      "@      $@      (@      .@      .@      $@      ,@      $@      3@      (@      7@      9@      ;@      :@      G@     �B@     �B@     �G@     �D@     �I@     �J@      O@      Q@      T@     �V@     �U@     @[@     @Y@     �Y@     �^@     �a@      b@     �b@     �c@     @]@     @\@     �V@      D@      9@      3@      (@      @        
�
conv1/biases_1*�	   @�o��   ���?      @@! @]���?)��;>M5�?2�I�������g�骿eiS�m��-Ա�L��uWy��r�;8�clp��l�P�`�E��{��^���|�~���MZ��K��;9��R�>���?�ګ>��[�?1��a˲?�7Kaa+?��VlQ.?d�\D�X=?���#@?�!�A?a�$��{E?
����G?�l�P�`?���%��b?���T}?>	� �?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?�/��?�uS��a�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?�������:�              �?              �?              �?              �?              �?              �?              �?              �?              �?      �?              �?              �?               @               @      �?      �?      �?              @               @      �?      �?      �?      �?       @               @        
�
conv2/weights_1*�	    �W��   �wV�?      �@!@Hs�bT@)�>"e~'F@2���]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾX$�z�>.��fc��>;�"�q�>['�?��>��>M|K�>�_�T�l�>�iD*L��>�ѩ�-�>���%�>�uE����>�f����>��(���>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?�������:�              �?      @     �N@      �@     ��@     ȝ@     �@     ��@     ��@     �@     x�@     ��@     X�@     Ў@     P�@     ��@     ��@     0�@     ��@     �@      }@     �y@     �x@     �w@      t@     �q@     pr@     `n@     @k@      j@     �g@     �b@     �d@     �a@     @^@     @Y@      V@     �V@     @X@     �N@      M@     �R@      L@     �N@     �H@      C@     �J@     �C@      ?@      9@      @@      8@      8@      8@      &@      ;@      2@      6@      3@      *@       @      &@      @      @      @      @      &@      @      "@      @      @      @      @      @      @      �?       @       @              @      �?       @      @      @      �?              �?       @      @              �?              �?              �?      �?              �?      �?      �?              �?              �?              �?      �?              �?       @              �?               @      �?               @       @      �?      @      @      @      �?      @      @      @      @               @      @      @      @      @      @       @      @      @      @      @      "@      @      (@      0@      (@      "@      1@      1@      2@      .@      =@      ?@      8@      6@      @@      F@     �G@     �D@      L@      L@     @P@     @S@     �N@     �T@     @V@     @Z@      ]@     �Z@     �^@      b@     @c@      h@     @i@     �k@     �k@     �l@     @q@     pr@     s@     �u@     �x@     �{@     �@     �@     �@     �@     ��@     ؈@     ��@     `�@     @�@     �@     ��@     ��@      �@     ��@     ��@     ؟@     n�@     ܖ@      y@     �f@      M@      "@      @        
�	
conv2/biases_1*�		   ����   ���?      P@!  �����?)���rP��?2��7c_XY��#�+(�ŉ����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�Tw��Nof�5Ucv0ed��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW��lDZrS�nK���LQ�k�1^�sO�
����G�a�$��{E�uܬ�@8���%�V6��u�w74���82��7Kaa+�I�I�)�(��[^:��"��S�F !���d�r�x?�x�������6�]����7Kaa+?��VlQ.?d�\D�X=?���#@?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�������:�              �?              �?              �?      �?              �?              �?      �?              �?              �?      �?              @              �?              �?               @              �?              �?              �?              �?              �?              �?               @              �?              �?              �?      �?              @              �?       @               @              �?               @              @      �?       @      �?      �?              @               @       @      @       @       @        
�
conv3/weights_1*�	   `Z���   @,@�?      �@!@�U+�eN@)_��l��U@2���]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a�h���`�8K�ߝ�a�Ϭ(���(��澢f������~]�[Ӿjqs&\�Ѿ
�/eq
Ⱦ����ž�����>
�/eq
�>K+�E���>jqs&\��>��~]�[�>��>M|K�>E��a�W�>�ѩ�-�>�f����>��(���>a�Ϭ(�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?�������:�              1@     ��@     �@     b�@     �@     �@     Ġ@     ��@     ��@     ��@     4�@     ��@     ��@     ȑ@     T�@     X�@     8�@     `�@     ��@     ��@     P�@     @�@     �~@     �{@      x@     �v@      v@     �r@      n@     �n@     �j@      i@     �d@     �c@     @b@     @a@      `@     @`@     �[@      Z@      V@     @T@     �T@      P@     �O@      J@      I@     �A@      F@      E@      E@     �A@      <@      :@      8@      7@      0@      .@      ,@      .@      ,@      &@       @      (@      "@      .@      $@       @      "@      $@      @      @      @      �?       @      �?              �?       @      @              �?      @      @      �?      @      �?              �?              �?      �?              �?              �?              �?              �?              �?              �?              �?      �?              �?               @              �?      �?      @      �?      �?      �?      @      @      �?       @      @      @       @      @      @       @      $@      @      @      ,@      @      $@      (@      $@      "@      4@      *@      1@      ;@      6@      9@      ?@      <@      9@      B@      ;@      @@     �G@     �E@     �O@     �K@     �M@      R@     �R@     �V@     �V@      ^@     �\@     �^@     �a@     �`@     �e@      i@     �j@      n@      n@     @p@      t@      w@     �v@     `x@     �{@     �@     �@     ȃ@     p�@     ��@     ��@     ��@     Ќ@     ܐ@     L�@     T�@     �@      �@     (�@     �@     :�@     ��@     ��@     ��@      �@     ��@     ��@      h@     �M@      5@      @       @        
�
conv3/biases_1*�	    5���   ��6�?      `@!  ?�P�?)܇�(�dr?2��#�h/���7c_XY��#�+(�ŉ�eiS�m������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@��u�w74���82�U�4@@�$��[^:��"���ڋ��vV�R9��T7���x?�x��>h�'����[���FF�G ��ߊ4F��>})�l a�>��d�r?�5�i}1?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?��%>��:?d�\D�X=?���#@?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?�������:�              �?              �?              �?      @      �?      �?      @      @      @      @       @      @               @      �?      �?              �?       @       @      �?       @               @              @      �?              �?              �?              �?              �?      �?              �?              �?              �?              �?               @       @              �?              �?               @      �?               @               @       @              @       @              �?      �?       @               @      �?               @      @      @      @      �?      �?      @      @      @      @      @      @       @      �?      �?               @        
�
conv4/weights_1*�	   �(��   ����?      �@!��|�w�5@)�U�m�~e@2�Ӗ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1������6�]����uE���⾮��%�;9��R���5�L�������>
�/eq
�>;�"�q�>['�?��>8K�ߝ�>�h���`�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�������:�             `a@     ��@     ��@     ��@     �@     ؎@     �@     @�@     �@     H�@     �@     ��@     P�@     �|@     �z@     �w@     pw@     `r@     Pq@     0p@     �l@      k@     �i@     �f@      c@     �c@     �b@     �_@     �`@     @\@      U@      V@     @W@     �S@      N@     �M@      F@     �E@      C@     �G@     �B@      7@     �@@      5@      5@      6@      2@      0@      *@       @      2@      ,@      (@      "@      @      @       @       @      @      @      @       @      @      @      @      @      @      @      �?      �?      �?      @       @      �?              �?      �?               @              �?              �?              �?              �?              �?              �?              �?              �?      �?              �?      �?      �?              �?      �?      �?      �?      �?      �?      �?      �?       @              �?              �?      @      @      �?      �?       @      @      @      @      @       @      @      @      @      @      (@      &@      "@      *@      @      0@      *@      .@      (@      &@      1@      <@      6@      6@      @@      D@     �E@     �D@      D@      H@     �I@     �J@      Q@     �Q@      T@     @Y@     �T@     @Z@      _@      a@     �`@     @d@     �c@     �h@     �k@     �i@     @m@     �q@     �r@     u@     �w@     �v@      y@     ~@      �@     �@     �@     @�@     �@     ��@     H�@     @�@     �@     ��@     X�@     4�@     �o@      3@      @        
�
conv4/biases_1*�	   `I��   ����?      p@!'j��u��?)���'K�?2�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6���bȬ�0���VlQ.��7Kaa+�+A�F�&�U�4@@�$��S�F !�ji6�9���.����ڋ��vV�R9��T7���1��a˲���[��})�l a��ߊ4F��8K�ߝ�a�Ϭ(�E��a�Wܾ�iD*L�پ�5�L�����]�����MZ��K���u��gr��ہkVl�p�w`f���n��
L�v�Q�28���FP���Ő�;F��`�}6D�/�p`B�p��Dp�@�p
T~�;�u 5�9��z��6�_"s�$1�7'_��+/����<�)�4��evk'���o�kJ%��i
�k���f��p��mm7&c��`����tO����f;H�\Q��        �-���q=�mm7&c>y�+pm>��-�z�!>4�e|�Z#>��o�kJ%>4��evk'>u 5�9>p
T~�;>����W_>>/�p`B>�`�}6D>28���FP>�
L�v�Q>�5�L�>;9��R�>��n����>�u`P+d�>0�6�/n�>
�/eq
�>;�"�q�>K+�E���>jqs&\��>��~]�[�>��>M|K�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>pz�w�7�>I��P=�>1��a˲?6�]��?����?f�ʜ�7
?>h�'�?��d�r?�5�i}1?�T7��?U�4@@�$?+A�F�&?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�������:�              �?      �?      �?               @      �?              @       @              �?      @       @       @      @       @       @              @      �?      @       @       @              @       @       @       @      �?       @               @       @      �?              �?      @      �?      �?      �?      @       @      �?              �?      �?              �?              �?       @      �?              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?              @              �?       @               @              �?      �?              �?              �?              �?              2@              �?               @              �?               @      �?              �?              �?              �?              �?      �?              �?              �?              �?               @              �?       @               @              �?              �?              �?      �?              �?      �?              �?              �?              �?      �?               @               @      �?      @       @      �?      �?      @      @      �?       @      @       @              �?      @       @      @      @      @      �?       @              @       @      @      �?      @      �?      �?      @      @      @      @      �?      @      @      �?      �?       @      �?       @      @       @      �?       @              �?      �?        
�
conv5/weights_1*�	    %ÿ   `��?      �@!  ��	=�)+��K��I@2�	�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��FF�G �>�?�s�����~]�[Ӿjqs&\�Ѿ�f����>��(���>��d�r?�5�i}1?�.�?ji6�9�?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�������:�	             �h@     �s@     �p@     Pp@     �o@     �l@     `f@     �f@     �c@     �b@     @\@     �[@     �\@     �X@      Y@     �U@      P@      J@     �K@     �J@     �O@     �M@      A@     �B@      =@      A@      :@      ;@      1@      8@      4@      8@      3@      ,@      2@      ,@      0@      (@       @      @      $@       @      @      @      @      @      @      @      @      �?      �?      �?       @      @      @       @      �?               @      @       @      �?              �?               @      �?               @              �?       @              �?              �?              �?              �?              �?              �?       @      �?      �?      �?              �?               @      @       @       @      @               @       @      @      �?      @       @      @      "@      $@      "@      @       @       @      (@       @      "@      &@      .@      8@      @      .@      5@      8@      6@      D@      ?@      8@      9@      :@     �F@      I@     �F@     �Q@      G@      L@      S@      R@     @X@     @Y@     �Z@     �\@     @`@     �b@     �a@     �e@     `a@     �g@      l@     �m@     Pp@      s@     @n@     �D@      @       @        
�
conv5/biases_1*�	   @~-S�   �~�R>      <@!   �a$>)i�B-�<2�H��'ϱS��
L�v�Q���8"uH���Ő�;F�/�p`B�p��Dp�@�u 5�9��z��6�6NK��2�_"s�$1��'v�V,����<�)�4��evk'���o�kJ%�4�e|�Z#���-�z�!�%�����i
�k��#���j�Z�TA[�����"���1���='j��p�=�mm7&c>y�+pm>RT��+�>��o�kJ%>4��evk'>���<�)>�'v�V,>_"s�$1>6NK��2>p
T~�;>����W_>>p��Dp�@>/�p`B>�`�}6D>��Ő�;F>6��>?�J>������M>�
L�v�Q>H��'ϱS>�������:�              �?               @              �?              �?               @              �?      �?               @      �?              �?              �?      �?              �?              �?      �?              �?              �?               @              �?              �?               @              �?              �?        �aV��W      ��7	`��*��A*�

step  pA

loss��.C
�
conv1/weights_1*�	   ��K��    J=�?     `�@! Pi�6@)��;wU"!@2�	Ӗ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.�+A�F�&�U�4@@�$�1��a˲���[���FF�G �ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?�������:�	               @      @      "@      ,@      2@      6@      C@     �G@     �U@     `b@     �c@     �`@     �[@     @_@     @_@     �S@     �Y@     �V@     @S@     �R@     �O@      P@     �L@      K@      G@     �F@      D@      8@      ;@      <@      B@      ;@      7@      7@      3@      .@      *@      ,@      $@      .@      &@      @      @      @      @      @      @       @      @      @      @      @      @      �?      @       @               @      @      @       @               @       @       @               @               @              �?      �?              @              �?              �?      �?              �?      �?       @      �?      �?      �?      @               @      �?      �?      @      �?      @      @      @      @      @      @      @      @       @      @      @      0@      "@      @      $@      "@      .@      0@      0@      5@      2@      6@      :@      >@      B@      B@      B@      I@      G@     �J@     �N@     �P@      O@      P@     @W@     @U@     �X@     @X@      Z@      X@     �b@      b@     `c@     `c@      Z@     �Z@     �X@     �T@     �F@      <@      7@      (@      @       @        
�
conv1/biases_1*�	   �y���   �k�?      @@! 1�*U�?)�Y�*2�?2�I�������g�骿#�+(�ŉ�eiS�m��hyO�s�uWy��r��l�P�`�E��{��^���|�~���MZ��K����n����>�u`P+d�>6�]��?����?��VlQ.?��bȬ�0?�T���C?a�$��{E?�qU���I?IcD���L?5Ucv0ed?Tw��Nof?����=��?���J�\�?-Ա�L�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?�������:�              �?              �?              �?              �?              �?              �?              �?              �?               @              �?              �?              �?      �?              �?               @      �?      �?              �?       @              �?      �?      �?      �?      �?       @      �?       @        
�
conv2/weights_1*�	   ��밿   @#�?      �@! �ك~Z@)*a�֗F@2���]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7���ߊ4F��h���`f�����uE����K+�E��Ͼ['�?�;�XQ�þ��~��¾��|�~���MZ��K��0�6�/n�>5�"�g��>['�?��>K+�E���>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>�ѩ�-�>���%�>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?�������:�               @      .@     �R@     x�@     |�@     \�@     ��@     ��@     d�@     ��@     ��@     x�@     �@     Ȏ@     �@     ��@     �@     h�@     �@     `�@     �{@      {@     �x@     �v@     �r@      r@     �r@     @o@      j@     �j@     �d@      e@      c@      b@      `@      \@     �Y@     �S@     �V@     �S@     @P@     �P@     @R@     �J@     �C@     �C@     �G@     �C@      >@      9@      :@      1@      9@      :@      ,@      7@      3@      3@      &@      ,@      &@      @      @       @      "@      @      @      @       @      @      �?      @       @       @      @      @       @              �?       @       @      �?       @      �?               @              �?              �?              �?              �?              �?              �?              �?              �?      �?      �?              �?              �?              �?              @      @      �?              �?      @      �?       @      �?       @      @      @      @       @       @      @       @      @      "@      @      @      $@      &@      @      &@      "@       @      $@      (@      &@      (@      .@      :@      1@      8@      7@      1@      <@      :@      6@     �C@      C@     �D@     @P@     �K@     @P@     �R@     �P@     �T@      Y@     �U@     �]@      Z@     �`@     �]@     �a@     �f@     `k@      m@     @m@     `n@     q@     �r@      t@      v@     �v@     �{@     �~@     �@     0�@     �@     ��@     p�@     ��@     ��@     @�@     �@     ��@     d�@     <�@     ��@     t�@     t�@     v�@     ��@     �|@     �m@     `a@     �I@      5@      @      �?        
�

conv2/biases_1*�
	   �o��   �s�?      P@!  C��G�?)
�(�w�?2����&���#�h/��>	� �����T}�o��5sz�*QH�x�&b՞
�u�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�k�1^�sO�IcD���L��qU���I�
����G��T���C��!�A�uܬ�@8���%�V6���82���bȬ�0���VlQ.�U�4@@�$��[^:��"���d�r�x?�x��G&�$��5�"�g�������?f�ʜ�7
?�vV�R9?��ڋ?ji6�9�?�S�F !?�T���C?a�$��{E?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?hyO�s?&b՞
�u?*QH�x?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?�������:�              �?              �?      �?              �?              �?       @              �?              �?              �?              �?              �?              �?              �?              �?      �?              �?              �?              �?              �?              �?              �?              �?              �?      �?               @              �?               @              �?      �?               @      �?              �?      �?              �?               @      @      �?      @              �?              �?      @               @      �?      @      @       @        
�
conv3/weights_1*�	   @�"��   �u�?      �@! W3��T@)��y��U@2��{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾙ѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�Ѿ�[�=�k���*��ڽ�0�6�/n���u`P+d����n��������]���>�5�L�>��n����>�u`P+d�>G&�$�>�*��ڽ>
�/eq
�>;�"�q�>jqs&\��>��~]�[�>�iD*L��>E��a�W�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?�������:�              �?      9@     ��@     ��@     �@     Τ@     ܢ@     ��@     ğ@      �@     �@     �@     P�@     4�@     ܑ@     ��@     �@     ��@     0�@     @�@     x�@     �@     `�@     @~@     �z@      x@     0w@     �t@     �s@     `o@      o@     �j@     `h@     �f@      c@     �a@      a@      a@     �^@     �W@     �T@     �X@     �S@     �W@     �J@      J@      I@     �I@     �I@      F@      C@      I@      A@      ;@      ;@      6@      4@      <@      2@      0@      &@      .@      (@      @      0@      (@       @       @      @      @       @      @      �?      $@      �?      �?      @      @       @      @              @      �?      @       @      �?              @       @       @      �?              �?      �?      �?              �?      �?      �?      �?               @              �?              �?      �?              �?              �?              �?              �?              �?              �?               @      �?       @      �?              �?      @      �?      @      @              @      �?      �?      @      �?              @              @      @      @      @      @      "@      (@      0@      $@      $@      .@      (@      1@      .@      8@      5@      0@      :@      =@      7@     �B@      :@     �G@      J@     �G@     �H@      K@      O@     �P@      T@      U@     @Y@      [@     �`@     @]@     �`@     �c@     `g@      f@      l@     `k@     0p@     @p@     @s@     0u@     0x@     �y@     0{@     �@     8�@     ؂@     �@     P�@     �@     �@     �@     ��@     �@     ��@     ��@     h�@     �@     �@     >�@     ~�@     ��@     ��@     �@     t�@     (�@     �r@     @]@     �D@      1@      @       @        
�
conv3/biases_1*�	   `�+��    ��?      `@!   ���?)�.8��w?2����&���#�h/���7c_XY��#�+(�ŉ����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E����#@�d�\D�X=���bȬ�0���VlQ.�I�I�)�(�+A�F�&��.����ڋ���d�r�x?�x��>h�'��f�ʜ�7
�I��P=��pz�w�7���f����>��(���>pz�w�7�>I��P=�>����?f�ʜ�7
?ji6�9�?�S�F !?U�4@@�$?+A�F�&?��VlQ.?��bȬ�0?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�������:�              �?              �?              �?      @      �?      �?      �?      @      @      �?      @      @      �?      �?       @      �?              �?       @      �?      @              �?      @      �?      �?              �?               @              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?      �?      �?      �?               @       @      @      �?      �?      �?       @       @      �?       @              �?      @      @      @      @               @      �?      @      @      @      @       @       @      �?               @        
�
conv4/weights_1*�	   `�v��   @=��?      �@! �qWʜ>@)��bH`�e@2��?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1�f�ʜ�7
������>�?�s���O�ʗ����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�MZ��K�>��|�~�>�����>
�/eq
�>E��a�W�>�ѩ�-�>���%�>�ߊ4F��>})�l a�>�FF�G ?��[�?1��a˲?6�]��?����?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�������:�              �?      b@     D�@     t�@     �@     ��@     ��@     X�@     �@     0�@     ��@     ��@     h�@     p�@     �{@     �{@     `w@      w@      s@     �q@     �n@      m@      l@      h@     �g@     �c@     �c@     @b@      ^@     �`@     @\@      U@     �V@     �V@     @T@      P@     �N@     �I@      H@      B@     �C@     �F@      9@      6@      ;@      7@      4@      3@      4@      ,@      (@      *@      .@      "@       @      @      @      &@      @      @      @      @       @      �?      @      @      @      @      @       @      @      @      @               @              �?      �?               @              �?              �?              �?      �?      �?              �?              �?              �?      �?              �?               @              �?      �?              �?              �?       @              �?              �?              �?      @       @      �?      @      �?      @      @       @              @      @      "@      @      "@      @      $@      *@      "@      $@      &@      1@      3@      *@      4@      :@      3@      <@      =@     �C@      E@     �D@      B@     �F@     �J@      K@     @P@     @P@     �R@     �Z@      U@     �[@      _@     @^@      b@     �c@     �d@     �h@     �j@     �i@     �m@     q@     s@     u@     w@      w@     �y@     `~@     �@     p�@     ��@     ؄@     x�@     �@     0�@     ��@     ��@     ��@     P�@     �@     �q@     �B@      *@      �?        
�
conv4/biases_1*�	   �8��   `p#�?      p@!j�r��?)x��:K��?2��"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:���82���bȬ�0��7Kaa+�I�I�)�(�U�4@@�$��[^:��"��S�F !���ڋ��vV�R9�>�?�s���O�ʗ����h���`�8K�ߝ���(��澢f���侮��%ᾙѩ�-߾;9��R���5�L����|�~���MZ��K��E'�/��x��i����v�H��'ϱS��
L�v�Q�6��>?�J���8"uH���Ő�;F��`�}6D�/�p`B�p��Dp�@�����W_>�p
T~�;�u 5�9��z��6��so쩾4�6NK��2�_"s�$1�7'_��+/����<�)�4��evk'�%�����i
�k���R����2!K�R��f;H�\Q������%��        �-���q=nx6�X� >�`��>4�e|�Z#>��o�kJ%>���<�)>�'v�V,>p
T~�;>����W_>>p��Dp�@>/�p`B>�`�}6D>��Ő�;F>H��'ϱS>��x��U>����>豪}0ڰ>�u`P+d�>0�6�/n�>5�"�g��>;�"�q�>['�?��>��~]�[�>��>M|K�>�_�T�l�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>O�ʗ��>>�?�s��>6�]��?����?��d�r?�5�i}1?�T7��?�.�?ji6�9�?�7Kaa+?��VlQ.?��bȬ�0?�u�w74?��%�V6?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�������:�              �?      �?      �?              �?       @      �?      @       @               @      @      @       @              @       @      �?       @      �?      @              @              @      �?      @      �?       @      �?               @       @               @      @      �?      �?      @              @              @              �?              �?      �?               @              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?      �?      �?               @      �?              �?              �?      �?              �?              �?              �?              �?              2@              �?               @              �?              �?      �?              �?      �?              �?              �?              �?      �?              �?              �?      �?              �?      �?              �?       @      �?       @              �?              �?              �?       @              �?              �?      �?               @              �?      �?      @               @      �?       @      @       @       @      @      @               @      @      @      @      @      @       @      �?      �?      �?      @       @      �?      @       @       @      @      @      @      @      �?      @       @      �?       @              @       @       @       @       @       @              �?      �?        
�
conv5/weights_1*�	   ���ÿ   �@(�?      �@!  O���@)A�
A�I@2�	yD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���VlQ.��7Kaa+�I�I�)�(�U�4@@�$��[^:��"��S�F !�ji6�9���vV�R9��T7���>h�'��f�ʜ�7
��.�?ji6�9�?�S�F !?�[^:��"?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%>��:?d�\D�X=?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�������:�	               @     �g@     Ps@     q@     �o@     @p@      k@     �g@     `f@      c@     �b@     @]@     �Z@     �Y@     �Z@      X@     @U@     �Q@     �M@     �G@      H@      N@      M@     �E@      D@      :@     �A@      8@      =@      1@      :@      8@      3@      0@      0@      2@      (@      $@      ,@      @      @      @      @       @      @      "@       @      �?      �?      @      @      �?       @       @      @       @       @      �?      �?      �?       @      �?       @               @      �?       @              �?      �?               @      �?      �?              �?              �?              �?              �?              �?      �?      �?      �?      �?               @              @      @      @      @      @              �?       @      @      @       @       @      @       @      *@       @      "@       @      @      @      .@      "@      @      $@      *@      6@      $@      *@      8@      7@      :@      A@      >@      9@      6@      @@      B@     �I@     �G@     �R@     �E@      M@      S@      P@     �Y@     @X@     @\@      ]@     @`@      b@      b@     �e@      a@      h@     �k@      m@     Pp@     @s@     �m@      J@      6@      @       @        
�
conv5/biases_1*�	   �
X�   �aS>      <@!   0�.�)�mAXP�<2�4�j�6Z�Fixі�W�6��>?�J���8"uH��`�}6D�/�p`B�u 5�9��z��6��so쩾4�6NK��2�7'_��+/��'v�V,����<�)�4��evk'���o�kJ%�4�e|�Z#�%�����i
�k�2!K�R���J��#���j��9�e����K��󽉊-��J�=�K���=y�+pm>RT��+�>��o�kJ%>4��evk'>�'v�V,>7'_��+/>_"s�$1>6NK��2>����W_>>p��Dp�@>�`�}6D>��Ő�;F>��8"uH>6��>?�J>������M>�
L�v�Q>H��'ϱS>�������:�              �?               @              �?              �?      �?      �?               @               @              �?              �?              �?      �?              �?              �?              �?              �?              �?               @              �?              �?       @              �?              �?        �s�TXX       �6�	�B�*��A*ɰ

step  �A

loss��B
�
conv1/weights_1*�	    �ɿ   ����?     `�@! �>���#@)
1�p''@2�
�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"�>h�'��f�ʜ�7
���~��¾�[�=�k��x?�x�?��d�r?�T7��?�vV�R9?ji6�9�?�S�F !?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�������:�
              �?       @      @      $@      .@      *@      ?@     �@@     �B@      B@      F@      I@     @Q@      W@     `b@     �a@      \@     @[@      `@     �X@     �R@      W@     @U@     @V@     �M@      O@     @Q@      M@      H@     �E@     �C@      A@      ;@      =@      8@      @@      6@      2@      1@      2@      2@      "@      .@      *@      ,@      "@      &@       @      @      @      @      �?       @      @      @      @      @      @      �?      @      �?       @       @      @       @               @      @      �?              �?      �?              �?               @              �?              �?              �?              �?              �?              �?      �?      �?      �?      �?      �?      �?              �?      �?       @      �?               @      @       @      @      @      @       @      @      @      @      @      @       @      @      .@       @      (@      (@      .@       @      $@      4@      .@      "@      6@      :@      ,@      8@      E@      B@      7@     �D@     �E@     �C@      J@     �L@      P@     @S@     �Q@      U@      X@     @U@      W@     �Z@     �`@     �_@     �a@     @d@     �Z@     �W@     @W@     @T@     @P@     �I@      @@      8@      ,@      @      @        
�
conv1/biases_1*�	    \��   ���?      @@! 6&����?)�Biܷ�?2���]$A鱿����iH���7c_XY��#�+(�ŉ�&b՞
�u�hyO�s��m9�H�[���bB�SY��*��ڽ�G&�$��0�6�/n�>5�"�g��>6�]��?����?��VlQ.?��bȬ�0?a�$��{E?
����G?k�1^�sO?nK���LQ?5Ucv0ed?Tw��Nof?>	� �?����=��?eiS�m�?#�+(�ŉ?^�S���?�"�uԖ?��<�A��?�v��ab�?�/��?�uS��a�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�������:�              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?               @              �?              �?       @      �?              �?               @      �?               @              �?       @       @               @        
�
conv2/weights_1*�	   ��~��   `%��?      �@!�L^��b@)�����G@2���]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ뾢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�����>
�/eq
�>��>M|K�>�_�T�l�>�iD*L��>���%�>�uE����>�f����>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?�������:�               @      9@      Z@     ��@     t�@     Ԝ@     ܚ@     x�@     ��@     ��@     �@     D�@     �@     (�@     ȉ@     ��@     @�@     ��@     Ѐ@     �~@     }@     pz@     Px@     �w@     �s@     �q@     �p@     �n@     �j@     @k@      d@      d@     �c@     @_@      `@      Y@     @Y@     @U@     @T@     �R@      O@      T@     @Q@     �I@      J@      B@     �D@      B@     �C@      ;@      =@      @@     �A@      5@      0@      7@      3@      4@      1@      (@      *@      "@       @      "@      @      @      @      @      @      @      @      @      @      @      @      @      �?       @      @       @              �?              @      �?       @      �?      �?               @              �?      �?      �?      �?              �?              �?      �?              �?       @              �?       @      @              �?      �?      �?              @      @       @      @      @      @       @      @       @      @      @      @      @      "@      @       @      $@      @      "@      "@      ,@      .@      ,@      4@      1@      2@      <@      4@      5@      @@      =@     �C@      F@      G@      M@      P@      O@     �U@     �O@     �Q@     �V@      W@      [@     �]@     �]@      a@     `b@     �e@      i@     �l@     `j@     `l@     0q@     �r@     �t@     @u@     0w@     �{@      }@     �@     ��@     (�@     ��@     x�@     p�@     ��@     �@     ��@     X�@     ��@     x�@     �@     (�@     ̟@     ��@     ��@      �@     �t@     �m@      c@     @X@      D@      3@      (@       @        
�	
conv2/biases_1*�		   �����    �s�?      P@!  ��=�?)K�=���?2�^�S�����Rc�ݒ�����=���>	� �����T}�o��5sz�*QH�x�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�IcD���L��qU���I���%>��:�uܬ�@8���%�V6��u�w74��[^:��"��S�F !��vV�R9��T7����5�i}1���d�r����%ᾙѩ�-߾��(���>a�Ϭ(�>�.�?ji6�9�?uܬ�@8?��%>��:?
����G?�qU���I?k�1^�sO?nK���LQ?�lDZrS?��bB�SY?�m9�H�[?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?*QH�x?o��5sz?���T}?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�������:�              �?              �?      �?              �?              �?      �?       @      �?              �?              �?              �?              �?              @              �?              �?              �?              �?              �?              �?              �?              �?              �?      �?              �?               @              �?      �?              �?               @               @      �?              �?              @      @      @      �?              �?              �?      @      �?      �?       @      @      �?       @        
�
conv3/weights_1*�	   @N���   @M��?      �@!��	n��`@)����_V@2��{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�a�Ϭ(���(��澮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ��>M|Kվ��~]�[Ӿ['�?�;;�"�qʾ['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>E��a�W�>�ѩ�-�>���%�>��(���>a�Ϭ(�>�h���`�>�ߊ4F��>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�������:�              �?     �F@     ��@     ��@     d�@     8�@     ��@     t�@     ��@     (�@     4�@     ��@     ��@     P�@     ؑ@     ��@     @�@      �@     ��@     ��@     (�@     ��@     0�@     �~@      z@     z@     v@     pt@     Pr@     �p@      m@     �i@     @f@     `f@     �c@     �c@     �c@     @_@     �`@     @Z@      W@     �V@     @U@     �V@     @P@      G@     �I@     �A@      I@      G@     �B@     �E@      A@      A@      6@      >@      >@      (@      *@      ,@      0@      1@      .@      "@       @      $@      @      @      @       @      @      @      @      "@               @       @      @      �?      @      @      @       @       @      �?              @      �?       @      �?              �?              @              �?               @              �?              �?              �?      �?              @      �?              �?              �?              �?      �?              �?      @               @       @       @              @      @       @      @      @      @      @      @      @      "@       @      $@      &@      2@       @      $@      *@      ,@      4@      0@      7@      :@      >@      4@      C@     �@@      H@      G@     �B@     �P@      P@     �Q@      N@      S@     �V@     �W@     �[@     �[@     @^@     �`@     �c@     `e@     �f@     `l@      n@     �p@     �o@     �r@     w@     �v@     @w@     �z@     P@     H�@     �@     (�@     ��@     8�@     �@     ȍ@     ��@     ��@     0�@     ��@     �@     ��@     �@     P�@     x�@     ��@     ��@     ܦ@     ֧@     ��@     �x@     `m@     @`@     �N@     �B@       @       @       @        
�
conv3/biases_1*�	   @k���   �<9�?      `@! �����?)}�ɫ�V}?2��Rc�ݒ����&���#�h/���7c_XY��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C�d�\D�X=���%>��:�I�I�)�(�+A�F�&�U�4@@�$��S�F !�ji6�9���T7����5�i}1���d�r��f�����uE���⾮��%�8K�ߝ�>�h���`�>O�ʗ��>>�?�s��>�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�u�w74?��%�V6?uܬ�@8?��%>��:?���#@?�!�A?nK���LQ?�lDZrS?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?�������:�              �?              �?              �?      @      �?      �?      �?      @      �?      @      �?      @       @      �?               @      �?       @       @               @      �?              �?              @               @      �?              �?              �?              �?      �?              �?              �?       @              �?      �?              �?              �?              �?              �?              �?              �?              �?               @              @      �?       @               @      �?       @       @      @              @      @      @      @       @      �?      �?      @      @      @       @      �?      @       @              �?      �?        
�
conv4/weights_1*�	   �r���   @ʒ�?      �@!��Mf��J@)�+���e@2��?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
��FF�G �>�?�s���O�ʗ����uE���⾮��%Ᾱ��?�ګ>����>�����>
�/eq
�>�f����>��(���>�h���`�>�ߊ4F��>pz�w�7�>I��P=�>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�������:�              @     �b@     Ԕ@      �@     ��@     ��@     (�@     ��@     �@     (�@     Ѓ@     x�@     ��@     @�@     �{@     �{@     �w@     �v@      r@     @r@     �n@     �l@     `k@     �g@     �g@      c@     @b@     @c@      `@      _@     �W@     �X@     @T@     �Y@     �R@     �P@     @P@      H@     �C@     �E@      D@     �E@      <@      =@      4@      ;@      9@      .@      3@      1@      &@      &@       @      0@      (@      @      @      $@      @      @      @      @      @       @       @      @       @       @       @      �?      @              @       @      @      @               @              �?      �?      �?              @              �?      �?              �?              �?              �?              �?              �?              �?              �?      �?              @              �?      �?              �?       @      �?      �?       @               @       @       @      @       @       @      @              @       @      @      @      @      @      @      @      @      "@      @      @      ,@      &@      &@      ,@      "@      4@       @      .@      ;@      :@      <@      ;@      D@      H@     �B@      F@     �I@     �J@      M@      P@      L@      T@     �X@     @W@      X@     �`@     �_@     `b@      b@     @e@     �h@     �i@     �h@     �m@     0q@     s@     pu@     pu@     �w@     �x@      @     0�@     ��@     ��@     p�@      �@     �@     X�@     p�@     �@     ��@     ��@     �@     `u@     @U@      ?@      0@      @      �?        
�
conv4/biases_1*�	   �d��   �銨?      p@!(.Գ���?)�0���?2�}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.�I�I�)�(�+A�F�&��[^:��"��S�F !��.����ڋ��T7����5�i}1�1��a˲���[���h���`�8K�ߝ뾢f�����uE���⾹��?�ګ�;9��R�����]������|�~��[#=�؏�������~�H��'ϱS��
L�v�Q�������M�6��>?�J���8"uH���Ő�;F��`�}6D�/�p`B�p��Dp�@�����W_>�p
T~�;�u 5�9��z��6��so쩾4�6NK��2����<�)�4��evk'���-�z�!�%������f��p�Łt�=	���-��J�'j��p�        �-���q==��]���=��1���=4�e|�Z#>��o�kJ%>���<�)>�'v�V,>����W_>>p��Dp�@>/�p`B>�`�}6D>��Ő�;F>��8"uH>��x��U>Fixі�W>��n����>�u`P+d�>0�6�/n�>5�"�g��>['�?��>K+�E���>��~]�[�>��>M|K�>�iD*L��>E��a�W�>�uE����>�f����>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>��[�?1��a˲?����?f�ʜ�7
?�T7��?�vV�R9?��ڋ?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?�������:�              �?               @              �?       @       @       @      �?      �?       @      @      @       @              @      �?       @       @       @      @      �?       @              @      @      �?      �?       @      �?      �?       @              �?      @              �?      @              �?               @      �?      �?              �?      �?              �?              �?              �?              �?              �?               @              �?              �?              �?              �?              �?              �?              �?              �?      �?      �?      �?               @      �?              �?      �?              �?              �?              �?              �?              2@              �?              �?               @              �?      �?              �?      �?              �?              �?               @              �?              �?              �?              �?              �?      �?      �?      @              �?              �?              �?      �?              �?              �?      �?               @      �?              �?      �?               @               @       @       @               @      @       @       @      @      �?      �?               @      @      �?      @      @      @      @      �?      �?      �?      @       @      �?      �?      @      @       @      @      @      @      �?      @       @      @      �?      �?      @       @      �?      @       @       @              �?              �?        
�
conv5/weights_1*�	   ��}Ŀ   `���?      �@! ��۴�0@)�����J@2�	yD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82�U�4@@�$��[^:��"��.����ڋ�>h�'��f�ʜ�7
��[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?�������:�	              &@     �g@     �r@     �p@     �n@     �n@     @k@     `h@      h@     �a@     @b@     �\@     @Y@      Z@      Y@      Y@      U@      P@     �P@      I@      L@      L@     �M@     �D@      C@      8@      ;@      <@      >@      3@      6@      7@      8@      5@      *@      3@      *@      $@      .@       @       @      &@      @      "@      @      @      @              @       @      �?      @      @      @      @      �?      @              �?       @      �?               @              �?              �?               @              �?              �?              �?              �?      �?              �?              �?              �?      �?      �?      �?      �?      @      @      �?      @       @       @       @       @      @      @      @       @      @      @      *@      $@      @      @      $@      @      (@       @      @      "@      1@      3@      0@      (@      8@      7@      4@      >@      ;@      7@      :@      A@     �F@      J@     �H@      P@      F@     �M@     �R@     �O@     �U@     @[@     �\@     @_@     @^@     �a@      b@      f@     �a@     @f@     `l@      l@      q@     �q@     `n@     �P@      F@      0@      ,@       @       @        
�
conv5/biases_1*�	    �[�    �,T>      <@!   ��3C�)�!��hu�<2���u}��\�4�j�6Z�6��>?�J���8"uH���Ő�;F��`�}6D�p
T~�;�u 5�9��z��6��so쩾4�_"s�$1�7'_��+/��'v�V,����<�)���-�z�!�%�����i
�k���f��p���R����2!K�R���J�=��]���=��1���=nx6�X� >�`��>4��evk'>���<�)>�'v�V,>7'_��+/>_"s�$1>p��Dp�@>/�p`B>��Ő�;F>��8"uH>6��>?�J>������M>28���FP>H��'ϱS>��x��U>�������:�              �?               @              �?              �?              @              �?      �?       @              �?              �?              �?      �?              �?              �?              �?              �?       @              �?              �?       @              �?              �?        K}E