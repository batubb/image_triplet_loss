       �K"	   ���Abrain.Event:22DS��     ���	�Y���A"��
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
.conv1/weights/Initializer/random_uniform/shapeConst*
_output_shapes
:*%
valueB"             * 
_class
loc:@conv1/weights*
dtype0
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
,conv1/weights/Initializer/random_uniform/subSub,conv1/weights/Initializer/random_uniform/max,conv1/weights/Initializer/random_uniform/min*
_output_shapes
: *
T0* 
_class
loc:@conv1/weights
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
conv1/weights/AssignAssignconv1/weights(conv1/weights/Initializer/random_uniform*&
_output_shapes
: *
use_locking(*
T0* 
_class
loc:@conv1/weights*
validate_shape(
�
conv1/weights/readIdentityconv1/weights*&
_output_shapes
: *
T0* 
_class
loc:@conv1/weights
�
conv1/biases/Initializer/zerosConst*
_output_shapes
: *
valueB *    *
_class
loc:@conv1/biases*
dtype0
�
conv1/biases
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
conv1/biases/AssignAssignconv1/biasesconv1/biases/Initializer/zeros*
_class
loc:@conv1/biases*
validate_shape(*
_output_shapes
: *
use_locking(*
T0
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
,conv2/weights/Initializer/random_uniform/minConst*
valueB
 *��L�* 
_class
loc:@conv2/weights*
dtype0*
_output_shapes
: 
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
,conv2/weights/Initializer/random_uniform/subSub,conv2/weights/Initializer/random_uniform/max,conv2/weights/Initializer/random_uniform/min* 
_class
loc:@conv2/weights*
_output_shapes
: *
T0
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
conv2/biases/readIdentityconv2/biases*
_output_shapes
:@*
T0*
_class
loc:@conv2/biases
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
:���������K@*
	dilations

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
,conv3/weights/Initializer/random_uniform/mulMul6conv3/weights/Initializer/random_uniform/RandomUniform,conv3/weights/Initializer/random_uniform/sub*'
_output_shapes
:@�*
T0* 
_class
loc:@conv3/weights
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
conv3/weights/AssignAssignconv3/weights(conv3/weights/Initializer/random_uniform*
use_locking(*
T0* 
_class
loc:@conv3/weights*
validate_shape(*'
_output_shapes
:@�
�
conv3/weights/readIdentityconv3/weights*'
_output_shapes
:@�*
T0* 
_class
loc:@conv3/weights
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
conv3/biases/readIdentityconv3/biases*
_class
loc:@conv3/biases*
_output_shapes	
:�*
T0
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
:���������&�
�
model/conv3/BiasAddBiasAddmodel/conv3/Conv2Dconv3/biases/read*
data_formatNHWC*0
_output_shapes
:���������&�*
T0
n
model/conv3/conv3/ReluRelumodel/conv3/BiasAdd*0
_output_shapes
:���������&�*
T0
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
seed2 *
dtype0*(
_output_shapes
:��*

seed *
T0* 
_class
loc:@conv4/weights
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
(conv4/weights/Initializer/random_uniformAdd,conv4/weights/Initializer/random_uniform/mul,conv4/weights/Initializer/random_uniform/min*(
_output_shapes
:��*
T0* 
_class
loc:@conv4/weights
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
conv4/biases/AssignAssignconv4/biasesconv4/biases/Initializer/zeros*
use_locking(*
T0*
_class
loc:@conv4/biases*
validate_shape(*
_output_shapes	
:�
r
conv4/biases/readIdentityconv4/biases*
_class
loc:@conv4/biases*
_output_shapes	
:�*
T0
j
model/conv4/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
�
model/conv4/Conv2DConv2Dmodel/conv3/MaxPool2D/MaxPoolconv4/weights/read*
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
.conv5/weights/Initializer/random_uniform/shapeConst*
_output_shapes
:*%
valueB"            * 
_class
loc:@conv5/weights*
dtype0
�
,conv5/weights/Initializer/random_uniform/minConst*
_output_shapes
: *
valueB
 *���* 
_class
loc:@conv5/weights*
dtype0
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
valueB"      *
dtype0*
_output_shapes
:
�
model/conv5/Conv2DConv2Dmodel/conv4/MaxPool2D/MaxPoolconv5/weights/read*
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
model/conv5/BiasAddBiasAddmodel/conv5/Conv2Dconv5/biases/read*
data_formatNHWC*/
_output_shapes
:���������
*
T0
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
+model/Flatten/flatten/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
�
#model/Flatten/flatten/strided_sliceStridedSlicemodel/Flatten/flatten/Shape)model/Flatten/flatten/strided_slice/stack+model/Flatten/flatten/strided_slice/stack_1+model/Flatten/flatten/strided_slice/stack_2*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
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
N*
_output_shapes
:*
T0*

axis 
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
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME
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
model_1/conv2/Conv2DConv2Dmodel_1/conv1/MaxPool2D/MaxPoolconv2/weights/read*
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
model_1/conv2/MaxPool2D/MaxPoolMaxPoolmodel_1/conv2/conv2/Relu*
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
model_1/conv3/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
�
model_1/conv3/Conv2DConv2Dmodel_1/conv2/MaxPool2D/MaxPoolconv3/weights/read*
strides
*
data_formatNHWC*
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
model_1/conv4/Conv2DConv2Dmodel_1/conv3/MaxPool2D/MaxPoolconv4/weights/read*
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
:���������

�
model_1/conv5/BiasAddBiasAddmodel_1/conv5/Conv2Dconv5/biases/read*
data_formatNHWC*/
_output_shapes
:���������
*
T0
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
model_1/Flatten/flatten/ShapeShapemodel_1/conv5/MaxPool2D/MaxPool*
_output_shapes
:*
T0*
out_type0
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
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
T0*
Index0
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
model_2/conv1/dilation_rateConst*
_output_shapes
:*
valueB"      *
dtype0
�
model_2/conv1/Conv2DConv2Dnegative_inputconv1/weights/read*
use_cudnn_on_gpu(*
paddingSAME*0
_output_shapes
:���������2� *
	dilations
*
T0*
strides
*
data_formatNHWC
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
model_2/conv2/MaxPool2D/MaxPoolMaxPoolmodel_2/conv2/conv2/Relu*
data_formatNHWC*
strides
*
ksize
*
paddingSAME*/
_output_shapes
:���������&@*
T0
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
T0*
strides
*
data_formatNHWC*
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
model_2/conv3/MaxPool2D/MaxPoolMaxPoolmodel_2/conv3/conv3/Relu*
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
model_2/conv4/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
�
model_2/conv4/Conv2DConv2Dmodel_2/conv3/MaxPool2D/MaxPoolconv4/weights/read*0
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
}
subSubmodel/Flatten/flatten/Reshapemodel_1/Flatten/flatten/Reshape*
T0*(
_output_shapes
:����������
J
Pow/yConst*
valueB
 *   @*
dtype0*
_output_shapes
: 
I
PowPowsubPow/y*
T0*(
_output_shapes
:����������
W
Sum/reduction_indicesConst*
value	B :*
dtype0*
_output_shapes
: 
u
SumSumPowSum/reduction_indices*'
_output_shapes
:���������*
	keep_dims(*

Tidx0*
T0
C
SqrtSqrtSum*
T0*'
_output_shapes
:���������

sub_1Submodel/Flatten/flatten/Reshapemodel_2/Flatten/flatten/Reshape*
T0*(
_output_shapes
:����������
L
Pow_1/yConst*
valueB
 *   @*
dtype0*
_output_shapes
: 
O
Pow_1Powsub_1Pow_1/y*
T0*(
_output_shapes
:����������
Y
Sum_1/reduction_indicesConst*
value	B :*
dtype0*
_output_shapes
: 
{
Sum_1SumPow_1Sum_1/reduction_indices*'
_output_shapes
:���������*
	keep_dims(*

Tidx0*
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
MaximumMaximumadd	Maximum/y*'
_output_shapes
:���������*
T0
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
dtype0*
_output_shapes
: *
	container *
shape: *
shared_name 
�
Variable/AssignAssignVariableVariable/initial_value*
_output_shapes
: *
use_locking(*
T0*
_class
loc:@Variable*
validate_shape(
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
gradients/Mean_grad/ProdProdgradients/Mean_grad/Shape_1gradients/Mean_grad/Const*
	keep_dims( *

Tidx0*
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
: *
	keep_dims( *

Tidx0*
T0
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
:*
	keep_dims( *

Tidx0*
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
/gradients/Maximum_grad/tuple/control_dependencyIdentitygradients/Maximum_grad/Reshape(^gradients/Maximum_grad/tuple/group_deps*1
_class'
%#loc:@gradients/Maximum_grad/Reshape*'
_output_shapes
:���������*
T0
�
1gradients/Maximum_grad/tuple/control_dependency_1Identity gradients/Maximum_grad/Reshape_1(^gradients/Maximum_grad/tuple/group_deps*
T0*3
_class)
'%loc:@gradients/Maximum_grad/Reshape_1*
_output_shapes
: 
]
gradients/add_grad/ShapeShapesub_2*
_output_shapes
:*
T0*
out_type0
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
gradients/add_grad/SumSum/gradients/Maximum_grad/tuple/control_dependency(gradients/add_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
gradients/add_grad/ReshapeReshapegradients/add_grad/Sumgradients/add_grad/Shape*
Tshape0*'
_output_shapes
:���������*
T0
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
-gradients/add_grad/tuple/control_dependency_1Identitygradients/add_grad/Reshape_1$^gradients/add_grad/tuple/group_deps*
_output_shapes
: *
T0*/
_class%
#!loc:@gradients/add_grad/Reshape_1
^
gradients/sub_2_grad/ShapeShapeSqrt*
T0*
out_type0*
_output_shapes
:
b
gradients/sub_2_grad/Shape_1ShapeSqrt_1*
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
gradients/Sqrt_grad/SqrtGradSqrtGradSqrt-gradients/sub_2_grad/tuple/control_dependency*'
_output_shapes
:���������*
T0
�
gradients/Sqrt_1_grad/SqrtGradSqrtGradSqrt_1/gradients/sub_2_grad/tuple/control_dependency_1*'
_output_shapes
:���������*
T0
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
gradients/Sum_grad/addAddSum/reduction_indicesgradients/Sum_grad/Size*
T0*+
_class!
loc:@gradients/Sum_grad/Shape*
_output_shapes
: 
�
gradients/Sum_grad/modFloorModgradients/Sum_grad/addgradients/Sum_grad/Size*+
_class!
loc:@gradients/Sum_grad/Shape*
_output_shapes
: *
T0
�
gradients/Sum_grad/Shape_1Const*
valueB *+
_class!
loc:@gradients/Sum_grad/Shape*
dtype0*
_output_shapes
: 
�
gradients/Sum_grad/range/startConst*
_output_shapes
: *
value	B : *+
_class!
loc:@gradients/Sum_grad/Shape*
dtype0
�
gradients/Sum_grad/range/deltaConst*
value	B :*+
_class!
loc:@gradients/Sum_grad/Shape*
dtype0*
_output_shapes
: 
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
gradients/Sum_grad/FillFillgradients/Sum_grad/Shape_1gradients/Sum_grad/Fill/value*

index_type0*+
_class!
loc:@gradients/Sum_grad/Shape*
_output_shapes
: *
T0
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
gradients/Sum_grad/MaximumMaximum gradients/Sum_grad/DynamicStitchgradients/Sum_grad/Maximum/y*
_output_shapes
:*
T0*+
_class!
loc:@gradients/Sum_grad/Shape
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
gradients/Sum_grad/TileTilegradients/Sum_grad/Reshapegradients/Sum_grad/floordiv*(
_output_shapes
:����������*

Tmultiples0*
T0
_
gradients/Sum_1_grad/ShapeShapePow_1*
out_type0*
_output_shapes
:*
T0
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
gradients/Sum_1_grad/modFloorModgradients/Sum_1_grad/addgradients/Sum_1_grad/Size*
_output_shapes
: *
T0*-
_class#
!loc:@gradients/Sum_1_grad/Shape
�
gradients/Sum_1_grad/Shape_1Const*
_output_shapes
: *
valueB *-
_class#
!loc:@gradients/Sum_1_grad/Shape*
dtype0
�
 gradients/Sum_1_grad/range/startConst*
dtype0*
_output_shapes
: *
value	B : *-
_class#
!loc:@gradients/Sum_1_grad/Shape
�
 gradients/Sum_1_grad/range/deltaConst*
value	B :*-
_class#
!loc:@gradients/Sum_1_grad/Shape*
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
gradients/Sum_1_grad/MaximumMaximum"gradients/Sum_1_grad/DynamicStitchgradients/Sum_1_grad/Maximum/y*
_output_shapes
:*
T0*-
_class#
!loc:@gradients/Sum_1_grad/Shape
�
gradients/Sum_1_grad/floordivFloorDivgradients/Sum_1_grad/Shapegradients/Sum_1_grad/Maximum*-
_class#
!loc:@gradients/Sum_1_grad/Shape*
_output_shapes
:*
T0
�
gradients/Sum_1_grad/ReshapeReshapegradients/Sqrt_1_grad/SqrtGrad"gradients/Sum_1_grad/DynamicStitch*
T0*
Tshape0*0
_output_shapes
:������������������
�
gradients/Sum_1_grad/TileTilegradients/Sum_1_grad/Reshapegradients/Sum_1_grad/floordiv*

Tmultiples0*
T0*(
_output_shapes
:����������
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
(gradients/Pow_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/Pow_grad/Shapegradients/Pow_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
p
gradients/Pow_grad/mulMulgradients/Sum_grad/TilePow/y*
T0*(
_output_shapes
:����������
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
m
gradients/Pow_grad/PowPowsubgradients/Pow_grad/sub*
T0*(
_output_shapes
:����������
�
gradients/Pow_grad/mul_1Mulgradients/Pow_grad/mulgradients/Pow_grad/Pow*
T0*(
_output_shapes
:����������
�
gradients/Pow_grad/SumSumgradients/Pow_grad/mul_1(gradients/Pow_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
gradients/Pow_grad/ReshapeReshapegradients/Pow_grad/Sumgradients/Pow_grad/Shape*
T0*
Tshape0*(
_output_shapes
:����������
a
gradients/Pow_grad/Greater/yConst*
valueB
 *    *
dtype0*
_output_shapes
: 
{
gradients/Pow_grad/GreaterGreatersubgradients/Pow_grad/Greater/y*
T0*(
_output_shapes
:����������
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
gradients/Pow_grad/ones_likeFill"gradients/Pow_grad/ones_like/Shape"gradients/Pow_grad/ones_like/Const*(
_output_shapes
:����������*
T0*

index_type0
�
gradients/Pow_grad/SelectSelectgradients/Pow_grad/Greatersubgradients/Pow_grad/ones_like*
T0*(
_output_shapes
:����������
k
gradients/Pow_grad/LogLoggradients/Pow_grad/Select*
T0*(
_output_shapes
:����������
b
gradients/Pow_grad/zeros_like	ZerosLikesub*(
_output_shapes
:����������*
T0
�
gradients/Pow_grad/Select_1Selectgradients/Pow_grad/Greatergradients/Pow_grad/Loggradients/Pow_grad/zeros_like*
T0*(
_output_shapes
:����������
p
gradients/Pow_grad/mul_2Mulgradients/Sum_grad/TilePow*(
_output_shapes
:����������*
T0
�
gradients/Pow_grad/mul_3Mulgradients/Pow_grad/mul_2gradients/Pow_grad/Select_1*(
_output_shapes
:����������*
T0
�
gradients/Pow_grad/Sum_1Sumgradients/Pow_grad/mul_3*gradients/Pow_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
gradients/Pow_grad/Reshape_1Reshapegradients/Pow_grad/Sum_1gradients/Pow_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
g
#gradients/Pow_grad/tuple/group_depsNoOp^gradients/Pow_grad/Reshape^gradients/Pow_grad/Reshape_1
�
+gradients/Pow_grad/tuple/control_dependencyIdentitygradients/Pow_grad/Reshape$^gradients/Pow_grad/tuple/group_deps*(
_output_shapes
:����������*
T0*-
_class#
!loc:@gradients/Pow_grad/Reshape
�
-gradients/Pow_grad/tuple/control_dependency_1Identitygradients/Pow_grad/Reshape_1$^gradients/Pow_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/Pow_grad/Reshape_1*
_output_shapes
: 
_
gradients/Pow_1_grad/ShapeShapesub_1*
_output_shapes
:*
T0*
out_type0
_
gradients/Pow_1_grad/Shape_1Const*
_output_shapes
: *
valueB *
dtype0
�
*gradients/Pow_1_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/Pow_1_grad/Shapegradients/Pow_1_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
v
gradients/Pow_1_grad/mulMulgradients/Sum_1_grad/TilePow_1/y*
T0*(
_output_shapes
:����������
_
gradients/Pow_1_grad/sub/yConst*
_output_shapes
: *
valueB
 *  �?*
dtype0
e
gradients/Pow_1_grad/subSubPow_1/ygradients/Pow_1_grad/sub/y*
T0*
_output_shapes
: 
s
gradients/Pow_1_grad/PowPowsub_1gradients/Pow_1_grad/sub*(
_output_shapes
:����������*
T0
�
gradients/Pow_1_grad/mul_1Mulgradients/Pow_1_grad/mulgradients/Pow_1_grad/Pow*
T0*(
_output_shapes
:����������
�
gradients/Pow_1_grad/SumSumgradients/Pow_1_grad/mul_1*gradients/Pow_1_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
gradients/Pow_1_grad/ReshapeReshapegradients/Pow_1_grad/Sumgradients/Pow_1_grad/Shape*(
_output_shapes
:����������*
T0*
Tshape0
c
gradients/Pow_1_grad/Greater/yConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
gradients/Pow_1_grad/GreaterGreatersub_1gradients/Pow_1_grad/Greater/y*(
_output_shapes
:����������*
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

index_type0*(
_output_shapes
:����������
�
gradients/Pow_1_grad/SelectSelectgradients/Pow_1_grad/Greatersub_1gradients/Pow_1_grad/ones_like*
T0*(
_output_shapes
:����������
o
gradients/Pow_1_grad/LogLoggradients/Pow_1_grad/Select*(
_output_shapes
:����������*
T0
f
gradients/Pow_1_grad/zeros_like	ZerosLikesub_1*
T0*(
_output_shapes
:����������
�
gradients/Pow_1_grad/Select_1Selectgradients/Pow_1_grad/Greatergradients/Pow_1_grad/Loggradients/Pow_1_grad/zeros_like*
T0*(
_output_shapes
:����������
v
gradients/Pow_1_grad/mul_2Mulgradients/Sum_1_grad/TilePow_1*
T0*(
_output_shapes
:����������
�
gradients/Pow_1_grad/mul_3Mulgradients/Pow_1_grad/mul_2gradients/Pow_1_grad/Select_1*
T0*(
_output_shapes
:����������
�
gradients/Pow_1_grad/Sum_1Sumgradients/Pow_1_grad/mul_3,gradients/Pow_1_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
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
#!loc:@gradients/Pow_1_grad/Reshape*(
_output_shapes
:����������
�
/gradients/Pow_1_grad/tuple/control_dependency_1Identitygradients/Pow_1_grad/Reshape_1&^gradients/Pow_1_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/Pow_1_grad/Reshape_1*
_output_shapes
: 
u
gradients/sub_grad/ShapeShapemodel/Flatten/flatten/Reshape*
_output_shapes
:*
T0*
out_type0
y
gradients/sub_grad/Shape_1Shapemodel_1/Flatten/flatten/Reshape*
_output_shapes
:*
T0*
out_type0
�
(gradients/sub_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/sub_grad/Shapegradients/sub_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
gradients/sub_grad/SumSum+gradients/Pow_grad/tuple/control_dependency(gradients/sub_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
gradients/sub_grad/ReshapeReshapegradients/sub_grad/Sumgradients/sub_grad/Shape*
T0*
Tshape0*(
_output_shapes
:����������
�
gradients/sub_grad/Sum_1Sum+gradients/Pow_grad/tuple/control_dependency*gradients/sub_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
Z
gradients/sub_grad/NegNeggradients/sub_grad/Sum_1*
T0*
_output_shapes
:
�
gradients/sub_grad/Reshape_1Reshapegradients/sub_grad/Neggradients/sub_grad/Shape_1*
Tshape0*(
_output_shapes
:����������*
T0
g
#gradients/sub_grad/tuple/group_depsNoOp^gradients/sub_grad/Reshape^gradients/sub_grad/Reshape_1
�
+gradients/sub_grad/tuple/control_dependencyIdentitygradients/sub_grad/Reshape$^gradients/sub_grad/tuple/group_deps*-
_class#
!loc:@gradients/sub_grad/Reshape*(
_output_shapes
:����������*
T0
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
gradients/sub_1_grad/SumSum-gradients/Pow_1_grad/tuple/control_dependency*gradients/sub_1_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
gradients/sub_1_grad/ReshapeReshapegradients/sub_1_grad/Sumgradients/sub_1_grad/Shape*
T0*
Tshape0*(
_output_shapes
:����������
�
gradients/sub_1_grad/Sum_1Sum-gradients/Pow_1_grad/tuple/control_dependency,gradients/sub_1_grad/BroadcastGradientArgs:1*
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
4gradients/model_1/Flatten/flatten/Reshape_grad/ShapeShapemodel_1/conv5/MaxPool2D/MaxPool*
_output_shapes
:*
T0*
out_type0
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
4gradients/model/Flatten/flatten/Reshape_grad/ReshapeReshapegradients/AddN2gradients/model/Flatten/flatten/Reshape_grad/Shape*
Tshape0*/
_output_shapes
:���������*
T0
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
;gradients/model/conv5/BiasAdd_grad/tuple/control_dependencyIdentity8gradients/model/conv5/MaxPool2D/MaxPool_grad/MaxPoolGrad4^gradients/model/conv5/BiasAdd_grad/tuple/group_deps*K
_classA
?=loc:@gradients/model/conv5/MaxPool2D/MaxPool_grad/MaxPoolGrad*/
_output_shapes
:���������
*
T0
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
=gradients/model_2/conv5/BiasAdd_grad/tuple/control_dependencyIdentity:gradients/model_2/conv5/MaxPool2D/MaxPool_grad/MaxPoolGrad6^gradients/model_2/conv5/BiasAdd_grad/tuple/group_deps*M
_classC
A?loc:@gradients/model_2/conv5/MaxPool2D/MaxPool_grad/MaxPoolGrad*/
_output_shapes
:���������
*
T0
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
7gradients/model_1/conv5/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput*gradients/model_1/conv5/Conv2D_grad/ShapeNconv5/weights/read=gradients/model_1/conv5/BiasAdd_grad/tuple/control_dependency*
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
<gradients/model_1/conv5/Conv2D_grad/tuple/control_dependencyIdentity7gradients/model_1/conv5/Conv2D_grad/Conv2DBackpropInput5^gradients/model_1/conv5/Conv2D_grad/tuple/group_deps*J
_class@
><loc:@gradients/model_1/conv5/Conv2D_grad/Conv2DBackpropInput*0
_output_shapes
:���������
�*
T0
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
4gradients/model_2/conv5/Conv2D_grad/tuple/group_depsNoOp9^gradients/model_2/conv5/Conv2D_grad/Conv2DBackpropFilter8^gradients/model_2/conv5/Conv2D_grad/Conv2DBackpropInput
�
<gradients/model_2/conv5/Conv2D_grad/tuple/control_dependencyIdentity7gradients/model_2/conv5/Conv2D_grad/Conv2DBackpropInput5^gradients/model_2/conv5/Conv2D_grad/tuple/group_deps*J
_class@
><loc:@gradients/model_2/conv5/Conv2D_grad/Conv2DBackpropInput*0
_output_shapes
:���������
�*
T0
�
>gradients/model_2/conv5/Conv2D_grad/tuple/control_dependency_1Identity8gradients/model_2/conv5/Conv2D_grad/Conv2DBackpropFilter5^gradients/model_2/conv5/Conv2D_grad/tuple/group_deps*'
_output_shapes
:�*
T0*K
_classA
?=loc:@gradients/model_2/conv5/Conv2D_grad/Conv2DBackpropFilter
�
gradients/AddN_1AddN?gradients/model_1/conv5/BiasAdd_grad/tuple/control_dependency_1=gradients/model/conv5/BiasAdd_grad/tuple/control_dependency_1?gradients/model_2/conv5/BiasAdd_grad/tuple/control_dependency_1*C
_class9
75loc:@gradients/model_1/conv5/BiasAdd_grad/BiasAddGrad*
N*
_output_shapes
:*
T0
�
:gradients/model_1/conv4/MaxPool2D/MaxPool_grad/MaxPoolGradMaxPoolGradmodel_1/conv4/conv4/Relumodel_1/conv4/MaxPool2D/MaxPool<gradients/model_1/conv5/Conv2D_grad/tuple/control_dependency*
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
gradients/AddN_2AddN>gradients/model_1/conv5/Conv2D_grad/tuple/control_dependency_1<gradients/model/conv5/Conv2D_grad/tuple/control_dependency_1>gradients/model_2/conv5/Conv2D_grad/tuple/control_dependency_1*K
_classA
?=loc:@gradients/model_1/conv5/Conv2D_grad/Conv2DBackpropFilter*
N*'
_output_shapes
:�*
T0
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
0gradients/model_2/conv4/conv4/Relu_grad/ReluGradReluGrad:gradients/model_2/conv4/MaxPool2D/MaxPool_grad/MaxPoolGradmodel_2/conv4/conv4/Relu*0
_output_shapes
:����������*
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
;gradients/model/conv4/BiasAdd_grad/tuple/control_dependencyIdentity.gradients/model/conv4/conv4/Relu_grad/ReluGrad4^gradients/model/conv4/BiasAdd_grad/tuple/group_deps*0
_output_shapes
:����������*
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
:����������
�
?gradients/model_2/conv4/BiasAdd_grad/tuple/control_dependency_1Identity0gradients/model_2/conv4/BiasAdd_grad/BiasAddGrad6^gradients/model_2/conv4/BiasAdd_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/model_2/conv4/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:�
�
*gradients/model_1/conv4/Conv2D_grad/ShapeNShapeNmodel_1/conv3/MaxPool2D/MaxPoolconv4/weights/read* 
_output_shapes
::*
T0*
out_type0*
N
�
7gradients/model_1/conv4/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput*gradients/model_1/conv4/Conv2D_grad/ShapeNconv4/weights/read=gradients/model_1/conv4/BiasAdd_grad/tuple/control_dependency*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*0
_output_shapes
:����������*
	dilations

�
8gradients/model_1/conv4/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFiltermodel_1/conv3/MaxPool2D/MaxPool,gradients/model_1/conv4/Conv2D_grad/ShapeN:1=gradients/model_1/conv4/BiasAdd_grad/tuple/control_dependency*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*(
_output_shapes
:��*
	dilations
*
T0
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
5gradients/model/conv4/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput(gradients/model/conv4/Conv2D_grad/ShapeNconv4/weights/read;gradients/model/conv4/BiasAdd_grad/tuple/control_dependency*
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
6gradients/model/conv4/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFiltermodel/conv3/MaxPool2D/MaxPool*gradients/model/conv4/Conv2D_grad/ShapeN:1;gradients/model/conv4/BiasAdd_grad/tuple/control_dependency*
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
2gradients/model/conv4/Conv2D_grad/tuple/group_depsNoOp7^gradients/model/conv4/Conv2D_grad/Conv2DBackpropFilter6^gradients/model/conv4/Conv2D_grad/Conv2DBackpropInput
�
:gradients/model/conv4/Conv2D_grad/tuple/control_dependencyIdentity5gradients/model/conv4/Conv2D_grad/Conv2DBackpropInput3^gradients/model/conv4/Conv2D_grad/tuple/group_deps*
T0*H
_class>
<:loc:@gradients/model/conv4/Conv2D_grad/Conv2DBackpropInput*0
_output_shapes
:����������
�
<gradients/model/conv4/Conv2D_grad/tuple/control_dependency_1Identity6gradients/model/conv4/Conv2D_grad/Conv2DBackpropFilter3^gradients/model/conv4/Conv2D_grad/tuple/group_deps*
T0*I
_class?
=;loc:@gradients/model/conv4/Conv2D_grad/Conv2DBackpropFilter*(
_output_shapes
:��
�
*gradients/model_2/conv4/Conv2D_grad/ShapeNShapeNmodel_2/conv3/MaxPool2D/MaxPoolconv4/weights/read*
T0*
out_type0*
N* 
_output_shapes
::
�
7gradients/model_2/conv4/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput*gradients/model_2/conv4/Conv2D_grad/ShapeNconv4/weights/read=gradients/model_2/conv4/BiasAdd_grad/tuple/control_dependency*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*0
_output_shapes
:����������*
	dilations

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
:gradients/model_1/conv3/MaxPool2D/MaxPool_grad/MaxPoolGradMaxPoolGradmodel_1/conv3/conv3/Relumodel_1/conv3/MaxPool2D/MaxPool<gradients/model_1/conv4/Conv2D_grad/tuple/control_dependency*
data_formatNHWC*
strides
*
ksize
*
paddingSAME*0
_output_shapes
:���������&�*
T0
�
8gradients/model/conv3/MaxPool2D/MaxPool_grad/MaxPoolGradMaxPoolGradmodel/conv3/conv3/Relumodel/conv3/MaxPool2D/MaxPool:gradients/model/conv4/Conv2D_grad/tuple/control_dependency*
data_formatNHWC*
strides
*
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
N*(
_output_shapes
:��*
T0*K
_classA
?=loc:@gradients/model_1/conv4/Conv2D_grad/Conv2DBackpropFilter
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
0gradients/model_2/conv3/conv3/Relu_grad/ReluGradReluGrad:gradients/model_2/conv3/MaxPool2D/MaxPool_grad/MaxPoolGradmodel_2/conv3/conv3/Relu*
T0*0
_output_shapes
:���������&�
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
6gradients/model/conv3/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFiltermodel/conv2/MaxPool2D/MaxPool*gradients/model/conv3/Conv2D_grad/ShapeN:1;gradients/model/conv3/BiasAdd_grad/tuple/control_dependency*'
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
:���������K@*
T0
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
:gradients/model_2/conv2/MaxPool2D/MaxPool_grad/MaxPoolGradMaxPoolGradmodel_2/conv2/conv2/Relumodel_2/conv2/MaxPool2D/MaxPool<gradients/model_2/conv3/Conv2D_grad/tuple/control_dependency*
data_formatNHWC*
strides
*
ksize
*
paddingSAME*/
_output_shapes
:���������K@*
T0
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
.gradients/model/conv2/BiasAdd_grad/BiasAddGradBiasAddGrad.gradients/model/conv2/conv2/Relu_grad/ReluGrad*
data_formatNHWC*
_output_shapes
:@*
T0
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
=gradients/model/conv2/BiasAdd_grad/tuple/control_dependency_1Identity.gradients/model/conv2/BiasAdd_grad/BiasAddGrad4^gradients/model/conv2/BiasAdd_grad/tuple/group_deps*A
_class7
53loc:@gradients/model/conv2/BiasAdd_grad/BiasAddGrad*
_output_shapes
:@*
T0
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
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME
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
:���������K *
T0*J
_class@
><loc:@gradients/model_1/conv2/Conv2D_grad/Conv2DBackpropInput
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
:���������K 
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
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*/
_output_shapes
:���������K 
�
8gradients/model_2/conv2/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFiltermodel_2/conv1/MaxPool2D/MaxPool,gradients/model_2/conv2/Conv2D_grad/ShapeN:1=gradients/model_2/conv2/BiasAdd_grad/tuple/control_dependency*
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
4gradients/model_2/conv2/Conv2D_grad/tuple/group_depsNoOp9^gradients/model_2/conv2/Conv2D_grad/Conv2DBackpropFilter8^gradients/model_2/conv2/Conv2D_grad/Conv2DBackpropInput
�
<gradients/model_2/conv2/Conv2D_grad/tuple/control_dependencyIdentity7gradients/model_2/conv2/Conv2D_grad/Conv2DBackpropInput5^gradients/model_2/conv2/Conv2D_grad/tuple/group_deps*
T0*J
_class@
><loc:@gradients/model_2/conv2/Conv2D_grad/Conv2DBackpropInput*/
_output_shapes
:���������K 
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
:gradients/model_1/conv1/MaxPool2D/MaxPool_grad/MaxPoolGradMaxPoolGradmodel_1/conv1/conv1/Relumodel_1/conv1/MaxPool2D/MaxPool<gradients/model_1/conv2/Conv2D_grad/tuple/control_dependency*
paddingSAME*0
_output_shapes
:���������2� *
T0*
data_formatNHWC*
strides
*
ksize

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
0gradients/model_1/conv1/conv1/Relu_grad/ReluGradReluGrad:gradients/model_1/conv1/MaxPool2D/MaxPool_grad/MaxPoolGradmodel_1/conv1/conv1/Relu*
T0*0
_output_shapes
:���������2� 
�
.gradients/model/conv1/conv1/Relu_grad/ReluGradReluGrad8gradients/model/conv1/MaxPool2D/MaxPool_grad/MaxPoolGradmodel/conv1/conv1/Relu*0
_output_shapes
:���������2� *
T0
�
0gradients/model_2/conv1/conv1/Relu_grad/ReluGradReluGrad:gradients/model_2/conv1/MaxPool2D/MaxPool_grad/MaxPoolGradmodel_2/conv1/conv1/Relu*0
_output_shapes
:���������2� *
T0
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
=gradients/model_2/conv1/BiasAdd_grad/tuple/control_dependencyIdentity0gradients/model_2/conv1/conv1/Relu_grad/ReluGrad6^gradients/model_2/conv1/BiasAdd_grad/tuple/group_deps*0
_output_shapes
:���������2� *
T0*C
_class9
75loc:@gradients/model_2/conv1/conv1/Relu_grad/ReluGrad
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
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*0
_output_shapes
:���������2�
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
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*0
_output_shapes
:���������2�
�
6gradients/model/conv1/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilteranchor_input*gradients/model/conv1/Conv2D_grad/ShapeN:1;gradients/model/conv1/BiasAdd_grad/tuple/control_dependency*&
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
7gradients/model_2/conv1/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput*gradients/model_2/conv1/Conv2D_grad/ShapeNconv1/weights/read=gradients/model_2/conv1/BiasAdd_grad/tuple/control_dependency*
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
<gradients/model_2/conv1/Conv2D_grad/tuple/control_dependencyIdentity7gradients/model_2/conv1/Conv2D_grad/Conv2DBackpropInput5^gradients/model_2/conv1/Conv2D_grad/tuple/group_deps*0
_output_shapes
:���������2�*
T0*J
_class@
><loc:@gradients/model_2/conv1/Conv2D_grad/Conv2DBackpropInput
�
>gradients/model_2/conv1/Conv2D_grad/tuple/control_dependency_1Identity8gradients/model_2/conv1/Conv2D_grad/Conv2DBackpropFilter5^gradients/model_2/conv1/Conv2D_grad/tuple/group_deps*&
_output_shapes
: *
T0*K
_classA
?=loc:@gradients/model_2/conv1/Conv2D_grad/Conv2DBackpropFilter
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
8conv1/weights/Momentum/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:* 
_class
loc:@conv1/weights*%
valueB"             
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
(conv1/weights/Momentum/Initializer/zerosFill8conv1/weights/Momentum/Initializer/zeros/shape_as_tensor.conv1/weights/Momentum/Initializer/zeros/Const*&
_output_shapes
: *
T0* 
_class
loc:@conv1/weights*

index_type0
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
'conv1/biases/Momentum/Initializer/zerosConst*
dtype0*
_output_shapes
: *
_class
loc:@conv1/biases*
valueB *    
�
conv1/biases/Momentum
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
dtype0*
_output_shapes
:* 
_class
loc:@conv2/weights*%
valueB"          @   
�
.conv2/weights/Momentum/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: * 
_class
loc:@conv2/weights*
valueB
 *    
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
dtype0*
_output_shapes
:* 
_class
loc:@conv3/weights*%
valueB"      @   �   
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
(conv4/weights/Momentum/Initializer/zerosFill8conv4/weights/Momentum/Initializer/zeros/shape_as_tensor.conv4/weights/Momentum/Initializer/zeros/Const*
T0* 
_class
loc:@conv4/weights*

index_type0*(
_output_shapes
:��
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
conv4/weights/Momentum/AssignAssignconv4/weights/Momentum(conv4/weights/Momentum/Initializer/zeros*
validate_shape(*(
_output_shapes
:��*
use_locking(*
T0* 
_class
loc:@conv4/weights
�
conv4/weights/Momentum/readIdentityconv4/weights/Momentum*
T0* 
_class
loc:@conv4/weights*(
_output_shapes
:��
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
VariableV2*
shape:�*
dtype0*
_output_shapes	
:�*
shared_name *
_class
loc:@conv4/biases*
	container 
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
+Momentum/update_conv1/weights/ApplyMomentumApplyMomentumconv1/weightsconv1/weights/MomentumMomentum/learning_rategradients/AddN_10Momentum/momentum*
use_locking( *
T0* 
_class
loc:@conv1/weights*
use_nesterov(*&
_output_shapes
: 
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
+Momentum/update_conv5/weights/ApplyMomentumApplyMomentumconv5/weightsconv5/weights/MomentumMomentum/learning_rategradients/AddN_2Momentum/momentum*
T0* 
_class
loc:@conv5/weights*
use_nesterov(*'
_output_shapes
:�*
use_locking( 
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
save/Assign_3Assignconv1/weightssave/RestoreV2:3*
validate_shape(*&
_output_shapes
: *
use_locking(*
T0* 
_class
loc:@conv1/weights
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
save/Assign_5Assignconv2/biasessave/RestoreV2:5*
T0*
_class
loc:@conv2/biases*
validate_shape(*
_output_shapes
:@*
use_locking(
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
save/Assign_8Assignconv2/weights/Momentumsave/RestoreV2:8*
validate_shape(*&
_output_shapes
: @*
use_locking(*
T0* 
_class
loc:@conv2/weights
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
save/Assign_14Assignconv4/biases/Momentumsave/RestoreV2:14*
T0*
_class
loc:@conv4/biases*
validate_shape(*
_output_shapes	
:�*
use_locking(
�
save/Assign_15Assignconv4/weightssave/RestoreV2:15*
T0* 
_class
loc:@conv4/weights*
validate_shape(*(
_output_shapes
:��*
use_locking(
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
save/Assign_17Assignconv5/biasessave/RestoreV2:17*
use_locking(*
T0*
_class
loc:@conv5/biases*
validate_shape(*
_output_shapes
:
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
: "�|�G�(     W>�	�y���AJ��
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
model/conv1/BiasAddBiasAddmodel/conv1/Conv2Dconv1/biases/read*
data_formatNHWC*0
_output_shapes
:���������2� *
T0
n
model/conv1/conv1/ReluRelumodel/conv1/BiasAdd*0
_output_shapes
:���������2� *
T0
�
model/conv1/MaxPool2D/MaxPoolMaxPoolmodel/conv1/conv1/Relu*
data_formatNHWC*
strides
*
ksize
*
paddingSAME*/
_output_shapes
:���������K *
T0
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
conv2/biases/Initializer/zerosConst*
_class
loc:@conv2/biases*
valueB@*    *
dtype0*
_output_shapes
:@
�
conv2/biases
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
conv2/biases/AssignAssignconv2/biasesconv2/biases/Initializer/zeros*
use_locking(*
T0*
_class
loc:@conv2/biases*
validate_shape(*
_output_shapes
:@
q
conv2/biases/readIdentityconv2/biases*
_output_shapes
:@*
T0*
_class
loc:@conv2/biases
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
model/conv2/MaxPool2D/MaxPoolMaxPoolmodel/conv2/conv2/Relu*
ksize
*
paddingSAME*/
_output_shapes
:���������&@*
T0*
data_formatNHWC*
strides

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
dtype0*
_output_shapes
: * 
_class
loc:@conv3/weights*
valueB
 *�[q�
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
conv3/weights/readIdentityconv3/weights*
T0* 
_class
loc:@conv3/weights*'
_output_shapes
:@�
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
model/conv3/Conv2DConv2Dmodel/conv2/MaxPool2D/MaxPoolconv3/weights/read*
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
T0*
strides
*
data_formatNHWC*
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
shape:��*
dtype0*(
_output_shapes
:��*
shared_name * 
_class
loc:@conv4/weights*
	container 
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
model/conv4/Conv2DConv2Dmodel/conv3/MaxPool2D/MaxPoolconv4/weights/read*
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
dtype0*
_output_shapes
: * 
_class
loc:@conv5/weights*
valueB
 *���
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
,conv5/weights/Initializer/random_uniform/mulMul6conv5/weights/Initializer/random_uniform/RandomUniform,conv5/weights/Initializer/random_uniform/sub*
T0* 
_class
loc:@conv5/weights*'
_output_shapes
:�
�
(conv5/weights/Initializer/random_uniformAdd,conv5/weights/Initializer/random_uniform/mul,conv5/weights/Initializer/random_uniform/min*'
_output_shapes
:�*
T0* 
_class
loc:@conv5/weights
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
model/conv5/MaxPool2D/MaxPoolMaxPoolmodel/conv5/BiasAdd*
data_formatNHWC*
strides
*
ksize
*
paddingSAME*/
_output_shapes
:���������*
T0
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
+model/Flatten/flatten/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
�
#model/Flatten/flatten/strided_sliceStridedSlicemodel/Flatten/flatten/Shape)model/Flatten/flatten/strided_slice/stack+model/Flatten/flatten/strided_slice/stack_1+model/Flatten/flatten/strided_slice/stack_2*
T0*
Index0*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
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
dtype0*
_output_shapes
:*
valueB"      
�
model_1/conv1/Conv2DConv2Dpositive_inputconv1/weights/read*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*0
_output_shapes
:���������2� *
	dilations
*
T0
�
model_1/conv1/BiasAddBiasAddmodel_1/conv1/Conv2Dconv1/biases/read*
data_formatNHWC*0
_output_shapes
:���������2� *
T0
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
model_1/conv2/BiasAddBiasAddmodel_1/conv2/Conv2Dconv2/biases/read*
data_formatNHWC*/
_output_shapes
:���������K@*
T0
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
model_1/conv3/Conv2DConv2Dmodel_1/conv2/MaxPool2D/MaxPoolconv3/weights/read*0
_output_shapes
:���������&�*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME
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
model_1/conv5/Conv2DConv2Dmodel_1/conv4/MaxPool2D/MaxPoolconv5/weights/read*/
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
model_1/conv5/BiasAddBiasAddmodel_1/conv5/Conv2Dconv5/biases/read*
T0*
data_formatNHWC*/
_output_shapes
:���������

�
model_1/conv5/MaxPool2D/MaxPoolMaxPoolmodel_1/conv5/BiasAdd*
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
model_1/Flatten/flatten/ShapeShapemodel_1/conv5/MaxPool2D/MaxPool*
_output_shapes
:*
T0*
out_type0
u
+model_1/Flatten/flatten/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
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
%model_1/Flatten/flatten/strided_sliceStridedSlicemodel_1/Flatten/flatten/Shape+model_1/Flatten/flatten/strided_slice/stack-model_1/Flatten/flatten/strided_slice/stack_1-model_1/Flatten/flatten/strided_slice/stack_2*
end_mask *
_output_shapes
: *
T0*
Index0*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask 
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
T0*
strides
*
data_formatNHWC*
ksize
*
paddingSAME*/
_output_shapes
:���������K 
l
model_2/conv2/dilation_rateConst*
dtype0*
_output_shapes
:*
valueB"      
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
T0*
strides
*
data_formatNHWC*
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
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME
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
dtype0*
_output_shapes
:*
valueB"      
�
model_2/conv4/Conv2DConv2Dmodel_2/conv3/MaxPool2D/MaxPoolconv4/weights/read*
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
model_2/conv4/BiasAddBiasAddmodel_2/conv4/Conv2Dconv4/biases/read*
data_formatNHWC*0
_output_shapes
:����������*
T0
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
model_2/conv5/Conv2DConv2Dmodel_2/conv4/MaxPool2D/MaxPoolconv5/weights/read*/
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
T0*
strides
*
data_formatNHWC
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
}
subSubmodel/Flatten/flatten/Reshapemodel_1/Flatten/flatten/Reshape*(
_output_shapes
:����������*
T0
J
Pow/yConst*
valueB
 *   @*
dtype0*
_output_shapes
: 
I
PowPowsubPow/y*
T0*(
_output_shapes
:����������
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
SqrtSqrtSum*'
_output_shapes
:���������*
T0

sub_1Submodel/Flatten/flatten/Reshapemodel_2/Flatten/flatten/Reshape*(
_output_shapes
:����������*
T0
L
Pow_1/yConst*
valueB
 *   @*
dtype0*
_output_shapes
: 
O
Pow_1Powsub_1Pow_1/y*
T0*(
_output_shapes
:����������
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
gradients/Mean_grad/ShapeShapeMaximum*
_output_shapes
:*
T0*
out_type0
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
gradients/Mean_grad/floordivFloorDivgradients/Mean_grad/Prodgradients/Mean_grad/Maximum*
_output_shapes
: *
T0
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
gradients/add_grad/ReshapeReshapegradients/add_grad/Sumgradients/add_grad/Shape*'
_output_shapes
:���������*
T0*
Tshape0
�
gradients/add_grad/Sum_1Sum/gradients/Maximum_grad/tuple/control_dependency*gradients/add_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
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
^
gradients/sub_2_grad/ShapeShapeSqrt*
T0*
out_type0*
_output_shapes
:
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
gradients/sub_2_grad/Reshape_1Reshapegradients/sub_2_grad/Neggradients/sub_2_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:���������
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
gradients/Sum_grad/addAddSum/reduction_indicesgradients/Sum_grad/Size*
T0*+
_class!
loc:@gradients/Sum_grad/Shape*
_output_shapes
: 
�
gradients/Sum_grad/modFloorModgradients/Sum_grad/addgradients/Sum_grad/Size*
T0*+
_class!
loc:@gradients/Sum_grad/Shape*
_output_shapes
: 
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
gradients/Sum_grad/range/deltaConst*
dtype0*
_output_shapes
: *+
_class!
loc:@gradients/Sum_grad/Shape*
value	B :
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
: *+
_class!
loc:@gradients/Sum_grad/Shape*
value	B :
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
T0*+
_class!
loc:@gradients/Sum_grad/Shape*
N*
_output_shapes
:
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
T0*(
_output_shapes
:����������
_
gradients/Sum_1_grad/ShapeShapePow_1*
_output_shapes
:*
T0*
out_type0
�
gradients/Sum_1_grad/SizeConst*-
_class#
!loc:@gradients/Sum_1_grad/Shape*
value	B :*
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
gradients/Sum_1_grad/Shape_1Const*
dtype0*
_output_shapes
: *-
_class#
!loc:@gradients/Sum_1_grad/Shape*
valueB 
�
 gradients/Sum_1_grad/range/startConst*-
_class#
!loc:@gradients/Sum_1_grad/Shape*
value	B : *
dtype0*
_output_shapes
: 
�
 gradients/Sum_1_grad/range/deltaConst*-
_class#
!loc:@gradients/Sum_1_grad/Shape*
value	B :*
dtype0*
_output_shapes
: 
�
gradients/Sum_1_grad/rangeRange gradients/Sum_1_grad/range/startgradients/Sum_1_grad/Size gradients/Sum_1_grad/range/delta*-
_class#
!loc:@gradients/Sum_1_grad/Shape*
_output_shapes
:*

Tidx0
�
gradients/Sum_1_grad/Fill/valueConst*-
_class#
!loc:@gradients/Sum_1_grad/Shape*
value	B :*
dtype0*
_output_shapes
: 
�
gradients/Sum_1_grad/FillFillgradients/Sum_1_grad/Shape_1gradients/Sum_1_grad/Fill/value*
T0*-
_class#
!loc:@gradients/Sum_1_grad/Shape*

index_type0*
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
gradients/Sum_1_grad/TileTilegradients/Sum_1_grad/Reshapegradients/Sum_1_grad/floordiv*(
_output_shapes
:����������*

Tmultiples0*
T0
[
gradients/Pow_grad/ShapeShapesub*
_output_shapes
:*
T0*
out_type0
]
gradients/Pow_grad/Shape_1Const*
dtype0*
_output_shapes
: *
valueB 
�
(gradients/Pow_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/Pow_grad/Shapegradients/Pow_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
p
gradients/Pow_grad/mulMulgradients/Sum_grad/TilePow/y*
T0*(
_output_shapes
:����������
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
m
gradients/Pow_grad/PowPowsubgradients/Pow_grad/sub*(
_output_shapes
:����������*
T0
�
gradients/Pow_grad/mul_1Mulgradients/Pow_grad/mulgradients/Pow_grad/Pow*(
_output_shapes
:����������*
T0
�
gradients/Pow_grad/SumSumgradients/Pow_grad/mul_1(gradients/Pow_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
gradients/Pow_grad/ReshapeReshapegradients/Pow_grad/Sumgradients/Pow_grad/Shape*
T0*
Tshape0*(
_output_shapes
:����������
a
gradients/Pow_grad/Greater/yConst*
valueB
 *    *
dtype0*
_output_shapes
: 
{
gradients/Pow_grad/GreaterGreatersubgradients/Pow_grad/Greater/y*
T0*(
_output_shapes
:����������
e
"gradients/Pow_grad/ones_like/ShapeShapesub*
_output_shapes
:*
T0*
out_type0
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

index_type0*(
_output_shapes
:����������
�
gradients/Pow_grad/SelectSelectgradients/Pow_grad/Greatersubgradients/Pow_grad/ones_like*
T0*(
_output_shapes
:����������
k
gradients/Pow_grad/LogLoggradients/Pow_grad/Select*
T0*(
_output_shapes
:����������
b
gradients/Pow_grad/zeros_like	ZerosLikesub*(
_output_shapes
:����������*
T0
�
gradients/Pow_grad/Select_1Selectgradients/Pow_grad/Greatergradients/Pow_grad/Loggradients/Pow_grad/zeros_like*
T0*(
_output_shapes
:����������
p
gradients/Pow_grad/mul_2Mulgradients/Sum_grad/TilePow*
T0*(
_output_shapes
:����������
�
gradients/Pow_grad/mul_3Mulgradients/Pow_grad/mul_2gradients/Pow_grad/Select_1*
T0*(
_output_shapes
:����������
�
gradients/Pow_grad/Sum_1Sumgradients/Pow_grad/mul_3*gradients/Pow_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
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
!loc:@gradients/Pow_grad/Reshape*(
_output_shapes
:����������
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
v
gradients/Pow_1_grad/mulMulgradients/Sum_1_grad/TilePow_1/y*
T0*(
_output_shapes
:����������
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
s
gradients/Pow_1_grad/PowPowsub_1gradients/Pow_1_grad/sub*(
_output_shapes
:����������*
T0
�
gradients/Pow_1_grad/mul_1Mulgradients/Pow_1_grad/mulgradients/Pow_1_grad/Pow*
T0*(
_output_shapes
:����������
�
gradients/Pow_1_grad/SumSumgradients/Pow_1_grad/mul_1*gradients/Pow_1_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
gradients/Pow_1_grad/ReshapeReshapegradients/Pow_1_grad/Sumgradients/Pow_1_grad/Shape*(
_output_shapes
:����������*
T0*
Tshape0
c
gradients/Pow_1_grad/Greater/yConst*
dtype0*
_output_shapes
: *
valueB
 *    
�
gradients/Pow_1_grad/GreaterGreatersub_1gradients/Pow_1_grad/Greater/y*(
_output_shapes
:����������*
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

index_type0*(
_output_shapes
:����������
�
gradients/Pow_1_grad/SelectSelectgradients/Pow_1_grad/Greatersub_1gradients/Pow_1_grad/ones_like*
T0*(
_output_shapes
:����������
o
gradients/Pow_1_grad/LogLoggradients/Pow_1_grad/Select*
T0*(
_output_shapes
:����������
f
gradients/Pow_1_grad/zeros_like	ZerosLikesub_1*
T0*(
_output_shapes
:����������
�
gradients/Pow_1_grad/Select_1Selectgradients/Pow_1_grad/Greatergradients/Pow_1_grad/Loggradients/Pow_1_grad/zeros_like*(
_output_shapes
:����������*
T0
v
gradients/Pow_1_grad/mul_2Mulgradients/Sum_1_grad/TilePow_1*(
_output_shapes
:����������*
T0
�
gradients/Pow_1_grad/mul_3Mulgradients/Pow_1_grad/mul_2gradients/Pow_1_grad/Select_1*
T0*(
_output_shapes
:����������
�
gradients/Pow_1_grad/Sum_1Sumgradients/Pow_1_grad/mul_3,gradients/Pow_1_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
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
#!loc:@gradients/Pow_1_grad/Reshape*(
_output_shapes
:����������
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
gradients/sub_grad/SumSum+gradients/Pow_grad/tuple/control_dependency(gradients/sub_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
�
gradients/sub_grad/ReshapeReshapegradients/sub_grad/Sumgradients/sub_grad/Shape*
T0*
Tshape0*(
_output_shapes
:����������
�
gradients/sub_grad/Sum_1Sum+gradients/Pow_grad/tuple/control_dependency*gradients/sub_grad/BroadcastGradientArgs:1*

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
Tshape0*(
_output_shapes
:����������
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
8gradients/model_1/conv5/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFiltermodel_1/conv4/MaxPool2D/MaxPool,gradients/model_1/conv5/Conv2D_grad/ShapeN:1=gradients/model_1/conv5/BiasAdd_grad/tuple/control_dependency*
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
8gradients/model_2/conv5/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFiltermodel_2/conv4/MaxPool2D/MaxPool,gradients/model_2/conv5/Conv2D_grad/ShapeN:1=gradients/model_2/conv5/BiasAdd_grad/tuple/control_dependency*
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
0gradients/model_2/conv4/conv4/Relu_grad/ReluGradReluGrad:gradients/model_2/conv4/MaxPool2D/MaxPool_grad/MaxPoolGradmodel_2/conv4/conv4/Relu*0
_output_shapes
:����������*
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
:����������
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
N* 
_output_shapes
::*
T0*
out_type0
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
8gradients/model_1/conv4/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFiltermodel_1/conv3/MaxPool2D/MaxPool,gradients/model_1/conv4/Conv2D_grad/ShapeN:1=gradients/model_1/conv4/BiasAdd_grad/tuple/control_dependency*
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
N* 
_output_shapes
::*
T0*
out_type0
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
:gradients/model/conv4/Conv2D_grad/tuple/control_dependencyIdentity5gradients/model/conv4/Conv2D_grad/Conv2DBackpropInput3^gradients/model/conv4/Conv2D_grad/tuple/group_deps*
T0*H
_class>
<:loc:@gradients/model/conv4/Conv2D_grad/Conv2DBackpropInput*0
_output_shapes
:����������
�
<gradients/model/conv4/Conv2D_grad/tuple/control_dependency_1Identity6gradients/model/conv4/Conv2D_grad/Conv2DBackpropFilter3^gradients/model/conv4/Conv2D_grad/tuple/group_deps*
T0*I
_class?
=;loc:@gradients/model/conv4/Conv2D_grad/Conv2DBackpropFilter*(
_output_shapes
:��
�
*gradients/model_2/conv4/Conv2D_grad/ShapeNShapeNmodel_2/conv3/MaxPool2D/MaxPoolconv4/weights/read*
T0*
out_type0*
N* 
_output_shapes
::
�
7gradients/model_2/conv4/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput*gradients/model_2/conv4/Conv2D_grad/ShapeNconv4/weights/read=gradients/model_2/conv4/BiasAdd_grad/tuple/control_dependency*0
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
:gradients/model_1/conv3/MaxPool2D/MaxPool_grad/MaxPoolGradMaxPoolGradmodel_1/conv3/conv3/Relumodel_1/conv3/MaxPool2D/MaxPool<gradients/model_1/conv4/Conv2D_grad/tuple/control_dependency*
data_formatNHWC*
strides
*
ksize
*
paddingSAME*0
_output_shapes
:���������&�*
T0
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
:gradients/model_2/conv3/MaxPool2D/MaxPool_grad/MaxPoolGradMaxPoolGradmodel_2/conv3/conv3/Relumodel_2/conv3/MaxPool2D/MaxPool<gradients/model_2/conv4/Conv2D_grad/tuple/control_dependency*
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
8gradients/model_1/conv3/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFiltermodel_1/conv2/MaxPool2D/MaxPool,gradients/model_1/conv3/Conv2D_grad/ShapeN:1=gradients/model_1/conv3/BiasAdd_grad/tuple/control_dependency*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME*'
_output_shapes
:@�*
	dilations
*
T0
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
5gradients/model/conv3/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput(gradients/model/conv3/Conv2D_grad/ShapeNconv3/weights/read;gradients/model/conv3/BiasAdd_grad/tuple/control_dependency*/
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
6gradients/model/conv3/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFiltermodel/conv2/MaxPool2D/MaxPool*gradients/model/conv3/Conv2D_grad/ShapeN:1;gradients/model/conv3/BiasAdd_grad/tuple/control_dependency*
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
2gradients/model/conv3/Conv2D_grad/tuple/group_depsNoOp7^gradients/model/conv3/Conv2D_grad/Conv2DBackpropFilter6^gradients/model/conv3/Conv2D_grad/Conv2DBackpropInput
�
:gradients/model/conv3/Conv2D_grad/tuple/control_dependencyIdentity5gradients/model/conv3/Conv2D_grad/Conv2DBackpropInput3^gradients/model/conv3/Conv2D_grad/tuple/group_deps*/
_output_shapes
:���������&@*
T0*H
_class>
<:loc:@gradients/model/conv3/Conv2D_grad/Conv2DBackpropInput
�
<gradients/model/conv3/Conv2D_grad/tuple/control_dependency_1Identity6gradients/model/conv3/Conv2D_grad/Conv2DBackpropFilter3^gradients/model/conv3/Conv2D_grad/tuple/group_deps*'
_output_shapes
:@�*
T0*I
_class?
=;loc:@gradients/model/conv3/Conv2D_grad/Conv2DBackpropFilter
�
*gradients/model_2/conv3/Conv2D_grad/ShapeNShapeNmodel_2/conv2/MaxPool2D/MaxPoolconv3/weights/read*
N* 
_output_shapes
::*
T0*
out_type0
�
7gradients/model_2/conv3/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput*gradients/model_2/conv3/Conv2D_grad/ShapeNconv3/weights/read=gradients/model_2/conv3/BiasAdd_grad/tuple/control_dependency*/
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
:gradients/model_1/conv2/MaxPool2D/MaxPool_grad/MaxPoolGradMaxPoolGradmodel_1/conv2/conv2/Relumodel_1/conv2/MaxPool2D/MaxPool<gradients/model_1/conv3/Conv2D_grad/tuple/control_dependency*
data_formatNHWC*
strides
*
ksize
*
paddingSAME*/
_output_shapes
:���������K@*
T0
�
8gradients/model/conv2/MaxPool2D/MaxPool_grad/MaxPoolGradMaxPoolGradmodel/conv2/conv2/Relumodel/conv2/MaxPool2D/MaxPool:gradients/model/conv3/Conv2D_grad/tuple/control_dependency*
T0*
strides
*
data_formatNHWC*
ksize
*
paddingSAME*/
_output_shapes
:���������K@
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
0gradients/model_2/conv2/conv2/Relu_grad/ReluGradReluGrad:gradients/model_2/conv2/MaxPool2D/MaxPool_grad/MaxPoolGradmodel_2/conv2/conv2/Relu*/
_output_shapes
:���������K@*
T0
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
;gradients/model/conv2/BiasAdd_grad/tuple/control_dependencyIdentity.gradients/model/conv2/conv2/Relu_grad/ReluGrad4^gradients/model/conv2/BiasAdd_grad/tuple/group_deps*/
_output_shapes
:���������K@*
T0*A
_class7
53loc:@gradients/model/conv2/conv2/Relu_grad/ReluGrad
�
=gradients/model/conv2/BiasAdd_grad/tuple/control_dependency_1Identity.gradients/model/conv2/BiasAdd_grad/BiasAddGrad4^gradients/model/conv2/BiasAdd_grad/tuple/group_deps*
_output_shapes
:@*
T0*A
_class7
53loc:@gradients/model/conv2/BiasAdd_grad/BiasAddGrad
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
*gradients/model_1/conv2/Conv2D_grad/ShapeNShapeNmodel_1/conv1/MaxPool2D/MaxPoolconv2/weights/read*
N* 
_output_shapes
::*
T0*
out_type0
�
7gradients/model_1/conv2/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput*gradients/model_1/conv2/Conv2D_grad/ShapeNconv2/weights/read=gradients/model_1/conv2/BiasAdd_grad/tuple/control_dependency*
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
8gradients/model_1/conv2/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFiltermodel_1/conv1/MaxPool2D/MaxPool,gradients/model_1/conv2/Conv2D_grad/ShapeN:1=gradients/model_1/conv2/BiasAdd_grad/tuple/control_dependency*
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
6gradients/model/conv2/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFiltermodel/conv1/MaxPool2D/MaxPool*gradients/model/conv2/Conv2D_grad/ShapeN:1;gradients/model/conv2/BiasAdd_grad/tuple/control_dependency*&
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
7gradients/model_2/conv2/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput*gradients/model_2/conv2/Conv2D_grad/ShapeNconv2/weights/read=gradients/model_2/conv2/BiasAdd_grad/tuple/control_dependency*
paddingSAME*/
_output_shapes
:���������K *
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
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
T0*
strides
*
data_formatNHWC
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
7gradients/model_1/conv1/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput*gradients/model_1/conv1/Conv2D_grad/ShapeNconv1/weights/read=gradients/model_1/conv1/BiasAdd_grad/tuple/control_dependency*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*0
_output_shapes
:���������2�*
	dilations
*
T0
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
(gradients/model/conv1/Conv2D_grad/ShapeNShapeNanchor_inputconv1/weights/read*
N* 
_output_shapes
::*
T0*
out_type0
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
8gradients/model_2/conv1/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilternegative_input,gradients/model_2/conv1/Conv2D_grad/ShapeN:1=gradients/model_2/conv1/BiasAdd_grad/tuple/control_dependency*
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
_class
loc:@conv2/weights*
	container *
shape: @*
dtype0*&
_output_shapes
: @*
shared_name 
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
Momentum/momentumConst*
dtype0*
_output_shapes
: *
valueB
 *�p}?
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
save/filename/inputConst*
dtype0*
_output_shapes
: *
valueB Bmodel
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
save/Assign_5Assignconv2/biasessave/RestoreV2:5*
use_locking(*
T0*
_class
loc:@conv2/biases*
validate_shape(*
_output_shapes
:@
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
save/Assign_7Assignconv2/weightssave/RestoreV2:7*
use_locking(*
T0* 
_class
loc:@conv2/weights*
validate_shape(*&
_output_shapes
: @
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
save/Assign_10Assignconv3/biases/Momentumsave/RestoreV2:10*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0*
_class
loc:@conv3/biases
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
save/Assign_17Assignconv5/biasessave/RestoreV2:17*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*
_class
loc:@conv5/biases
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
dtype0*
_output_shapes
: *
valueB Bconv1/biases_1
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
conv5/biases:0conv5/biases/Assignconv5/biases/read:02 conv5/biases/Initializer/zeros:08~HV:�9      �,U	u"����A*�s

step    

lossa��>
�
conv1/weights_1*�	   `�D��    �H�?     `�@!   �u�@)DÛ�@2�����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0�I�I�)�(�+A�F�&�U�4@@�$��vV�R9��T7�����d�r�x?�x��1��a˲���[��a�Ϭ(���(����5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?�������:�              E@     `l@     �h@      d@     �b@     �`@     @a@     �Y@     @^@     �Y@     @R@      S@      M@      N@      J@      M@      H@     �B@     �C@      C@     �C@      ;@      ?@      8@      8@      9@      ,@      4@      :@      2@      $@      &@      @      ,@      "@      @      (@      @      @      @      @      @      @      @      @      @       @      �?       @      @      �?       @              @       @      �?      �?      @               @      �?              �?              �?              �?              �?              �?              �?              �?       @              �?      �?      �?              �?      @      @       @      @      �?       @      @       @      @      @      @      @      @      @      @      @      @      @      @      "@      2@      "@      .@      @      @      "@      3@      "@      *@      6@      3@      0@      2@      6@      @@      :@     �C@     �B@     �C@      D@      H@     �R@      K@     �M@     �Q@     @U@     �W@     �Z@      Y@     @\@     �a@      d@     �c@     `e@     �g@     �k@     �J@        
O
conv1/biases_1*=      @@2        �-���q=�������:              @@        
�
conv2/weights_1*�	    ����   ����?      �@!  @���@)�a��.E@2����g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ�XQ�þ��~��¾�*��ڽ�G&�$���*��ڽ>�[�=�k�>K+�E���>jqs&\��>��>M|K�>�_�T�l�>E��a�W�>�ѩ�-�>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?�������:�             �@     >�@     Н@     ��@     �@     ��@     d�@     Г@     ��@     ��@     P�@     ��@     `�@     �@     `�@     ��@     (�@     @~@     �|@     �w@     �w@     �s@     @t@     �r@      q@     �l@     @j@     �g@     �g@     �b@      c@     �`@      Z@     �V@     �X@     �U@     �P@     �N@      K@      I@     �L@     �B@      A@      C@     �D@      A@      A@      ?@      4@      8@      ;@      .@      3@      $@      6@      @      (@      *@       @      *@      @      "@      @      �?      @      @      @      @      @       @      @      @      @               @      �?      �?       @      @      �?      �?       @      �?      �?              �?      �?              @      �?       @              �?              �?      �?              �?              �?               @              �?               @              �?              �?      �?       @      @      �?      @      �?      �?      @              @      �?      �?      @      @              �?      @       @      @      @      �?       @      @      @      @      $@      @       @      (@       @       @      $@       @      .@      0@      2@      0@      2@      =@      ;@      6@      <@      :@      >@      7@     �K@      H@     �H@      G@     �S@     @Q@      Q@     �S@     @X@      V@      Y@     �]@     @_@     �b@     �d@     �d@      g@     �k@      i@     �m@     `q@     @t@      t@     �u@     �x@     �|@     p�@     �@      �@     H�@     �@     ��@     ��@     H�@     d�@     ��@     p�@     ܕ@     �@     ��@     ��@     p�@     �@     ��@        
O
conv2/biases_1*=      P@2        �-���q=�������:              P@        
�
conv3/weights_1*�	   �W+��   �	+�?      �@!  ���n@)�Z@�?U@2�����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%��_�T�l׾��>M|Kվ���]������|�~���MZ��K�>��|�~�>�*��ڽ>�[�=�k�>�iD*L��>E��a�W�>���%�>�uE����>��(���>a�Ϭ(�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?�������:�             ��@     H�@     �@     Z�@     ��@     2�@     d�@     �@     �@     H�@     ��@     ��@     D�@     �@     h�@     �@     �@     ��@     @�@     �@     �@     �}@     p{@     `z@     `x@     �s@     t@     pr@     p@     @o@      k@      f@     �e@     �b@     �a@      a@     �^@     �^@     @[@     �S@     �P@     @X@     �S@     �P@      G@      L@      G@      J@      E@      ;@      3@      F@      8@      2@      7@      3@      5@      9@      &@      2@      @      *@      "@      &@      $@      "@      @      *@       @      @      (@      @      @      @      @       @       @      @      �?      @      �?       @       @      �?       @      �?      @      �?      @      �?              �?      �?      �?              �?              �?               @              �?              �?              �?              �?              �?               @       @      @      �?              �?       @      @      �?       @      @      �?      @      @      @      @      @      $@      "@       @      *@      1@      @      @      .@      *@      .@      (@      0@      6@      3@      .@      6@      <@      3@      C@      A@     �A@      D@      E@      F@     �H@      G@     @S@      G@     �U@      W@     �U@      _@      `@      `@     �`@     �b@     @e@      h@     @j@     p@     0p@     pq@     �q@     �v@     w@     Py@     �{@     P~@     @�@     Ѓ@     8�@     ��@      �@     ��@      �@     p�@     ��@     ,�@     ؕ@     l�@     �@      �@     �@     ��@     �@     ��@     >�@     v�@     X�@        
O
conv3/biases_1*=      `@2        �-���q=�������:              `@        
�
conv4/weights_1*�	   �����    ���?      �@!   Y��8�)Q��u.&e@2�Ӗ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
�6�]���1��a˲�})�l a��ߊ4F��a�Ϭ(���(��澢f����jqs&\�ѾK+�E��ϾK+�E���>jqs&\��>�f����>��(���>�h���`�>�ߊ4F��>1��a˲?6�]��?����?f�ʜ�7
?>h�'�?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�������:�             �a@     �@     ��@     В@     �@     ��@     P�@     ��@     0�@      �@     (�@     ��@     �@     |@     0{@     �v@      x@     @v@     �s@     �q@     �n@     �j@     `h@     �e@     �f@     �c@      `@     �a@     �_@      \@     @T@     @V@      V@     �Q@      Q@      P@     �J@     �G@      I@     �B@      >@      :@     �A@      9@      6@      ?@      5@      7@      5@      .@      "@      3@      *@      @      .@      @      @      @      �?       @      @       @      @      @       @       @      @      @      @              @      �?      �?      �?      @      �?      �?               @      �?              �?      �?              @              �?              �?       @              �?              �?              �?              �?               @      �?      @       @               @       @      �?              @      @       @      �?      @      @      @      @      @       @      @      @       @      @      @      "@       @       @       @      &@      &@      *@      (@      (@      *@      .@      $@      *@      9@      7@      0@      3@      9@     �@@      7@     �F@      B@      K@     �N@      J@     �M@     �Q@     �S@      W@     @Y@     �W@     �[@     �[@     �a@     �b@     �c@     `f@     `i@     `k@     �l@     `r@      r@     @t@     0t@     �x@     0y@     P}@     ��@     ؁@     ؃@     ��@     Ї@     ��@     X�@     ��@     ��@     �@     �@     Е@     �_@        
O
conv4/biases_1*=      p@2        �-���q=�������:              p@        
�
conv5/weights_1*�	    j�¿   `���?      �@!   �
@)_/%�/I@2�	�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%�V6��u�w74���82���bȬ�0���VlQ.��[^:��"��S�F !�>h�'��f�ʜ�7
���[���FF�G �1��a˲?6�]��?�vV�R9?��ڋ?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?�������:�	             �j@      s@     p@     �p@     �n@     `k@     �e@     �e@     �e@     @]@      Z@      ]@     �Z@     @Y@      V@     �V@     �Q@     @Q@      P@     �J@      G@     �B@      H@      E@     �@@      @@      <@      :@      2@      =@      1@      1@      ,@      ,@      2@      "@      $@      *@       @      @      @       @      @      "@      @      @      @      @      @      �?       @       @      @      @      @      @       @               @      �?      @               @              �?      �?              �?              �?              �?              �?              �?               @              �?              �?              �?              �?               @      �?       @      �?      �?               @      �?      @      �?      @      �?      @       @      @      @      @      �?      @      @      "@      @      *@      $@      7@       @      0@      &@      (@      2@      8@      5@      ;@      2@      :@      9@     �B@      @@     �C@     �G@     �J@      M@     �H@     �P@     �Q@      W@      Z@      Y@     �\@     �]@     `a@     @d@     �c@     �b@     �g@     `i@      n@     Pq@     �o@     r@     �m@        
O
conv5/biases_1*=      <@2        �-���q=�������:              <@        �}�G�T      1�H	"�=���A*��

step  �?

loss���>
�
conv1/weights_1*�	   �HT��   �~Z�?     `�@! @����@)�tl�@2�����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���VlQ.��7Kaa+�I�I�)�(�+A�F�&��vV�R9��T7����5�i}1���d�r�})�l a��ߊ4F��8K�ߝ�>�h���`�>>h�'�?x?�x�?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?�������:�              E@      l@      i@     @d@     �b@     �`@      a@     �Y@     @]@     @Y@     @R@     �R@      O@      L@     �K@      L@      K@      B@     �A@      D@      D@      <@      <@      <@      4@      ;@      *@      5@      ;@      1@      &@      @      &@      $@       @      "@       @       @      @      @      @      @      @      @      @      @      @       @      @              @       @      @      @      �?              �?               @               @              �?              �?              �?              �?              �?              �?              �?      �?              �?       @       @      �?      @       @      �?       @      �?      �?      @               @              @      @      @       @      @       @      @      @      @      @      @      @      ,@      *@      (@      "@      $@      $@      @      2@      "@      0@      6@      0@      .@      3@      8@      <@      @@      ?@     �D@     �C@     �E@      I@     @Q@      L@     �J@     �Q@     @V@     �X@     @Z@     @W@     @^@     @a@     @d@     �c@     @e@     �g@     �k@     �K@        
�
conv1/biases_1*�	   @�2�    T8@?      @@!  ���[?)�{�u�>2���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$�ji6�9���.���vV�R9��T7�����d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[��O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a�iD*L�پ�_�T�l׾pz�w�7�>I��P=�>��[�?1��a˲?>h�'�?x?�x�?�5�i}1?�T7��?�vV�R9?�S�F !?�[^:��"?U�4@@�$?+A�F�&?�7Kaa+?��VlQ.?��bȬ�0?�u�w74?��%�V6?uܬ�@8?��%>��:?���#@?�!�A?�������:�              �?              �?              �?      �?              �?              �?               @              �?      �?              �?      �?              �?       @              �?              �?              �?              �?              �?              �?       @              �?              �?              �?      �?              �?      �?       @              �?        
�
conv2/weights_1*�	   �;���   ` ��?      �@! 	X��@)n�'/E@2����g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��8K�ߝ�a�Ϭ(龢f�����uE���⾄iD*L�پ�_�T�l׾��~]�[Ӿjqs&\�Ѿ['�?�;;�"�qʾ�XQ�þ��~��¾5�"�g���0�6�/n�����]������|�~��;9��R�>���?�ګ>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>��~]�[�>��>M|K�>�_�T�l�>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?�������:�             Ԑ@     P�@     ԝ@     l�@     H�@     x�@     d�@     ��@     ��@      �@      �@     ��@     x�@     І@     ��@     Ђ@     �@     �~@     �|@      x@     �v@     �s@     �t@     s@     �p@      l@     �j@     `g@     �g@      b@     @e@     �_@     �\@      U@     �W@     �U@     �P@      M@      J@      J@     �K@     �E@      ?@     �B@      D@      A@      C@     �@@      .@      3@      >@      1@      2@      ,@      0@      &@      0@      .@       @      (@      "@      @      @      @      @      @      @      @      @      @      @      �?       @      �?      �?      �?      �?      �?      @      @      �?               @               @              @               @              �?               @              �?              �?              �?              �?              �?              �?              @              �?      �?              �?              @       @      �?      @               @       @      �?      �?              �?      @      @      @              @      @      @      �?      �?       @      @       @      @      $@       @      *@      @      $@       @      �?      0@      .@      ,@      &@      6@      8@      ;@      ;@      ,@      ?@     �A@      =@      =@      C@      M@      E@      I@     @S@     �P@      R@      S@      W@     �W@     �X@     �]@     �_@     @b@      e@     @d@     �h@     @j@     �h@     �n@     �q@      s@     �t@     Pv@     x@     �|@     ��@     ��@     `�@     �@     P�@     H�@     ��@     @�@     p�@     ��@     x�@     ̕@     �@     ��@     ��@     |�@     Ƞ@     ��@        
�	
conv2/biases_1*�		    ��D�   ���=?      P@!  �'?�U?)5	T���>2�a�$��{E��T���C��!�A�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+��S�F !�ji6�9���.���T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
��������[���FF�G ��h���`�8K�ߝ���(��澢f����E��a�Wܾ�iD*L�پK+�E��Ͼ['�?�;�[�=�k���*��ڽ�G&�$��5�"�g���0�6�/n����n�����豪}0ڰ�[#=�؏�������~�        �-���q=��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>O�ʗ��>>�?�s��>>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?��%>��:?d�\D�X=?���#@?�������:�              �?      �?               @              �?      �?      �?      �?              �?      �?              �?               @      �?      �?      �?              �?               @              �?              �?              �?              �?              �?      �?              �?              �?               @              �?      �?              �?              �?              �?       @              @      �?      @              �?      �?      @      �?       @               @       @              �?      �?      �?              �?      �?        
�
conv3/weights_1*�	    s3��   ��:�?      �@! ؾ&g�@)~F}+@U@2�����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(��uE���⾮��%ᾙѩ�-߾�_�T�l׾��>M|KվK+�E��Ͼ['�?�;;�"�qʾ
�/eq
Ⱦ����ž���]������|�~��G&�$�>�*��ڽ>�[�=�k�>jqs&\��>��~]�[�>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?�������:�             ��@     D�@     �@     \�@     |�@     <�@     t�@     ��@     �@     h�@     ��@     ��@     @�@     �@     (�@     �@     0�@     ؄@     ��@     ��@     �@     ~@     `{@     �z@     Px@     �s@     @t@     �r@     p@     �n@     �k@     `f@     @e@     �b@     �`@      b@     �^@      `@     @[@     �Q@     �S@     �V@     �R@     �Q@     �G@      L@     �G@     �G@     �E@      ;@      8@     �A@      ?@      3@      6@      6@      6@      7@      "@      ,@      &@      .@      ,@      $@      &@      @       @      $@      $@      @      @      @      @      @      @      @      @      �?       @      @       @       @      �?      @       @              �?      �?              �?              �?       @              �?              �?      �?              �?               @              �?      �?              �?              �?               @              �?       @      @      �?       @       @      @       @      �?      @              @      @      @       @      @      "@      @      @      @      "@      ,@      "@      (@      .@      (@      .@      (@      1@      *@      6@      1@      3@     �@@      6@      A@      @@      A@     �F@     �F@     �B@      G@      I@     @S@      I@      U@     �V@      V@      ]@      a@     �_@     @a@     �b@      d@     @i@     `j@      o@      p@     pq@     �q@     pw@     �v@      y@     @|@      ~@     `�@     ��@     0�@     ��@     ��@     ��@     P�@     X�@     t�@     8�@     ؕ@     p�@      �@     ��@     �@     ��@     �@     ��@     <�@     x�@     ��@        
�
conv3/biases_1*�	   @��?�    ��??      `@!  `ş�b?)ťP�+��>2����#@�d�\D�X=���%>��:���%�V6��u�w74���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%��_�T�l׾��>M|KվG&�$��5�"�g����MZ��K���u��gr��        �-���q=豪}0ڰ>��n����>��~���>�XQ��>��>M|K�>�_�T�l�>��(���>a�Ϭ(�>8K�ߝ�>I��P=�>��Zr[v�>O�ʗ��>�FF�G ?��[�?1��a˲?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�������:�              �?      �?              �?               @      @       @      �?      @      �?      �?       @       @       @      @       @              �?       @      @       @       @      �?              @       @              �?               @      �?      �?       @      �?               @              �?       @      �?              �?              �?              �?              @              �?              �?              �?              �?      �?              �?      @              �?      �?              �?      @      @      �?               @      �?      �?      @       @       @       @      @      �?      @       @      �?      �?      �?      @       @      �?              �?        
�
conv4/weights_1*�	   �9��   `M�?      �@! �MY�v8�)��A&e@2�Ӗ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.���T7����5�i}1�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲�O�ʗ�����Zr[v��})�l a��ߊ4F��h���`�8K�ߝ���(��澢f������~]�[Ӿjqs&\�Ѿ�����>
�/eq
�>K+�E���>jqs&\��>�f����>��(���>�h���`�>�ߊ4F��>pz�w�7�>I��P=�>6�]��?����?f�ʜ�7
?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�������:�             �a@     �@     ��@     Ԓ@     �@     ��@     `�@     ��@     8�@     �@     8�@     ��@     �@     |@     P{@     pv@     Px@      v@     �s@     �q@     �n@     `j@     �h@     �e@     �f@     `c@     �_@     �a@     �_@     �[@     �T@     @V@     @V@     �Q@     �Q@     �N@      J@     �H@     �I@      B@      >@      ;@     �A@      7@      8@      ?@      2@      :@      2@      2@      @      3@      *@      @      ,@      @      @      @      �?      @      @       @      @      @      @      @      @      @      @      �?      @               @       @      @               @              �?      �?               @      �?              �?              �?              �?               @              �?              �?              �?              �?              �?              �?               @       @               @              �?       @              @      �?      @      @      �?      @      @      @      @       @      @      @              @      @      @       @      &@      @      "@      &@      (@      $@      ,@      *@      &@      0@      "@      (@      <@      6@      0@      3@      :@      @@      7@      G@     �A@      J@      O@      J@     �M@      R@     @S@     �V@     �Y@     �W@     �[@      \@      a@     �b@      d@     @f@     �i@     @k@     `l@     �r@     r@     0t@     @t@     Px@     �y@     @}@     ��@     ؁@     �@     �@     ��@     ��@     `�@     ��@     ��@     �@     �@     ؕ@     @_@        
�
conv4/biases_1*�	   ��J�   `�I??      p@!���DK?)����\��>2�
IcD���L��qU���I��!�A����#@�d�\D�X=���%>��:���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a�h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%���~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;��~��¾�[�=�k���*��ڽ�G&�$����������?�ګ��5�L�����]������|�~��RT��+��y�+pm��mm7&c��`���nx6�X� ��f׽r����tO����f;H�\Q����-��J�'j��p���1���=��]����/�4��ݟ��uy�z������Qu�R"�PæҭUݽH����ڽ���X>ؽ��
"
ֽ�|86	Խ�!p/�^˽�d7���Ƚ�b1�ĽK?�\��½�
6������Bb�!澽5%�����Į#�������/��        �-���q=����/�=�Į#��=5%���=�Bb�!�=K?�\���=�b1��=�|86	�=��
"
�=���X>�=H�����=PæҭU�=�Qu�R"�=i@4[��=z�����=ݟ��uy�=�/�4��==��]���=��1���='j��p�=��-��J�=�K���=�9�e��=����%�=f;H�\Q�=�tO���=�f׽r��=nx6�X� >�`��>�mm7&c>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>jqs&\��>��~]�[�>E��a�W�>�ѩ�-�>���%�>�f����>��(���>a�Ϭ(�>�h���`�>�ߊ4F��>})�l a�>O�ʗ��>>�?�s��>�FF�G ?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�������:�
              �?              �?              �?               @      �?               @       @      @      �?      @      @      @      �?      �?      �?      @              @      @      �?              @               @              �?      �?              �?      �?               @      �?      �?              �?      �?              �?              �?              �?      �?      �?              �?              �?      �?              �?               @              �?      @      �?               @              �?      @               @              �?              @      �?      @              �?              �?              �?      �?              �?              O@              �?              �?               @              @              �?              @               @              @      �?              �?      �?      �?      �?               @              �?              �?       @              �?      �?              �?               @              �?      �?              @      @              @      �?              �?       @              @      �?      @       @              @      @      @      �?      �?       @       @      �?       @      @       @      �?       @      @      �?       @       @      �?      �?       @               @        
�
conv5/weights_1*�	    j�¿   `���?      �@!  @F9q@)����\/I@2�	�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0��[^:��"��S�F !�>h�'��f�ʜ�7
��XQ��>�����>1��a˲?6�]��?�vV�R9?��ڋ?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?�������:�	             �j@      s@     p@     �p@     �n@     `k@     �e@     �e@     �e@     �]@     @Y@     @]@      [@     @Y@      V@      W@     @Q@     @Q@      P@     �J@      G@     �B@      H@      E@      A@      @@      <@      9@      2@      =@      1@      1@      ,@      ,@      3@      @      &@      *@       @      @      @       @      @      $@      @      @       @      @      @      �?       @       @      @      @       @      @       @               @              @       @              �?       @               @              �?              �?              �?              �?              �?              �?      �?      �?              �?              �?              �?               @       @      �?       @              �?       @      �?       @      @      �?      @      @      @      @      @      �?      @       @      "@      @      *@      $@      7@       @      .@      *@      (@      0@      9@      5@      :@      2@      ;@      9@      B@     �@@     �C@     �G@      J@     �M@      H@      Q@     �Q@      W@      Z@     @Y@     @\@     �]@     `a@      d@     �c@      c@     �g@     `i@      n@     @q@      p@      r@     �m@        
�
conv5/biases_1*�	   ����   �)z>      <@!   N0�>)��ml5R<2��#���j�Z�TA[�����"�RT��+������%���9�e����K��󽉊-��J�'j��p���1���=��]����Qu�R"�PæҭUݽH����ڽ���X>ؽ��
"
ֽ�|86	Խ�b1�ĽK?�\��½�Į#�������/������/�=�Į#��=K?�\���=�b1��=�|86	�=��
"
�=PæҭU�=�Qu�R"�=z�����=ݟ��uy�=�/�4��==��]���=��1���='j��p�=�K���=�9�e��=����%�=f;H�\Q�=�tO���=�f׽r��=�`��>�mm7&c>�#���j>�J>�������:�              �?              �?              �?              �?      �?      �?      �?              �?              �?              �?              �?              �?              �?              �?              �?              �?               @              �?              �?              �?      �?      �?              @              �?              �?        ���q�V      ���o	�w����A*�

step   @

loss�7�>
�
conv1/weights_1*�	    ����   ��o�?     `�@! `z`@)}�;bu!@2�	����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@���%>��:�uܬ�@8��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r���[���FF�G �6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?��ڋ?�.�?ji6�9�?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?�������:�	              F@     �k@     �h@      e@     @b@     ``@      a@      [@     @[@      [@     �R@      S@     �K@      O@     �M@      K@      G@      A@      D@     �F@      C@      8@      @@      9@      7@      :@      1@      6@      4@      2@      &@      @      &@      "@      @      $@      @      &@       @      @      @      @       @      @       @      @       @      @      @       @      @              @              �?      @       @               @              �?              �?      @               @               @      �?      �?              �?              �?              �?      �?              �?              �?      �?               @      �?      �?      �?      �?       @              �?              @              @      @      @      �?      @      �?      @      @      @      @      @      &@      @      ,@      @      $@      *@      @      @      0@      *@      .@      (@      2@      6@      .@      4@      ;@      3@      B@     �@@      ?@     �G@      F@     �M@     �K@     �L@      O@     �P@     �V@      X@     @Z@     �V@      `@     @a@     �c@     �c@     �e@     @g@      l@     �I@        
�
conv1/biases_1*�	   @p�I�    ��\?      @@!  `�ǻt?)����IP�>2��qU���I�
����G��!�A����#@�d�\D�X=���%>��:��u�w74���82���bȬ�0���VlQ.�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9����ڋ��vV�R9�>�?�s���O�ʗ�����Zr[v��I��P=��E��a�Wܾ�iD*L�پ�XQ�þ��~��¾+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?uܬ�@8?��%>��:?d�\D�X=?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?�m9�H�[?E��{��^?�������:�              �?              �?               @              �?              �?              �?              �?       @      �?              �?              �?              �?              �?              �?              �?       @              �?      �?       @              �?      �?              �?              �?      @              �?              �?        
�
conv2/weights_1*�	    �᩿   @3ȩ?      �@! ����@)�j�m�/E@2����g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�_�T�l׾��>M|Kվ�XQ�þ��~��¾�u��gr��R%������豪}0ڰ>��n����>0�6�/n�>5�"�g��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?�������:�             ̐@     T�@     ĝ@     p�@     @�@     ��@     D�@     ԓ@     ��@     ��@     �@     ��@     ��@     ��@     ؃@     ��@     (�@     �~@     �|@     0x@     �v@     �s@     @u@     0r@     `p@      n@     @i@     �h@     �g@      b@     �d@      ^@     @]@      V@     @V@      U@     �R@      O@      H@     �C@      N@     �A@     �D@      D@      D@     �D@      <@      :@      ,@      >@      4@      7@      1@      (@      .@      "@      3@      (@      ,@      ,@       @       @      @      @      $@      @      @      @      @      @      �?       @      @      �?      @       @      �?      �?       @       @               @              �?              �?              �?      �?      �?              �?              �?              �?              �?              �?              �?              �?      �?               @      �?              �?      �?              �?              �?      �?               @      @      �?      @       @      �?       @      �?      @       @      @      �?               @      �?      @      @      @      @       @      "@      @      &@      @      $@      @      &@      @      $@      (@      (@      1@      3@      ,@      5@      9@      5@      ;@      ?@      8@     �@@      ?@     �B@     �H@     �I@     �K@     �S@      P@      S@     �R@     �T@     �X@     �Y@     �]@      _@     �c@     @d@      d@     �h@      j@     �h@     �n@     �q@     �s@      t@     �v@     �w@     �|@     ��@     Ѐ@     (�@     8�@     H�@     p�@     ��@     @�@     h�@     ��@     T�@     ��@     �@     ��@     ��@     X�@     ؠ@     ��@        
�

conv2/biases_1*�
	   �xV�   ��MT?      P@!  	��3h?)�}kW��>2�ܗ�SsW�<DKc��T�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8��u�w74���82���bȬ�0�I�I�)�(�+A�F�&��[^:��"��S�F !�ji6�9���.���5�i}1���d�r�f�ʜ�7
������6�]���O�ʗ�����Zr[v��I��P=��pz�w�7���ߊ4F��h���`���(��澢f����
�/eq
Ⱦ����ž�XQ�þ���?�ګ�;9��R��u��6
��K���7��        �-���q=G&�$�>�*��ڽ>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>f�ʜ�7
?>h�'�?�vV�R9?��ڋ?�.�?ji6�9�?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?�lDZrS?<DKc��T?�������:�               @              �?              �?              �?              �?      �?      �?              �?               @      �?              �?               @               @              �?              �?      �?              �?              �?              �?              �?              �?      �?              �?              �?              �?              �?              �?              �?              �?      �?              �?              �?              �?               @              �?      �?       @      �?      �?      �?      �?       @      @       @      @      �?               @      �?      �?      �?              �?        
�
conv3/weights_1*�	    SK��   �}Q�?      �@! ��w�@)ah:�x@U@2�����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��8K�ߝ�a�Ϭ(���(����uE���⾮��%��_�T�l׾��>M|Kվ��~]�[Ӿ����ž�XQ�þ�5�L�����]������|�~���_�T�l�>�iD*L��>�uE����>�f����>��(���>a�Ϭ(�>�h���`�>�ߊ4F��>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?�������:�             X�@     :�@     ��@     T�@     |�@     <�@     T�@     �@     D�@     D�@     ��@     ȓ@     �@     ��@     ��@     X�@      �@     �@     ؃@      �@     �@     ~@      {@     �z@     `x@     ps@     �t@     0r@     Pp@     �m@      l@     `f@     `e@     �b@     ``@      b@     @^@     ``@     @[@     @S@     @R@     �V@     �Q@     �Q@     �F@     �L@     �F@      L@      A@      :@      A@      A@      6@      8@      4@      1@      <@      ;@      (@      &@      *@      ,@      $@      *@      "@      &@       @      @      @      @      @      @      @      $@       @       @      �?      @       @      �?       @      @      @       @       @              �?               @       @              �?               @      �?              �?              �?      �?              �?              �?              �?              �?               @      @      �?       @              @       @      @      @      �?      @      @       @      @      @      @      (@      @      @      "@      $@      (@      0@      "@      0@      ,@      1@      (@      2@      0@      0@      5@      ?@      :@     �B@      >@      C@     �@@      G@      G@     �F@      I@     @Q@      M@      R@     �W@     @W@     �[@     `a@     �_@     @a@     @c@      e@     �g@     �j@      o@     Pp@      q@     �q@     �w@     �v@     Py@     P|@     �}@     ��@     �@     ��@     ��@     ��@     ��@     @�@     d�@     d�@     L�@     ȕ@     |�@     �@     ��@     �@     ��@     �@     z�@     @�@     v�@     ��@        
�
conv3/biases_1*�	   @ʌO�   @[?      `@!  h��v?)O�I�G�>2�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(龄iD*L�پ�_�T�l׾        �-���q=5�"�g��>G&�$�>�[�=�k�>��~���>�uE����>�f����>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?�������:�              �?       @              �?      �?      �?              �?       @               @      �?       @      @      �?              �?       @       @      @      �?      �?      @       @       @       @       @               @      �?       @      �?      �?      @      �?               @      �?      �?              �?              �?       @      �?      �?      �?              �?              @              �?              �?              �?              �?               @      �?              �?               @      �?      �?      �?      �?      �?      �?               @      �?      @      @      @       @              �?      @       @       @      �?      �?      @              @      �?      �?      @       @      �?              �?              �?        
�
conv4/weights_1*�	   ����    ��?      �@! �FT�I8�)�b�tl&e@2�Ӗ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
�6�]���1��a˲�})�l a��ߊ4F��h���`�8K�ߝ���(��澢f���侮��%ᾙѩ�-߾�iD*L�پ�_�T�l׾��~]�[Ӿjqs&\�ѾK+�E���>jqs&\��>�f����>��(���>�h���`�>�ߊ4F��>��Zr[v�>O�ʗ��>����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�������:�             �a@     �@     ��@     Ԓ@     �@     Ȏ@     @�@     ��@     0�@      �@     @�@     ��@     �@     `|@     {@     Pv@     `x@     v@      t@     pq@     �n@     �j@     @h@     �e@     �f@     �c@      _@      b@     �_@      \@      T@     @V@     �V@     �Q@      Q@      N@      L@     �F@     �H@     �E@      =@      :@      B@      6@      :@      <@      0@      :@      2@      4@      @      3@      *@      @      ,@      @      @      @      �?      @      �?      @      @      @      @      @      @      @       @       @       @               @       @      @      �?              @      �?      �?      �?       @              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?              @              �?               @      �?       @      �?       @      �?      @       @      @       @      @       @       @      @       @      @              @      @      "@      @      $@       @      &@      $@      $@      &@      0@      *@      (@      ,@      $@      &@      ;@      5@      1@      4@      8@      @@      8@     �G@      B@      I@     �O@      J@     �L@      R@     �S@      W@     @Y@     �W@      \@     �\@     �`@     @c@      d@      f@     `i@     @k@     �l@     pr@      r@     @t@     0t@     `x@     py@     @}@     ��@     ؁@     ��@     ��@     ��@     ��@     8�@     ��@     ��@     �@     �@     ��@     �_@        
�
conv4/biases_1*�	   ��GW�   ��yQ?      p@!.v�ni?)�&�>2���bB�SY�ܗ�SsW�k�1^�sO�IcD���L��qU���I�
����G��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��8K�ߝ�a�Ϭ(��uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��~]�[Ӿjqs&\�Ѿ����ž�XQ�þ�MZ��K���u��gr��[#=�؏�������~���-�z�!�%�����i
�k���f��p�Łt�=	�2!K�R���J��#���j�Z�TA[�����"�RT��+���mm7&c��`���nx6�X� ��f׽r����tO����f;H�\Q������%����-��J�'j��p���1���=��]����/�4��i@4[���Qu�R"���
"
ֽ�|86	Խ;3���н��.4NνK?�\��½�
6�����        �-���q=�b1��=��؜��=PæҭU�=�Qu�R"�=�/�4��==��]���=��1���='j��p�=��-��J�=�K���=�9�e��=����%�=f;H�\Q�=�f׽r��=nx6�X� >�mm7&c>y�+pm>RT��+�>���">Z�TA[�>�#���j>2!K�R�>��R���>��-�z�!>4�e|�Z#>��o�kJ%>7'_��+/>_"s�$1>39W$:��>R%�����>�*��ڽ>�[�=�k�>��~���>
�/eq
�>;�"�q�>K+�E���>jqs&\��>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>���%�>�uE����>�f����>��(���>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?1��a˲?6�]��?����?f�ʜ�7
?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?�������:�              �?              �?              �?              @      �?              @      @      �?      �?      �?      �?       @      @       @               @      @              �?       @      �?      @              �?              �?       @      @      �?              �?      �?       @      �?               @              �?              �?      �?               @      �?              �?              �?              �?              �?              �?              �?      �?              �?              �?      �?       @               @              �?      �?              �?               @       @      �?       @              �?              �?               @              �?             �L@              �?              �?              �?      �?       @      �?      �?      �?       @      �?              @              �?      @       @      �?      �?              �?              �?      �?              �?              �?              �?      �?              �?              �?              �?               @              �?      @       @               @      �?              �?       @       @       @      �?              �?              �?               @       @       @      @               @       @      @      @      @      �?      @              @      @               @       @      �?       @      �?      �?      �?       @       @              @               @              �?        
�
conv5/weights_1*�	    j�¿   �v��?      �@! ��ke%@)�}BB0I@2�	�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:���%�V6��u�w74���82���bȬ�0�I�I�)�(�+A�F�&��[^:��"��S�F !�>h�'��f�ʜ�7
��FF�G ?��[�?x?�x�?��d�r?�vV�R9?��ڋ?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��bȬ�0?��82?�u�w74?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?�������:�	             �j@     s@     p@     �p@     �n@     �k@     `e@     �e@     �e@     �]@     �X@     �]@     �Z@     �Y@      V@      W@     �Q@      Q@      P@      J@     �G@     �B@      G@      F@      @@      ?@      ?@      8@      2@      =@      2@      0@      .@      ,@      3@      @       @      1@       @      @      @       @      @      "@      @      @       @      @      @              @      �?      @      @       @      @      @               @      �?       @      �?              �?       @       @              �?              �?              �?              �?              �?              �?              �?               @              �?      �?              �?               @       @      �?       @              �?       @       @              @      �?      �?      @      @      @      @       @      @       @      "@      @      (@      &@      6@      @      ,@      (@      &@      1@      9@      5@      ;@      0@      =@      9@     �A@      A@     �C@      G@     �I@     �N@      I@      Q@     @Q@      W@     @Y@     �Y@     @\@     @]@     �a@      d@     �c@      c@     �g@      i@     @n@     Pq@     �o@      r@      n@        
�
conv5/biases_1*�	   �� �    /'>      <@!   lP>)OlP��*}<2���-�z�!�%������f��p�Łt�=	�2!K�R���J��#���j�Z�TA[�����"�RT��+��y�+pm��mm7&c��`����tO����f;H�\Q���9�e����K���=��]����/�4��ݟ��uy影�.4N�=;3����=(�+y�6�=�|86	�=z�����=ݟ��uy�=�/�4��==��]���=��1���='j��p�=��-��J�=�tO���=�f׽r��=nx6�X� >�`��>Z�TA[�>�#���j>�J>2!K�R�>Łt�=	>��f��p>��o�kJ%>4��evk'>�������:�              �?              �?               @              �?      �?              �?              �?              �?              �?              �?      �?              �?              �?              �?      �?              �?      �?      �?              �?      �?      �?               @               @              �?              �?        E+B�V      ���	� i���A*ɭ

step  @@

loss�%�>
�
conv1/weights_1*�	    �ͮ�   `��?     `�@!  5��@)��Cӥ*@2�	����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1�O�ʗ�����Zr[v��O�ʗ��>>�?�s��>>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?�������:�	              G@     @k@     @i@     `d@     �b@      a@     ``@     @Z@     �Z@     �\@     @R@     @R@     �K@     �N@     �M@     �N@     �E@      @@      F@     �C@     �C@      =@      >@      7@      9@      ;@      8@      *@      2@      5@       @      $@       @      "@       @      .@      @       @      @      @      @      @      �?      @      �?      @      @              @       @      @      �?      @       @      �?      �?      @              �?              �?       @              �?      �?              �?      �?       @              �?              �?              �?              �?              �?      �?              �?      �?      �?              �?      �?       @      �?              �?              �?      �?       @      �?      �?      @       @      �?      �?      @      @      @      "@      @      @      @       @      $@      @      $@      @      .@       @      ,@      *@      *@      1@      2@      1@      8@      2@      6@      6@      @@     �C@      ?@      F@     �G@     �K@     �M@      K@     �M@     @R@      U@     @W@     �Y@     @X@      a@     �_@     @d@      d@     `e@     �f@      m@     �H@        
�
conv1/biases_1*�	   ���Q�   �n?      @@!  ���[�?)ꘇ�J ?2��lDZrS�nK���LQ�k�1^�sO�IcD���L�a�$��{E��T���C��!�A����#@�d�\D�X=���%�V6��u�w74���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�x?�x��>h�'��f�ʜ�7
������f�ʜ�7
?>h�'�?+A�F�&?I�I�)�(?��bȬ�0?��82?�u�w74?��%�V6?��%>��:?d�\D�X=?���#@?�!�A?a�$��{E?
����G?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�N�W�m?;8�clp?�������:�              �?              �?              �?      �?      �?      �?              �?               @      �?              �?              �?              �?              �?              �?              �?               @              �?      @      �?              �?              �?      �?               @      �?      �?              �?              �?        
�
conv2/weights_1*�	   ��+��   `nݩ?      �@! 	qf�} @)�K0,	1E@2����g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ뾢f�����uE���⾕XQ�þ��~��¾�u`P+d����n�������|�~���MZ��K���u��gr����z!�?��T�L<����n����>�u`P+d�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?�������:�             ��@     V�@     ܝ@     P�@     X�@     ��@     T�@     ��@     ��@      �@     ��@     ��@     ��@     h�@     (�@     P�@     ��@     �}@     �|@      x@     Pv@     Pt@      u@     �q@     �p@      m@     �j@     @h@      g@     �c@     �b@     �`@     �Y@     �Y@      U@     �T@     @T@      M@      K@     �E@      H@     �B@     �C@      G@      A@     �A@      =@      ?@      5@      9@      0@      7@      &@      4@      &@      ,@      *@      .@      *@      .@      (@      $@      @      @      @      @      @      @      @      @      @       @      �?      @       @       @      @       @      @              @      �?      �?      �?      �?              �?               @              �?              �?              �?      �?              �?              �?              �?              �?               @              �?       @      @      �?      �?       @      �?       @       @       @      @      @      @       @       @      @      @      �?      @      @      @      @      @      "@      @      @      $@      @      "@      @      $@      @      .@      ,@      4@      2@      4@      4@      =@      9@      <@      >@     �@@     �A@      <@      I@      E@     �P@     �R@     @P@     �R@     @Q@     �W@     �V@     @Y@     �_@     �^@     �b@      e@      d@     �g@     @j@      j@     �o@     �p@     �s@     @t@     �v@     �w@     �|@     ��@     �@     8�@     8�@     ��@     Ј@     p�@     8�@     x�@     �@     H�@     ��@      �@     ��@     ��@     d�@      @     <�@        
�

conv2/biases_1*�		   ��`�   ���\?      P@!  d��t?)���)?2��l�P�`�E��{��^��m9�H�[���bB�SY��lDZrS�nK���LQ�k�1^�sO��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%�V6��u�w74���82���bȬ�0���VlQ.�I�I�)�(�+A�F�&�U�4@@�$��[^:��"���d�r�x?�x��>h�'��6�]���1��a˲���Zr[v��I��P=��pz�w�7���XQ�þ��~��¾        �-���q=��~���>�XQ��>
�/eq
�>;�"�q�>E��a�W�>�ѩ�-�>a�Ϭ(�>8K�ߝ�>pz�w�7�>I��P=�>>�?�s��>�FF�G ?��d�r?�5�i}1?�T7��?�vV�R9?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?�m9�H�[?E��{��^?�������:�              �?      �?      �?               @      �?              �?               @      �?              �?               @               @      �?              �?              �?              �?       @              �?              �?      �?              �?              �?              �?              �?              �?              �?              �?              �?              @              �?              �?       @              �?      �?      �?      �?              �?       @      �?      �?              �?       @       @      @      �?      �?       @       @              �?        
�
conv3/weights_1*�	   �B^��   �@v�?      �@! ��$@)���AU@2�����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�a�Ϭ(���(��澢f�����uE���⾮��%ᾄiD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�Ѿ
�/eq
Ⱦ����ž�*��ڽ�G&�$��5�"�g�����������?�ګ����]������|�~����|�~�>���]���>K+�E���>jqs&\��>���%�>�uE����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?�������:�             ��@     T�@     �@     J�@     ��@     B�@     0�@     �@     �@     d�@     ��@     ��@     ,�@     0�@     ��@     p�@     ؈@     8�@     ��@     ��@     �@      ~@     �z@     �z@     0x@     �s@     Pt@     `r@     �p@     �l@      l@     �e@     �e@     �b@      `@     �a@     �^@      `@      \@     �T@     �S@      V@     @Q@     �Q@      H@      J@     �H@      H@      F@      2@      @@      @@      <@      :@      4@      6@      6@      7@      $@      1@      "@      ,@      .@      "@      @      "@      "@      @       @      @      @      @      @      @      @      @              �?       @      @       @       @      @      �?       @       @      �?      �?              �?      �?              �?              �?      �?              �?              �?              �?      �?              �?              �?              �?              �?              �?              @      �?               @              �?      @      �?      @       @      @      @      @      @      @      @       @      @      @      @      @      @       @      &@      @      &@       @      *@      @      .@      5@      $@      *@      4@      :@      ,@      3@      B@      4@     �D@      @@     �B@      @@      G@      H@     �E@     �H@      R@     �K@      R@     @W@     �W@     �[@     �`@     @`@      a@     @b@     �e@      h@     `j@     �n@      q@     �p@     �q@     �w@     pv@     �x@     �|@     �}@     ��@     8�@     ��@     ��@     x�@     ��@     ��@     l�@     p�@     D�@     ��@     ��@     $�@     ̜@      �@     ��@     �@     h�@     L�@     l�@     H�@        
�
conv3/biases_1*�	   �V[Z�   �'�f?      `@!  @\
�?).v��BN	?2��m9�H�[���bB�SY�ܗ�SsW�<DKc��T�nK���LQ�k�1^�sO�IcD���L�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
�1��a˲���[��>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��[�=�k���*��ڽ�        �-���q=�uE����>�f����>a�Ϭ(�>8K�ߝ�>})�l a�>pz�w�7�>�FF�G ?��[�?1��a˲?6�]��?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?�.�?ji6�9�?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?Tw��Nof?P}���h?�������:�               @              �?              @      �?              �?      @      �?      �?      @              @              @       @               @      �?       @       @      @               @      �?      �?               @               @               @      @      �?              �?              �?       @      �?              �?      �?              �?              @              �?              �?              �?              �?      �?      �?               @       @              �?      @              @              �?               @      @      @      @      �?       @      @              @              �?      @      �?      �?       @       @               @       @       @      @      �?              �?      �?              �?        
�
conv4/weights_1*�	   �>
��     �?      �@!��''8�)�77��&e@2�Ӗ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������1��a˲���[����Zr[v��I��P=��})�l a��ߊ4F��a�Ϭ(���(��澢f������~]�[Ӿjqs&\�ѾK+�E��ϾK+�E���>jqs&\��>�ѩ�-�>���%�>�uE����>�f����>8K�ߝ�>�h���`�>�ߊ4F��>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?����?f�ʜ�7
?>h�'�?x?�x�?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�������:�             @a@     $�@     ��@     �@      �@     ��@     `�@     ��@     @�@      �@     X�@     ��@     �@     P|@     P{@     v@     �x@      v@     t@     Pq@     �n@     �j@     @h@     �e@     �f@      c@     @_@      b@     �_@      \@      T@      V@     @W@     �Q@      Q@      N@     �L@     �E@     �I@     �E@      <@      <@      >@      :@      <@      :@      2@      7@      1@      3@      (@      ,@      .@      @      .@      @      @      @      @      @      @      @      @       @      @       @      @       @      @      �?       @      �?      �?      @      @               @              @       @              �?               @              �?              �?              �?              �?      �?               @      �?              �?              �?              �?              �?      �?              �?               @              @              �?              �?      �?      �?      �?              @      @      @      �?      @      @      @      @       @       @      �?       @      @      @      @      (@      @      *@      "@      $@      &@      3@      $@      (@      1@      $@      $@      :@      5@      *@      6@      ;@      ?@      6@     �F@      C@     �I@     @P@     �J@      L@     �Q@      S@     �W@     �Y@      W@     @\@     @\@     �`@     @c@     �c@     �f@     `h@      k@      m@     Pr@     @r@     @t@     @t@     0x@     �y@     `}@     ��@     �@     ��@      �@     ��@     ��@     H�@     ��@     ��@     �@     �@     ĕ@     �`@        
�
conv4/biases_1*�	   @��^�   ��=f?      p@!���9��s?)�H�*?2��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7�����d�r�x?�x��>h�'��f�ʜ�7
������6�]���>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(����uE���⾮��%ᾙѩ�-߾��>M|Kվ��~]�[Ӿ
�/eq
Ⱦ����ž5�"�g���0�6�/n����z!�?��T�L<����o�kJ%�4�e|�Z#���-�z�!�%�����i
�k���f��p�Łt�=	���R����2!K�R���J��#���j����"�RT��+��y�+pm��mm7&c��`����tO����f;H�\Q������%���9�e����K���=��]����/�4��z�����i@4[���Qu�R"�PæҭUݽ�
6������Bb�!澽5%����G�L���        �-���q=K?�\���=�b1��=�!p/�^�=��.4N�=��
"
�=���X>�=H�����=PæҭU�=��1���='j��p�=��-��J�=�K���=f;H�\Q�=�tO���=�`��>�mm7&c>y�+pm>RT��+�>���">�#���j>�J>2!K�R�>��R���>Łt�=	>��f��p>�i
�k>%���>��-�z�!>��o�kJ%>4��evk'>���<�)>�'v�V,>6NK��2>�so쩾4>���]���>�5�L�>豪}0ڰ>��n����>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>a�Ϭ(�>8K�ߝ�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?��bB�SY?�m9�H�[?E��{��^?5Ucv0ed?Tw��Nof?�������:�              �?              �?      �?              @      @              �?      �?      @              @      @              @      @       @      �?       @      �?              @               @      @      �?              �?              �?      �?      �?      �?      �?              �?      �?              �?      �?              �?       @              �?              �?      �?              �?              �?              �?              �?               @      �?      �?              �?       @       @              �?       @              �?      �?              �?              �?      �?               @              �?              �?      �?      �?              �?              �?             �I@              �?              �?              �?              �?              �?       @      �?              �?              �?              �?      �?               @      �?               @       @              �?      �?              �?              �?              �?              �?              �?              �?              �?      �?              �?              @       @       @       @              �?      �?               @       @      �?       @      @              �?      �?      @       @       @       @              �?      @              @      �?       @       @       @       @       @      @      �?       @       @       @      �?       @              @      �?       @      �?       @              �?      �?              �?        
�
conv5/weights_1*�	    j�¿    &��?      �@! ��4�@)ǵ�1I@2�	�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.�U�4@@�$��[^:��"��S�F !�O�ʗ��>>�?�s��>��[�?1��a˲?��ڋ?�.�?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?�������:�	             �j@      s@      p@     �p@     �n@     �k@     `e@      f@     �e@     �]@     �X@     �]@     �Z@      Z@     �U@     @V@     �Q@     �Q@      P@      I@      H@     �B@      G@      F@      ?@      >@      @@      8@      2@      =@      3@      .@      0@      ,@      3@      @       @      .@      &@      @      @       @       @      "@      @      @      @      @       @              �?      @      @      @      �?      @       @              @       @      �?              �?      �?               @       @              �?      �?              �?              �?              �?              �?              �?              @              �?              �?               @       @       @      �?              �?      �?      �?      �?              @       @       @      @      @      @      @      @      @      "@      $@       @      &@      &@      4@      @      ,@      *@      &@      2@      9@      4@      9@      2@      =@      7@      C@     �@@     �C@      F@      I@      P@     �I@      Q@      Q@     �W@     �X@     �Y@     �\@      ]@     �a@      d@      d@      c@     @g@      i@     @n@     pq@     �o@      r@      n@        
�
conv5/biases_1*�	   `�#�    ��0>      <@!   t3�%>)�TN̂�<2���o�kJ%�4�e|�Z#���-�z�!�%�����i
�k���f��p�2!K�R���J��#���j�Z�TA[�����"�RT��+��'j��p���1����b1��=��؜��=H�����=PæҭU�==��]���=��1���=��-��J�=�K���=����%�=f;H�\Q�=�tO���=�`��>�mm7&c>y�+pm>�J>2!K�R�>%���>��-�z�!>���<�)>�'v�V,>7'_��+/>_"s�$1>�������:�              �?              �?       @      �?              �?      �?               @      �?              �?              �?              �?              �?               @              �?       @              �?      @              �?               @              �?              �?        @䴐hV      
O�	;����A*٬

step  �@

loss��>
�
conv1/weights_1*�	   ��P��    �X�?     `�@! @�+9@)�����8@2�	����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !���ڋ��vV�R9��T7����5�i}1���d�r�>h�'��f�ʜ�7
�������FF�G �>�?�s�����n����>�u`P+d�>��~]�[�>��>M|K�>f�ʜ�7
?>h�'�?ji6�9�?�S�F !?�[^:��"?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?�������:�	              J@     `k@     �h@      e@     �b@     @`@      `@      ]@     �Y@     �[@     �Q@     �R@     �M@      L@     �L@     �O@      F@      D@      F@     �A@     �C@      ?@      6@      @@      5@      =@      2@      2@      ,@      1@      .@      "@      $@       @      (@      @       @       @      "@      @      &@      @      @       @       @      @       @              �?      �?       @       @       @       @              �?              @              �?      �?      �?      �?      �?       @              �?      �?              �?              �?       @              �?              �?              �?              �?              �?      �?               @              �?      �?              �?      �?              @      @              �?       @       @       @      @      @      @      "@      @      @       @      @      @      (@      @      $@      "@      $@      .@      &@      $@      2@      2@      1@      .@      8@      2@      :@      9@      7@      C@     �@@     �G@      G@      K@     �M@     �J@     @P@     �P@     �U@     �W@     @Y@     �W@     �a@     ``@     `c@     �c@     �d@     `g@     @m@      H@        
�
conv1/biases_1*�	   @k_�   `�su?      @@!  @ji��?)ڃw�́?2��l�P�`�E��{��^�ܗ�SsW�<DKc��T�k�1^�sO�IcD���L��qU���I�
����G��T���C��!�A���%>��:�uܬ�@8���%�V6��u�w74�+A�F�&�U�4@@�$��vV�R9��T7����.�?ji6�9�?��82?�u�w74?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?hyO�s?&b՞
�u?�������:�              �?              �?              @              �?              �?               @               @              �?              �?              �?              �?              �?              �?      �?       @      �?      �?              �?               @              @      �?      �?      �?              �?        
�
conv2/weights_1*�	   �Bw��   @u��?      �@! ;|1{"@)>�p�2E@2����g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ���(��澢f���侙ѩ�-߾E��a�Wܾ
�/eq
Ⱦ����ž�XQ�þ��~��¾0�6�/n���u`P+d��.��fc���X$�z���[�=�k�>��~���>;�"�q�>['�?��>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?�������:�             ��@     J�@     �@     �@     ��@     ��@     <�@     ��@     ��@     X�@     �@     x�@     ��@     Ȇ@     Ѓ@     (�@     Ѐ@     `}@     �|@     @y@     pu@     �t@     �t@     `q@     �p@     �o@     �j@     �f@     `g@     �b@     �c@     @`@      \@      W@     �V@     �U@     �R@     �M@     �L@      E@     �I@      C@     �B@      F@     �A@      ;@      9@      C@      .@      <@      4@      7@      (@      6@      (@      3@      *@      0@      ,@      (@      ,@      @      @      @      @      @      @      �?      @      �?      �?      @       @       @       @      @      @      @       @       @       @              �?      �?      �?      �?       @               @              �?              �?      �?      �?              �?              �?              �?              �?               @              �?              �?      �?      �?      �?      @      �?       @       @              �?      @       @      @      @      @      @      �?       @      @       @      @      @      @      @      @      @      @      @      @      $@      @      @      $@      @      3@      .@      5@      0@      4@      :@      >@      0@     �A@      :@      ;@      >@     �A@     �H@     �I@     �N@     @P@     �P@     �S@     �S@     �V@     �W@     @Z@     @]@      _@      c@     `e@     `c@      g@     �j@     �j@     `o@     �p@     @s@     �t@     �v@     @w@     �|@     �@     ��@      �@     @�@     Ѕ@     ��@     p�@     ��@     H�@     Ē@     h�@     ��@     �@     ��@     |�@     `�@     Ԡ@     p�@        
�

conv2/biases_1*�
	   �^Xe�   @D�c?      P@!  \k>Ԃ?)h�i�9?2�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.�U�4@@�$��[^:��"��vV�R9��T7���>h�'��f�ʜ�7
�1��a˲���[���_�T�l׾��>M|Kվ����ž�XQ�þ        �-���q=E��a�W�>�ѩ�-�>���%�>�uE����>�f����>a�Ϭ(�>8K�ߝ�>})�l a�>pz�w�7�>��Zr[v�>O�ʗ��>f�ʜ�7
?>h�'�?�T7��?�vV�R9?��ڋ?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?���#@?�!�A?�T���C?a�$��{E?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?���%��b?5Ucv0ed?�������:�              �?      �?              �?      �?              �?              �?      �?      �?               @              �?      �?              �?              �?       @      �?      �?              �?              �?              �?              �?              �?              �?              �?              �?              �?      �?              �?              �?              �?              �?              �?              �?      �?              �?              �?      �?               @      �?       @      �?              �?               @              �?      �?       @       @      @       @       @      �?      �?               @        
�
conv3/weights_1*�	    냮�    ��?      �@!��(\�@)݄���AU@2�����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F����(��澢f�����uE���⾮��%�E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~��¾�[�=�k�����]������|�~��;9��R�>���?�ګ>��~���>�XQ��>�����>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?�������:�             ��@     D�@     �@     N�@     ��@     D�@     H�@     ��@     �@     |�@     x�@     ��@     $�@     0�@     ��@     ��@     ��@     H�@     ��@     (�@     �@     �}@     `{@     �z@     �w@     �t@     �s@     @r@     Pq@      l@      k@     @g@      e@     �c@      `@      `@     ``@     �\@      [@     �V@     �U@     �U@     �Q@     @Q@      I@      M@     �H@      F@      C@      6@      ?@      =@      ;@      =@      7@      1@      6@      6@      *@      2@      &@      ,@      .@      (@      @       @       @      &@      @      @      @      @      �?      @      @      @      �?      �?              @       @      �?       @      @      �?      @      �?              �?              �?              �?              �?              �?              �?              �?              �?      �?              �?              �?               @              �?      �?      �?       @       @       @      @      �?       @      �?      @       @      @       @       @       @      @      @      @      @      @      @      $@      $@      @      *@      "@      *@      (@      ,@      0@      @      3@      *@      4@      .@      6@      B@      6@     �D@      9@      F@     �A@     �C@      H@      J@     �L@     @P@     �K@     �P@      V@      X@     @]@     �`@     �_@     @`@      d@     �d@     �h@      k@     �n@      p@     0p@     0s@     pv@     Pw@     �x@     �|@      ~@     0�@     ��@     ��@     ��@     �@     h�@     0�@      �@     ̒@     (�@     t�@     ��@     ,�@     ��@     4�@     ��@     �@     ��@     <�@     n�@     ��@        
�
conv3/biases_1*�	   ���b�   ��Ko?      `@!  [-W�?)]��u�?2�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�IcD���L��qU���I�
����G�a�$��{E��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��5�i}1���d�r�6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��豪}0ڰ������        �-���q=
�}���>X$�z�>�h���`�>�ߊ4F��>})�l a�>1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�T7��?�vV�R9?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?5Ucv0ed?Tw��Nof?�N�W�m?;8�clp?�������:�              �?      �?              �?       @              �?       @      �?              @       @      @              @      @              �?      @       @      �?      @      �?       @      �?              �?              @      �?      �?              �?              �?              �?               @      �?      �?              �?      �?              �?              @              �?              �?      �?              �?       @              �?      �?      �?              �?              �?              @       @              �?      �?      �?      @      @       @      �?              @      @       @       @      @      �?      @       @      �?      @      �?               @      @      �?      @              �?              �?        
�
conv4/weights_1*�	   ����   ���?      �@! �xj��7�)$�-'e@2�Ӗ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.���T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������pz�w�7��})�l a��ߊ4F����(��澢f�����uE������>M|Kվ��~]�[Ӿ;�"�qʾ
�/eq
Ⱦ�[�=�k���*��ڽ�0�6�/n���u`P+d��R%�����>�u��gr�>�XQ��>�����>jqs&\��>��~]�[�>8K�ߝ�>�h���`�>�ߊ4F��>pz�w�7�>I��P=�>�FF�G ?��[�?6�]��?����?f�ʜ�7
?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�������:�              a@     (�@     ��@     �@     ��@     ��@     P�@     ��@      �@      �@     H�@     ��@     �@     p|@     P{@     �u@     �x@     �u@     �s@     �q@     �n@     �j@      h@      f@     �f@      c@     �^@     `b@     �^@      \@      U@     �U@     @W@     �Q@      Q@     �L@      N@      E@     �J@      E@      >@      :@      >@      <@      8@      <@      2@      8@      ,@      3@      (@      1@      ,@      @      *@      @      @      @       @      @      @       @      @      @      @      @      @      @      @      �?      @              @      @      @               @      @      �?              �?       @              �?      �?              �?      �?              �?              �?              �?              �?              �?              �?              �?              �?      �?              �?              �?              �?       @              �?               @      �?              @       @       @       @      @      @      @       @      @      @       @       @      @       @      @      *@      @      (@      &@      ,@      "@      *@      (@      &@      4@      &@      @      9@      6@      .@      3@      =@      =@      9@     �E@     �C@      G@      P@      M@      L@     �Q@     �S@     �W@     @Y@     �V@     @[@     �]@     @a@     �b@     @d@     @f@      h@     `k@     �l@     0r@     �r@     pt@      t@     `x@     @y@     �}@     ��@     Ё@     ��@     �@     ��@     ��@     `�@     Ȏ@     ��@     �@     �@     ԕ@     ``@        
�
conv4/biases_1*�	   `rd�   ��q?      p@!0�M妪s?).�g�l�?2�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���})�l a��ߊ4F����(��澢f���侮��%ᾙѩ�-߾['�?�;;�"�qʾ5�"�g���0�6�/n���MZ��K���u��gr��w`f���n�=�.^ol�7'_��+/��'v�V,����<�)�4��evk'�4�e|�Z#���-�z�!�%�����i
�k�2!K�R���J�Z�TA[�����"�RT��+��y�+pm��mm7&c��`���nx6�X� ��f׽r����tO����i@4[���Qu�R"�K?�\��½�
6�����        �-���q=�d7����=�!p/�^�=��.4N�=;3����==��]���=��1���='j��p�=����%�=f;H�\Q�=nx6�X� >�`��>�mm7&c>y�+pm>RT��+�>�#���j>�J>2!K�R�>��R���>��f��p>�i
�k>%���>��-�z�!>4�e|�Z#>�'v�V,>7'_��+/>6NK��2>�so쩾4>p
T~�;>����W_>>ڿ�ɓ�i>=�.^ol>�XQ��>�����>;�"�q�>['�?��>��~]�[�>��>M|K�>���%�>�uE����>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�l�P�`?���%��b?5Ucv0ed?;8�clp?uWy��r?�������:�               @              �?      �?              �?      @       @       @               @       @               @      @      �?      �?      @      @      �?               @      @      �?              �?       @      @      �?       @      �?              @      �?               @              �?      �?       @              �?      �?       @               @              �?              �?              �?              �?              �?              �?               @      �?      @              �?               @              �?              �?              �?      �?      �?       @      �?      �?              �?              �?              H@              �?              �?              �?      �?              �?              �?               @      @              �?              �?               @      @               @              �?              �?              �?              �?              �?              �?               @              �?              @              �?              �?       @      �?      �?              �?       @      �?      �?              @      �?       @       @       @      �?      @      �?              �?      @      @       @              �?       @              @      @      @      �?       @       @      @      �?       @      @       @      @      �?      �?      �?      @      �?              �?       @              �?        
�
conv5/weights_1*�	    j�¿   `��?      �@! ��;m�@)�Fq;�2I@2�	�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(��[^:��"��S�F !�I��P=�>��Zr[v�>�.�?ji6�9�?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?�������:�	             �j@      s@     0p@     �p@      n@     �k@     �e@     �e@     `e@     �^@     �X@     @]@      [@     �Y@     �U@     �V@     �Q@     �P@     �P@      I@     �G@      C@      G@     �E@      @@      ?@      @@      8@      2@      >@      1@      .@      0@      .@      3@      @      (@      $@      *@      "@      @      @      @      @      @      @       @      @      @      �?      �?       @      @       @      @      "@       @      �?       @       @      �?              �?      �?      �?      �?      �?      �?               @              �?              �?              �?               @               @              �?      �?      �?       @      @      @              �?      �?      �?              @       @       @      @      @       @       @       @      @      @      &@      "@      *@      *@      2@      @      &@      .@      $@      3@      7@      3@      <@      2@      <@      8@     �B@      @@     �D@     �E@     �H@     �P@     �I@      P@      R@      W@      Y@     @Y@     �\@     @]@     @a@     `d@     �c@     @c@      g@     `i@     @n@     �q@     �o@     �q@     �n@        
�
conv5/biases_1*�	   @ح)�   ��6>      <@!   dJ�$>)��M�̟<2����<�)�4��evk'���o�kJ%���-�z�!�%�����i
�k���f��p�Łt�=	���R����2!K�R���J��#���j�Z�TA[����-��J�'j��p���1���H����ڽ���X>ؽz�����=ݟ��uy�='j��p�=��-��J�=�K���=�9�e��=nx6�X� >�`��>�mm7&c>y�+pm>Z�TA[�>�#���j>�J>2!K�R�>��R���>%���>��-�z�!>4�e|�Z#>��o�kJ%>�'v�V,>7'_��+/>�z��6>u 5�9>�������:�               @       @              �?              �?      �?      �?              �?              �?              �?      �?              �?              �?               @              �?              �?      �?      �?              �?      �?              �?              �?              �?               @              �?        B�9(V      �+	������A*��

step  �@

loss��k>
�
conv1/weights_1*�	   �ů�   �3��?     `�@!  �p��@)%G��L@2�����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���bȬ�0���VlQ.��7Kaa+�I�I�)�(�U�4@@�$��[^:��"��S�F !���ڋ��vV�R9��T7�����d�r�x?�x��
�/eq
�>;�"�q�>�iD*L��>E��a�W�>x?�x�?��d�r?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?�������:�             �K@     @k@      i@     �d@     �b@     ``@      `@     @[@     @[@     �Z@     @S@     �Q@      N@     �H@     @P@      M@     �I@     �B@      D@      H@      A@      :@      9@      8@      :@      ;@      3@      1@      4@      1@      "@      1@      @      "@      @      $@      @      @      "@      @      @      @      @      @       @      @      @       @       @      �?       @       @       @      �?      @      �?               @              @              �?      �?              �?      �?              �?              �?              �?              �?              �?               @              �?      @      �?      �?              @      �?       @       @      �?               @      @      @      @      @      @       @      @      @      @       @      $@      @      &@      ,@       @      $@      ,@      &@      &@      .@      8@      &@      0@      9@      1@      :@      <@      3@     �D@      @@      E@      I@      N@     �I@     �J@     @P@     @Q@      U@     @Y@     �Y@     �U@     �a@      `@      c@     �c@      f@     `f@     �l@      N@        
�
conv1/biases_1*�	   ���f�   �_�~?      @@!  `�n�?)4ܙ��'?2�P}���h�Tw��Nof�5Ucv0ed����%��b�ܗ�SsW�<DKc��T�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:��u�w74���82���bȬ�0�+A�F�&?I�I�)�(?a�$��{E?
����G?�qU���I?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?hyO�s?&b՞
�u?���T}?>	� �?�������:�              �?              �?              �?              �?      �?      �?              �?      �?      �?      �?              �?              �?      �?              �?              �?      �?              @               @      �?              �?      �?              �?      @      �?              �?              �?              �?        
�
conv2/weights_1*�	    Cת�   ��7�?      �@!�+��%@)�]�O5E@2����g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ���pz�w�7��})�l a��ߊ4F��h���`�a�Ϭ(���(��澢f�����uE���⾙ѩ�-߾E��a�Wܾ;�"�qʾ
�/eq
Ⱦ�XQ�þ��~��¾���?�ګ�;9��R��f^��`{>�����~>K+�E���>jqs&\��>�_�T�l�>�iD*L��>�ѩ�-�>���%�>�f����>��(���>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?�������:�             ��@     H�@     0�@     �@     l�@     ��@     4�@     ��@     ��@     `�@     H�@     ؊@     8�@     ��@     ��@     ��@     ��@     }@     �|@     0y@     �u@     �t@     `t@     �q@     q@     @p@      i@     �g@     �f@     �b@      b@      a@      ]@     �X@     �T@      U@     �R@     �L@     @P@      H@      G@      A@     �D@      E@      C@     �@@      @@      9@      .@      :@      :@      1@      .@      2@      1@      1@      .@      &@      &@      "@       @      &@      @      @      @      @      @      @      @      @       @      @      �?       @      �?      @      �?      �?       @      �?               @      �?      �?              �?      �?      �?               @              @              �?              �?              �?               @              �?              �?              �?              @      @      @              �?              @      �?      @      @      @      @       @       @       @       @       @      @      @      @      �?      @      $@      @      @      @      "@      $@      @      @      (@      2@      1@      0@      .@      6@      5@      <@      0@      =@      @@     �@@     �@@      :@      I@     �F@     �P@     �Q@     �R@      R@     @T@     �U@     �X@      Z@     �Y@     �`@     �b@     `d@     �e@     �f@     �h@     �k@     @o@     0q@     �r@     �t@     �v@     �v@     �|@     ��@     ��@     p�@     h�@     ��@     ��@     ��@     ��@     8�@     ��@     ��@     x�@     D�@     ��@     X�@     T�@     ��@     ��@        
�

conv2/biases_1*�
	   ��-h�   ��Gm?      P@!  #���?)	n�Dn?2�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���bȬ�0���VlQ.��S�F !�ji6�9���.����ڋ��vV�R9���d�r�x?�x��6�]���1��a˲���[���f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>��Zr[v�>O�ʗ��>>�?�s��>�T7��?�vV�R9?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?�������:�              �?      �?      �?       @              �?      �?      �?      �?      �?              �?      �?               @      @              �?      �?              �?              �?       @      �?      �?              �?              �?      �?              �?      �?              �?      �?              �?              �?      �?              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?      @      @              �?      �?      �?       @       @       @      �?      �?      �?              �?        
�
conv3/weights_1*�	    ����    o�?      �@!@&�pK @)#S�b"CU@2�����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(����uE���⾮��%��_�T�l׾��>M|Kվ��~]�[Ӿ���]������|�~������>豪}0ڰ>0�6�/n�>5�"�g��>K+�E���>jqs&\��>�uE����>�f����>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?�������:�             H�@     B�@     �@     j�@     r�@     H�@     T�@     ��@     �@     ��@     l�@     ��@     �@      �@     `�@     Ȋ@      �@     0�@     ��@     8�@     �@      ~@     {@     @z@     `x@     �s@     t@     0r@      q@      m@     �i@     �g@     �g@     �b@      _@     �]@     �`@     @Z@     �\@     @V@     �U@     �U@     �R@      R@      J@     �K@      H@      F@     �A@      @@      8@      =@      <@     �B@      3@      *@      5@      :@      *@      (@      $@      ,@      "@      (@       @      @       @      "@      @      @       @      @       @      @      @       @      @      @      @      @      @      �?      @      @       @      �?      �?       @      �?              @               @              �?      �?              �?              �?              �?              �?              �?              �?       @               @       @      @      @              @      @      @      �?      @      @      @      @      @      @      @      @      "@      @      @      @      $@      $@       @      *@      3@      *@      *@      0@      .@      7@      0@      5@      <@      6@      F@      9@     �B@      C@     �F@      J@      J@      P@      O@      J@      Q@     �T@     @Y@     �[@      `@     @a@     �_@     �b@     �f@      g@     �j@      o@     �p@     p@     �r@     �v@      w@     �y@     P|@     P}@     ��@     8�@     �@     ��@     Ї@     �@     X�@     4�@     ܒ@     @�@     $�@     ��@     8�@     |�@     4�@     ��@     2�@     ��@     (�@     V�@     ��@        
�
conv3/biases_1*�	   ��Eh�   `��s?      `@!  ���b�?)� [h��&?2�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !���ڋ��vV�R9��T7���x?�x��>h�'��f�ʜ�7
������6�]���1��a˲��FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7����~��¾�[�=�k��        �-���q=O�ʗ��>>�?�s��>6�]��?����?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?hyO�s?&b՞
�u?�������:�              �?      �?      �?       @              �?       @      �?      �?               @      @      �?       @      @      �?      �?      �?       @      �?      �?       @      �?      �?      �?              �?      �?       @      @      @               @               @       @              �?              �?       @      �?              �?              �?              �?              �?              @              �?               @               @              �?              �?              �?      �?      �?       @              �?               @      �?               @      @      @      @      @       @              �?      �?       @      @       @      @      @       @      �?       @      @              �?      �?       @       @       @              �?              �?        
�
conv4/weights_1*�	   ��+��    �&�?      �@!��br7�)+�m�'e@2�Ӗ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1������6�]���1��a˲���[���FF�G �})�l a��ߊ4F����(��澢f�����uE���⾮��%��_�T�l׾��>M|KվK+�E��Ͼ['�?�;jqs&\��>��~]�[�>�ѩ�-�>���%�>�h���`�>�ߊ4F��>I��P=�>��Zr[v�>��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�������:�             @a@      �@     ��@     �@     �@     ��@     `�@     ��@     �@     X�@     0�@     ��@     �@     �|@     �{@     0u@     py@     �u@     �s@     �q@     �n@     �j@     �g@     �e@     �g@     �b@      `@     �a@     �_@     �Z@     �U@      U@     @W@     �R@     �P@     �L@      N@      F@     �K@     �D@      =@      <@      @@      6@      6@      =@      4@      :@      .@      ,@      *@      .@      1@      @      ,@      @      @      @      @      @      @      @      @      @      �?      @      @      @      @      �?      @      @       @      �?      @              �?       @      @              �?              �?      �?              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?      �?              �?      �?              �?              @       @              �?      �?      @      @      �?      �?      @      @       @       @      @       @      @       @      @      @      @      "@      $@      $@      *@      &@       @      ,@      (@      .@      1@       @      "@      8@      8@      ,@      4@      ;@      ?@      7@      F@     �B@     �G@     @P@     �M@      L@     @Q@     �T@     �U@     @[@     �V@      Y@      _@     �a@     @b@     `d@     `f@     �g@      k@     �m@      r@     �r@     t@     �s@     y@     �x@     �}@     ��@     ��@     ؃@     (�@     ��@     ��@     0�@     ��@     ��@     �@     �@     ̕@      a@        
�
conv4/biases_1*�	   ��i�   �:�v?      p@! ��qf?)���3 z&?2�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
�6�]���1��a˲�>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��8K�ߝ�a�Ϭ(��uE���⾮��%ᾙѩ�-߾��>M|Kվ��~]�[Ӿ��������?�ګ�;9��R���5�L���so쩾4�6NK��2�_"s�$1�7'_��+/��'v�V,���o�kJ%�4�e|�Z#��i
�k���f��p����"�RT��+��y�+pm��mm7&c��`���nx6�X� ��f׽r����9�e����K��󽉊-��J�'j��p�        �-���q=�b1��=��؜��=�d7����=�!p/�^�='j��p�=��-��J�=f;H�\Q�=�tO���=�`��>�mm7&c>RT��+�>���">Z�TA[�>�#���j>�J>2!K�R�>��R���>Łt�=	>��f��p>��-�z�!>4�e|�Z#>��o�kJ%>4��evk'>���<�)>�'v�V,>7'_��+/>_"s�$1>6NK��2>u 5�9>p
T~�;>p��Dp�@>/�p`B>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>E��a�W�>�ѩ�-�>�uE����>�f����>��(���>a�Ϭ(�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?��d�r?�5�i}1?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?&b՞
�u?*QH�x?�������:�              �?       @              �?      �?      �?      @      �?      �?       @      @      @       @      �?      �?       @       @              �?      @      @       @              @              @      @       @       @      @               @              �?       @       @      @      �?      �?       @              �?       @              �?              �?      �?               @              �?              �?       @              �?              �?              �?              �?      @       @      �?              �?               @              �?       @      �?               @      �?              �?              �?             �F@              �?              �?              �?               @              �?               @      �?      �?              �?      �?      �?      �?               @      �?              �?               @              �?              �?              �?              �?               @              �?              �?              �?              �?      �?       @              �?              �?              @      �?              �?              @               @      �?      �?       @      @              @      �?               @      �?      �?       @       @      @               @      @      @      �?      �?      �?       @      @      @      @              @              �?      @               @      �?              �?      �?              �?        
�
conv5/weights_1*�	    ɚ¿    4��?      �@! ��,�@)_�aƫ4I@2�	�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(��[^:��"��S�F !�>h�'��f�ʜ�7
�6�]���1��a˲�8K�ߝ�a�Ϭ(��[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?�������:�	             `j@      s@     �o@     q@     @n@     �k@     @e@      f@     �e@      ^@     �X@     �]@      [@      Y@     �U@     @W@     �Q@     �P@     �P@     �H@      H@     �C@     �F@      E@      @@     �@@     �A@      2@      3@      ?@      0@      1@      0@      ,@      4@      @      $@      ,@      "@       @      @      @      @       @      @      @      @       @      @               @      @       @      @       @       @      @              @      @       @      �?              �?      �?              �?              �?              �?              �?              �?              �?              �?       @              �?              �?      �?               @               @       @      @      �?              �?      �?      �?              @              �?      @      @      @      @              @      @      $@       @      ,@      (@      2@      @      *@      ,@      (@      3@      7@      3@      6@      6@      =@      7@      C@      >@      E@      F@     �H@     �O@      K@     @P@     @Q@      W@     @Y@      Y@      ]@     @]@      a@      d@      d@     @c@      g@     @i@     @n@     �q@     �o@     �q@     �n@        
�
conv5/biases_1*�	   `�1�   �L*=>      <@!   z9�!>)�b�cR̩<2�6NK��2�_"s�$1�7'_��+/��'v�V,�4��evk'���o�kJ%���-�z�!�%�����i
�k���f��p���R����2!K�R��Z�TA[�����"�y�+pm��mm7&c�'j��p���1����/�4��ݟ��uy��
6����=K?�\���=��1���='j��p�=����%�=f;H�\Q�=�mm7&c>y�+pm>RT��+�>���">�#���j>�J>��R���>Łt�=	>�i
�k>%���>4��evk'>���<�)>�'v�V,>7'_��+/>�so쩾4>�z��6>p
T~�;>����W_>>�������:�               @              �?              �?              �?      @      �?              �?              �?              �?              �?              �?              �?              �?              �?              �?               @              �?              �?              �?               @              �?              �?              �?        ��L�U      �TE	�e8���A*��

step  �@

loss��>>
�
conv1/weights_1*�	   �V:��    j��?     `�@!  ��x @)so���g@2�����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0�ji6�9���.����ڋ��vV�R9�x?�x��>h�'��O�ʗ�����Zr[v���u`P+d�>0�6�/n�>�FF�G ?��[�?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?�������:�              Q@     @k@     �h@     @d@     `b@      a@      _@     @\@     @]@     �X@     @R@     @R@     �K@     �K@      Q@     �K@      J@     �B@      H@      C@      @@      >@      5@      ?@      5@      9@      6@      3@      4@      1@      (@      $@      .@      @      "@      @      $@       @      @       @      @      @       @      @       @      �?      @       @      �?      �?       @              @              �?      �?      @      �?               @      �?      �?               @              �?              �?              �?              �?      �?      �?               @      �?      �?              �?      �?       @      @      @       @      �?      �?      �?       @      �?      @      @      @      @      @      @      @      @       @      $@      &@      $@      @      "@      (@      @      $@      ,@      6@      &@      1@      1@      6@      7@      8@      7@      3@     �D@     �C@      E@     �I@     �M@     �E@     �K@     @Q@      Q@     �T@      Y@      Z@     �W@      a@     ``@     �b@     �c@     �d@      h@      k@      P@        
�
conv1/biases_1*�	   @��n�   ����?      @@!  @�<̦?)I����H5?2�;8�clp��N�W�m�ߤ�(g%k����%��b��l�P�`�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G��!�A����#@�uܬ�@8���%�V6��u�w74���82���VlQ.��7Kaa+��S�F !?�[^:��"?��VlQ.?��bȬ�0?k�1^�sO?nK���LQ?�lDZrS?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?&b՞
�u?*QH�x?���T}?>	� �?���J�\�?-Ա�L�?�������:�              �?      �?              �?              �?      �?      �?              �?              �?              �?              �?              �?              �?              �?              �?              �?      �?              �?      �?      �?      �?      �?              �?      �?      �?      @      �?      �?              �?              �?              �?        
�
conv2/weights_1*�	   @FY��   @B�?      �@! �ޤu(@)��d��8E@2�I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%��_�T�l׾��>M|Kվ;�"�qʾ
�/eq
Ⱦ����ž�XQ�þ��~��¾�*��ڽ�G&�$��Fixі�W���x��U��_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?�������:�              @     H�@     J�@     d�@     ��@     p�@     З@     Ԕ@     ��@     ��@     ��@     ��@      �@     P�@     І@     ��@     ��@     ��@     p~@     �{@     @y@     �u@     0t@     `t@     �r@     @p@     �o@     �i@      g@      g@     �b@     @c@      `@     @[@     �Y@     @V@      S@      U@     �K@     �M@     �J@     �I@      E@      G@     �D@      <@      A@      ?@      7@      5@      0@      ?@      ,@      2@      6@      (@      1@      4@      1@      @       @      @      &@      @      @      @      @       @      @       @      @      @       @              @      @               @      @       @      �?      �?      �?      �?       @       @              �?       @      �?              �?              �?              �?              �?      �?              �?              �?              �?      �?              �?               @      @       @      �?      @      @      �?      �?              @      @      @      �?      @      �?      �?      @       @      @      @      @      "@      @      @       @      @       @      "@      @      0@      "@      0@      *@      1@      *@      8@      9@      :@      1@      9@      :@     �@@     �B@      @@     �E@      I@     @P@     �R@     �P@     �R@     �W@     �R@      Y@     �X@     �Z@     �`@     �a@      d@     �f@      g@      h@     �k@     �p@     q@      r@     �t@      w@      w@     @|@     �@     ��@     �@     ��@     ��@     X�@     ��@     H�@     \�@     `�@     ��@     ĕ@      �@     ��@     T�@     4�@     �@     �@        
�	
conv2/biases_1*�		    �}l�   ��r?      P@!  X�KU�?)��т�s'?2��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b�E��{��^��m9�H�[���bB�SY�nK���LQ�k�1^�sO�a�$��{E��T���C��!�A���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&��[^:��"��S�F !�ji6�9���vV�R9��T7����5�i}1���d�r��FF�G �>�?�s���pz�w�7�>I��P=�>��[�?1��a˲?6�]��?����?>h�'�?x?�x�?��ڋ?�.�?�7Kaa+?��VlQ.?��bȬ�0?��82?��%�V6?uܬ�@8?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?�������:�              �?      �?      �?               @              �?      @               @              �?      @              @              �?      �?              �?       @              �?      �?              �?               @              �?               @               @              �?              �?              �?              �?      �?      �?              �?              �?               @      �?      �?              �?              @       @              �?              @      �?      �?              @               @      �?              �?        
�
conv3/weights_1*�	    ���   ����?      �@! h�l%@)j�>"�DU@2�����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾�_�T�l׾��>M|Kվ���]������|�~����n����>�u`P+d�>�[�=�k�>��~���>K+�E���>jqs&\��>E��a�W�>�ѩ�-�>���%�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?�������:�             �@     "�@     ��@     b�@     h�@     N�@     4�@     ��@     $�@     l�@     h�@     �@     �@     @�@     P�@     �@     ��@     P�@     (�@     �@      @     �~@     @{@     �z@     �w@     `t@     �s@     pr@      q@     @m@      k@     �f@      d@     `e@     @^@     �`@     �^@     @Y@     @]@      V@     �U@     @T@      S@     �N@      J@     �L@      I@     �D@     �F@      :@     �@@     �@@      :@      <@      <@      3@      1@      :@      &@      &@      &@      *@      @      $@      @      &@      @      @      @      @      @      @      @      @      �?      �?       @      @      @       @              �?       @      �?       @       @      �?              �?      �?      �?      �?       @      �?      �?               @              �?              �?              �?              �?              �?       @              �?       @      �?      �?      @      �?      @      @              @       @      �?      @      @      @      @      @       @       @      $@      (@      "@       @      @      (@      ,@      3@      $@      *@      "@      $@      0@      4@      5@      5@      5@     �A@      3@     �G@      4@      D@      C@      F@     �F@      M@     �N@      Q@     �J@     �P@     @X@     �W@     �[@     @^@      a@     �`@     `b@     `d@     `i@      j@     `m@     0q@     �p@      r@     pv@      w@      z@      |@     0~@     `�@     X�@     �@     ��@     ؇@     X�@      �@     l�@     p�@     ��@     �@     �@     @�@     D�@     H�@     ��@     <�@     z�@     �@     z�@     @�@        
�
conv3/biases_1*�	    �n�   ���w?      `@!  �#Б�?) �M��2?2�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[��        �-���q=����?f�ʜ�7
?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?&b՞
�u?*QH�x?�������:�              �?              �?               @       @      �?      @               @       @       @      �?              @      �?       @               @       @      �?      �?       @      �?       @      �?              @       @              �?      �?              �?      @      �?      �?      �?      �?       @              �?              �?      �?              �?              @              �?              �?       @      �?      �?              �?      �?               @      �?      @      �?              @      �?       @      �?      @      �?              @      �?      @      �?      @      @      �?      �?      @      �?      @               @      �?      �?       @      �?       @              �?        
�
conv4/weights_1*�	   �XC��    �B�?      �@! ����6�)K�=V(e@2�Ӗ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1������6�]���})�l a��ߊ4F��h���`�8K�ߝ���(��澢f����E��a�Wܾ�iD*L�پ�_�T�l׾���?�ګ>����>jqs&\��>��~]�[�>�h���`�>�ߊ4F��>����?f�ʜ�7
?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�������:�             @a@     $�@     ��@     ؒ@     �@     ��@     (�@     Ȋ@     �@     @�@     �@     Ȃ@     �@     P|@     �{@     0u@     `y@     Pu@     t@     �q@      o@     �j@     �g@      e@      h@     `b@      `@     `a@     �`@     @Z@      U@     �U@     @V@     �R@     @P@      N@      N@     �H@     �G@      F@      A@      9@      >@      7@      6@      ;@      7@      3@      1@      .@      1@      ,@      ,@      @      (@      "@      @      @      @      @      �?      @      @      @      @      @      @      @      @      @      @       @      �?      �?       @      �?      �?       @      �?              �?              �?              �?              �?              �?       @              �?              �?               @              �?              �?       @      �?      �?      �?      �?      @      @              @      @      @       @      @      @      �?      �?      @      @      @      (@      ,@       @      $@      ,@      "@      0@       @      0@      .@      "@       @      8@      9@      *@      7@      ;@      <@      9@      H@     �@@     �H@      L@      N@      N@     �Q@      T@     @W@     �Z@     �U@     �X@     �^@      b@     @b@      d@     �f@     �g@     �j@     �m@     pr@     �r@      t@     �s@     �y@     �x@     �}@     ��@     ��@      �@     (�@     ��@     ��@     (�@     ��@     ��@      �@     �@     ��@     `a@        
�
conv4/biases_1*�	   ��q�   �q�{?      p@!@��{�T?)6ˌ�Q�2?2�uWy��r�;8�clp��N�W�m�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.���vV�R9��T7����5�i}1���d�r�x?�x��>h�'��1��a˲���[��O�ʗ�����Zr[v��I��P=��pz�w�7���ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澕XQ�þ��~��¾�[�=�k����������?�ګ�p��Dp�@�����W_>�p
T~�;�u 5�9��z��6��so쩾4�6NK��2�_"s�$1����<�)�4��evk'���o�kJ%�4�e|�Z#���-�z�!���f��p�Łt�=	���R����2!K�R���J��#���j�Z�TA[�����"�RT��+��y�+pm��`���nx6�X� �����%���9�e�����-��J�'j��p�ݟ��uy�z������Qu�R"�PæҭUݽ        �-���q=�/�4��==��]���=��1���='j��p�=�tO���=�f׽r��=y�+pm>RT��+�>2!K�R�>��R���>Łt�=	>��f��p>�i
�k>%���>��-�z�!>4�e|�Z#>��o�kJ%>4��evk'>�'v�V,>7'_��+/>_"s�$1>6NK��2>�so쩾4>�z��6>u 5�9>����W_>>p��Dp�@>/�p`B>�`�}6D>5�"�g��>G&�$�>�XQ��>�����>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>�ѩ�-�>���%�>�uE����>a�Ϭ(�>8K�ߝ�>�h���`�>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?o��5sz?���T}?�������:�              �?       @               @              @      @       @       @       @       @      �?              �?      �?       @       @      �?       @       @      �?              @      @               @      �?      @      @      �?      @       @      �?       @      @       @       @      �?               @       @      �?              �?              �?               @              �?               @      �?      �?      �?              �?      �?              �?              �?              �?              �?       @      �?              �?              �?      �?              �?              �?              �?       @              �?      �?              �?               @              �?              �?              �?             �D@              �?              �?              �?              �?               @               @               @      �?              �?      �?              �?              �?      �?              �?              �?              �?              �?              �?              �?              �?      �?              �?      �?              �?      �?              �?              �?      �?               @      �?              �?      �?       @               @      �?              �?      �?       @               @       @      �?      �?      @       @       @      �?      �?      @      @       @      �?      @      �?       @              @              �?      �?      @       @       @      @      @               @       @              �?      @              �?              �?              �?        
�
conv5/weights_1*�	   @ם¿   �+,�?      �@!  Y���@)����7I@2�	�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%�V6��u�w74���82�+A�F�&�U�4@@�$��[^:��"��S�F !��T7����5�i}1�>h�'��f�ʜ�7
�����?f�ʜ�7
?>h�'�?x?�x�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��bȬ�0?��82?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A�