       �K"	  �u'��Abrain.Event:2�)��     �:E	�S�u'��A"��
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
conv1/weights/AssignAssignconv1/weights(conv1/weights/Initializer/random_uniform* 
_class
loc:@conv1/weights*
validate_shape(*&
_output_shapes
: *
use_locking(*
T0
�
conv1/weights/readIdentityconv1/weights* 
_class
loc:@conv1/weights*&
_output_shapes
: *
T0
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
model/conv1/dilation_rateConst*
_output_shapes
:*
valueB"      *
dtype0
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
_class
loc:@conv2/biases*
	container *
shape:@*
dtype0*
_output_shapes
:@*
shared_name 
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
.conv3/weights/Initializer/random_uniform/shapeConst*
_output_shapes
:*%
valueB"      @   �   * 
_class
loc:@conv3/weights*
dtype0
�
,conv3/weights/Initializer/random_uniform/minConst*
_output_shapes
: *
valueB
 *�[q�* 
_class
loc:@conv3/weights*
dtype0
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
,conv3/weights/Initializer/random_uniform/subSub,conv3/weights/Initializer/random_uniform/max,conv3/weights/Initializer/random_uniform/min*
_output_shapes
: *
T0* 
_class
loc:@conv3/weights
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
model/conv3/conv3/ReluRelumodel/conv3/BiasAdd*
T0*0
_output_shapes
:���������&�
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
,conv4/weights/Initializer/random_uniform/mulMul6conv4/weights/Initializer/random_uniform/RandomUniform,conv4/weights/Initializer/random_uniform/sub*(
_output_shapes
:��*
T0* 
_class
loc:@conv4/weights
�
(conv4/weights/Initializer/random_uniformAdd,conv4/weights/Initializer/random_uniform/mul,conv4/weights/Initializer/random_uniform/min*(
_output_shapes
:��*
T0* 
_class
loc:@conv4/weights
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
conv4/weights/AssignAssignconv4/weights(conv4/weights/Initializer/random_uniform*(
_output_shapes
:��*
use_locking(*
T0* 
_class
loc:@conv4/weights*
validate_shape(
�
conv4/weights/readIdentityconv4/weights*
T0* 
_class
loc:@conv4/weights*(
_output_shapes
:��
�
conv4/biases/Initializer/zerosConst*
dtype0*
_output_shapes	
:�*
valueB�*    *
_class
loc:@conv4/biases
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
model/conv4/conv4/ReluRelumodel/conv4/BiasAdd*0
_output_shapes
:����������*
T0
�
model/conv4/MaxPool2D/MaxPoolMaxPoolmodel/conv4/conv4/Relu*
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
(conv5/weights/Initializer/random_uniformAdd,conv5/weights/Initializer/random_uniform/mul,conv5/weights/Initializer/random_uniform/min*'
_output_shapes
:�*
T0* 
_class
loc:@conv5/weights
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
_class
loc:@conv5/biases*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name 
�
conv5/biases/AssignAssignconv5/biasesconv5/biases/Initializer/zeros*
_class
loc:@conv5/biases*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
q
conv5/biases/readIdentityconv5/biases*
_output_shapes
:*
T0*
_class
loc:@conv5/biases
j
model/conv5/dilation_rateConst*
_output_shapes
:*
valueB"      *
dtype0
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
model/Flatten/flatten/ShapeShapemodel/conv5/MaxPool2D/MaxPool*
T0*
out_type0*
_output_shapes
:
s
)model/Flatten/flatten/strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB: 
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
#model/Flatten/flatten/strided_sliceStridedSlicemodel/Flatten/flatten/Shape)model/Flatten/flatten/strided_slice/stack+model/Flatten/flatten/strided_slice/stack_1+model/Flatten/flatten/strided_slice/stack_2*
new_axis_mask *
end_mask *
_output_shapes
: *
T0*
Index0*
shrink_axis_mask*
ellipsis_mask *

begin_mask 
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
model_1/conv2/Conv2DConv2Dmodel_1/conv1/MaxPool2D/MaxPoolconv2/weights/read*
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
model_1/conv2/MaxPool2D/MaxPoolMaxPoolmodel_1/conv2/conv2/Relu*
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
model_1/conv3/conv3/ReluRelumodel_1/conv3/BiasAdd*0
_output_shapes
:���������&�*
T0
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
model_1/conv4/Conv2DConv2Dmodel_1/conv3/MaxPool2D/MaxPoolconv4/weights/read*
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
model_1/conv5/MaxPool2D/MaxPoolMaxPoolmodel_1/conv5/BiasAdd*/
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
%model_1/Flatten/flatten/strided_sliceStridedSlicemodel_1/Flatten/flatten/Shape+model_1/Flatten/flatten/strided_slice/stack-model_1/Flatten/flatten/strided_slice/stack_1-model_1/Flatten/flatten/strided_slice/stack_2*
Index0*
T0*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
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
model_1/Flatten/flatten/ReshapeReshapemodel_1/conv5/MaxPool2D/MaxPool%model_1/Flatten/flatten/Reshape/shape*(
_output_shapes
:����������*
T0*
Tshape0
l
model_2/conv1/dilation_rateConst*
_output_shapes
:*
valueB"      *
dtype0
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
model_2/conv1/conv1/ReluRelumodel_2/conv1/BiasAdd*
T0*0
_output_shapes
:���������2� 
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
model_2/conv2/conv2/ReluRelumodel_2/conv2/BiasAdd*/
_output_shapes
:���������K@*
T0
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
model_2/conv3/Conv2DConv2Dmodel_2/conv2/MaxPool2D/MaxPoolconv3/weights/read*
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
model_2/conv4/dilation_rateConst*
_output_shapes
:*
valueB"      *
dtype0
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
model_2/conv5/Conv2DConv2Dmodel_2/conv4/MaxPool2D/MaxPoolconv5/weights/read*
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
+model_2/Flatten/flatten/strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB: 
w
-model_2/Flatten/flatten/strided_slice/stack_1Const*
_output_shapes
:*
valueB:*
dtype0
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
T0*

axis *
N*
_output_shapes
:
�
model_2/Flatten/flatten/ReshapeReshapemodel_2/conv5/MaxPool2D/MaxPool%model_2/Flatten/flatten/Reshape/shape*(
_output_shapes
:����������*
T0*
Tshape0

mulMulmodel_1/Flatten/flatten/Reshapemodel_2/Flatten/flatten/Reshape*(
_output_shapes
:����������*
T0
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
Pow/yConst*
_output_shapes
: *
valueB
 *   @*
dtype0
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
Sum_2/reduction_indicesConst*
value	B :*
dtype0*
_output_shapes
: 
w
Sum_2SumPow_1Sum_2/reduction_indices*#
_output_shapes
:���������*
	keep_dims( *

Tidx0*
T0
C
Sqrt_1SqrtSum_2*
T0*#
_output_shapes
:���������
H
mul_1MulSqrtSqrt_1*#
_output_shapes
:���������*
T0
H
divRealDivSummul_1*#
_output_shapes
:���������*
T0

mul_2Mulmodel_1/Flatten/flatten/Reshapemodel/Flatten/flatten/Reshape*(
_output_shapes
:����������*
T0
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
Sum_4SumPow_2Sum_4/reduction_indices*
	keep_dims( *

Tidx0*
T0*#
_output_shapes
:���������
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
Pow_3Powmodel/Flatten/flatten/ReshapePow_3/y*(
_output_shapes
:����������*
T0
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
Pow_4/yConst*
_output_shapes
: *
valueB
 *   @*
dtype0
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
Pow_5Powsub_1Pow_5/y*(
_output_shapes
:����������*
T0
Y
Sum_7/reduction_indicesConst*
value	B :*
dtype0*
_output_shapes
: 
{
Sum_7SumPow_5Sum_7/reduction_indices*
	keep_dims(*

Tidx0*
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
!gradients/Mean_grad/Reshape/shapeConst*
_output_shapes
:*
valueB"      *
dtype0
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
gradients/Mean_grad/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
�
gradients/Mean_grad/Prod_1Prodgradients/Mean_grad/Shape_2gradients/Mean_grad/Const_1*
	keep_dims( *

Tidx0*
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
gradients/Maximum_grad/Select_1Select#gradients/Maximum_grad/GreaterEqualgradients/Maximum_grad/zerosgradients/Mean_grad/truediv*'
_output_shapes
:���������*
T0
�
gradients/Maximum_grad/SumSumgradients/Maximum_grad/Select,gradients/Maximum_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
gradients/Maximum_grad/ReshapeReshapegradients/Maximum_grad/Sumgradients/Maximum_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
gradients/Maximum_grad/Sum_1Sumgradients/Maximum_grad/Select_1.gradients/Maximum_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
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
-gradients/add_grad/tuple/control_dependency_1Identitygradients/add_grad/Reshape_1$^gradients/add_grad/tuple/group_deps*/
_class%
#!loc:@gradients/add_grad/Reshape_1*
_output_shapes
: *
T0
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
gradients/sub_2_grad/Sum_1Sum+gradients/add_grad/tuple/control_dependency,gradients/sub_2_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
^
gradients/sub_2_grad/NegNeggradients/sub_2_grad/Sum_1*
_output_shapes
:*
T0
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
gradients/Sum_6_grad/SizeConst*
value	B :*-
_class#
!loc:@gradients/Sum_6_grad/Shape*
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
gradients/Sum_6_grad/Shape_1Const*
dtype0*
_output_shapes
: *
valueB *-
_class#
!loc:@gradients/Sum_6_grad/Shape
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
N*
_output_shapes
:*
T0*-
_class#
!loc:@gradients/Sum_6_grad/Shape
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
 gradients/Sum_7_grad/range/deltaConst*
dtype0*
_output_shapes
: *
value	B :*-
_class#
!loc:@gradients/Sum_7_grad/Shape
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
gradients/Pow_4_grad/mulMulgradients/Sum_6_grad/TilePow_4/y*(
_output_shapes
:����������*
T0
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
gradients/Pow_4_grad/Greater/yConst*
valueB
 *    *
dtype0*
_output_shapes
: 

gradients/Pow_4_grad/GreaterGreatersubgradients/Pow_4_grad/Greater/y*(
_output_shapes
:����������*
T0
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
gradients/Pow_4_grad/zeros_like	ZerosLikesub*(
_output_shapes
:����������*
T0
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
gradients/Pow_4_grad/mul_3Mulgradients/Pow_4_grad/mul_2gradients/Pow_4_grad/Select_1*(
_output_shapes
:����������*
T0
�
gradients/Pow_4_grad/Sum_1Sumgradients/Pow_4_grad/mul_3,gradients/Pow_4_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
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
gradients/Pow_5_grad/Sum_1Sumgradients/Pow_5_grad/mul_3,gradients/Pow_5_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
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
6gradients/model_1/Flatten/flatten/Reshape_grad/ReshapeReshape-gradients/sub_grad/tuple/control_dependency_14gradients/model_1/Flatten/flatten/Reshape_grad/Shape*/
_output_shapes
:���������*
T0*
Tshape0
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
N* 
_output_shapes
::*
T0*
out_type0
�
5gradients/model/conv5/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput(gradients/model/conv5/Conv2D_grad/ShapeNconv5/weights/read;gradients/model/conv5/BiasAdd_grad/tuple/control_dependency*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME*0
_output_shapes
:���������
�*
	dilations
*
T0
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
6gradients/model/conv4/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFiltermodel/conv3/MaxPool2D/MaxPool*gradients/model/conv4/Conv2D_grad/ShapeN:1;gradients/model/conv4/BiasAdd_grad/tuple/control_dependency*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME*(
_output_shapes
:��*
	dilations
*
T0
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
T0*
out_type0*
N* 
_output_shapes
::
�
7gradients/model_2/conv4/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput*gradients/model_2/conv4/Conv2D_grad/ShapeNconv4/weights/read=gradients/model_2/conv4/BiasAdd_grad/tuple/control_dependency*
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
0gradients/model_1/conv3/BiasAdd_grad/BiasAddGradBiasAddGrad0gradients/model_1/conv3/conv3/Relu_grad/ReluGrad*
data_formatNHWC*
_output_shapes	
:�*
T0
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
8gradients/model_1/conv3/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFiltermodel_1/conv2/MaxPool2D/MaxPool,gradients/model_1/conv3/Conv2D_grad/ShapeN:1=gradients/model_1/conv3/BiasAdd_grad/tuple/control_dependency*
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
T0*
data_formatNHWC*
strides
*
ksize
*
paddingSAME
�
8gradients/model/conv2/MaxPool2D/MaxPool_grad/MaxPoolGradMaxPoolGradmodel/conv2/conv2/Relumodel/conv2/MaxPool2D/MaxPool:gradients/model/conv3/Conv2D_grad/tuple/control_dependency*
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
=gradients/model/conv2/BiasAdd_grad/tuple/control_dependency_1Identity.gradients/model/conv2/BiasAdd_grad/BiasAddGrad4^gradients/model/conv2/BiasAdd_grad/tuple/group_deps*
_output_shapes
:@*
T0*A
_class7
53loc:@gradients/model/conv2/BiasAdd_grad/BiasAddGrad
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
7gradients/model_1/conv2/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput*gradients/model_1/conv2/Conv2D_grad/ShapeNconv2/weights/read=gradients/model_1/conv2/BiasAdd_grad/tuple/control_dependency*
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
8gradients/model_1/conv2/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFiltermodel_1/conv1/MaxPool2D/MaxPool,gradients/model_1/conv2/Conv2D_grad/ShapeN:1=gradients/model_1/conv2/BiasAdd_grad/tuple/control_dependency*
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
7gradients/model_2/conv2/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput*gradients/model_2/conv2/Conv2D_grad/ShapeNconv2/weights/read=gradients/model_2/conv2/BiasAdd_grad/tuple/control_dependency*
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
8gradients/model_2/conv2/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFiltermodel_2/conv1/MaxPool2D/MaxPool,gradients/model_2/conv2/Conv2D_grad/ShapeN:1=gradients/model_2/conv2/BiasAdd_grad/tuple/control_dependency*
strides
*
data_formatNHWC*
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
8gradients/model/conv1/MaxPool2D/MaxPool_grad/MaxPoolGradMaxPoolGradmodel/conv1/conv1/Relumodel/conv1/MaxPool2D/MaxPool:gradients/model/conv2/Conv2D_grad/tuple/control_dependency*
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
N* 
_output_shapes
::*
T0*
out_type0
�
7gradients/model_1/conv1/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput*gradients/model_1/conv1/Conv2D_grad/ShapeNconv1/weights/read=gradients/model_1/conv1/BiasAdd_grad/tuple/control_dependency*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME*0
_output_shapes
:���������2�*
	dilations

�
8gradients/model_1/conv1/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterpositive_input,gradients/model_1/conv1/Conv2D_grad/ShapeN:1=gradients/model_1/conv1/BiasAdd_grad/tuple/control_dependency*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME*&
_output_shapes
: *
	dilations
*
T0
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
T0*
data_formatNHWC*
strides
*
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
T0*
data_formatNHWC*
strides
*
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
dtype0*
_output_shapes
:* 
_class
loc:@conv2/weights*%
valueB"          @   
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
:* 
_class
loc:@conv3/weights*%
valueB"      @   �   
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
conv3/weights/Momentum/AssignAssignconv3/weights/Momentum(conv3/weights/Momentum/Initializer/zeros*
validate_shape(*'
_output_shapes
:@�*
use_locking(*
T0* 
_class
loc:@conv3/weights
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
dtype0*
_output_shapes
:* 
_class
loc:@conv4/weights*%
valueB"      �      
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
(conv4/weights/Momentum/Initializer/zerosFill8conv4/weights/Momentum/Initializer/zeros/shape_as_tensor.conv4/weights/Momentum/Initializer/zeros/Const*
T0* 
_class
loc:@conv4/weights*

index_type0*(
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
conv5/weights/Momentum/AssignAssignconv5/weights/Momentum(conv5/weights/Momentum/Initializer/zeros*
validate_shape(*'
_output_shapes
:�*
use_locking(*
T0* 
_class
loc:@conv5/weights
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
+Momentum/update_conv1/weights/ApplyMomentumApplyMomentumconv1/weightsconv1/weights/MomentumMomentum/learning_rategradients/AddN_10Momentum/momentum*
use_nesterov(*&
_output_shapes
: *
use_locking( *
T0* 
_class
loc:@conv1/weights
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
*Momentum/update_conv2/biases/ApplyMomentumApplyMomentumconv2/biasesconv2/biases/MomentumMomentum/learning_rategradients/AddN_7Momentum/momentum*
use_locking( *
T0*
_class
loc:@conv2/biases*
use_nesterov(*
_output_shapes
:@
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
save/Assign_10Assignconv3/biases/Momentumsave/RestoreV2:10*
T0*
_class
loc:@conv3/biases*
validate_shape(*
_output_shapes	
:�*
use_locking(
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
save/Assign_12Assignconv3/weights/Momentumsave/RestoreV2:12*
validate_shape(*'
_output_shapes
:@�*
use_locking(*
T0* 
_class
loc:@conv3/weights
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
save/Assign_19Assignconv5/weightssave/RestoreV2:19*
validate_shape(*'
_output_shapes
:�*
use_locking(*
T0* 
_class
loc:@conv5/weights
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
dtype0*
_output_shapes
: * 
valueB Bconv1/weights_1
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
conv4/biases_1HistogramSummaryconv4/biases_1/tagconv4/biases/read*
_output_shapes
: *
T0
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
conv5/biases_1HistogramSummaryconv5/biases_1/tagconv5/biases/read*
T0*
_output_shapes
: 
�
Merge/MergeSummaryMergeSummarysteplossconv1/weights_1conv1/biases_1conv2/weights_1conv2/biases_1conv3/weights_1conv3/biases_1conv4/weights_1conv4/biases_1conv5/weights_1conv5/biases_1*
N*
_output_shapes
: "�;�+
5     ��B�	\�u'��AJ��
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
negative_inputPlaceholder*%
shape:���������2�*
dtype0*0
_output_shapes
:���������2�
�
.conv1/weights/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:* 
_class
loc:@conv1/weights*%
valueB"             
�
,conv1/weights/Initializer/random_uniform/minConst*
_output_shapes
: * 
_class
loc:@conv1/weights*
valueB
 *�Er�*
dtype0
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
conv1/weights/readIdentityconv1/weights*
T0* 
_class
loc:@conv1/weights*&
_output_shapes
: 
�
conv1/biases/Initializer/zerosConst*
dtype0*
_output_shapes
: *
_class
loc:@conv1/biases*
valueB *    
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
conv1/biases/AssignAssignconv1/biasesconv1/biases/Initializer/zeros*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*
_class
loc:@conv1/biases
q
conv1/biases/readIdentityconv1/biases*
_class
loc:@conv1/biases*
_output_shapes
: *
T0
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
model/conv1/BiasAddBiasAddmodel/conv1/Conv2Dconv1/biases/read*0
_output_shapes
:���������2� *
T0*
data_formatNHWC
n
model/conv1/conv1/ReluRelumodel/conv1/BiasAdd*0
_output_shapes
:���������2� *
T0
�
model/conv1/MaxPool2D/MaxPoolMaxPoolmodel/conv1/conv1/Relu*
paddingSAME*/
_output_shapes
:���������K *
T0*
data_formatNHWC*
strides
*
ksize
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
dtype0*
_output_shapes
: * 
_class
loc:@conv2/weights*
valueB
 *��L=
�
6conv2/weights/Initializer/random_uniform/RandomUniformRandomUniform.conv2/weights/Initializer/random_uniform/shape*

seed *
T0* 
_class
loc:@conv2/weights*
seed2 *
dtype0*&
_output_shapes
: @
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
conv2/biases/Initializer/zerosConst*
_class
loc:@conv2/biases*
valueB@*    *
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
conv2/biases/AssignAssignconv2/biasesconv2/biases/Initializer/zeros*
_output_shapes
:@*
use_locking(*
T0*
_class
loc:@conv2/biases*
validate_shape(
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
T0*
data_formatNHWC*
strides
*
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
model/conv2/MaxPool2D/MaxPoolMaxPoolmodel/conv2/conv2/Relu*
data_formatNHWC*
strides
*
ksize
*
paddingSAME*/
_output_shapes
:���������&@*
T0
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
VariableV2*
shape:@�*
dtype0*'
_output_shapes
:@�*
shared_name * 
_class
loc:@conv3/weights*
	container 
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
conv3/biases/Initializer/zerosConst*
_output_shapes	
:�*
_class
loc:@conv3/biases*
valueB�*    *
dtype0
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
valueB"      *
dtype0*
_output_shapes
:
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
conv4/weights/AssignAssignconv4/weights(conv4/weights/Initializer/random_uniform*
T0* 
_class
loc:@conv4/weights*
validate_shape(*(
_output_shapes
:��*
use_locking(
�
conv4/weights/readIdentityconv4/weights*(
_output_shapes
:��*
T0* 
_class
loc:@conv4/weights
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
conv4/biases/AssignAssignconv4/biasesconv4/biases/Initializer/zeros*
T0*
_class
loc:@conv4/biases*
validate_shape(*
_output_shapes	
:�*
use_locking(
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
model/conv4/MaxPool2D/MaxPoolMaxPoolmodel/conv4/conv4/Relu*
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
,conv5/weights/Initializer/random_uniform/maxConst*
_output_shapes
: * 
_class
loc:@conv5/weights*
valueB
 *��>*
dtype0
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
conv5/biases/AssignAssignconv5/biasesconv5/biases/Initializer/zeros*
use_locking(*
T0*
_class
loc:@conv5/biases*
validate_shape(*
_output_shapes
:
q
conv5/biases/readIdentityconv5/biases*
_output_shapes
:*
T0*
_class
loc:@conv5/biases
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
model/conv5/MaxPool2D/MaxPoolMaxPoolmodel/conv5/BiasAdd*
T0*
strides
*
data_formatNHWC*
ksize
*
paddingSAME*/
_output_shapes
:���������
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
model_1/conv2/Conv2DConv2Dmodel_1/conv1/MaxPool2D/MaxPoolconv2/weights/read*
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
model_1/conv3/conv3/ReluRelumodel_1/conv3/BiasAdd*0
_output_shapes
:���������&�*
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
:����������
l
model_1/conv4/dilation_rateConst*
dtype0*
_output_shapes
:*
valueB"      
�
model_1/conv4/Conv2DConv2Dmodel_1/conv3/MaxPool2D/MaxPoolconv4/weights/read*0
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
dtype0*
_output_shapes
:*
valueB"      
�
model_2/conv3/Conv2DConv2Dmodel_2/conv2/MaxPool2D/MaxPoolconv3/weights/read*
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
T0*
strides
*
data_formatNHWC
l
model_2/conv4/dilation_rateConst*
dtype0*
_output_shapes
:*
valueB"      
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
model_2/conv5/Conv2DConv2Dmodel_2/conv4/MaxPool2D/MaxPoolconv5/weights/read*
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
-model_2/Flatten/flatten/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
�
%model_2/Flatten/flatten/strided_sliceStridedSlicemodel_2/Flatten/flatten/Shape+model_2/Flatten/flatten/strided_slice/stack-model_2/Flatten/flatten/strided_slice/stack_1-model_2/Flatten/flatten/strided_slice/stack_2*
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
Sum/reduction_indicesConst*
dtype0*
_output_shapes
: *
value	B :
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
PowPowmodel_1/Flatten/flatten/ReshapePow/y*
T0*(
_output_shapes
:����������
Y
Sum_1/reduction_indicesConst*
dtype0*
_output_shapes
: *
value	B :
u
Sum_1SumPowSum_1/reduction_indices*
T0*#
_output_shapes
:���������*

Tidx0*
	keep_dims( 
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
Sum_4SumPow_2Sum_4/reduction_indices*#
_output_shapes
:���������*

Tidx0*
	keep_dims( *
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
Sum_5SumPow_3Sum_5/reduction_indices*

Tidx0*
	keep_dims( *
T0*#
_output_shapes
:���������
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
Pow_4/yConst*
dtype0*
_output_shapes
: *
valueB
 *   @
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
Sum_6SumPow_4Sum_6/reduction_indices*

Tidx0*
	keep_dims(*
T0*'
_output_shapes
:���������
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
Pow_5Powsub_1Pow_5/y*(
_output_shapes
:����������*
T0
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
Variable/AssignAssignVariableVariable/initial_value*
T0*
_class
loc:@Variable*
validate_shape(*
_output_shapes
: *
use_locking(
a
Variable/readIdentityVariable*
_output_shapes
: *
T0*
_class
loc:@Variable
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
gradients/FillFillgradients/Shapegradients/grad_ys_0*
_output_shapes
: *
T0*

index_type0
r
!gradients/Mean_grad/Reshape/shapeConst*
dtype0*
_output_shapes
:*
valueB"      
�
gradients/Mean_grad/ReshapeReshapegradients/Fill!gradients/Mean_grad/Reshape/shape*
_output_shapes

:*
T0*
Tshape0
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
gradients/Maximum_grad/Sum_1Sumgradients/Maximum_grad/Select_1.gradients/Maximum_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
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
gradients/add_grad/Reshape_1Reshapegradients/add_grad/Sum_1gradients/add_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
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
gradients/sub_2_grad/Sum_1Sum+gradients/add_grad/tuple/control_dependency,gradients/sub_2_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
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
gradients/Sum_6_grad/SizeConst*-
_class#
!loc:@gradients/Sum_6_grad/Shape*
value	B :*
dtype0*
_output_shapes
: 
�
gradients/Sum_6_grad/addAddSum_6/reduction_indicesgradients/Sum_6_grad/Size*
_output_shapes
: *
T0*-
_class#
!loc:@gradients/Sum_6_grad/Shape
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
gradients/Sum_6_grad/rangeRange gradients/Sum_6_grad/range/startgradients/Sum_6_grad/Size gradients/Sum_6_grad/range/delta*

Tidx0*-
_class#
!loc:@gradients/Sum_6_grad/Shape*
_output_shapes
:
�
gradients/Sum_6_grad/Fill/valueConst*-
_class#
!loc:@gradients/Sum_6_grad/Shape*
value	B :*
dtype0*
_output_shapes
: 
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
N*
_output_shapes
:*
T0*-
_class#
!loc:@gradients/Sum_6_grad/Shape
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
gradients/Sum_7_grad/Shape_1Const*
dtype0*
_output_shapes
: *-
_class#
!loc:@gradients/Sum_7_grad/Shape*
valueB 
�
 gradients/Sum_7_grad/range/startConst*-
_class#
!loc:@gradients/Sum_7_grad/Shape*
value	B : *
dtype0*
_output_shapes
: 
�
 gradients/Sum_7_grad/range/deltaConst*
dtype0*
_output_shapes
: *-
_class#
!loc:@gradients/Sum_7_grad/Shape*
value	B :
�
gradients/Sum_7_grad/rangeRange gradients/Sum_7_grad/range/startgradients/Sum_7_grad/Size gradients/Sum_7_grad/range/delta*

Tidx0*-
_class#
!loc:@gradients/Sum_7_grad/Shape*
_output_shapes
:
�
gradients/Sum_7_grad/Fill/valueConst*-
_class#
!loc:@gradients/Sum_7_grad/Shape*
value	B :*
dtype0*
_output_shapes
: 
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
gradients/Sum_7_grad/MaximumMaximum"gradients/Sum_7_grad/DynamicStitchgradients/Sum_7_grad/Maximum/y*
_output_shapes
:*
T0*-
_class#
!loc:@gradients/Sum_7_grad/Shape
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
gradients/Pow_4_grad/SumSumgradients/Pow_4_grad/mul_1*gradients/Pow_4_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
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
gradients/Pow_4_grad/LogLoggradients/Pow_4_grad/Select*(
_output_shapes
:����������*
T0
d
gradients/Pow_4_grad/zeros_like	ZerosLikesub*(
_output_shapes
:����������*
T0
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
gradients/Pow_5_grad/GreaterGreatersub_1gradients/Pow_5_grad/Greater/y*(
_output_shapes
:����������*
T0
i
$gradients/Pow_5_grad/ones_like/ShapeShapesub_1*
_output_shapes
:*
T0*
out_type0
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
gradients/Pow_5_grad/LogLoggradients/Pow_5_grad/Select*(
_output_shapes
:����������*
T0
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
gradients/Pow_5_grad/Sum_1Sumgradients/Pow_5_grad/mul_3,gradients/Pow_5_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
gradients/Pow_5_grad/Reshape_1Reshapegradients/Pow_5_grad/Sum_1gradients/Pow_5_grad/Shape_1*
_output_shapes
: *
T0*
Tshape0
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
(gradients/sub_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/sub_grad/Shapegradients/sub_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
gradients/sub_grad/SumSum-gradients/Pow_4_grad/tuple/control_dependency(gradients/sub_grad/BroadcastGradientArgs*

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
gradients/sub_1_grad/SumSum-gradients/Pow_5_grad/tuple/control_dependency*gradients/sub_1_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
gradients/sub_1_grad/ReshapeReshapegradients/sub_1_grad/Sumgradients/sub_1_grad/Shape*(
_output_shapes
:����������*
T0*
Tshape0
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
/gradients/sub_1_grad/tuple/control_dependency_1Identitygradients/sub_1_grad/Reshape_1&^gradients/sub_1_grad/tuple/group_deps*(
_output_shapes
:����������*
T0*1
_class'
%#loc:@gradients/sub_1_grad/Reshape_1
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
4gradients/model_2/Flatten/flatten/Reshape_grad/ShapeShapemodel_2/conv5/MaxPool2D/MaxPool*
_output_shapes
:*
T0*
out_type0
�
6gradients/model_2/Flatten/flatten/Reshape_grad/ReshapeReshape/gradients/sub_1_grad/tuple/control_dependency_14gradients/model_2/Flatten/flatten/Reshape_grad/Shape*/
_output_shapes
:���������*
T0*
Tshape0
�
:gradients/model_1/conv5/MaxPool2D/MaxPool_grad/MaxPoolGradMaxPoolGradmodel_1/conv5/BiasAddmodel_1/conv5/MaxPool2D/MaxPool6gradients/model_1/Flatten/flatten/Reshape_grad/Reshape*
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
:���������

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
:���������
*
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
7gradients/model_1/conv5/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput*gradients/model_1/conv5/Conv2D_grad/ShapeNconv5/weights/read=gradients/model_1/conv5/BiasAdd_grad/tuple/control_dependency*0
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
<gradients/model/conv5/Conv2D_grad/tuple/control_dependency_1Identity6gradients/model/conv5/Conv2D_grad/Conv2DBackpropFilter3^gradients/model/conv5/Conv2D_grad/tuple/group_deps*'
_output_shapes
:�*
T0*I
_class?
=;loc:@gradients/model/conv5/Conv2D_grad/Conv2DBackpropFilter
�
*gradients/model_2/conv5/Conv2D_grad/ShapeNShapeNmodel_2/conv4/MaxPool2D/MaxPoolconv5/weights/read*
N* 
_output_shapes
::*
T0*
out_type0
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
>gradients/model_2/conv5/Conv2D_grad/tuple/control_dependency_1Identity8gradients/model_2/conv5/Conv2D_grad/Conv2DBackpropFilter5^gradients/model_2/conv5/Conv2D_grad/tuple/group_deps*'
_output_shapes
:�*
T0*K
_classA
?=loc:@gradients/model_2/conv5/Conv2D_grad/Conv2DBackpropFilter
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
7gradients/model_1/conv4/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput*gradients/model_1/conv4/Conv2D_grad/ShapeNconv4/weights/read=gradients/model_1/conv4/BiasAdd_grad/tuple/control_dependency*
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
<gradients/model_1/conv4/Conv2D_grad/tuple/control_dependencyIdentity7gradients/model_1/conv4/Conv2D_grad/Conv2DBackpropInput5^gradients/model_1/conv4/Conv2D_grad/tuple/group_deps*0
_output_shapes
:����������*
T0*J
_class@
><loc:@gradients/model_1/conv4/Conv2D_grad/Conv2DBackpropInput
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
7gradients/model_2/conv4/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput*gradients/model_2/conv4/Conv2D_grad/ShapeNconv4/weights/read=gradients/model_2/conv4/BiasAdd_grad/tuple/control_dependency*
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
8gradients/model_2/conv4/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFiltermodel_2/conv3/MaxPool2D/MaxPool,gradients/model_2/conv4/Conv2D_grad/ShapeN:1=gradients/model_2/conv4/BiasAdd_grad/tuple/control_dependency*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME*(
_output_shapes
:��*
	dilations
*
T0
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
>gradients/model_2/conv4/Conv2D_grad/tuple/control_dependency_1Identity8gradients/model_2/conv4/Conv2D_grad/Conv2DBackpropFilter5^gradients/model_2/conv4/Conv2D_grad/tuple/group_deps*K
_classA
?=loc:@gradients/model_2/conv4/Conv2D_grad/Conv2DBackpropFilter*(
_output_shapes
:��*
T0
�
gradients/AddN_3AddN?gradients/model_1/conv4/BiasAdd_grad/tuple/control_dependency_1=gradients/model/conv4/BiasAdd_grad/tuple/control_dependency_1?gradients/model_2/conv4/BiasAdd_grad/tuple/control_dependency_1*
T0*C
_class9
75loc:@gradients/model_1/conv4/BiasAdd_grad/BiasAddGrad*
N*
_output_shapes	
:�
�
:gradients/model_1/conv3/MaxPool2D/MaxPool_grad/MaxPoolGradMaxPoolGradmodel_1/conv3/conv3/Relumodel_1/conv3/MaxPool2D/MaxPool<gradients/model_1/conv4/Conv2D_grad/tuple/control_dependency*
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
8gradients/model_1/conv3/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFiltermodel_1/conv2/MaxPool2D/MaxPool,gradients/model_1/conv3/Conv2D_grad/ShapeN:1=gradients/model_1/conv3/BiasAdd_grad/tuple/control_dependency*
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
6gradients/model/conv3/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFiltermodel/conv2/MaxPool2D/MaxPool*gradients/model/conv3/Conv2D_grad/ShapeN:1;gradients/model/conv3/BiasAdd_grad/tuple/control_dependency*
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
=gradients/model_1/conv2/BiasAdd_grad/tuple/control_dependencyIdentity0gradients/model_1/conv2/conv2/Relu_grad/ReluGrad6^gradients/model_1/conv2/BiasAdd_grad/tuple/group_deps*C
_class9
75loc:@gradients/model_1/conv2/conv2/Relu_grad/ReluGrad*/
_output_shapes
:���������K@*
T0
�
?gradients/model_1/conv2/BiasAdd_grad/tuple/control_dependency_1Identity0gradients/model_1/conv2/BiasAdd_grad/BiasAddGrad6^gradients/model_1/conv2/BiasAdd_grad/tuple/group_deps*
_output_shapes
:@*
T0*C
_class9
75loc:@gradients/model_1/conv2/BiasAdd_grad/BiasAddGrad
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
0gradients/model_2/conv2/BiasAdd_grad/BiasAddGradBiasAddGrad0gradients/model_2/conv2/conv2/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes
:@
�
5gradients/model_2/conv2/BiasAdd_grad/tuple/group_depsNoOp1^gradients/model_2/conv2/BiasAdd_grad/BiasAddGrad1^gradients/model_2/conv2/conv2/Relu_grad/ReluGrad
�
=gradients/model_2/conv2/BiasAdd_grad/tuple/control_dependencyIdentity0gradients/model_2/conv2/conv2/Relu_grad/ReluGrad6^gradients/model_2/conv2/BiasAdd_grad/tuple/group_deps*C
_class9
75loc:@gradients/model_2/conv2/conv2/Relu_grad/ReluGrad*/
_output_shapes
:���������K@*
T0
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
(gradients/model/conv2/Conv2D_grad/ShapeNShapeNmodel/conv1/MaxPool2D/MaxPoolconv2/weights/read* 
_output_shapes
::*
T0*
out_type0*
N
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
7gradients/model_2/conv2/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput*gradients/model_2/conv2/Conv2D_grad/ShapeNconv2/weights/read=gradients/model_2/conv2/BiasAdd_grad/tuple/control_dependency*
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
8gradients/model_2/conv2/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFiltermodel_2/conv1/MaxPool2D/MaxPool,gradients/model_2/conv2/Conv2D_grad/ShapeN:1=gradients/model_2/conv2/BiasAdd_grad/tuple/control_dependency*
strides
*
data_formatNHWC*
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
.gradients/model/conv1/conv1/Relu_grad/ReluGradReluGrad8gradients/model/conv1/MaxPool2D/MaxPool_grad/MaxPoolGradmodel/conv1/conv1/Relu*0
_output_shapes
:���������2� *
T0
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
7gradients/model_1/conv1/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput*gradients/model_1/conv1/Conv2D_grad/ShapeNconv1/weights/read=gradients/model_1/conv1/BiasAdd_grad/tuple/control_dependency*
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
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(
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
7gradients/model_2/conv1/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput*gradients/model_2/conv1/Conv2D_grad/ShapeNconv1/weights/read=gradients/model_2/conv1/BiasAdd_grad/tuple/control_dependency*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME*0
_output_shapes
:���������2�*
	dilations

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
VariableV2*
shape: *
dtype0*&
_output_shapes
: *
shared_name * 
_class
loc:@conv1/weights*
	container 
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
'conv1/biases/Momentum/Initializer/zerosConst*
valueB *    *
_class
loc:@conv1/biases*
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
8conv2/weights/Momentum/Initializer/zeros/shape_as_tensorConst*%
valueB"          @   * 
_class
loc:@conv2/weights*
dtype0*
_output_shapes
:
�
.conv2/weights/Momentum/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    * 
_class
loc:@conv2/weights
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
	container *
shape:@*
dtype0*
_output_shapes
:@*
shared_name *
_class
loc:@conv2/biases
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
conv3/weights/Momentum/AssignAssignconv3/weights/Momentum(conv3/weights/Momentum/Initializer/zeros*
validate_shape(*'
_output_shapes
:@�*
use_locking(*
T0* 
_class
loc:@conv3/weights
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
dtype0*
_output_shapes
:*
shared_name *
_class
loc:@conv5/biases*
	container *
shape:
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
Momentum/valueConst^Momentum/update*
dtype0*
_output_shapes
: *
_class
loc:@Variable*
value	B :
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
dtype0*
_output_shapes
: * 
valueB Bconv1/weights_1
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
conv5/biases_1HistogramSummaryconv5/biases_1/tagconv5/biases/read*
_output_shapes
: *
T0
�
Merge/MergeSummaryMergeSummarysteplossconv1/weights_1conv1/biases_1conv2/weights_1conv2/biases_1conv3/weights_1conv3/biases_1conv4/weights_1conv4/biases_1conv5/weights_1conv5/biases_1*
N*
_output_shapes
: ""�
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
conv5/biases:0conv5/biases/Assignconv5/biases/read:02 conv5/biases/Initializer/zeros:08"
train_op


Momentumu@�b�9      �t	Ĕ�v'��A*�s

step    

lossB.�>
�
conv1/weights_1*�	   `�E��   @�E�?     `�@!  @�1� �)�p����@2�����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���VlQ.��7Kaa+�I�I�)�(�+A�F�&��S�F !�ji6�9���.����ڋ�pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?ji6�9�?�S�F !?�[^:��"?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?�������:�             �G@     `g@     �h@     `g@     �b@     `b@     �`@     �`@     �[@     �U@     �W@     @S@      R@     �L@      O@      J@     @Q@      J@     �I@      E@      ;@      F@      ;@      8@      5@      8@      :@      6@      2@      ,@      0@      "@      &@      @      @      $@       @      "@       @      @       @      $@      @              @      @      @      @      @      @      @      @       @               @       @       @              �?              �?               @      �?      �?              �?              �?              �?      �?              �?      �?      �?               @      �?              �?      �?      �?       @      �?              @      �?               @      @      �?       @       @       @      @      @      @      �?      @      @      $@      @       @      "@       @      "@      ,@      &@      *@      .@      1@      4@      2@      ;@      5@      =@      :@      4@     �@@      D@     �K@     �D@      D@     �Q@      R@     �M@      S@      N@     �T@     �T@     @]@     �Y@     ``@     @b@     @d@      d@     �f@     �i@      J@        
O
conv1/biases_1*=      @@2        �-���q=�������:              @@        
�
conv2/weights_1*�	   �����   �[��?      �@!   �lk'@)�Y�%%E@2����g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7���h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ��~]�[Ӿjqs&\�ѾK+�E��Ͼ�u��gr�>�MZ��K�>
�/eq
�>;�"�q�>['�?��>K+�E���>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?�������:�             �@     �@     �@     ��@     ��@     x�@     �@     ܒ@     ��@     \�@     ��@     ��@     ��@     @�@     ��@     �@     p�@     p~@     �{@     `x@     �u@     pu@     �r@      p@     �p@     �l@     �k@     �g@     �h@     @d@      a@     �]@     �\@      ^@     �Z@      Z@     @S@     �M@      N@     �L@      P@      N@     �F@      ?@     �B@     �B@      7@      9@      8@      3@      6@      0@      5@      4@      1@      ,@      *@      $@      &@      &@      @      @      @      @       @      @      @      @      @      @      @      @       @      �?       @      @      �?               @      @              @      @               @      �?       @      �?               @              �?               @      �?              �?              �?              �?              �?      �?              �?      �?              @              �?              �?               @      �?      �?       @       @       @      @      @      �?       @      �?       @      @      @      @      @      @      @      @      $@      @      @      &@      *@      "@      &@      *@      .@      5@      3@      4@      ;@      8@      7@      E@      =@     �F@      D@     �D@     �D@     �J@      L@     @U@     @Q@     @X@     �V@     @S@     @\@     @_@      d@     �b@      e@     �g@     �g@     @n@     �p@      q@     pu@     �u@      u@     �w@     �{@      �@     ��@     0�@     �@      �@     �@     ��@     ��@     ��@     ��@     ��@     ��@     L�@     d�@     ��@     ,�@     Ҡ@     (�@        
O
conv2/biases_1*=      P@2        �-���q=�������:              P@        
�
conv3/weights_1*�	   �w+��   `+�?      �@!  �JPT'�)q���6_U@2�����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%�jqs&\�ѾK+�E��Ͼ5�"�g���0�6�/n���u`P+d����n�����X$�z��
�}�����u`P+d�>0�6�/n�>;�"�q�>['�?��>��~]�[�>��>M|K�>�ѩ�-�>���%�>�uE����>�f����>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?�������:�             �@      �@     �@     �@     H�@     ��@     r�@     �@     @�@     8�@     �@     t�@     x�@     \�@     ��@     `�@     0�@     ��@     ��@     ��@     ؀@     �~@     z@     `y@     �x@     �t@     t@     �p@     �l@     @k@     �i@     `j@     @c@      e@     �d@     �a@     �Y@     @^@      X@     �X@     �S@     @T@     @Q@      L@      J@     �L@     �B@      B@     �E@      <@      A@      :@      3@      .@      4@      0@      7@      3@      1@      $@      *@       @       @       @      @      @      @      @      @      @      @      @      @              @      �?      @       @      �?      @              �?      �?      �?      �?       @      �?      �?      �?              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?               @      �?      �?               @      @      @      @       @      �?       @       @       @       @       @       @      @      @      @      @      "@       @       @      @      @      @      @      *@      ,@      ,@      .@      *@      1@      *@      4@      .@      3@      0@      @@      @@      E@      A@      E@      G@      F@      J@      J@     @P@     �S@     �T@     @Z@     @W@     �Z@     �^@     �`@     �c@     �a@     @c@      i@     �l@     �k@      n@     �q@     �q@     r@     @x@     0x@      ~@     P~@     p@     (�@     Є@     ��@     �@     `�@     ��@     h�@     ��@     <�@     Е@     $�@     �@     ��@     �@     ��@     ��@     @�@     R�@     �@     �@        
O
conv3/biases_1*=      `@2        �-���q=�������:              `@        
�
conv4/weights_1*�	   �����   ����?      �@!   Zz9�)f��ɑme@2�Ӗ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.���vV�R9��T7����5�i}1���d�r�x?�x��I��P=��pz�w�7��})�l a�ѩ�-߾E��a�Wܾ�����>
�/eq
�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�������:�             �a@     ,�@     �@     ��@     `�@     �@     `�@     ��@     H�@     ؅@     `�@     ��@     �@     P}@      |@     pw@     �t@     0t@     t@     �q@     �m@     �j@     �h@     `i@     @g@      e@     �_@     �V@     �]@     @Z@     @R@     @U@     @R@     �R@      P@     @P@      F@     �J@      E@      B@     �D@      9@      8@      =@      >@      6@      :@      2@      4@      5@      ,@      *@      0@       @      $@      @      (@       @      @      @      $@      @      @      @      @       @       @      @      �?      @       @      �?       @               @              �?              @       @              �?      �?              �?              �?              �?              �?      �?      �?              �?              @      �?      �?       @               @              �?       @      �?      �?      �?      @       @       @              �?      �?      �?       @      �?      �?      @      @      @       @       @       @      "@      $@      @      @      ,@      &@      .@      1@      0@      &@      2@      5@      9@      >@      A@      =@      E@     �@@      F@     �K@     �H@     �L@     �P@     �R@      Q@     �P@      T@     �U@     @`@      [@     �b@     �c@      f@     `d@     @h@      j@     �j@     Pp@     �s@     t@     v@     {@     �y@     @{@     p�@      �@     ��@     ��@     �@     �@     Ќ@     0�@     ��@     ��@     (�@     ܖ@     `a@        
O
conv4/biases_1*=      p@2        �-���q=�������:              p@        
�
conv5/weights_1*�	    Ț¿    x��?      �@!  ����@)�����xI@2�	�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�a�$��{E��T���C��!�A�d�\D�X=���%>��:���%�V6��u�w74��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9����ڋ��vV�R9��5�i}1���d�r�>h�'��f�ʜ�7
��T7��?�vV�R9?��ڋ?�.�?�S�F !?�[^:��"?U�4@@�$?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?�������:�	             �l@     �q@      r@     `n@     �k@      m@      h@     �c@     �`@     @c@     �`@     �[@      Z@     @X@     �Y@     �R@     @T@     �S@     �P@      L@      D@      ?@     �G@      9@     �A@      =@      B@      2@      =@      ,@      2@      ,@      5@      (@      ,@      (@      @      (@      @       @      (@       @      @      @      @      @      @      @       @      @      @       @      @       @      @              �?      �?               @              �?              �?              �?              �?       @              �?              �?              �?              �?              �?              �?      �?              �?      �?              �?      �?      �?      �?       @      �?      �?      @      @              @      �?      @       @      @      @       @      @      @      @      @      @      @      @      @      "@      "@      $@      2@      .@      7@       @      .@      &@      4@      8@      8@      8@      @@      >@     �E@     �F@      H@      E@      N@      P@      P@     �R@      T@     �W@      [@     @]@     @]@      `@      d@     �b@      e@     �b@      i@     �m@     �o@     `s@     0t@     `l@        
O
conv5/biases_1*=      <@2        �-���q=�������:              <@        ğ1�W      ��-l	�Ux'��A*��

step  �?

loss���>
�
conv1/weights_1*�	   ��`��   `�{�?     `�@! �7e� �)�����@2�	����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�U�4@@�$��[^:��"��5�i}1���d�r�x?�x���FF�G �>�?�s���pz�w�7��})�l a�E��a�Wܾ�iD*L�پ��n����>�u`P+d�>��[�?1��a˲?6�]��?x?�x�?��d�r?�T7��?�vV�R9?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?�������:�	              G@     �g@     �h@     �g@     �b@     `b@     �`@     ``@     �Z@      W@     @X@     �Q@     �R@     �L@     @P@     �J@     @Q@      I@     �J@     �D@      =@     �D@      <@      6@      6@      6@      :@      8@      0@      ,@      3@      "@      @      $@      @      &@      @      @      @      $@      @      @      @      "@      @      @              @      @       @      @      @      @      �?       @      �?               @              �?              �?              �?      �?              �?              �?              �?              �?              �?      �?              �?              �?               @      �?              �?      �?      �?       @      �?      �?              �?      �?      �?              �?      @      �?      @      @      @       @      @       @      @      @      @      "@      "@      "@      @      $@      .@      $@      1@      (@      0@      7@      2@      8@      9@      7@      @@      .@     �B@      C@      I@      F@     �E@      Q@     �Q@     �O@      S@      P@     �S@     �S@     @^@      Y@     �`@     �b@     �c@      d@      g@     `i@     �J@        
�
conv1/biases_1*�	    ��<�   `jJ?      @@!  `#�g?)�qf�I�>2�d�\D�X=���%>��:��u�w74���82���bȬ�0���VlQ.��T7����5�i}1�����ž�XQ�þ��>M|K�>�_�T�l�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>f�ʜ�7
?>h�'�?x?�x�?��d�r?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?�qU���I?IcD���L?�������:�               @              �?              �?              �?              �?              �?              �?              �?      �?      �?              �?              �?      �?      �?              �?      �?      �?      �?              �?       @      �?              @      �?               @               @              �?        
�
conv2/weights_1*�	   @Ӥ��   �N��?      �@! $�(9(@)�����%E@2����g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(龢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�Ѿ
�/eq
Ⱦ����ž�XQ�þ��~��¾���?�ګ�;9��R��.��fc���X$�z���_�T�l�>�iD*L��>���%�>�uE����>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?�������:�             �@     ڠ@      �@     P�@     ��@     p�@     �@     �@     p�@     X�@     ��@     ��@     @�@     x�@     p�@      �@     x�@     �~@     �{@      x@     �u@     pu@     @r@     Pp@     `p@     �m@      k@      h@     @h@     �c@      b@     �\@     @]@     @^@      Z@     �Y@     �S@      O@      M@      M@      O@     �O@      I@      :@      C@      @@      @@      7@      8@      9@      *@      3@      .@      :@      ,@      5@       @      @      *@      $@      $@       @      @      @      @      @      @       @      @      @      @      @      @      @      �?      @       @      �?              �?       @      @              �?               @      @              �?      �?      �?              �?              �?      �?      �?              �?              �?              �?              �?              �?               @              �?               @               @              �?      �?       @      @      �?       @      @      @              @              @      @      @      @      @      @      @      @      @      $@      @      (@      0@       @      &@      0@      2@      0@      5@      0@      ;@      6@      >@      ;@     �D@     �F@      ?@      I@      C@     �L@      L@     �T@      Q@     @X@      V@     �T@     �\@      ^@     �d@     `b@     �d@     �g@     @h@     �m@     �p@     �q@      u@     �u@     u@     `x@     p{@     �@      �@     ȁ@     0�@     �@     �@     ��@     ��@     ��@     đ@     ��@     ��@     @�@     P�@     �@     @�@     ܠ@     ,�@        
�

conv2/biases_1*�
	    ]�C�   �NO?      P@!  j�[?)�ѐ��>2�a�$��{E��T���C�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9�x?�x��>h�'�������6�]���1��a˲���[���FF�G �>�?�s����f�����uE����jqs&\�ѾK+�E��Ͼ�[�=�k���*��ڽ���ӤP�����z!�?��        �-���q=
�/eq
�>;�"�q�>�iD*L��>E��a�W�>��(���>a�Ϭ(�>�ߊ4F��>})�l a�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?
����G?�qU���I?IcD���L?k�1^�sO?�������:�              �?              �?       @      �?      �?      �?      �?               @              @      �?      �?      �?      �?      �?               @              �?              �?              �?      �?      �?              �?              �?              �?              �?               @              �?              �?              �?              �?              �?              �?      �?      �?              �?               @      �?              �?      �?       @      @              �?      �?      �?               @      �?              �?       @              �?      �?      �?              �?              �?        
�
conv3/weights_1*�	    O7��    B5�?      �@!��ni&�)-�Toc_U@2�����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a�h���`�8K�ߝ���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾['�?�;;�"�qʾ0�6�/n���u`P+d���MZ��K���u��gr��5�"�g��>G&�$�>�*��ڽ>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>�_�T�l�>�iD*L��>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?�������:�             Є@     �@     �@     �@     N�@     ��@     x�@     �@     H�@     `�@     �@     t�@     l�@     <�@     �@     Њ@     Љ@     ؇@     Ȅ@     ��@     ��@     �~@     �z@     py@     �x@     pt@      t@     0q@     �l@     `k@     @j@     �i@     �c@     @e@     �d@     �a@     @Y@      _@      W@     �X@     �S@     �R@     @R@      O@     �K@     �I@      B@      @@      G@      :@     �@@      7@      7@      0@      .@      3@      7@      ,@      5@      (@      (@      *@      @      @      @      @      $@              @       @      @      @      @      @              @               @       @      @      @      �?      �?      �?              �?              �?               @      �?       @              �?      �?      �?              �?              �?              �?              �?      �?              �?              �?      �?              �?              �?      @      �?      �?      �?      �?      �?      @       @      @       @       @      @       @      @      @      @      @       @       @      @      @      @      @       @      "@      "@      @      *@      &@      ,@      *@      0@      0@      7@      ,@      :@      1@      4@     �A@      D@      ?@     �D@     �I@     �H@      I@     �H@     @Q@     �R@     �T@     �Y@     �W@     @Z@     @^@     `a@     �c@     `a@     �c@      i@     �l@      l@     �l@     Pr@     �q@     0r@     0x@     @x@     �}@     0~@     �@     (�@     ��@     ��@     �@     h�@     ��@     p�@     �@     H�@     ��@     �@     �@     ��@     ��@     ~�@     ��@     :�@     D�@     �@     8�@        
�
conv3/biases_1*�	   @ڞG�    W	F?      `@!  8]/]d?)=�X�C�>2�
����G�a�$��{E��T���C��!�A�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�>h�'��f�ʜ�7
������6�]���1��a˲���[��>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;;�"�qʾ
�/eq
Ⱦ����ž�XQ�þ���?�ګ�;9��R��        �-���q=G&�$�>�*��ڽ>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>��Zr[v�>O�ʗ��>�FF�G ?��[�?6�]��?����?f�ʜ�7
?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?a�$��{E?
����G?�������:�              �?              �?              �?      �?      �?              @              @      @       @      �?      @      �?      �?       @       @              �?      �?      �?       @              �?       @      �?              @               @               @              �?              �?      �?               @              �?      �?              �?              �?              �?              �?              �?              �?              @              �?              �?      �?      �?              �?      �?      �?               @              �?      �?      @              �?               @              @      �?              @       @      @      �?      �?       @      �?      @      �?      @      �?      @      �?       @       @      �?      �?      �?       @      @               @      �?              �?        
�
conv4/weights_1*�	    \���   �[��?      �@!�}�Uf9�)"Nv�me@2�Ӗ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���vV�R9��T7����5�i}1���d�r�x?�x��f�ʜ�7
������I��P=��pz�w�7��})�l a�
�/eq
�>;�"�q�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?>h�'�?x?�x�?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�������:�             �a@      �@     (�@     ��@     `�@      �@     X�@     x�@     P�@     ؅@     X�@     ��@     �@     0}@      |@     �w@     �t@     0t@     t@     �q@     �m@     �j@     �h@     @i@     �g@     �d@     @_@      W@     �]@      Z@     @R@     �U@      R@     �R@     �P@      P@     �E@     �J@      E@     �B@      D@      8@      :@      ;@      @@      6@      :@      3@      4@      4@      ,@      ,@      ,@       @      (@      @      $@       @      @      @      &@       @              @      @      �?       @      �?       @      @      �?      �?      �?      @              �?      �?      @      �?              �?              �?      �?              �?              �?              �?      �?              �?       @      �?       @              @              �?               @      �?      �?              @      @       @              �?      �?       @      @      �?              @      @      @      @      "@      @      $@      &@      @      @      &@      (@      0@      ,@      2@      $@      4@      5@      9@      =@     �A@      >@      D@      A@     �E@      L@      H@      M@     �P@     �R@     �P@     @Q@     �S@     @U@      `@     �[@     �b@     �c@     �e@     �d@      h@     @j@     �j@     @p@     �s@     0t@     �u@     0{@     �y@     @{@     p�@      �@     �@     ��@     �@      �@     ��@     0�@     ��@     ��@      �@     ؖ@     �a@        
�
conv4/biases_1*�	   �$pM�    7�B?      p@!RV.�{?^?)�'�q�O�>2�k�1^�sO�IcD���L�a�$��{E��T���C��!�A����#@�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;;�"�qʾ
�/eq
Ⱦ�XQ�þ��~��¾�[�=�k���*��ڽ�G&�$��5�"�g����u`P+d����n�����豪}0ڰ����]������|�~���MZ��K��X$�z��
�}����T�L<��u��6
��f^��`{�E'�/��x�6NK��2�_"s�$1��J��#���j�RT��+��y�+pm��mm7&c��`���nx6�X� ��f׽r����tO����f;H�\Q������%���9�e����K�����1���=��]���ݟ��uy�z������Qu�R"�PæҭUݽH����ڽ���X>ؽ(�+y�6ҽ;3���н��.4Nν�!p/�^˽�Į#�������/�����:������z5��        �-���q=K?�\���=�b1��=�d7����=�!p/�^�=�|86	�=��
"
�=���X>�=H�����=PæҭU�=�Qu�R"�=i@4[��=z�����=ݟ��uy�=�/�4��=��-��J�=�K���=�9�e��=����%�=f;H�\Q�=�tO���=�f׽r��=nx6�X� >�`��>�mm7&c>y�+pm>RT��+�>p
T~�;>����W_>>���?�ګ>����>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�����>
�/eq
�>;�"�q�>['�?��>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?��%>��:?d�\D�X=?���#@?�!�A?�T���C?�������:�              �?              �?               @              �?              �?              �?              @      �?               @      �?               @       @       @      �?      @       @       @       @      �?      �?       @      �?      �?       @      �?              @       @      �?              �?      �?      �?      �?               @      �?      �?       @      �?              �?      �?       @      �?              �?      �?              �?      �?              �?      @               @      �?              �?              �?              �?              �?              �?              �?      �?       @       @              �?              �?       @      �?               @              @              �?       @      �?              �?              �?              �?              �?              G@              �?              �?              �?      �?              �?       @              �?              �?              �?              �?      �?      �?      �?      �?               @               @              �?              �?              �?      �?      �?       @      �?              �?              �?               @       @      �?       @      �?      �?      �?      �?      @      �?      �?      �?               @       @               @       @              �?      @      �?      @               @      @      �?       @      @      �?       @      �?       @      @      @      �?              @      �?      @      �?       @               @      �?       @              �?      @              �?        
�
conv5/weights_1*�	    Ț¿   @%��?      �@! ���@)���*�xI@2�	�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74��7Kaa+�I�I�)�(�U�4@@�$��[^:��"��S�F !�ji6�9����ڋ��vV�R9��5�i}1���d�r�x?�x��>h�'����ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?�������:�	             `l@     �q@     0r@     @n@     �k@     �l@     `h@     �c@     �`@     @c@     �`@     �[@     @Z@      X@      Z@     �R@     @T@     �S@      Q@     �K@      D@      @@      G@      7@     �B@      =@      B@      2@      =@      ,@      2@      ,@      4@      *@      ,@      (@      @      *@      @       @      &@      "@       @      @      @      @       @      @      @      @       @      @       @       @      @      �?      �?              �?      �?              �?      �?              �?              �?              �?               @              �?              �?              �?              �?      �?              �?       @              �?      �?              �?      �?               @       @      �?      �?      @      @              @       @       @      @      @       @       @      @      @      @      @      @      @      @      @      "@      "@      $@      2@      .@      7@       @      .@      (@      3@      8@      8@      8@      >@      @@     �E@     �F@     �G@     �E@      N@      P@      P@     �R@     @T@      W@     @[@     @]@     @]@      `@      d@     �b@      e@     �b@      i@     �m@     `o@     �s@     0t@     `l@        
�
conv5/biases_1*�	    ���   `�`>      <@!  ��4>)�?��RX<2��`���nx6�X� ��tO����f;H�\Q������%���9�e����K���'j��p���1���=��]����/�4���Į#�������/����.4N�=;3����=(�+y�6�=�|86	�=��
"
�=z�����=ݟ��uy�=�/�4��==��]���=��1���='j��p�=��-��J�=�K���=�9�e��=����%�=f;H�\Q�=�tO���=�f׽r��=nx6�X� >�`��>Z�TA[�>�#���j>�J>2!K�R�>��R���>�������:�              �?              �?              �?      �?               @              �?              �?              �?              �?      @              �?              �?              �?               @      �?      �?               @      �?              �?              �?      �?              �?        ib8T      ��bg	�H�y'��A*��

step   @

loss@i�>
�
conv1/weights_1*�	   �K���   `ʦ�?     `�@! ���a���)��5�@2�����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���VlQ.��7Kaa+�I�I�)�(��S�F !�ji6�9����d�r�x?�x��>h�'����Zr[v�>O�ʗ��>�FF�G ?��[�?6�]��?����?��d�r?�5�i}1?��ڋ?�.�?�[^:��"?U�4@@�$?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?�������:�             �H@     �g@     �g@      h@     �b@      b@     �`@     �`@     @[@     �V@      W@     �Q@     �S@      M@      N@     �K@     �Q@     �H@     �G@     �F@      <@      E@      ?@      4@      3@      9@      ;@      4@      2@      3@      &@      &@      @      $@      "@      $@       @       @      @      @      @      $@      @      @       @      @               @      @       @      �?      @       @      �?      @       @      �?               @       @               @              �?      �?              �?              �?              �?              �?               @              �?              �?      �?               @      �?      �?       @      �?       @       @      @      @      �?              @      @      @      @      "@      @       @      $@      $@       @      @      @      *@      0@      *@      3@      .@      5@      3@      8@      6@      9@      9@      :@      ?@      D@      I@      G@     �D@     @Q@      P@     @P@     �S@     �P@     �S@     @T@     �[@     @[@     ``@     �b@     �b@     �d@     �f@     �i@      J@        
�
conv1/biases_1*�	   �|N�   �f�U?      @@!  `.�(�?)��	7S��>2�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E����#@�d�\D�X=�I�I�)�(�+A�F�&�>�?�s��>�FF�G ?��[�?f�ʜ�7
?>h�'�?��d�r?�5�i}1?��ڋ?�.�?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?�������:�              �?      �?              �?              �?              �?              �?      �?              �?              �?              �?              �?      �?       @       @              @              �?      �?               @      �?      �?      �?       @              �?              �?              �?      �?        
�
conv2/weights_1*�	   `ة�    �̩?      �@!�sHX�9)@)w@;&E@2����g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%�['�?�;;�"�qʾ�[�=�k�>��~���>�XQ��>��~]�[�>��>M|K�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?�������:�             �@     Ҡ@      �@     P�@     ��@     L�@     �@     ܒ@     p�@     t�@     x�@     �@     `�@     X�@     x�@      �@     ��@     �~@     @{@     @x@     @v@     �u@      r@     Pp@      p@     �m@      k@     @g@     �g@     �d@     �a@     �^@     �\@      ]@     @[@     �V@      U@     �P@     �L@      M@      N@     �M@     �G@      C@     �B@      ?@      <@      6@      ?@      9@      2@      ,@      ,@      8@      1@      1@      "@      "@      &@      $@       @      �?      @      $@      @      $@      �?      @      @       @      @      �?      @      @      @       @       @      �?      �?       @              @      �?              �?              @      �?      �?      �?              �?               @              �?      �?              �?               @              �?              �?      �?      �?              @       @              �?      @      �?      @      @      �?       @      @              @      �?       @      @      @       @      "@      $@      "@       @      @      &@      *@      $@      "@      0@      5@      .@      1@      6@      9@      6@      :@      <@     �G@     �C@      D@     �E@      ?@     �N@      M@     �R@     @S@     @X@     @V@     @U@     �Z@     �]@      f@      a@     �d@      i@     �h@     `l@     �p@     @q@     �u@      v@     �t@     `x@     �{@     ؀@     ��@     ��@     ��@     x�@     ��@     ��@     p�@     8�@     �@     ��@     ��@     8�@     8�@     ��@     L�@     ܠ@     H�@        
�	
conv2/biases_1*�		    gB[�    �a?      P@!  $f7+p?)�4.���>2��m9�H�[���bB�SY�IcD���L��qU���I�
����G�a�$��{E��T���C�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���bȬ�0���VlQ.��7Kaa+�I�I�)�(���ڋ��vV�R9��T7����5�i}1�>h�'��f�ʜ�7
�O�ʗ�����Zr[v���h���`�8K�ߝ뾢f�����uE����E��a�Wܾ�iD*L�پ�[�=�k���*��ڽ�        �-���q=I��P=�>��Zr[v�>����?f�ʜ�7
?>h�'�?x?�x�?�5�i}1?�T7��?�.�?ji6�9�?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��bȬ�0?��82?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?�l�P�`?���%��b?�������:�              �?               @      �?      @       @               @       @       @      �?               @      �?       @               @       @      �?               @              �?              �?              �?              �?              �?               @              �?              �?              �?              �?               @               @      �?       @              @              �?              �?      @      �?              �?      �?      �?      �?      �?              �?      �?      �?              �?        
�
conv3/weights_1*�	   �K��   ��d�?      �@!�>����$�)В�g�_U@2�����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7���h���`�8K�ߝ�a�Ϭ(龢f�����uE���⾮��%ᾄiD*L�پ�_�T�l׾['�?�;;�"�qʾ
�/eq
ȾG&�$��5�"�g�����n�����豪}0ڰ��*��ڽ>�[�=�k�>�XQ��>�����>;�"�q�>['�?��>jqs&\��>��~]�[�>�iD*L��>E��a�W�>�uE����>�f����>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?�������:�             ��@     
�@     �@     �@     H�@     ̡@     d�@     �@     T�@     H�@     �@     ��@     H�@     T�@      �@     ��@     ��@     8�@     ��@     H�@     ��@     0~@     0z@     �y@     �x@     �t@     t@     @q@     �k@      k@     �j@      i@     �d@      d@     �d@     �a@     �\@     @[@     �X@     �W@     �U@     @Q@     �R@     �K@      K@     �J@      E@     �@@      A@      ;@     �@@      :@      9@      .@      &@      3@      7@      1@      7@      $@      *@      *@      "@      "@      @      &@      @      @      @      @      �?      @      @              �?      @       @      @      �?       @      @      @      �?               @              @      �?              �?      �?              �?               @      �?              �?              �?              �?              �?              �?              �?              �?              �?              �?      �?              �?               @      @       @       @      �?      @      @      @       @      @      @      @      "@      @      @       @      @      @      @      @      $@      @      ,@       @      ,@      *@      .@      .@      5@      2@      ;@      ,@      6@      E@      ?@     �A@      D@      I@      H@     �H@     �H@      N@     @S@     �U@     �Z@     @W@     �Y@     @`@      `@     �c@      b@      c@     �h@     �l@     �k@     �m@     @r@     �q@     �r@     @x@     �x@     p}@     �}@     �@     ؀@     H�@     h�@     8�@     ��@     ��@     8�@     �@     H�@     ̕@     $�@     ܙ@     ��@     Ԟ@     ~�@     ��@     ,�@     N�@     ��@     h�@        
�
conv3/biases_1*�	   `�[\�   ��2\?      `@!  ��=�{?)�9+����>2�E��{��^��m9�H�[�ܗ�SsW�<DKc��T�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]�����[���FF�G �>�?�s���O�ʗ����h���`�8K�ߝ���>M|Kվ��~]�[Ӿ��~��¾�[�=�k��        �-���q=��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>O�ʗ��>>�?�s��>����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?U�4@@�$?+A�F�&?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?nK���LQ?�lDZrS?�m9�H�[?E��{��^?�������:�              �?              �?              �?      �?      @              �?      �?      �?      @      �?       @      �?      @      �?      �?              �?              �?              �?      @              �?      @              �?              �?      �?      �?              �?      �?               @              @              �?              �?              �?              @              �?       @      �?       @              �?               @              �?               @      �?       @       @      �?              �?      �?      �?              @              @      �?      @      @      @      @      @      @       @      @      @      �?       @       @               @              �?        
�
conv4/weights_1*�	   �����   ���?      �@!�ǁ�{@9�)�S��me@2�Ӗ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���I��P=��pz�w�7��})�l a�
�/eq
�>;�"�q�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�������:�             �a@     $�@     �@     ��@     `�@     $�@     X�@     ��@     (�@      �@     P�@     ��@     �@     p}@     0|@     pw@     �t@     �s@     0t@     �q@     @n@     �j@     �h@     `i@     `g@      e@      _@     �V@     �]@     �Y@     @S@     �T@     @R@      R@     @Q@     �O@      E@     �J@      F@     �B@      C@      :@      9@      ;@     �@@      4@      <@      2@      4@      6@      ,@      .@      $@      &@      "@       @      $@      @      "@       @      $@       @       @      @      @      �?      �?      @      �?      @      �?      �?      �?      �?              �?       @              �?       @       @               @              �?              �?      �?              �?              �?               @      �?              �?      �?      �?       @      �?              @              �?              �?      �?       @              �?      @       @      �?      �?       @       @       @       @      �?      @      @      @      @       @      @      $@       @      @      @      ,@      $@      .@      1@      0@      (@      3@      4@      :@      >@      A@      =@     �F@      =@     �F@     �L@      H@     �M@      O@      S@     �P@      R@      S@      V@      `@      [@     �b@     �c@     �e@     �d@     `h@      j@     �j@     0p@     �s@     @t@     �u@     @{@     �y@      {@     ��@     ��@     �@     ��@     �@     �@     ��@     �@     ��@     ��@     4�@     ��@     @a@        
�
conv4/biases_1*�	   @�g�    �nS?      p@!������k?)$n����>2�P}���h�Tw��Nof�E��{��^��m9�H�[�ܗ�SsW�<DKc��T��lDZrS�nK���LQ��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !��.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�Ѿ['�?�;;�"�qʾ����ž�XQ�þ��������?�ګ�;9��R��39W$:���.��fc���X$�z���i
�k���f��p�Łt�=	���R����2!K�R���J��#���j�Z�TA[�����"�RT��+��y�+pm��mm7&c��`���nx6�X� ��f׽r����tO����f;H�\Q���9�e����K���=��]����/�4���b1�ĽK?�\��½        �-���q=�EDPq�=����/�=�d7����=�!p/�^�=���X>�=H�����=�Qu�R"�=i@4[��==��]���=��1���=�K���=�9�e��=����%�=f;H�\Q�=�tO���=nx6�X� >�`��>�mm7&c>y�+pm>RT��+�>���">Z�TA[�>�#���j>�J>X$�z�>.��fc��>39W$:��>��|�~�>���]���>�[�=�k�>��~���>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��>M|K�>�_�T�l�>���%�>�uE����>�f����>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>I��P=�>��Zr[v�>>�?�s��>�FF�G ?��[�?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?nK���LQ?�lDZrS?<DKc��T?�������:�              �?              �?              �?              �?              �?       @              �?       @      @       @      �?      �?              @               @      @               @       @      @      �?      �?       @      �?      @               @       @              �?      @       @      �?       @              �?      �?               @              �?      �?              �?      �?      �?      �?      �?       @       @               @              �?              �?      �?              �?      �?              �?      �?              �?      �?       @              �?       @      @               @               @      �?      �?               @              �?              �?             �C@              �?              �?               @              �?              �?               @              �?      �?              �?      �?               @      �?      �?      �?      �?              �?      �?               @              �?              �?      �?      �?      �?      �?              @              �?       @              @       @      �?      �?               @              @      �?               @       @       @      @      �?       @      @              @       @      �?               @      �?      �?      @       @      @      @      @       @               @              @       @      �?      �?      �?      �?      @               @      �?        
�
conv5/weights_1*�	    Ț¿   ����?      �@! `�ᩦ@)�Ȩ�yI@2�	�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9����ڋ��vV�R9��5�i}1���d�r���ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?�������:�	             @l@      r@      r@     `n@     `k@      m@     @h@     �c@     �`@      c@      a@     �[@     @Z@      X@      Z@     @R@     �T@     @S@     �Q@     �K@      D@      @@      F@      7@     �C@      =@     �A@      3@      <@      .@      2@      ,@      3@      *@      .@      &@      @      *@      @       @      &@      "@      "@       @      @      @      @      @      @      @      �?      @       @       @      @       @      �?              �?              �?              �?              �?               @              �?              �?               @              �?              �?              �?      �?              �?      �?              �?      �?      �?              �?       @              @      �?      �?      @       @               @      @       @      �?      @       @       @      @      "@      @       @      @      @      @      @      @      (@      "@      2@      0@      5@      $@      *@      *@      4@      8@      7@      8@      =@     �@@     �E@     �F@     �G@     �E@      N@      P@     �O@      S@      T@     @W@     @[@      ]@     �]@     ``@     �c@     �b@      e@     �b@      i@     �m@     @o@     ps@     Pt@     `l@        
�
conv5/biases_1*�	   `z�   ��f">      <@!  ���5>)�m���q<2�2!K�R���J��#���j�y�+pm��mm7&c��`���nx6�X� ��f׽r�������%���9�e���'j��p���1����/�4��ݟ��uy��-���q�        %�f*=\��$�=V���Ұ�=y�訥=�8�4L��=�EDPq�=PæҭU�=�Qu�R"�=��1���='j��p�=��-��J�=�K���=�f׽r��=nx6�X� >�`��>�mm7&c>y�+pm>RT��+�>���">Z�TA[�>2!K�R�>��R���>��f��p>�i
�k>��-�z�!>4�e|�Z#>�������:�              �?      �?              �?      �?       @      �?              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?               @               @      �?      �?      �?      �?              �?              �?              �?        �B�xU      �ח�	�EF{'��A*�

step  @@

loss&��>
�
conv1/weights_1*�	   ��5��   ��n�?     `�@! �������)�=L'�@2�	����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"���ڋ��vV�R9������6�]���a�Ϭ(�>8K�ߝ�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?ji6�9�?�S�F !?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?�������:�	             �F@     �g@     �g@     �g@      c@     �b@     �`@     �`@     �\@     �V@     @U@     �Q@     �S@     �P@      M@     �I@     �P@     �K@     �F@      H@      ;@      B@      =@      :@      6@      4@      :@      4@      .@      4@      &@      (@       @      @      ,@       @      "@      (@      $@      @      @       @      @       @       @      @      �?      @      @      @      @       @      �?      �?      �?       @              �?       @      �?      �?       @               @              �?               @              �?              �?              �?              �?              �?       @              �?              �?      �?      �?              �?      �?              @      �?       @      �?       @       @      @       @      @      @      @       @      @      @       @      &@      @      @      $@      $@       @      ,@      .@      ,@      3@      8@      .@      7@      =@      3@      ;@      8@     �A@      E@      D@      H@      G@     @Q@     @Q@      L@     �S@     �Q@      S@     �S@     �\@     @\@     �_@     �b@     �b@      e@     `f@     �i@      L@        
�
conv1/biases_1*�	   `>�V�   ��i?      @@!  ����?)��R�?2�ܗ�SsW�<DKc��T�k�1^�sO�IcD���L�
����G�a�$��{E��!�A����#@�1��a˲?6�]��?����?f�ʜ�7
?�.�?ji6�9�?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?P}���h?ߤ�(g%k?�������:�              �?              �?               @              �?              �?              �?              �?              �?              �?      �?      �?              �?               @              @      �?      @       @              �?              �?               @              �?      �?              �?        
�
conv2/weights_1*�	   @��   ���?      �@!�5�kO+@)]c��`'E@2����g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(��uE���⾮��%ᾄiD*L�پ�_�T�l׾��>M|Kվ['�?�;;�"�qʾjqs&\��>��~]�[�>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?�������:�             Ԑ@     �@     �@     �@     ̙@     t�@     ��@     �@     X�@     ��@     ��@     Њ@     `�@     @�@     `�@     ��@     ��@     �~@     �{@     �w@     �v@     @u@     �r@     �n@     `p@     �m@     �j@     @h@     �g@     @e@     �a@     �^@     �]@     @[@      _@     @V@     �R@     �N@      M@     �N@      Q@      L@      E@      C@     �@@      >@      A@      @@      9@      8@      1@      (@      4@      4@      ,@      0@       @       @      ,@      $@      "@      @      @      "@      @      @      @      @      @      @      @               @      @      �?      �?      �?              �?      @      �?      @       @      �?      �?      �?      �?      �?              �?              �?      �?              �?              �?               @              �?      �?               @       @              @      @       @      @      @       @      �?       @      @      @      @      "@      @      @      @      @      @      &@      @      $@      *@      *@      @      (@      2@      0@      ,@      3@      1@      7@      ?@      @@     �@@      B@      E@      C@      E@     �D@     �E@     �K@     @U@     �S@      W@      W@     @S@     �Z@      a@     �c@      b@      e@     �i@     @g@     `m@     pp@     q@     0u@     Pw@     `t@     �x@     �{@     ��@     �@     ��@     ��@     ��@     ��@     ��@     @�@     x�@      �@     ��@     ��@     X�@     4�@     ̝@     \�@     �@     d�@        
�

conv2/biases_1*�		   ���f�   @$�i?      P@! ��{��?)����?2�P}���h�Tw��Nof���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�+A�F�&�U�4@@�$�ji6�9���.����d�r�x?�x��>h�'��6�]���1��a˲���[���FF�G �I��P=��pz�w�7��        �-���q=�4[_>��>
�}���>�u��gr�>�MZ��K�>����?f�ʜ�7
?>h�'�?x?�x�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?Tw��Nof?P}���h?ߤ�(g%k?�������:�              �?              �?              �?      @      �?      �?      �?       @      �?      �?              @      �?              �?              �?      �?      �?              �?              @              �?              �?      �?              �?              �?              �?              �?              �?              �?              �?              �?              �?       @      �?              �?               @      �?      @      �?      �?              �?       @               @      �?      �?      �?              �?              �?      �?              @      �?        
�
conv3/weights_1*�	   �Ih��   `���?      �@!�4��V"�)j�ESG`U@2�����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ;�"�qʾ
�/eq
Ⱦ�XQ�þ��~��¾0�6�/n���u`P+d��39W$:��>R%�����>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>���%�>�uE����>�f����>��(���>a�Ϭ(�>�ߊ4F��>})�l a�>pz�w�7�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?�������:�             @�@     �@     �@     ڥ@     j�@     ��@     r�@     �@     8�@     X�@     ��@     |�@     `�@     X�@     8�@     ��@     ��@     ��@     �@     p�@     ��@      ~@     �y@      {@     `x@     �t@     `t@     q@     @k@     �k@     �k@     �i@      d@      e@      d@     �a@      ]@     �Z@      W@     �Y@     �T@     @S@     @P@     �M@     �G@      L@     �D@      B@      C@      9@      A@      7@      ;@      1@      &@      3@      7@      &@      4@      (@      ,@      ,@      ,@      $@      @      "@       @      �?      @      @      �?      @      @      @      �?      @       @      @      �?      �?      �?       @               @       @       @              �?      �?              �?      �?       @      �?              �?      �?      �?              �?              �?              �?              �?              �?              �?      �?      �?              �?              �?              �?              �?      �?              @      @              @               @      @      @       @      @       @       @      @      @      @      @      @      @      @      @      @      @      (@      $@      $@      $@      4@       @      2@      .@      2@      .@      9@      1@      =@      @@     �@@      B@     �D@     �H@     �F@     �G@      I@     �N@     @Q@      W@     �X@     �X@      [@     �`@     ``@     �a@     �c@     @b@     �i@     @l@     �k@     �l@     r@     pr@     �q@     �x@     0x@     �}@     �}@     �@     ؀@     ��@     �@     @�@     ��@     ��@     P�@     �@     \�@     ԕ@     $�@     ԙ@     ��@     ��@     n�@     ̣@     0�@     ,�@     �@     ��@        
�
conv3/biases_1*�	   �Ϲg�   @*�c?      `@!  �4�o�?)H�ld�?2�P}���h�Tw��Nof����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���h���`�8K�ߝ�a�Ϭ(龙ѩ�-߾E��a�Wܾ�iD*L�پ        �-���q=�u`P+d�>0�6�/n�>;�"�q�>['�?��>��~]�[�>��>M|K�>pz�w�7�>I��P=�>��Zr[v�>6�]��?����?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?���%��b?5Ucv0ed?�������:�              �?              �?              �?               @      �?       @              �?              @      �?      �?      @              �?      �?       @      �?       @      �?      @              �?               @      �?      @       @      �?              �?       @              �?              �?              �?      �?              �?              �?      �?              �?      �?              @              �?              �?              �?              �?       @               @               @      �?      �?      �?       @      �?               @      �?       @      @              �?       @      �?      @      @      @      @      @      @      @      @      �?       @               @      �?              �?        
�
conv4/weights_1*�	    9��   �3�?      �@!������8�)\ehne@2�Ӗ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��T7����5�i}1���d�r�x?�x��>h�'��I��P=��pz�w�7��})�l a�
�/eq
�>;�"�q�>��~]�[�>��>M|K�>�ѩ�-�>���%�>8K�ߝ�>�h���`�>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�������:�             �a@     $�@     $�@     ��@     h�@     �@     8�@     ��@     �@      �@     8�@     ��@     `@     �}@     @|@     @w@      u@     �s@      t@     �q@     �m@     �j@     �h@     �i@      g@     @e@      _@     �V@     @^@     �Y@      S@     �T@      S@     @R@      O@     �P@     �D@     �I@      F@     �B@      E@      9@      8@      <@      ?@      4@      =@      4@      4@      6@      .@      0@      @      $@      "@      @      $@      @      "@      @      $@      @               @       @      @       @      �?       @      @              �?       @       @       @       @              �?              @      �?              �?       @              �?              �?              �?              �?              �?      �?               @               @              �?       @       @      �?              �?      �?              @              @      @       @       @      �?       @      @       @              @      @      @      @      "@       @       @       @      @      "@      $@      "@      0@      1@      .@      &@      2@      ;@      7@      @@      A@      :@      G@      >@      E@     �M@     �H@     �M@     �N@     @R@      Q@     �R@     @R@     �V@     @_@      [@     �b@     �c@      f@     @d@     �h@     @j@     �j@     @p@     �s@     �t@     �u@     @{@     �y@     {@     ��@      �@     ��@     ��@     ��@     �@     �@     0�@     ��@     ��@     <�@     �@     @a@        
�
conv4/biases_1*�	   �^�q�   ��a?      p@!�
�_�?)�`��?2�uWy��r�;8�clp�Tw��Nof�5Ucv0ed����%��b��l�P�`�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[��>�?�s���O�ʗ�����Zr[v��I��P=���ߊ4F��h���`�8K�ߝ�a�Ϭ(龢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;;�"�qʾ0�6�/n���u`P+d����n��������?�ګ�;9��R����|�~���MZ��K��H��'ϱS��
L�v�Q��i
�k���f��p�Łt�=	���R����2!K�R���J��#���j�Z�TA[�����"�RT��+��y�+pm��mm7&c��`���nx6�X� ��f׽r����tO������-��J�'j��p�ݟ��uy�z����㽳�s�����:��        �-���q=_�H�}��=�>�i�E�=H�����=PæҭU�=�/�4��==��]���='j��p�=��-��J�=�K���=�9�e��=����%�=f;H�\Q�=�tO���=�f׽r��=nx6�X� >�`��>�mm7&c>y�+pm>RT��+�>Z�TA[�>�#���j>2!K�R�>��R���>Łt�=	>��f��p>�����>
�/eq
�>['�?��>K+�E���>jqs&\��>��~]�[�>�_�T�l�>�iD*L��>���%�>�uE����>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>1��a˲?6�]��?����?f�ʜ�7
?>h�'�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?�������:�              �?              �?      �?      �?              �?              �?              �?               @      �?      @      �?              �?      �?       @      �?      �?      �?      �?      �?      @      �?       @      �?      @      �?      �?       @      �?       @       @      �?      �?              �?      �?      @      @       @              �?              �?              @       @      @               @              �?      �?      @               @              �?              �?      �?               @       @              �?              �?              �?              �?      �?       @              �?       @      �?       @       @       @              �?              �?      �?              �?              �?              �?             �A@              �?              �?              �?              �?              �?              �?               @              �?      �?              @               @              �?              �?              �?              �?              �?              �?               @              �?      �?      �?      �?      �?      �?       @              �?              �?       @       @       @              �?       @      @      @      �?              �?      @       @       @              @      �?      �?      @      @      @       @      @      @              �?      �?      �?      @       @      �?      @      �?       @      �?       @      �?               @               @        
�
conv5/weights_1*�	   ���¿   `]��?      �@! ��q�Y@)�NQ�zI@2�	�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G��T���C��!�A�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9���d�r�x?�x���FF�G ?��[�?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?�������:�	             @l@      r@      r@     `n@     `k@      m@      h@      d@     �`@      c@     �`@     �[@      Z@      X@     @Z@      R@     �T@      S@     �Q@     �J@      E@      @@     �E@      9@      C@      ;@     �B@      2@      ;@      3@      .@      &@      4@      .@      ,@      *@      @      ,@      @      @      &@      (@      @       @      @      @      @      @      @      @       @       @       @      @      �?      @              �?              �?               @      �?              �?              �?      �?       @      �?              �?              �?              �?              �?              �?              �?              �?       @              @              �?       @       @       @      @      @               @      �?      @      �?      @       @      �?       @       @       @      @       @      @      @      @      @      &@      "@      3@      *@      7@      &@      &@      .@      3@      8@      7@      8@      ?@      ?@     �E@      G@     �G@     �D@     �N@      P@      O@     �S@     �S@     @W@     @[@     @\@      ^@     �`@     �c@     �b@      e@      c@     �h@     �m@      o@     ps@     Pt@     �l@        
�
conv5/biases_1*�	   ����   @7`)>      <@!  @;(m4>)	��G�<2�%�����i
�k���f��p�Łt�=	���R�����J��#���j�Z�TA[�����"�RT��+��y�+pm��mm7&c�f;H�\Q������%���9�e����K��󽉊-��J�;3���н��.4Nν��؜�ƽ�b1�Ľ;3����=(�+y�6�=�|86	�=i@4[��=z�����=��-��J�=�K���=�`��>�mm7&c>RT��+�>���">Z�TA[�>�#���j>�J>2!K�R�>��R���>Łt�=	>��-�z�!>4�e|�Z#>4��evk'>���<�)>�������:�              �?               @      �?              �?              �?              �?      �?              �?              �?      �?              �?              �?              �?      �?              �?              �?              �?              �?              �?      �?       @      �?       @              �?              �?        `t:�U      ��q�	���|'��A*ɫ

step  �@

loss���>
�
conv1/weights_1*�	   �����   ����?     `�@! ��L����)ݜ��@2�	����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0�+A�F�&�U�4@@�$��.����ڋ��vV�R9�x?�x��>h�'�������6�]���1��a˲��h���`�8K�ߝ�jqs&\��>��~]�[�>��>M|K�>�f����>��(���>a�Ϭ(�>��Zr[v�>O�ʗ��>�FF�G ?��[�?1��a˲?����?f�ʜ�7
?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?�������:�	             �F@     �g@     @g@     @g@      d@     �b@      `@     ``@      ]@     @W@     �U@     �P@     �R@     �P@      N@     �L@     �M@      K@     �I@      F@     �@@      @@      8@      ?@      5@      6@      9@      5@      .@      .@      (@      (@      "@       @      ,@      "@      (@      "@      (@      @      @      @       @      @       @       @      @      @      @       @      @              �?      �?       @              �?      �?               @              �?      �?              �?              �?      �?              �?              �?      �?              �?      �?              �?              �?       @              �?              �?      �?              �?              �?              �?      �?               @               @      @      �?      �?              �?      @              �?      @       @       @      @      @      @       @      @      @      @      @      $@      $@      @      &@      2@      "@      0@      0@      8@      3@      :@      9@      5@      7@      >@     �@@     �E@      C@     �C@     �M@     �O@     @P@     �O@      S@      R@     �R@     @S@      ]@      [@      `@     `b@      c@     `e@     @f@      i@      P@        
�
conv1/biases_1*�	   ��9_�   `^Xq?      @@!   ��o�?)�����G?2��l�P�`�E��{��^��lDZrS�nK���LQ�
����G�a�$��{E�d�\D�X=���%>��:�1��a˲?6�]��?��d�r?�5�i}1?U�4@@�$?+A�F�&?�u�w74?��%�V6?��%>��:?d�\D�X=?���#@?�!�A?�T���C?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?;8�clp?uWy��r?�������:�              �?               @              �?              �?              �?              �?              �?              �?              �?      �?              �?               @              @      �?              @      �?       @              �?      �?               @              �?              �?        
�
conv2/weights_1*�	   @*l��   @҉�?      �@!�ݗMw .@)��)E@2����g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(����uE���⾮��%�E��a�Wܾ�iD*L�پ����ž�XQ�þ�[�=�k�>��~���>�uE����>�f����>��(���>a�Ϭ(�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?�������:�             ��@     �@     ��@     �@     ��@     P�@      �@     В@     `�@     ��@     H�@      �@     h�@     ��@     X�@     @�@     x�@     �~@     p{@     pw@     �v@     Pu@     �r@      n@     �p@      n@     �k@     �f@     �h@     �c@     �b@      ^@     �^@     �]@     @\@     @W@      Q@      P@     �M@     @P@      L@      P@      E@     �C@      A@      B@      9@      ;@      9@      2@      3@      8@      .@      7@      "@      3@      @       @      ,@      0@       @       @      @      @      @      @      @      @      @      @      @      �?      @      @               @      @      �?      �?      �?      �?      �?      �?              �?              �?              @               @              �?              �?              �?              �?               @              �?              �?      �?              �?      �?       @              @      @      @      @       @      �?      @       @       @      @      @      @      @      @      @      "@      @      (@      &@      2@      $@      *@      0@      .@      :@      1@      4@      =@      @@      C@      B@      F@      B@     �G@      @@      E@     �L@     �T@     �S@     �V@     @U@     �V@     �Z@     @`@     �c@      c@     `e@      j@     �f@     �k@     �p@     0q@     u@     w@     �t@      y@     @{@     ��@     `�@     X�@     p�@     Ї@     �@     ��@     �@     ��@     �@     ��@     h�@     `�@     T�@     ��@     `�@     �@     ��@        
�

conv2/biases_1*�
	   ��p�   �A6s?      P@!  ����?)X�'�#?2�uWy��r�;8�clp����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L�
����G�a�$��{E��T���C��!�A����#@���%>��:�uܬ�@8���82���bȬ�0�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�1��a˲���[����~]�[Ӿjqs&\�Ѿ        �-���q=����?f�ʜ�7
?��d�r?�5�i}1?ji6�9�?�S�F !?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?�������:�              �?              �?              �?              @      @      �?      �?              �?               @       @      �?       @               @              �?              �?      �?              �?              �?      �?              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?               @      �?              �?      �?               @       @      �?       @               @      �?      �?      �?              �?              �?               @              �?       @              �?        
�
conv3/weights_1*�	    ���   �H��?      �@!�`���)�H�aU@2�����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���I��P=��pz�w�7���ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾄiD*L�پ�_�T�l׾�[�=�k���*��ڽ���������?�ګ���n����>�u`P+d�>�����>
�/eq
�>;�"�q�>['�?��>��~]�[�>��>M|K�>�iD*L��>E��a�W�>�ѩ�-�>���%�>a�Ϭ(�>8K�ߝ�>�h���`�>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?�������:�             �@      �@     �@     ĥ@     |�@     ��@     h�@     ��@     `�@     ,�@     ܕ@     ��@     H�@     D�@     `�@     Ȋ@     ؉@     ��@     ȃ@     (�@     ؀@     0~@     �y@     �z@      y@     �t@     �s@     �q@      k@     `l@     `j@     �i@     �c@     �f@      c@     �a@     @\@      ^@      V@     �X@     @U@     �P@     �P@      L@      G@      J@     �G@      @@      C@      >@     �D@      >@      2@      2@      2@      7@      5@      &@      0@      2@      .@      (@      (@      "@      @      @      @      @       @      @      @       @      @       @       @       @      @      @      �?      @      @              �?               @      �?      �?               @               @              �?              �?              �?              �?               @               @              �?              �?              �?               @      �?              @       @      �?      @              �?       @      @       @       @       @       @       @      @      @      �?      @      @      @      @      @      *@      @      "@      *@      0@      0@      "@      2@      0@      2@      .@      5@      8@      7@      >@     �A@      A@     �D@      H@      G@      G@      M@     �M@     �P@      T@     �X@      Z@     @Y@     @a@     `b@      a@     �c@      c@     `i@     �j@     �l@     �l@      r@     �q@     pr@     �x@     Px@      ~@     �}@     �@     ��@     `�@     �@     �@     ��@     Ѝ@     @�@     $�@     @�@     ܕ@      �@      �@     ��@     $�@     d�@     ��@     .�@     D�@     �@     ��@        
�
conv3/biases_1*�	   �THq�   `�hg?      `@! ����?)����h�#?2�uWy��r�;8�clp�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"�ji6�9���.���vV�R9��T7����5�i}1���d�r�x?�x��>h�'�������6�]���a�Ϭ(���(��澪�ӤP�����z!�?��        �-���q=E��a�W�>�ѩ�-�>���%�>pz�w�7�>I��P=�>1��a˲?6�]��?f�ʜ�7
?>h�'�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?ji6�9�?�S�F !?�[^:��"?+A�F�&?I�I�)�(?�7Kaa+?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?�������:�              �?              �?               @      �?               @               @       @      �?              �?      �?      �?      �?       @      @       @              �?       @       @      �?      @      @       @              �?              �?      �?              �?              �?              �?       @       @              �?              �?              �?              @              �?      �?              �?              �?              �?              �?               @      �?              �?       @              �?      @              �?      �?      �?      �?      @       @       @      @       @       @      @       @      @      @      @      @       @      �?      @               @      �?              �?        
�
conv4/weights_1*�	    T��   �!�?      �@!�W�w��8�)��nne@2�Ӗ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��8K�ߝ�a�Ϭ(�39W$:���.��fc��������>
�/eq
�>�f����>��(���>8K�ߝ�>�h���`�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>6�]��?����?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�������:�              a@     ,�@     $�@     ��@     h�@     �@     `�@     ��@      �@     �@     H�@     ��@     p@     `}@     @|@     0w@     �t@     0t@      t@     �q@      n@     `j@     @i@     �h@     �g@     @e@     �^@     �W@     �]@      Y@     �R@     @U@     @R@     @R@     �P@     �O@     �E@      J@     �C@     �D@     �C@      9@      <@      :@      :@      :@      9@      8@      3@      9@      (@      .@      "@      &@      "@      @      &@      @      $@      @      $@      @      �?       @      @      �?      �?       @       @      @      @               @      �?              �?              �?       @               @               @              �?               @              �?              �?              �?              �?              �?              �?              �?              �?              �?               @      �?      �?      �?               @      �?      �?      �?      �?      @      �?      @       @      �?      @      �?       @      �?      @      @      @      @      &@      @      "@      @      @      &@      @      &@      *@      2@      0@      (@      ,@      :@      9@     �A@      >@      >@     �F@      =@      F@     �K@      J@      N@      P@     @Q@      P@     �S@     �R@      V@      `@     @Z@     �b@     `d@     �e@     `d@     �h@     �i@      k@     0p@     �s@     �t@     �u@     {@     �y@     �z@     ��@     ��@     ��@      �@     ��@     �@     �@     �@     ��@     ��@     (�@      �@     `a@        
�
conv4/biases_1*�	   @޷w�   `?�g?      p@!P������?)+SU�z'?2�*QH�x�&b՞
�u�;8�clp��N�W�m�ߤ�(g%k�P}���h����%��b��l�P�`�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]�����[���FF�G �>�?�s���I��P=��pz�w�7��})�l a��ߊ4F��8K�ߝ�a�Ϭ(���(��澮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ;�"�qʾ
�/eq
Ⱦ�*��ڽ�G&�$��5�"�g���0�6�/n���u`P+d��;9��R���5�L��%�����i
�k���f��p�Łt�=	���R����2!K�R���J��#���j�Z�TA[��RT��+��y�+pm��mm7&c��f׽r����tO����f;H�\Q��'j��p���1����/�4��ݟ��uy�z������
6������Bb�!澽        �-���q=�|86	�=��
"
�=i@4[��=z�����=�tO���=�f׽r��=�mm7&c>y�+pm>RT��+�>���">Z�TA[�>�#���j>�J>Łt�=	>��f��p>4��evk'>���<�)>E'�/��x>f^��`{>���]���>�5�L�>�����>
�/eq
�>;�"�q�>['�?��>jqs&\��>��~]�[�>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?�������:�              �?              �?      �?      �?              �?              �?      �?       @       @       @              �?       @      �?       @      �?      @              �?               @      �?       @      @      @      �?       @       @      �?               @      �?      @      �?      �?              �?      �?      �?       @      �?      �?               @       @              @      �?      @               @      �?              @              �?      �?      �?      �?      �?      �?              �?              �?      �?              �?              �?               @               @       @      �?      �?      �?      �?              �?      �?              �?      �?              �?              �?      �?              �?              @@              �?              �?               @              �?      �?       @      �?              �?              �?              �?              �?              �?              �?      �?      �?              �?              �?              �?              �?      �?              �?      �?              �?              �?      �?      @      �?       @      �?               @      �?       @       @       @      �?       @              �?      �?      �?      @      @               @      @       @              @      @       @      @       @      �?      @      �?      @       @      @      �?       @      @      @      �?      �?       @      �?      �?       @               @      @        
�
conv5/weights_1*�	    �¿   `���?      �@! `��tT@)0�.��{I@2�	�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��!�A����#@���%�V6��u�w74���bȬ�0���VlQ.��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9�x?�x��>h�'���5�i}1?�T7��?��ڋ?�.�?�S�F !?�[^:��"?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?�������:�	              l@      r@      r@      n@     �k@      m@     `h@     �c@     �`@     @c@     �`@     @[@      Z@     @X@     @Z@     �Q@     @U@      S@      R@     �H@      F@      @@     �E@      :@      B@      <@     �B@      1@      =@      3@      .@      (@      3@      .@      ,@      $@      @      (@      @      @      *@      &@      @       @      @      @       @      @      @      @      �?      �?       @      @       @       @       @              �?              �?               @              �?      �?              �?       @              �?              �?               @              �?              �?              �?      �?      �?       @              �?      �?       @      @      @      @      �?               @      �?      @      �?      @              @      �?      @      @      @       @       @      @      @      @      $@      $@      4@      (@      5@      &@      *@      *@      5@      8@      7@      7@      ?@      @@     �E@      G@      G@      E@     �M@     @P@     �N@     @T@     �S@     @W@      [@     �\@      ^@     @`@      d@     �b@     �d@      c@     `i@     �m@      o@     �s@      t@     �l@        
�
conv5/biases_1*�	   ��&�    ��.>      <@!   ���8>)���NE�<2�4��evk'���o�kJ%��i
�k���f��p���R����2!K�R���J��#���j�Z�TA[��RT��+��y�+pm��9�e����K��󽉊-��J�'j��p���1����Qu�R"�PæҭUݽ(�+y�6ҽ;3���нH�����=PæҭU�=�K���=�9�e��=�f׽r��=nx6�X� >Z�TA[�>�#���j>�J>2!K�R�>��R���>Łt�=	>��f��p>�i
�k>%���>���<�)>�'v�V,>7'_��+/>�������:�              �?              �?              @      �?      �?      �?              �?              �?      �?               @              �?              �?              �?              �?              �?              �?      �?      �?               @              �?       @              �?      �?        �L�U      ����	��m~'��A*��

step  �@

loss�5�>
�
conv1/weights_1*�	   `�$��    .��?     `�@!  ��i���)���}�@2�����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0�I�I�)�(�+A�F�&�ji6�9���.��f�ʜ�7
������6�]���a�Ϭ(���(���pz�w�7�>I��P=�>O�ʗ��>>�?�s��>f�ʜ�7
?>h�'�?�vV�R9?��ڋ?�7Kaa+?��VlQ.?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?�������:�             �D@     �h@     �g@     �f@     @e@      b@     @`@      _@     �\@      Y@     �S@      Q@     @S@     @S@     �G@     �M@     �M@      L@      J@     �@@     �B@      C@      ;@      :@      8@      7@      3@      2@      2@      &@      .@      (@      (@      $@      "@      &@      &@       @      &@      &@      @      @      @      @       @       @      @      @       @      �?      @      �?               @       @      @               @              �?              �?              �?      �?              �?              �?              �?               @               @              �?              �?      @      @      �?       @      @       @      @       @      @      @       @       @      �?      �?      @      @       @       @      @      @       @      @      @      &@      .@      (@      4@      5@      4@      4@      5@      9@      8@      <@      ;@      >@     �D@      E@     �B@     �O@     �J@     �R@     �N@     �R@     �Q@     �T@     @R@     �\@      [@     �_@     �a@     �c@     `e@     �f@      h@      R@        
�
conv1/biases_1*�	   �Za�    ~Hu?      @@!  dt�h�?)8�_1>"?2����%��b��l�P�`���bB�SY�ܗ�SsW��lDZrS�nK���LQ��T���C��!�A��u�w74���82�I��P=�>��Zr[v�>��ڋ?�.�?��VlQ.?��bȬ�0?d�\D�X=?���#@?�!�A?�T���C?
����G?�qU���I?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?hyO�s?&b՞
�u?�������:�              �?              �?              �?              �?              �?              �?              �?              �?               @              �?              �?              �?      �?      �?      @      �?              �?      @      @      �?               @      �?      �?              �?        
�
conv2/weights_1*�	   �3���    <�?      �@! c(�0@)�Z�+E@2����g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲��FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(����uE���⾮��%ᾙѩ�-߾�iD*L�پ�_�T�l׾��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;;�"�qʾ��������?�ګ���~]�[�>��>M|K�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?�������:�             ��@     �@     ��@     �@     ��@     H�@     T�@     ��@     T�@     Ȑ@     x�@     ؊@     8�@     (�@     h�@     (�@     ��@     �}@     0{@     �x@     v@     �u@     �q@     p@     �o@     �m@     �m@      e@     �h@     �d@     �a@     @`@      `@     @^@      \@      U@     �P@     @P@     @P@      M@     �O@      J@      F@      C@      =@     �G@      9@      9@      9@      6@      7@      2@      ,@      3@      *@      3@      (@      (@      @      $@      $@       @      $@      @      @      @       @       @      @      @      @      @       @      �?      @       @               @      �?      �?      �?      �?       @              �?              @      �?               @      �?              �?              �?      �?              �?              �?              �?              �?      �?               @               @              �?              �?      �?      �?              �?              �?       @              @      �?      @       @      @      @      �?      �?      @      @      @      @      @       @      @       @      @      @      &@       @      (@      (@      &@      5@      2@      8@      4@      ;@      =@      9@     �B@     �C@     �C@     �D@     �H@     �@@     �G@     �M@     �R@     �S@     �T@     @V@     @W@     �[@     @]@     �d@     �c@      e@      i@     �h@     �k@     �p@     �o@     �u@     �v@     �u@     py@     z@     ��@     ��@     ��@     ��@     ��@     �@     ��@     ��@     8�@     ��@     ԓ@     d�@      �@     l�@     Н@     0�@     ޠ@     4�@        
�	
conv2/biases_1*�		    ��u�    @ey?      P@!  ����?)�˱�S1?2�&b՞
�u�hyO�s�ߤ�(g%k�P}���h����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��FF�G �>�?�s���;�"�qʾ
�/eq
Ⱦ        �-���q=�5�i}1?�T7��?ji6�9�?�S�F !?U�4@@�$?+A�F�&?��bȬ�0?��82?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?*QH�x?o��5sz?�������:�              �?              �?              @       @       @      �?      �?      �?               @              �?      �?      �?       @              �?      �?      �?               @              �?              �?      �?              �?              �?              �?              �?              �?              �?              �?               @              �?              �?      �?       @      �?       @              �?      �?               @       @               @      �?              �?      �?              �?      �?      �?       @       @              �?        
�
conv3/weights_1*�	   @���   ��ݮ?      �@!�/��#�)�7�=bU@2�����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾��>M|Kվ��~]�[Ӿ;�"�qʾ
�/eq
Ⱦ39W$:���.��fc���d�V�_���u}��\�����>豪}0ڰ>
�/eq
�>;�"�q�>['�?��>jqs&\��>��~]�[�>�_�T�l�>�iD*L��>E��a�W�>��(���>a�Ϭ(�>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?�������:�             ��@     �@     �@     ĥ@     z�@     ��@     h�@     �@     \�@     �@     �@     ԓ@     ,�@     H�@     P�@      �@     ��@     ��@     �@     �@     ��@     @~@     �y@     �z@     �x@      u@     �s@     r@      k@     �k@     �j@     `j@      d@     �e@     �b@     �a@     �\@     �\@     �W@     @Z@      V@     �P@     �L@      M@      F@     �L@      B@     �C@     �B@     �B@     �@@     �@@      :@      6@       @      2@      3@      2@      (@      1@      1@      ,@      $@       @      @      "@      $@       @      @       @      @      @      @               @      @      �?       @      �?      @       @              �?               @       @      �?              �?      �?              �?      �?      �?              �?              �?              �?              �?              �?              �?      �?              �?              �?       @              �?              �?      �?      �?       @      �?      @       @      �?              @              @       @      @      @       @      @      @      @      @      @      @      @      *@      $@      *@      2@      ,@      ,@      4@      1@      .@      8@      8@      7@      9@     �A@      B@      E@      G@      E@     �K@     �M@      P@     �Q@     �P@     �X@     �V@      _@      a@     �a@     �`@     �b@     �d@     @i@     `j@     �k@     `n@     �q@     �p@     �r@     �x@     @y@     �|@     �~@     �@     �@     0�@     X�@     ��@     ��@     Ѝ@     �@     0�@     4�@     ��@     ��@     ,�@     t�@     $�@     J�@     £@     0�@     D�@     ��@     ��@        
�
conv3/biases_1*�	   �>9w�   ��@m?      `@!  :3֋�?)�ʃm�*2?2�*QH�x�&b՞
�u��N�W�m�ߤ�(g%k�P}���h�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0�+A�F�&�U�4@@�$��S�F !�ji6�9���.����ڋ��vV�R9��5�i}1���d�r�>h�'��f�ʜ�7
��������[���FF�G �pz�w�7��})�l a���������?�ګ�        �-���q=�[�=�k�>��~���>jqs&\��>��~]�[�>O�ʗ��>>�?�s��>f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?�������:�              �?               @       @              �?      �?              @              �?      @      @       @      �?               @              �?      �?      @       @      �?      @      �?       @              �?              �?      �?              �?              �?              �?      �?              �?              �?              �?              @              �?              �?              �?              �?      �?              �?              �?              �?              �?              �?              �?               @      @      �?      �?      @      �?       @      �?      �?      @      @       @      �?      @      @       @      @      @      @              @      �?      �?       @        
�
conv4/weights_1*�	   `}��   `&'�?      �@!��8�)2�9�ne@2�Ӗ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&��S�F !�ji6�9���.����ڋ��T7����5�i}1���d�r�x?�x��1��a˲���[���FF�G �>�?�s�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`��%ᾙѩ�-߾�_�T�l׾��>M|KվK+�E���>jqs&\��>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�������:�              a@     ,�@     �@     ��@     \�@     (�@     `�@     ��@     �@     �@     H�@     ��@     �@     P}@     @|@     w@     �t@     @t@     @t@     0q@     �n@     �i@     �i@     �h@     �g@     `e@     @_@      W@      ^@     @X@     �R@     �U@      S@     @Q@     �P@      M@     �H@     �G@     �D@     �E@     �C@      8@      <@      7@      =@      9@      <@      8@      5@      6@      ,@      ,@      @      *@      .@      @      @      @      "@      @      "@      @       @       @       @       @              @      �?       @              �?      �?      �?               @      �?      �?               @              �?              �?      �?       @              �?              �?              �?              �?              �?               @              �?               @      �?               @      �?               @              �?               @      @      �?      @      �?              @       @       @      �?      @      @      @      @       @      "@      "@      $@      @      "@       @      "@      .@      1@      1@      &@      *@      9@      :@     �B@      ?@      =@     �F@      >@     �E@      L@     �J@      L@     �Q@     @P@     �O@      T@     �R@      V@     �_@     �Z@     �b@     `d@     �e@     �d@     �h@     �i@      k@     0p@     �s@     pt@     pu@     �z@     �y@     p{@     ��@      �@     ��@      �@     ��@     �@     ،@     ��@     đ@     x�@     ,�@     �@     �a@        
�
conv4/biases_1*�	   ���}�   �fXv?      p@!�������?)3��J/�5?2�>	� �����T}�&b՞
�u�hyO�s�uWy��r��N�W�m�ߤ�(g%k�P}���h��l�P�`�E��{��^��m9�H�[���bB�SY�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F����(��澢f�����uE���⾙ѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;�XQ�þ��~��¾0�6�/n���u`P+d����������?�ګ�X$�z��
�}������-�z�!�%�����i
�k���f��p�2!K�R���J��#���j����"�RT��+���f׽r����tO������1���=��]���i@4[���Qu�R"�����/���EDPq��        �-���q=��s�=������=���X>�=H�����=�/�4��==��]���=����%�=f;H�\Q�=�tO���=�f׽r��=�J>2!K�R�>��R���>��f��p>�i
�k>��-�z�!>4�e|�Z#>_"s�$1>6NK��2>K���7�>u��6
�>T�L<�>��z!�?�>0�6�/n�>5�"�g��>��~���>�XQ��>�����>
�/eq
�>K+�E���>jqs&\��>��>M|K�>�_�T�l�>�iD*L��>�f����>��(���>8K�ߝ�>�h���`�>�ߊ4F��>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?&b՞
�u?*QH�x?�������:�              �?              �?      �?              �?      �?              �?      �?      @              �?      �?       @      �?              �?       @      �?      @      �?      �?      �?       @      @      @       @      �?      �?       @      @      �?      �?      �?       @      �?      @       @      �?      �?      �?       @              �?      �?       @      �?      �?      �?               @      �?       @      �?      �?       @      �?              �?      @              �?      �?              �?      �?      �?              �?              �?              �?              �?              �?              @      �?      �?              �?      �?              �?              �?              @              �?              �?              <@              �?              �?              �?               @              �?               @      �?              �?              �?              �?              �?      �?      �?              �?              �?      �?      �?              �?              �?      �?              �?               @       @              �?      �?      �?      �?      �?               @               @       @      �?              �?       @      �?      �?      �?      @      �?      @      @      �?      @       @      @      @      �?              @      @       @      @              �?      @       @      �?      @      @      @      �?      @      �?      @      @               @       @       @              @      �?              �?        
�
conv5/weights_1*�	   �;�¿   �o��?      �@! p��@)���}I@2�	�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@���%>��:�uܬ�@8���%�V6��u�w74��[^:��"��S�F !�ji6�9����ڋ��vV�R9���d�r�x?�x��>h�'��f�ʜ�7
�1��a˲���[���5�i}1?�T7��?�vV�R9?��ڋ?�.�?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?�������:�	              l@     r@     �q@     `n@     �k@     �l@     �h@     �c@     �`@      c@     �`@     �[@     �Z@      X@     �Y@     �Q@     @U@     @S@     �Q@      I@     �E@     �@@      F@      8@     �A@      <@      C@      4@      :@      1@      2@      (@      2@      0@      &@      (@      @      (@      @       @      (@      &@      @      @      @      @       @      @      @      @      @      �?      �?      @      �?       @      �?      @      �?      �?              �?              �?              �?      �?              �?              �?              �?              �?              �?      �?              �?              �?      �?      �?      @              �?               @               @      @      �?      �?      @      @      @      �?      @              �?       @      "@      @      @      @      @      @      @       @      $@      $@      1@      .@      3@      (@      *@      .@      6@      4@      9@      5@      @@      @@     �E@      G@      G@     �D@     �N@     �O@      O@      T@     �S@     �W@     @Z@      ]@      ^@     @`@      d@     �b@     �d@      c@      i@     �m@     �n@     �s@     t@      m@        
�
conv5/biases_1*�	   ��*�   `9$4>      <@!    �>A>)?�6���<2��'v�V,����<�)�%�����i
�k���f��p�Łt�=	���R�����#���j�Z�TA[��RT��+��y�+pm��mm7&c��`����9�e����K������X>ؽ��
"
ֽ�|86	Խ'j��p�=��-��J�=�`��>�mm7&c>y�+pm>RT��+�>�J>2!K�R�>��R���>Łt�=	>�i
�k>%���>��-�z�!>4�e|�Z#>7'_��+/>_"s�$1>6NK��2>�so쩾4>�������:�              �?              @              �?      �?               @               @              �?              �?              �?      �?              �?              �?              �?              �?       @      �?               @              �?              �?               @        ��xU      �ח�	��'��A*�

step  �@

loss�K>
�
conv1/weights_1*�	   �)d��   �m�?     `�@! ���_���)�Fe�v�@2�	��]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82�+A�F�&�U�4@@�$��[^:��"��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��1��a˲���[���f�����uE����a�Ϭ(�>8K�ߝ�>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�vV�R9?��ڋ?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?�������:�	              �?     �F@      h@      i@      f@     �d@     �a@      `@      `@     @Z@     �Z@     �T@      Q@      R@      Q@      K@     �L@      N@     �M@     �G@      C@      C@      D@      :@      4@      9@      9@      0@      3@      4@      2@      ,@      ,@      @      *@      (@      &@       @      (@      @      @      @      @      @      @      @      @      @      �?      @      �?      �?      �?      �?      �?               @       @              �?      �?              �?      �?              �?      �?              �?              �?              �?              �?              �?              �?               @              �?       @      �?      �?      @      @       @      �?              @      �?       @      �?       @      @      @       @      @      @      @      @      @      @      @      @      ,@      @      "@      1@      @      4@      0@      9@      7@      :@      4@      9@      =@      ;@      :@     �F@     �B@     �E@     �K@      M@      R@     �N@     �R@     �R@     �T@     @R@     �Z@     �[@     �`@      a@     �c@     �e@     �f@     �h@     �Q@        
�
conv1/biases_1*�	   `�N`�   �$�x?      @@!  ]�ׯ?)� ���Y0?2��l�P�`�E��{��^���bB�SY�ܗ�SsW�<DKc��T��lDZrS��qU���I�
����G�+A�F�&�U�4@@�$��4[_>��>
�}���>ji6�9�?�S�F !?uܬ�@8?��%>��:?d�\D�X=?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?�������:�              �?              �?              �?              �?              �?              �?              �?              �?      �?              �?               @              �?               @               @       @       @              @       @               @              �?      �?      �?        
�
conv2/weights_1*�	   @�#��   ��?�?      �@! �u��2@)]���.E@2�I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پK+�E��Ͼ['�?�;;�"�qʾ
�/eq
Ⱦ0�6�/n�>5�"�g��>�����>
�/eq
�>;�"�q�>jqs&\��>��~]�[�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?�������:�              @     ,�@     �@     ��@     ��@     ę@     <�@     x�@     Ē@     �@     ̐@     ��@     �@     ȇ@     Ї@     Ѓ@     h�@     x�@     �}@     p{@     �x@     �v@     Pu@     `q@     q@      n@     `n@     @l@     �e@     �f@     �e@     `b@     �`@     �`@      Z@     �\@      V@     @R@     �N@      M@     @P@     �M@     �H@      C@     �D@      D@     �D@     �@@      <@      .@      :@      6@       @      4@      .@      2@      (@      0@      @      &@       @       @      @      (@      @      @      @      @      @      �?      @      @      �?      �?       @      �?      �?       @       @      �?      �?              �?      @              �?       @              �?              �?      �?      �?              �?               @              �?              �?              �?      �?              �?              �?              �?              �?              �?              �?      �?       @       @       @              �?       @      �?       @      �?       @      @       @      �?       @      @      @      @      @      @       @      @      @       @      $@       @      *@      &@      "@      (@      *@      $@      0@      ,@      5@      9@      1@      ;@      >@      @@      A@      F@     �C@     �I@      B@     �K@      N@     �P@     @S@     �V@     @U@     @W@     �Y@     ``@     �c@     @c@     �d@      i@      i@     @m@     0p@     @p@     �u@     @v@     Pw@     px@     �y@     ��@     ��@     `�@     �@     X�@      �@     8�@     ��@     |�@     T�@     �@     t�@      �@     X�@     ��@     h�@     ��@     ��@      @        
�	
conv2/biases_1*�		   ��z�    ��}?      P@!  ?��?)g��	BG:?2����T}�o��5sz�hyO�s�uWy��r�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I��T���C��!�A����#@�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�        �-���q=['�?��>K+�E���>>h�'�?x?�x�?�T7��?�vV�R9?I�I�)�(?�7Kaa+?�u�w74?��%�V6?uܬ�@8?��%>��:?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?���T}?>	� �?�������:�              �?              �?              �?      @      �?       @      �?      �?               @              �?      �?      @      �?              �?      �?              �?      �?      �?      �?      �?      �?      �?              �?              �?              �?              �?              �?              �?              �?              �?       @      �?               @       @      �?              �?      @              �?              �?      �?      �?      �?      �?       @               @      @               @              �?        
�
conv3/weights_1*�	   ��֮�   �c>�?      �@! �V�*��)�����cU@2�����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=���ߊ4F��h���`�8K�ߝ�a�Ϭ(���(����uE���⾮��%��_�T�l׾��>M|Kվ�[�=�k���*��ڽ�G&�$�>�*��ڽ>�XQ��>�����>;�"�q�>['�?��>��>M|K�>�_�T�l�>�ѩ�-�>���%�>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?�������:�             �@     �@     ,�@     ��@     ��@     ơ@     p�@     ��@     ��@     �@     ԕ@     �@     D�@     $�@     ��@     �@     (�@     �@      �@     8�@     x�@     �}@     `z@      z@     `y@      u@     �r@     pr@     @m@      j@     `i@     �j@     �d@     @e@     �d@     �`@      ]@     �Z@     �W@     @W@      Y@     @R@     �P@     �M@      E@      F@      E@     �C@     �C@     �A@     �D@      1@      :@      1@      2@      4@      8@      6@      0@      "@      *@      1@      @       @      @      &@      @      @       @      @      @      @      @      @      @       @      @      �?      @      @       @      �?               @              �?      �?              �?              �?              �?              �?              �?              �?               @              �?               @              �?              �?       @              �?       @              �?      �?      �?      @       @       @      �?      �?      @      @      �?      @      @      @      @      @      @      @      @      "@      @      "@      $@      1@      1@      ,@      *@      ,@      2@      .@      9@      9@      B@      ;@      @@      E@     �A@     �E@      D@      K@      M@     �N@     �R@      Q@     �W@     �X@     @]@     �`@     `a@     �a@     `c@      e@     �h@      k@     �j@     �n@     0q@     @q@     `r@     �x@     0y@     �}@     �}@     �@     �@     h�@     0�@      �@     H�@     �@     ��@     |�@     x�@     ĕ@     �@     ԙ@     ��@     �@     D�@     أ@     2�@     0�@     ��@     Ȉ@        
�
conv3/biases_1*�	   `��|�    �t?      `@!  eh�s�?)�����>?2����T}�o��5sz�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.���vV�R9��T7�����[���FF�G �        �-���q=;�"�q�>['�?��>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>1��a˲?6�]��?f�ʜ�7
?>h�'�?x?�x�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?�7Kaa+?��VlQ.?��bȬ�0?�u�w74?��%�V6?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?�������:�              �?              @      �?              �?      �?               @       @               @      �?              @      @      @       @               @      �?              �?      @       @       @      �?              @              �?      �?              �?              �?       @      �?              �?              �?              @              �?              �?      �?      �?              �?              �?      �?              �?              �?               @       @              �?               @       @       @      �?               @               @               @      @      �?      @      @      @      @      @      @      @      @              @       @              �?        
�
conv4/weights_1*�	   ��+��   �2A�?      �@! 1�vp7�)��f�oe@2�Ӗ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$�ji6�9���.���T7����5�i}1���d�r�x?�x��>h�'����[���FF�G �I��P=��pz�w�7��})�l a��ߊ4F��h���`�
�/eq
Ⱦ����ž���?�ګ�;9��R���_�T�l�>�iD*L��>�uE����>�f����>a�Ϭ(�>8K�ߝ�>})�l a�>pz�w�7�>��Zr[v�>O�ʗ��>6�]��?����?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�������:�              a@     $�@      �@     ��@     T�@     0�@     8�@     ��@     (�@     Ѕ@     P�@     ��@     �@     @}@     |@     0w@     �t@     0t@     �t@     �p@     �o@     �i@     �h@      i@     @g@     �e@      `@     �W@     �\@     @Y@     @R@     �U@     �R@     �Q@      O@     �P@     �D@     �I@     �B@      F@     �C@      7@      <@      8@      :@      =@      =@      7@      6@      6@      ,@      $@      $@      "@      $@      $@      (@       @      $@      @       @      @      @      @      @      �?              @       @      @      �?              @              �?      �?      �?      �?              �?              �?      �?              �?              �?              �?              �?              �?               @               @              �?              �?               @      �?              �?      �?      �?               @      �?       @      @      �?       @              @      �?      @       @      @      @      @      @       @      @       @      @      $@      @      *@      $@      ,@      1@      2@      "@      ,@      6@      :@     �B@      @@      ?@     �F@     �@@     �E@      H@     �M@     �M@     �O@     �P@     �O@     �S@      S@     @W@     @^@      Z@     �b@     `d@      f@      e@     �g@      j@      k@     Pp@     �s@     �t@     0u@     �z@     �y@     �{@     p�@      �@     ��@      �@     ��@     �@     ��@     �@     ܑ@     l�@     $�@      �@     @b@        
�
conv4/biases_1*�	   �e傿   `�0�?      p@!0�܎r�?)��Н)LB?2����J�\������=���o��5sz�*QH�x�&b՞
�u�uWy��r�;8�clp��N�W�m�ߤ�(g%k�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]�����[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��8K�ߝ�a�Ϭ(���(��澢f���侮��%ᾙѩ�-߾�iD*L�پ�_�T�l׾['�?�;;�"�qʾ
�/eq
Ⱦ����ž�XQ�þ�*��ڽ�G&�$����n�����豪}0ڰ���o�kJ%�4�e|�Z#���-�z�!�%������f��p�Łt�=	��J��#���j�Z�TA[�����"�RT��+��nx6�X� ��f׽r����tO��������%���9�e�����-��J�'j��p���
"
ֽ�|86	Խ�EDPq���8�4L���        �-���q=��1���='j��p�=���">Z�TA[�>�#���j>2!K�R�>��R���>%���>��-�z�!>4�e|�Z#>��o�kJ%>4��evk'>���<�)>�z��6>u 5�9>��z!�?�>��ӤP��>���?�ګ>����>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�ߊ4F��>})�l a�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�T7��?�vV�R9?��ڋ?�.�?�S�F !?�[^:��"?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?>	� �?����=��?�������:�              �?              �?      �?              �?      �?      �?              �?      �?      �?       @               @      �?       @      �?              @       @       @      �?       @      @      �?      @       @              �?      �?      @      @      �?      �?              �?      �?       @              �?      �?       @              �?       @      �?      �?      �?      �?      �?              �?              @       @       @              @      �?       @              @              �?              �?              �?      �?              �?              �?               @      �?      �?              �?              �?              �?      @              �?      �?              �?              �?              �?              �?              :@              �?              �?       @              �?              �?              �?      �?       @              �?              �?              �?              �?              �?      �?      �?      �?      �?              �?      �?      �?      �?              �?              �?      �?      �?      �?              �?              �?      �?      �?      �?       @              �?      @      �?               @               @       @      @      @      �?      �?       @      @       @      �?       @      @      @      �?              @      �?       @      @      �?       @      @      @      �?      @       @       @      @       @      @      �?      �?      �?      @      �?      �?              �?        
�
conv5/weights_1*�	   ��¿    ���?      �@! h���� @)k2c���I@2�	�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74��[^:��"��S�F !�ji6�9����ڋ��vV�R9�6�]���1��a˲��h���`�8K�ߝ�pz�w�7�>I��P=�>>h�'�?x?�x�?��ڋ?�.�?+A�F�&?I�I�)�(?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?�������:�	             �k@      r@      r@     @n@     �k@     @l@     @i@     �c@     �`@     �b@     `a@     �[@     @Z@     @X@     @Y@     �Q@      V@     @S@     �Q@      I@     �F@      >@      F@      7@      C@      :@     �B@      6@      ;@      0@      2@      ,@      ,@      3@      $@      $@      "@      $@      @      @      *@      (@      @       @      @      @       @      @      @      @      �?      @       @      �?       @      �?              �?      �?      @       @      �?               @              �?      �?              �?              �?              �?              �?              �?              �?              �?              �?      �?               @      �?      �?              �?      �?       @      @      �?      @       @      �?      @              @      �?      @       @       @      @      @      "@      @      @      @       @      $@      &@      1@      (@      5@      &@      *@      1@      4@      6@      7@      5@      @@     �@@     �E@     �F@     �F@     �D@     �O@      O@      P@     �S@     �S@      X@     �Y@     @]@     @^@     @`@     �c@      c@     �d@     �b@      i@     �m@     �n@     �s@      t@     `m@        
�
conv5/biases_1*�	    �7�   �l�:>      <@!   IuJ:>)�֏�`�<2�u 5�9��z��6�7'_��+/��'v�V,���o�kJ%�4�e|�Z#���-�z�!�%�����i
�k���f��p�Łt�=	���R����2!K�R���J��#���j����"�RT��+���tO����f;H�\Q����-��J�'j��p�f;H�\Q�=�tO���=�f׽r��=���">Z�TA[�>�#���j>�J>2!K�R�>��R���>Łt�=	>��f��p>�i
�k>%���>��-�z�!>6NK��2>�so쩾4>�z��6>u 5�9>p
T~�;>�������:�              �?              �?              �?              �?               @      �?      �?              �?      �?              �?              �?              �?              �?       @              �?              �?              @      �?              �?      �?      �?              �?      �?              �?        �K&(T      �z	�`|�'��A*��

step  �@

lossȉ>
�
conv1/weights_1*�	   �o���   `jP�?     `�@! @�8��)�dI���@2���]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !��T7����5�i}1�>h�'��f�ʜ�7
������f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?��ڋ?�.�?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�������:�              @     �I@     �g@      i@     @f@     `d@     �a@     �`@     �_@      Z@     �Y@     @U@      Q@     �Q@     @Q@     �J@     �M@     �N@     �G@     �M@     �D@      =@      E@      <@      2@      7@      3@      9@      5@      1@      3@      &@      *@      $@      1@      $@      (@      @      @      @      @      @      @      @      @       @      �?      @      �?      @              @       @       @       @      �?              @       @               @              �?      �?              �?              �?      �?              �?      �?      �?              �?               @               @      �?       @      �?              �?       @      �?       @      �?       @              �?      �?       @      @      @      @       @      @       @      @      @      @      @      @       @      @      0@       @      $@      &@      *@      7@      4@      7@      7@      ;@      8@      7@      >@      A@      D@      B@      F@     �J@     �M@     @R@     �N@     @R@     @S@     @T@     @P@      [@      ^@     ``@     �`@     �c@      e@     �f@     �g@      T@      �?        
�
conv1/biases_1*�	    �`^�   `��?      @@!  �ᤵ?)��	~0�<?2�E��{��^��m9�H�[���bB�SY�<DKc��T��lDZrS�nK���LQ�k�1^�sO��[^:��"��S�F !��5�i}1���d�r�ji6�9�?�S�F !?�T���C?a�$��{E?�qU���I?IcD���L?nK���LQ?�lDZrS?ܗ�SsW?��bB�SY?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?*QH�x?o��5sz?���T}?>	� �?����=��?�������:�              �?      �?              �?              �?              �?              �?              �?               @              �?              �?               @              �?      �?      �?       @       @      @      @       @              �?               @      �?        
�
conv2/weights_1*�	   �l���    Eë?      �@!�p�Pq5@)��Њ�2E@2�I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F���uE���⾮��%�����ž�XQ�þ��n�����豪}0ڰ�K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�uE����>�f����>a�Ϭ(�>8K�ߝ�>})�l a�>pz�w�7�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?�������:�              @     Џ@     ޠ@     �@     ԛ@     ��@     \�@     L�@     ؒ@      �@     А@     ��@     0�@     ��@     Ȇ@     H�@     �@     h�@     �}@     `{@     �w@     `v@     Pv@     �q@     �p@      n@     �n@     �k@     `e@      g@     �e@      d@     �a@     @\@     �]@      Z@      V@     �Q@      I@     �R@     �P@      K@      K@     �C@     �A@      ?@      F@      =@      =@      ;@      7@      1@      3@      (@      5@      1@      ,@      0@      *@      (@      @      "@       @      @      @       @      @      @      @       @      @      @      @      �?       @      �?      �?      �?       @       @       @              �?              �?      �?               @              �?              �?              �?              �?      �?              �?              �?              �?               @      �?      @      @       @      �?              @       @      �?      @      @      @      @      @      @      @      @      @      @      "@      (@      (@      $@      (@      ,@      *@      "@      3@      *@      1@      9@      6@      8@     �A@      ?@      9@     �E@     �D@     �G@     �D@      O@      N@     @R@     �R@     �S@      W@     @U@     �Y@      b@      c@      b@     �d@     `h@     �j@     �m@      p@     `o@     �v@     �u@     �v@     `x@     �y@     ��@     ؀@     ��@     ��@     (�@     ��@     ��@      �@     ��@     H�@     �@     P�@     �@     h�@     ��@     Ȟ@     ��@     �@      @        
�	
conv2/biases_1*�		   �}v�   @�ۀ?      P@!  l驟?)Z�����B?2�>	� �����T}�*QH�x�&b՞
�u�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�
����G�a�$��{E��!�A����#@���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.�})�l a��ߊ4F��        �-���q=8K�ߝ�>�h���`�>�.�?ji6�9�?�S�F !?��VlQ.?��bȬ�0?��82?�u�w74?uܬ�@8?��%>��:?d�\D�X=?���#@?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?�������:�              �?              �?              @      �?      �?      �?      �?       @              �?      �?      �?       @       @      �?              �?              �?              �?      �?      �?              �?      �?              �?              �?              �?              �?      �?              �?              �?              �?               @              �?      �?      �?               @              �?      @              �?      �?               @      �?              �?       @      �?      �?               @      �?       @      �?      �?              �?        
�
conv3/weights_1*�	   @j	��    ���?      �@! H�g���?)O~�v�eU@2�����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a�h���`�8K�ߝ�a�Ϭ(��uE���⾮��%�E��a�Wܾ�iD*L�پ['�?�;;�"�qʾ;�"�q�>['�?��>K+�E���>jqs&\��>�f����>��(���>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?�������:�             x�@     ��@     �@     ��@     ��@     ��@     p�@     ��@     ��@     �@     �@     �@      �@     t�@     �@     H�@     h�@     h�@     ��@     0�@     ��@     �~@     0z@     py@      z@     pt@      s@     �q@     �n@     @l@      i@      h@     @e@     �e@      b@     �c@     �\@     @[@     �W@     �V@     @U@     �R@      T@      K@     �D@     �I@      D@      D@     �D@     �C@      C@      5@      9@      4@      *@      3@      6@      .@      ,@       @      ,@      0@      "@      @       @       @      "@      @      $@      �?      @      @      @      @              @              @      �?      @       @      @      �?      �?      �?      �?              �?      �?              �?               @               @              �?              �?              �?              �?      �?              �?               @      �?       @      @       @       @      @      @      @       @      �?      @      @      @      @      @       @       @      @      @      @      @      ,@      (@      ,@      3@      $@      .@      :@      1@      1@      =@      =@      C@      B@      E@      ?@     �J@      C@      J@      K@     �Q@      L@     �Q@     �X@     �W@     �]@      a@      _@     @b@     �b@     �e@     �i@      k@     �j@     �m@     �q@     �q@     �q@     �x@     �y@     p}@     �}@     �@     ��@     (�@     ��@     ��@     (�@      �@     ��@     |�@     ��@     ��@     l�@     |�@     ��@     ,�@     B�@     ޣ@     �@     H�@     ة@     ��@        
�
conv3/biases_1*�	   �
5��   @ww{?      `@!  ~��O�?)��$�SH?2�����=���>	� ��&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9���d�r�x?�x��pz�w�7��})�l a�        �-���q=a�Ϭ(�>8K�ߝ�>��Zr[v�>O�ʗ��>�FF�G ?��[�?ji6�9�?�S�F !?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?�!�A?�T���C?a�$��{E?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?�������:�              �?              @      �?              �?      �?      �?      �?       @      �?      �?              �?              @      @      @      @              �?              �?       @      @      �?              �?      �?      �?      @      �?              �?      �?      �?       @              �?      �?      �?      �?              �?              �?              @              �?              �?              �?              �?              @               @              �?       @              �?      @              �?      �?      @               @      @       @       @      @      @      @      @      @      @      @      �?               @       @       @              �?        
�
conv4/weights_1*�	   ��;��   ��]�?      �@! ��`��6�)D��tdpe@2�Ӗ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��T7����5�i}1���d�r�x?�x��f�ʜ�7
��������[���FF�G �I��P=��pz�w�7��['�?��>K+�E���>�ѩ�-�>���%�>})�l a�>pz�w�7�>��Zr[v�>O�ʗ��>6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�������:�              a@     $�@     �@     ��@     \�@     8�@     8�@     ��@     (�@     ȅ@     8�@     ��@     �@     `}@     �{@     0w@      u@     pt@     t@     �p@     `p@     �h@      i@     `h@     �g@     @e@      `@     �W@     �\@     �X@     @S@     @V@     �Q@      R@      N@      P@      E@     �J@      C@      B@     �E@      :@      :@      9@      <@      <@      ;@      8@      6@      5@      ,@      (@      "@      (@      @      @      (@      @      @      @      @      @      �?      @      @      @              @       @      @      �?      �?      �?       @      �?      �?              �?      �?      �?              �?              �?               @              �?              �?              �?               @              �?              �?      @              @      �?      �?      �?      �?       @       @       @      @              �?      �?       @      @              @      �?      @      @      @      @      @      "@       @      @      @      $@      &@       @      1@      *@      .@      $@      1@      7@      ?@      =@      @@      <@     �G@      B@      D@     �G@      O@     �O@     �O@     �O@     �P@      S@     �R@      W@     �_@     @Y@     �b@     `d@     �e@     @e@     �g@      k@     �j@     @p@     �s@     pt@     pu@     �z@     �y@     �{@     ��@     Ё@     ��@     �@     ��@     �@     ��@     0�@     ̑@     L�@     0�@     �@      c@        
�
conv4/biases_1*�	   @`���   �ӏ�?      p@!H�����?)c趧�+L?2�eiS�m��-Ա�L�����T}�o��5sz�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(����uE���⾮��%�E��a�Wܾ�iD*L�پ['�?�;;�"�qʾ�u`P+d����n�������������?�ګ�;9��R���5�L��_"s�$1�7'_��+/��'v�V,����<�)�4��evk'���o�kJ%�4�e|�Z#���-�z�!�2!K�R���J��#���j�Z�TA[�����"�RT��+��nx6�X� ��f׽r����tO�����9�e����K���        �-���q=��@��=V���Ұ�=��R���>Łt�=	>��f��p>%���>��-�z�!>4�e|�Z#>��o�kJ%>4��evk'>���<�)>�'v�V,>7'_��+/>����W_>>p��Dp�@>�
�%W�>���m!#�>�u��gr�>�MZ��K�>��|�~�>���]���>��>M|K�>�_�T�l�>E��a�W�>�ѩ�-�>���%�>�uE����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>��Zr[v�>O�ʗ��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?���T}?>	� �?-Ա�L�?eiS�m�?�������:�              �?               @              �?      �?              �?      �?               @      �?      �?      �?       @      �?       @      �?      �?      �?      @       @              �?      @      �?      @               @              �?      @       @      @       @      �?               @      �?       @      �?              �?              �?      @      �?       @              �?      �?       @              �?              �?      @      �?       @               @      �?               @      �?       @              �?               @              �?              �?              �?              �?              �?      �?      �?              �?      �?       @              �?      �?      �?              �?              �?      �?              �?              8@              �?              �?      �?              �?              �?      �?      �?              �?              �?              �?              �?              �?               @              �?               @              �?              �?               @              �?               @      �?       @      �?               @       @      �?      �?      �?              �?      @              �?              �?      @      �?      �?      @       @      @       @       @      @       @       @      @      @      @              @              @       @       @      �?              @      �?      @      @       @      @      @      @      �?       @      @              �?      @              �?              �?        
�
conv5/weights_1*�	   �1�¿   `"��?      �@! X�x9|!@)�(���I@2�	�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C����#@�d�\D�X=���%>��:���%�V6��u�w74���82���bȬ�0���VlQ.�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9����ڋ��vV�R9���[���FF�G ���Zr[v�>O�ʗ��>��[�?1��a˲?�T7��?�vV�R9?��ڋ?�.�?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?�������:�	             �k@      r@     �q@      n@     @l@      l@     @i@     @c@      a@     �b@     `a@     �[@     �Z@      X@     �X@     �Q@     �U@     �S@     @Q@      I@      G@      @@     �E@      7@     �C@      :@      @@      9@      >@      .@      2@      *@      ,@      3@      &@      $@      @      ,@      @      @      *@      "@      "@       @       @      @      @      @      �?      @      @      @       @       @      �?       @              �?              �?      �?              �?      �?              �?               @      �?      �?      �?              �?              �?              �?              �?              �?              �?              �?      �?       @      �?       @              �?       @              @       @      �?       @      @      �?      @              @              �?      @      @      @      @      @      @      @      @      "@      "@      (@      2@      &@      2@      (@      .@      0@      5@      6@      8@      4@      @@      @@      F@     �G@     �E@      D@     @P@      N@      O@     @T@     �S@      X@     �Y@     �]@      ^@     �`@     �c@     �b@     �d@      c@     `i@     �m@     `n@     �s@     0t@     �m@        
�
conv5/biases_1*�	    �bF�   �֦=>      <@!   ���<>)(ֽ�,6�<2���8"uH���Ő�;F�6NK��2�_"s�$1��'v�V,����<�)�4��evk'���o�kJ%�%�����i
�k���f��p�Łt�=	�Z�TA[�����"�RT��+���K��󽉊-��J�ݟ��uy�z�������؜�ƽ�b1�Ľ'j��p�=��-��J�=�f׽r��=nx6�X� >y�+pm>RT��+�>���">Z�TA[�>2!K�R�>��R���>Łt�=	>�i
�k>%���>��o�kJ%>4��evk'>�'v�V,>7'_��+/>�so쩾4>�z��6>p
T~�;>����W_>>�������:�              �?              �?              �?              �?               @      �?      �?              �?      �?              �?              �?              �?              �?              �?              �?              �?              �?      @               @              �?              �?              �?               @         KU      ���	#��'��A*��

step   A

loss���=
�
conv1/weights_1*�	   �2���   @���?     `�@! ������)b6rh	@2�	��]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !��vV�R9��T7����5�i}1�x?�x��>h�'���_�T�l�>�iD*L��>�uE����>�f����>��(���>a�Ϭ(�>�ߊ4F��>})�l a�>�FF�G ?��[�?>h�'�?x?�x�?�T7��?�vV�R9?��ڋ?�.�?U�4@@�$?+A�F�&?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�������:�	              @      M@     �f@     @i@     `f@     �d@     �`@     �`@     �_@     @Y@     @Z@     @U@      Q@     �Q@      O@      O@      K@     �N@      H@     �N@      D@     �@@      B@      >@      1@      :@      2@      9@      5@      (@      4@      *@      ,@      *@      &@      *@      @      $@      @      "@       @      @       @      @      @      @       @       @      @      @       @      �?      @       @              �?       @      �?              @      �?      �?              @              �?      �?              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?               @      �?      �?      @              �?       @       @      @      �?      @      @      �?      @      @      @      @      @      @      "@      "@      $@      (@      @      *@      *@      &@      3@      7@      5@      9@      =@      3@      9@      ?@     �A@     �B@     �C@     �E@     �J@     �O@     �P@      O@     �Q@      U@     �S@      P@     @Z@      ]@     �a@     �`@     �c@     @e@     �e@     �h@     @U@      �?        
�
conv1/biases_1*�	   ��q\�   ��ۄ?      @@!  xgǻ?)�(Y�"�F?2�E��{��^��m9�H�[���bB�SY�<DKc��T��lDZrS�nK���LQ��.����ڋ�U�4@@�$?+A�F�&?�u�w74?��%�V6?
����G?�qU���I?IcD���L?k�1^�sO?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?5Ucv0ed?Tw��Nof?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?>	� �?����=��?���J�\�?-Ա�L�?�������:�              �?      �?              �?      �?              �?              �?              �?              �?               @              �?              �?      �?              �?              @      �?      �?       @       @      @              �?      �?       @        
�
conv2/weights_1*�	    m䫿   ��I�?      �@!�%�H�7@)�/�6E@2�I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ����ߊ4F��h���`��uE���⾮��%�E��a�Wܾ�iD*L�پ['�?�;;�"�qʾ����ž�XQ�þ5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>['�?��>K+�E���>�iD*L��>E��a�W�>�uE����>�f����>a�Ϭ(�>8K�ߝ�>�ߊ4F��>})�l a�>pz�w�7�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?�������:�              @     H�@     Ԡ@     �@     ��@     �@     �@     t�@     ��@     D�@     Đ@     ؍@     H�@     ��@     x�@     ��@     P�@     �@      }@     �{@     �w@     �v@     `v@     q@     pq@      n@     �n@     @k@     `e@     `g@     �f@     �b@      `@      b@     @[@      Y@     �R@     @P@     �R@     �N@      K@      O@      H@      B@     �H@      D@      ?@      ?@      C@      5@      ;@      3@      0@      .@      1@      5@      0@      @      *@      @      (@      $@      $@      @      @      &@      @      @      @      @      @       @      @       @       @      �?              �?      �?      �?      @              �?              �?              �?              �?              �?              �?              �?      �?               @              �?               @               @               @      �?              �?       @       @      �?       @       @      @      @      �?       @      @      @      @      @      @      �?      @      @       @      @      @      @      *@      $@      ,@      ,@      (@      $@      1@      :@      $@      4@      4@      7@      :@     �B@     �B@     �E@      B@     �F@      E@      N@      O@     �R@     �T@     �S@     �S@     @X@     �Y@      a@     �c@     ``@      e@      h@     �k@     �l@     pp@     �o@     �v@     �u@     Pv@     `x@     �z@     �@     ȁ@     H�@     ��@     h�@     ��@     0�@     �@     ��@     |�@     �@     4�@     ��@     ܚ@     h�@     О@     ��@     |�@      &@        
�	
conv2/biases_1*�		    q(��    ��?      P@!  J�X�?)Z�W�	�I?2����J�\������=������T}�o��5sz�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T�k�1^�sO�IcD���L��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0����%ᾙѩ�-߾        �-���q=>�?�s��>�FF�G ?�T7��?�vV�R9?�7Kaa+?��VlQ.?��82?�u�w74?d�\D�X=?���#@?�T���C?a�$��{E?
����G?�qU���I?IcD���L?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?�������:�              �?              �?              �?      @      �?              �?      �?       @      �?              �?              @      �?       @              �?              �?      �?              �?               @              �?              �?              �?              �?              �?              �?              �?              �?               @       @              �?              @              �?       @       @      �?      �?              @              @      �?       @      �?               @      �?      �?               @        
�
conv3/weights_1*�	   @�@��    ��?      �@! t;}ؚ@)�a��gU@2�����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ��>M|Kվ��~]�[Ӿ;�"�q�>['�?��>jqs&\��>��~]�[�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?�������:�             `�@     ʩ@     �@     Х@     ��@     ��@     ��@     ��@     ��@     �@     Е@      �@     �@     <�@     (�@     ��@     H�@     P�@     �@     (�@     ��@     �~@     0z@     @z@     px@      u@     �s@     �q@     �m@      l@     �h@     �k@     @b@     �e@     �c@     �a@     �\@     �\@     �Y@     @U@     �S@     @U@      R@      N@      G@     �J@     �E@     �A@      C@      ;@      D@      =@      9@      5@      2@      2@      ,@      ,@      .@      $@      0@      $@      &@      (@      "@       @      (@      @      @       @      @      @      @      �?       @      @       @      @       @      @      �?      �?              �?       @      �?      �?              �?       @      �?      �?              �?              �?              �?              �?              �?              �?               @              �?              �?      �?      �?      �?      �?              �?      �?       @      �?      @       @       @      @       @       @      @      @      @      @      �?      @      @      @      @      @      @      @       @      (@      .@      0@      &@      3@      9@      (@      <@      4@      7@     �D@      F@     �H@      <@     �K@      C@     �H@     �J@      P@      Q@     �R@      W@      Z@     �X@     �_@      `@     �b@     @c@     `e@     �h@     �j@     �l@     �m@     Pq@     �q@     �q@     �w@     �y@     @~@     �}@     �~@     ��@     ��@     Ȇ@      �@     �@     �@     ��@     P�@     ��@     |�@     ��@     p�@     ��@     X�@     (�@     �@     6�@     ,�@     ҩ@     �@        
�
conv3/biases_1*�	   `n    ?�?      `@!  I\�Q�?)�c�MQ?2�-Ա�L�����J�\��*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��5�i}1���d�r�I��P=��pz�w�7���f�����uE����        �-���q=pz�w�7�>I��P=�>1��a˲?6�]��?�[^:��"?U�4@@�$?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?�!�A?�T���C?a�$��{E?
����G?�qU���I?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?�������:�              �?              @      �?               @               @       @      �?      �?              �?      @      @       @               @      �?              �?       @       @      �?       @              �?      �?       @              �?      �?              �?      �?               @              �?      �?               @      �?              �?               @              �?              @              �?              �?               @               @       @              �?      �?              �?              @      �?              �?      �?       @      �?       @      @       @      @       @      @      @      @      @      @      @       @       @      �?      @      �?      �?      �?      �?        
�
conv4/weights_1*�	   �FK��   @�y�?      �@! ���'6�)Ԭ�)7qe@2�Ӗ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&��[^:��"��S�F !��.����ڋ��5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���O�ʗ�����Zr[v��I��P=��pz�w�7���uE����>�f����>��(���>�h���`�>�ߊ4F��>})�l a�>��Zr[v�>O�ʗ��>6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�������:�              a@      �@     �@     ��@     h�@     $�@     8�@     ��@     8�@     �@      �@     ��@     �@     p}@     �{@     �w@      u@     0t@      t@     �p@     pp@     �i@     �h@      h@     @g@     �e@     �`@     @W@     �\@     �W@     @S@     �U@     �S@      Q@     �M@      O@      G@      I@     �B@      C@      D@      =@      ;@      9@      =@      ;@      9@      8@      6@      4@      *@      ,@      @      ,@       @      @      $@      @      "@      @      @      @      @       @      @       @      �?      �?      @      @              @               @              �?      @      �?              �?      �?              �?              �?              �?      �?              �?      �?              �?              �?              �?       @              �?      �?      �?       @               @      @       @       @      �?              @      �?      @              @      @      @       @      @      @      @      ,@      @      $@       @       @      *@      @      1@      ,@      *@      @      3@      ;@      9@     �A@      :@      ?@     �E@     �C@      F@      G@      M@     �Q@      O@     �M@     �O@     @S@     �R@     @X@     �^@     �Y@     �a@     @d@      f@     @e@      h@     �j@     �j@     Pp@     �s@     `t@     �u@     `z@     Py@     @|@     ��@     ��@     ��@     0�@     �@     Ȉ@     ��@     (�@     ��@     `�@     (�@     �@     `c@        
�
conv4/biases_1*�	    B.��   �e]�?      p@!�RļW�?)��U�� T?2��7c_XY��#�+(�ŉ�����=���>	� ��o��5sz�*QH�x�&b՞
�u�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=���ߊ4F��h���`�8K�ߝ�a�Ϭ(龢f�����uE���⾙ѩ�-߾E��a�Wܾjqs&\�ѾK+�E��Ͼ�XQ�þ��~��¾5�"�g���0�6�/n��豪}0ڰ������;9��R���5�L���u��gr��R%�������so쩾4�6NK��2�_"s�$1�7'_��+/��'v�V,����<�)�4��evk'���o�kJ%���-�z�!�%�����i
�k���f��p�Łt�=	���R�����J��#���j�Z�TA[�����"�RT��+��y�+pm�        �-���q=PæҭU�=�Qu�R"�=��f��p>�i
�k>%���>4�e|�Z#>��o�kJ%>���<�)>�'v�V,>7'_��+/>6NK��2>�so쩾4>p��Dp�@>/�p`B>�4[_>��>
�}���>���?�ګ>����>豪}0ڰ>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>8K�ߝ�>�h���`�>�ߊ4F��>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>�FF�G ?��[�?1��a˲?6�]��?����?x?�x�?��d�r?�5�i}1?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?����=��?���J�\�?#�+(�ŉ?�7c_XY�?�������:�              �?               @              �?      �?              �?              �?              @       @       @       @              �?       @       @      �?       @      �?              �?      @      @      �?       @               @      �?      @      @      �?      �?              �?      @              �?       @      �?              �?      �?      �?              �?       @              @       @      �?      �?               @      �?      @      �?               @      �?               @       @       @              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?      �?               @      �?              �?              �?              �?              �?      �?      �?              �?              8@              �?              �?      �?              �?              �?      �?              �?              �?              �?              �?      �?              �?               @      �?              �?      �?              �?      �?      �?              �?      �?      �?      �?              @      �?               @      @      �?       @       @              �?              �?       @       @       @      @               @      @      @      @      �?      @       @      @      @      @              @      @      �?      �?      �?       @       @       @      @      @      �?      @      @      �?       @       @       @      �?      @       @              �?              �?        
�
conv5/weights_1*�	   ���¿   ���?      �@! Ȝ��Z"@)�(��ΆI@2�	�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%�V6��u�w74���82���bȬ�0���VlQ.�U�4@@�$��[^:��"��S�F !�ji6�9����ڋ��vV�R9�>�?�s���O�ʗ�����Zr[v��8K�ߝ�>�h���`�>f�ʜ�7
?>h�'�?��ڋ?�.�?�[^:��"?U�4@@�$?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?�������:�	             @k@     Pr@     �q@      n@     @l@      l@     @i@      c@      a@     �b@     �a@      [@     �Z@     @X@     @X@      R@     �U@     �S@     @Q@      I@     �F@      A@      D@      :@      C@      :@      A@      7@      <@      1@      1@      ,@      .@      0@      *@      $@       @      $@      @      @      ,@       @      "@      �?      @      @      @      @      �?      @       @      @      @       @      �?      @              �?      �?              �?              �?              �?      �?               @              �?              �?              �?      �?              �?              �?              �?               @              �?      �?      �?      �?       @      �?              �?      �?       @      �?       @      �?       @       @       @      @      �?      @              �?      @      "@      @       @      @      "@      @      @      @      (@      &@      0@      .@      2@      $@      .@      4@      2@      5@      7@      7@      ?@      =@      G@     �H@      E@      C@     �P@      M@      P@     �T@     @R@      Y@     �Y@      ]@     �^@     @`@     �c@     �b@     �d@     �b@     �i@     @m@     �n@     �s@     �t@     `m@        
�
conv5/biases_1*�	   @L�N�   @B�A>      <@!   X�w>>)n��ׯ��<2�28���FP�������M��z��6��so쩾4�_"s�$1�7'_��+/����<�)�4��evk'���o�kJ%�4�e|�Z#���-�z�!�%�����i
�k��#���j�Z�TA[���`���nx6�X� �f;H�\Q������%��PæҭUݽH����ڽf;H�\Q�=�tO���=�f׽r��=nx6�X� >Z�TA[�>�#���j>�J>2!K�R�>��R���>Łt�=	>��f��p>%���>��-�z�!>4�e|�Z#>���<�)>�'v�V,>�z��6>u 5�9>p��Dp�@>/�p`B>�������:�              �?              �?              �?              �?      �?      �?              �?      �?              �?              �?              �?              �?              �?              �?              �?      �?      �?      �?      �?       @              �?      �?              �?               @               @        �t_�U      �o_�	�kc�'��A	*�

step  A

lossv��=
�
conv1/weights_1*�	   �lP��   @y�?     `�@!  Zԅ��)�3n�)@2�	��]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�ji6�9���.����ڋ��vV�R9��T7����5�i}1�x?�x��>h�'����[���FF�G ��ߊ4F��>})�l a�>��d�r?�5�i}1?�vV�R9?��ڋ?ji6�9�?�S�F !?�[^:��"?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�������:�	              @      M@     �f@     �i@     �e@     �c@     �a@     ``@      ^@     �[@     �Y@     @U@      Q@      Q@      P@      O@     �K@     �N@      I@      K@     �E@      B@      B@      <@      7@      1@      3@      9@      2@      2@      ,@      1@      $@      3@      .@      @      @      $@      @       @      @      @      @      @      @      @      �?      @      �?      @       @      @      �?       @      �?      �?      �?      �?              �?              �?      �?              �?              @      �?      �?              �?              �?              �?              �?              �?              �?      �?              �?       @       @              �?              @              �?      �?       @       @               @       @      @      �?       @       @      @      @      @      @      @       @      @      @      (@      $@      $@      0@      2@      &@      &@      4@      8@      5@      >@      ;@      :@      3@     �D@      ?@     �F@     �E@     �K@     �K@     �P@     @Q@     �P@     �S@      U@     �P@     �X@      \@     @b@     �a@      c@      d@      f@     �h@     @W@       @        
�
conv1/biases_1*�	    �"^�   �f;�?      @@!  �L7�?)�U���P?2�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T�nK���LQ�k�1^�sO��[^:��"��S�F !�+A�F�&?I�I�)�(?IcD���L?k�1^�sO?nK���LQ?�lDZrS?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�������:�              �?              �?      �?              �?              �?              �?               @      �?      �?              �?              �?              �?              �?              �?       @      �?      @      @       @      �?              �?              @        
�
conv2/weights_1*�	   ` A��   ��լ?      �@!��'��:@)?S�;E@2�I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ��uE���⾮��%���n�����豪}0ڰ��*��ڽ>�[�=�k�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?�������:�              &@     ��@     ܠ@      �@     ��@      �@     8�@     ��@     ��@      �@     ��@     0�@     H�@     ��@     `�@     ��@     ��@      @     0~@     �z@     x@     Pw@     0u@     pr@     �p@     �m@     �o@     �k@     �d@     �g@     �e@     �c@      `@      a@     @\@     �X@     �V@     �R@     �I@     �O@      O@      G@     �H@     �D@     �D@     �C@     �E@      ?@     �@@      8@      <@      6@      3@      &@      *@      1@      2@      (@      (@      *@      "@      &@      @       @      @      @      @               @      @      @      @       @       @      �?       @       @       @      �?               @      �?              �?              �?      �?       @              �?              �?              �?              �?              �?              �?      �?              �?      �?              �?       @               @      �?               @              �?      @              @      �?      @      @      @      @      @       @      @       @      �?      @      "@      @       @      @      1@       @      &@      $@      &@      *@      .@      1@      .@      3@      3@      :@      3@      8@     �B@      @@     �E@      E@     �G@      B@      O@      O@     �Q@     �R@     @T@      W@      W@     �[@     �_@     �b@     �b@     �b@     �h@     `i@     �m@      q@      p@     �v@     v@     @v@      w@     �{@     �@     ��@     `�@     ��@     `�@     ��@     ��@     �@     Ԑ@     ��@     ��@     H�@     ��@     ��@     ��@     ��@     ��@     ܔ@      7@        
�	
conv2/biases_1*�		    V���    �=�?      P@!  O �?)@*Qf�P?2�-Ա�L�����J�\������=���>	� ��*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�k�1^�sO�IcD���L�a�$��{E��T���C����#@�d�\D�X=���%>��:���%�V6��u�w74���82���bȬ�0���ڋ��vV�R9�})�l a��ߊ4F��        �-���q=��[�?1��a˲?����?f�ʜ�7
?d�\D�X=?���#@?�T���C?a�$��{E?
����G?�qU���I?IcD���L?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?�������:�              �?              �?              �?       @      �?      �?              �?      �?      @              �?              �?      @       @              �?              �?              �?       @              �?              �?              �?              �?              �?              �?              �?              �?              �?               @      @              @      �?      �?      �?      �?       @      �?      �?       @              �?       @      @      �?      �?      �?      �?      �?               @        
�
conv3/weights_1*�	   �wz��   @}Z�?      �@! �\�q@)���jU@2�����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ���I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ��>M|Kվ��~]�[Ӿ����ž�XQ�þ:�AC)8g>ڿ�ɓ�i>;�"�q�>['�?��>�_�T�l�>�iD*L��>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�������:�             `�@     ��@     �@     ��@     ��@     ��@     ��@     ��@     d�@     ��@      �@     �@     @�@     ,�@     H�@     ȋ@     (�@     8�@     (�@     (�@     ؀@     p~@     �z@     py@     �y@     �t@     �s@     `q@     �n@     `j@      i@      l@     �a@     �e@      c@     �b@     �[@     �\@     @Z@      T@     �R@     �V@     �Q@      M@     �L@      L@     �F@      @@     �D@      >@     �B@      6@      4@      1@      4@      0@      ,@      ,@      3@      *@      ,@      $@      ,@       @      @      ,@       @      @      "@       @      @      @      @      �?      @      @      @       @       @       @      �?      @              �?              @      �?              �?      �?      �?               @       @              �?              �?              �?              �?              �?              �?              �?      �?      �?              �?              �?      �?      �?      �?      �?      �?       @      @               @       @              @      @      @       @      @      @      @      @      @      "@      @      @      &@      @      "@      0@      ,@      6@      0@      5@      6@      1@      9@      @@      @@      A@     �I@     �B@     �D@      C@      M@      I@     �O@     �R@      S@     �W@     �X@     �W@      _@     �]@     @c@     �c@      e@     �g@     �l@     @k@      n@     �q@     �q@     Pr@     pw@     x@      @      ~@     �~@     ��@     ��@     p�@     P�@     H�@     �@     ��@     L�@     ��@     ��@     �@     h�@     ��@     H�@     4�@     ��@     J�@     (�@     ȩ@     @�@       @        
�
conv3/biases_1*�	   �����   ����?      `@!  T�?��?)8�I0XW?2�eiS�m��-Ա�L�����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(��S�F !�ji6�9���.���vV�R9��T7���>h�'��f�ʜ�7
���[���FF�G ��iD*L�پ�_�T�l׾K+�E��Ͼ['�?�;        �-���q=>h�'�?x?�x�?��d�r?�vV�R9?��ڋ?U�4@@�$?+A�F�&?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?�������:�              �?              �?      @              �?      �?       @       @      �?              �?      �?      �?              @      @              �?               @      �?       @      �?       @               @       @      �?      �?               @      �?      �?              �?      �?      �?              �?      �?              �?              �?              �?              �?              �?              @              �?      �?              �?              �?              �?               @               @              �?       @              �?      �?              �?      �?              @      @      �?      @       @       @       @      @      @      @      @      @      @      @      �?       @       @      @      �?              �?      �?        
�
conv4/weights_1*�	   � Z��    ��?      �@! jˏB5�)ϸ�re@2�Ӗ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
�1��a˲���[���FF�G ���Zr[v��I��P=��pz�w�7���h���`�8K�ߝ뾞[�=�k�>��~���>a�Ϭ(�>8K�ߝ�>�ߊ4F��>})�l a�>��Zr[v�>O�ʗ��>6�]��?����?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�������:�             �`@     (�@      �@     ��@     D�@     0�@     h�@     x�@     H�@     ؅@     (�@     ��@     �@     �}@     �{@     pw@     0u@      t@     Pt@     �p@     Pp@     �i@     �h@     `g@      g@     �e@      a@      W@      \@      X@     �Q@     �V@      T@     �Q@      O@     �L@     �F@      G@      E@      A@      E@      ?@      =@      ;@      9@      ;@      8@      ;@      9@      ,@      &@      *@      @      ,@       @      @      $@      @      $@      @      @      @       @      @      @      @               @       @      @              �?      �?      �?      �?      �?      �?      �?      �?      �?      �?       @      �?              �?      �?              �?      �?              �?              �?              �?              �?              �?              �?               @      �?       @       @              �?               @       @              �?      �?       @      �?       @      @       @       @      @      @       @      @      @      &@      $@      $@      "@      @      &@      "@      @      2@      &@      ,@      "@      3@      >@      4@      ?@      ;@     �@@     �D@     �B@      H@     �H@     �J@      R@      O@      N@      O@      S@     �S@     �W@     �^@     @[@      a@     �c@      f@     �e@     �h@      j@     �j@     �p@     �s@     @t@     �u@     Pz@     Py@      |@     ��@     ��@     ؄@     �@     ��@     Ј@     ،@      �@     ��@     T�@     �@     (�@     �c@        
�
conv4/biases_1*�	   �ޫ��   `�"�?      p@!J����?)����9[?2��#�h/���7c_XY��-Ա�L�����J�\������=������T}�o��5sz�&b՞
�u�hyO�s��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[��>�?�s���O�ʗ���pz�w�7��})�l a�h���`�8K�ߝ�a�Ϭ(���(��澢f���侮��%ᾙѩ�-߾��>M|Kվ��~]�[Ӿjqs&\�Ѿ['�?�;;�"�qʾG&�$��5�"�g�����|�~���MZ��K���z��6��so쩾4�6NK��2�_"s�$1�7'_��+/��'v�V,����<�)�4��evk'�4�e|�Z#���-�z�!��i
�k���f��p�Łt�=	���R�����J��#���j�Z�TA[�����"�        �-���q==��]���=��1���=�i
�k>%���>��-�z�!>4��evk'>���<�)>�'v�V,>7'_��+/>6NK��2>�so쩾4>�z��6>u 5�9>�`�}6D>��Ő�;F>
�}���>X$�z�>��n����>�u`P+d�>0�6�/n�>�iD*L��>E��a�W�>���%�>�uE����>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?�������:�              �?              �?      �?               @              �?               @       @      @               @       @              @       @              �?              �?       @      �?      @       @      �?       @              �?      @      �?      @      �?              @      �?              �?               @              �?      �?      @       @       @              �?      �?      @      �?      �?              @              �?              �?              @              �?      �?              �?              �?       @              �?              �?              �?              �?      �?      �?      �?      �?              �?              �?              �?      �?       @              �?              �?              8@              �?              �?      �?              �?              �?              �?              �?              �?              �?              �?       @              �?               @              �?      �?              �?      �?              �?              �?               @      �?       @              �?      �?       @              �?              @      �?       @               @               @      �?      �?       @       @      @      �?      �?       @      @      @      @       @      @      @      @       @      @       @      @       @      �?               @       @      �?      @      @      �?      @      @       @      �?      �?       @      �?      @      @              �?              �?              �?        
�
conv5/weights_1*�	   `7�¿   ��=�?      �@! ?i�9#@)
Q��6�I@2�	�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A�d�\D�X=���%>��:���%�V6��u�w74�U�4@@�$��[^:��"��S�F !�ji6�9����ڋ��vV�R9��T7����5�i}1���Zr[v��I��P=���ߊ4F��h���`iD*L�پ�_�T�l׾�MZ��K�>��|�~�>O�ʗ��>>�?�s��>��ڋ?�.�?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�������:�	              k@     `r@     �q@     �m@     �l@     @l@      i@     `c@      a@     �b@     �a@      [@     �Z@     @X@      X@     @R@     �T@     @T@     @Q@     �I@     �E@      A@     �D@      ;@      C@      9@     �A@      5@      <@      3@      3@      (@      &@      2@      1@      @      @      *@      @      @      .@      $@       @      @      @      @      @      @       @      @              @      @      @      @              �?               @               @              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?      �?              �?               @      �?       @      �?              �?      �?      @              �?       @      �?       @      �?      @       @       @       @      �?      @       @      $@      @      @      @      @      @      @      "@      $@      &@      3@      *@      0@      (@      0@      3@      4@      5@      6@      8@     �@@      :@      G@      H@      F@     �C@     �P@     �K@     @P@     �T@      R@      Y@     �Y@     �\@      _@     @`@     `c@      c@     �d@      c@     �i@      m@     �n@     �s@     pt@     �m@      �?        
�
conv5/biases_1*�	   ��/S�   `��C>      <@!  �.�>>)�3S��<2�H��'ϱS��
L�v�Q�p
T~�;�u 5�9��so쩾4�6NK��2��'v�V,����<�)�4��evk'���o�kJ%�4�e|�Z#���-�z�!����"�RT��+���tO����f;H�\Q������%���Bb�!澽5%�����f׽r��=nx6�X� >�J>2!K�R�>��R���>Łt�=	>��f��p>�i
�k>%���>��-�z�!>4�e|�Z#>��o�kJ%>7'_��+/>_"s�$1>�so쩾4>�z��6>p��Dp�@>/�p`B>�`�}6D>�������:�              �?              �?              �?               @      �?              �?      �?              �?              �?      �?              �?               @              �?      �?      �?       @      �?              �?      �?      �?              �?              �?              �?       @        Z}4�XV      �q�	 �ޅ'��A
*ɬ

step   A

loss��=
�
conv1/weights_1*�	   ����   ��P�?     `�@!  �X"�߿)!��lDL@2�	��]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.���vV�R9��T7���
�/eq
�>;�"�q�>['�?��>�h���`�>�ߊ4F��>��Zr[v�>O�ʗ��>��ڋ?�.�?ji6�9�?�S�F !?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�������:�	              $@     �N@      f@     �i@     �e@     �d@     �`@      a@     �^@      [@     @Y@      T@     �S@     �N@      Q@     �I@      P@      N@     �K@      H@      G@     �C@      A@      :@      7@      3@      .@      :@      ,@      4@      0@      4@      &@      &@      $@      &@      ,@      @      @       @      @      @      @       @      @       @      @      @       @       @       @      @       @      �?       @      �?      �?      �?      �?               @      �?              �?      �?      �?              �?              �?      �?              �?              �?               @               @              �?              �?              �?      @       @      �?              �?      @              @      @      �?      @      @      @      @      @      @      @       @      @      @      @      @      (@      .@      .@      &@      (@      5@      (@      5@      8@      9@      @@      <@      7@     �A@      8@      L@     �F@      E@     �O@     �P@     �Q@     @P@     @S@     �S@     �Q@     �X@     @[@     `b@     �a@     �b@     `d@     `f@     �f@     @[@      @        
�
conv1/biases_1*�	    ��_�   ����?      @@!  @Z�=�?)�/�l,/W?2��l�P�`�E��{��^���bB�SY�ܗ�SsW�<DKc��T��lDZrS�IcD���L��qU���I�+A�F�&�U�4@@�$�I�I�)�(?�7Kaa+?IcD���L?k�1^�sO?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?ߤ�(g%k?�N�W�m?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?�������:�              �?              �?              �?              �?              �?              �?              �?              �?               @              �?              �?               @              @      �?       @      @      @       @              �?              @        
�
conv2/weights_1*�	    �Ѭ�   ��\�?      �@! �� O=@)�PhW�@E@2�I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[��>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a�h���`�8K�ߝ�a�Ϭ(��uE���⾮��%�jqs&\�ѾK+�E��Ͼ0�6�/n���u`P+d��f^��`{>�����~>��~]�[�>��>M|K�>�_�T�l�>�ѩ�-�>���%�>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?�������:�              ,@     �@     �@     ��@     p�@     ܙ@     <�@     ��@     p�@      �@     Đ@     (�@     Ћ@     ȇ@     ��@     ��@     ��@     �~@     �~@     �y@     �x@     0w@     @u@     �r@     �o@     �n@     �o@      k@      h@     �f@     �c@     `d@     ``@     @^@      _@      Y@     @U@     �T@     �N@      L@     �L@      E@     �I@      E@      @@     �@@     �I@      =@      B@      7@      <@      4@      9@      2@      7@      &@      (@      &@      $@      "@       @      $@      &@      @      (@       @       @      @      �?      @      @      @      @       @       @      @              �?               @      �?       @              �?              �?      �?              �?              �?              �?              �?              �?      �?              �?              �?               @      �?      �?       @              �?      �?       @      �?       @       @       @      @      �?      @      @              �?      @      @      @      �?      "@      @      &@      @      @      &@      "@      3@      $@      ,@      *@      6@      .@      1@      9@      3@      :@      ;@      ?@      @@      D@     �D@      I@      E@     �J@     �O@     @Q@     @P@     �U@     �T@     �X@     �]@     �`@     �b@     �a@      d@     @g@     �h@     �n@     `q@     @o@     �u@     w@     �u@     �v@     �|@      @     P�@     ��@      �@      �@     ��@     x�@     h�@     Ԑ@     X�@     �@     8�@     ��@     К@     `�@     �@     ��@     $�@      A@        
�	
conv2/biases_1*�		   `����    �~�?      P@!  Ę�?)�̍h/RU?2�eiS�m��-Ա�L�����J�\������=���o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[�nK���LQ�k�1^�sO�
����G�a�$��{E����#@�d�\D�X=���%>��:�uܬ�@8��u�w74���82���bȬ�0�+A�F�&�U�4@@�$�        �-���q=O�ʗ��>>�?�s��>��[�?1��a˲?�T���C?a�$��{E?�qU���I?IcD���L?k�1^�sO?nK���LQ?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�������:�              �?              �?              �?       @      �?      �?              �?       @       @      �?              �?               @      @              �?              �?               @              �?              �?      �?               @              �?              �?              �?               @              �?      �?       @              �?      �?      �?       @              �?      �?      �?      @      �?      �?      �?              @       @       @      �?      �?               @              �?      �?        
�
conv3/weights_1*�	   `_���   �m��?      �@! ���e$@)�A|�lU@2�����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7���h���`�8K�ߝ�a�Ϭ(��uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;����ž�XQ�þ�[�=�k���*��ڽ�G&�$��5�"�g��>G&�$�>�����>
�/eq
�>;�"�q�>['�?��>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�������:�             p�@     j�@     ,�@     ��@     ��@     ��@     n�@     �@     h�@     h�@     �@     �@     $�@     T�@      �@     ��@     �@     8�@     H�@      �@     ��@     �~@     0z@     �y@      z@     �t@     `t@     �p@     �n@     �k@     @i@      j@     �b@      d@      e@     @`@      ]@     �]@     @V@     �Y@     �R@      U@     �Q@      M@      L@      O@     �A@     �A@      ?@     �A@      @@      :@      5@      2@      :@      4@      0@      (@      2@      &@      ,@      @       @      $@      &@      &@       @      $@      @      @      @      @      @      @      @       @      @              �?      @      �?      @      �?              �?              �?      �?               @      �?              �?               @              �?              �?              �?      �?              �?              �?              �?              �?              �?              �?              �?      �?              �?              �?      �?      �?      @       @      �?      @      �?      @      �?       @              @       @      @       @      @      @       @              "@      @      @      @      @      @       @      @      (@      &@      0@      3@      .@      3@      3@      5@      <@      >@      ;@     �D@     �H@      B@      K@      A@     �F@      O@     �I@     @Q@     �T@      X@      X@     @\@     �[@     �[@     �b@     @d@      e@     �g@     `l@      l@     �m@     Pq@     0r@     pr@     �w@     �w@     �~@     ~@     @      �@     ��@     ��@     @�@     ��@     ��@     ��@     d�@     `�@     ��@     Ԙ@     ��@     ��@     ܞ@     \�@     ��@     @�@     :�@     ��@     ��@      @        
�
conv3/biases_1*�	   �A��   ���?      `@!  �����?)%�{�j�^?2�#�+(�ŉ�eiS�m��>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G��T���C��!�A���%>��:�uܬ�@8���%�V6���82���bȬ�0���VlQ.��7Kaa+�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.���vV�R9��T7�����d�r�x?�x��        �-���q=��~]�[�>��>M|K�>x?�x�?��d�r?�vV�R9?��ڋ?+A�F�&?I�I�)�(?�7Kaa+?��82?�u�w74?��%�V6?��%>��:?d�\D�X=?���#@?�!�A?a�$��{E?
����G?�qU���I?IcD���L?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�������:�              �?               @       @              �?      �?      @       @               @              �?       @       @      @      �?      �?       @              �?      �?      @       @      @      �?              �?              �?       @               @      �?      �?              �?              �?              �?              �?              �?              @              �?              �?              �?              �?      �?              �?       @               @      �?       @               @              �?              �?               @       @      �?       @      @       @      �?       @      @      @       @      @      @       @      @      �?      �?       @       @       @      �?              �?      �?        
�
conv4/weights_1*�	   `�h��    L��?      �@! )H�<�4�)�6�se@2�Ӗ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"���ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
�1��a˲���[��O�ʗ�����Zr[v��I��P=��pz�w�7���uE����>�f����>8K�ߝ�>�h���`�>�ߊ4F��>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�������:�             �`@     8�@     �@     ��@     L�@     8�@     `�@     ��@     �@     ��@     �@     ��@      �@     �}@     �{@     0w@     0u@     0t@     `t@     �p@      p@     �i@     @i@     �g@     �f@     �e@      a@      W@     �[@     �X@     @Q@      W@     �S@     �Q@     @P@      K@      E@      I@      E@      @@      C@      =@     �A@      =@      7@      ;@      :@      ;@      5@      2@      &@      &@      $@      &@      @      @      @      @      @      @      &@      @      @      @      @      @       @      �?       @      @      �?      �?               @       @       @       @      �?               @              �?              �?              �?              �?              �?      �?              �?      �?              �?      �?      �?      �?      �?      �?       @              �?      �?              �?      @              �?              �?       @              �?      �?      @      @      @      @      @      @       @      �?      &@      (@      *@      @      @      "@      *@      @      .@      $@      ,@      @      6@      =@      3@      =@     �A@      9@     �E@      C@     �E@      I@     �O@     �P@      P@      N@     �M@     @T@      S@     @W@     �^@     �\@     �`@      d@     �f@     �d@     �h@     �i@     �j@     Pp@     �s@      t@     �u@     `z@     �y@     0|@     ��@     ��@     ��@     �@     �@     ��@     ��@     p�@     ��@     H�@     �@      �@     �d@        
�
conv4/biases_1*�	   `����   ���?      p@!��aBv�?)ڏ�:�a?2����&���#�h/��eiS�m��-Ա�L�����J�\������=���>	� �����T}�*QH�x�&b՞
�u�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C����#@�d�\D�X=���%>��:�uܬ�@8��u�w74���82��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
�6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7���ߊ4F��h���`�8K�ߝ�a�Ϭ(���(����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿ�*��ڽ�G&�$�����]������|�~��p
T~�;�u 5�9��z��6��so쩾4�6NK��2�_"s�$1�7'_��+/��'v�V,����<�)�4��evk'���o�kJ%���-�z�!�%������f��p�Łt�=	��#���j�Z�TA[�����"�RT��+��        �-���q=�9�e��=����%�=%���>��-�z�!>4�e|�Z#>��o�kJ%>�'v�V,>7'_��+/>_"s�$1>�z��6>u 5�9>p
T~�;>��Ő�;F>��8"uH>.��fc��>39W$:��>���?�ګ>����>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�XQ��>�����>['�?��>K+�E���>E��a�W�>�ѩ�-�>�uE����>�f����>��(���>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?#�+(�ŉ?�7c_XY�?���&�?�Rc�ݒ?�������:�              �?              �?      �?              �?      �?              �?               @      @      @       @      �?      �?              �?      @      �?              �?      �?       @              @      @       @       @      �?              @      @       @              @               @              �?               @      �?      �?       @      �?              @              �?       @               @              �?              �?      �?              �?              @      �?      �?       @              �?              �?              �?              �?              �?              �?              �?               @              �?              �?      �?              �?               @               @              �?              �?              7@              �?              �?              �?              �?      �?              �?      �?              �?              �?              �?              �?              �?              �?              �?              �?              �?      �?               @      �?      �?      �?              �?              �?              �?      �?       @      �?              �?       @              �?       @      �?       @              �?      @              @      @      �?       @              @      @      @       @      @      @      @      @      @      �?       @      @      @       @      �?              @      �?              @      @       @       @      @       @       @      �?      �?      �?      @      @              �?              �?              �?        
�
conv5/weights_1*�	    =�¿    �_�?      �@! �M^�$@)��}��I@2�	�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0�U�4@@�$��[^:��"��S�F !�ji6�9����ڋ��vV�R9�>�?�s���O�ʗ���a�Ϭ(���(��澮��%ᾙѩ�-߾��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?�[^:��"?U�4@@�$?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�������:�	              k@      r@     @r@     `m@     `l@     `l@     `i@      c@     `a@     �b@     @a@     �[@      Z@     �X@      X@     �R@     �T@     �S@     �Q@     �H@     �F@      A@      D@      =@      C@      9@      A@      6@      >@      1@      3@      (@      *@      0@      ,@      $@      @      ,@      @      @      (@      "@       @      @      $@      @      @      @      �?       @              @      @       @      @       @              �?              �?              @              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?              @              �?       @              @      �?               @       @               @       @      @      @      @      @      �?      @      @       @      @      @      @      @      @       @       @      @      $@      4@      *@      3@      $@      1@      1@      4@      5@      7@      9@      A@      9@      F@      J@      E@     �C@     �O@      N@      P@      T@     �R@      X@      [@     @[@     @`@      `@     @c@     �b@      e@     �b@      i@     �m@     �n@     ps@     pt@      n@      �?        
�
conv5/biases_1*�	    ��V�   �6
G>      <@!   I�@>),]HC���<2�Fixі�W���x��U�����W_>�p
T~�;��z��6��so쩾4�_"s�$1�7'_��+/��'v�V,����<�)���o�kJ%�4�e|�Z#�RT��+��y�+pm��tO����f;H�\Q������%���9�e�������/�=�Į#��=f;H�\Q�=�tO���=�f׽r��=nx6�X� >�J>2!K�R�>��R���>Łt�=	>��f��p>�i
�k>��-�z�!>4�e|�Z#>��o�kJ%>4��evk'>���<�)>6NK��2>�so쩾4>�z��6>�`�}6D>��Ő�;F>��8"uH>�������:�              �?              �?              �?              �?       @      �?              �?              �?              �?              �?              �?              �?              �?              �?              �?      �?      @              �?              �?      �?              �?      �?              �?       @        >suk�V      �ž2	y$P�'��A*��

step  0A

lossl2>
�
conv1/weights_1*�	   @1��   `ը�?     `�@!  %^>ֿ)����p@2�	�{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��vV�R9��T7����5�i}1���d�r�>�?�s���O�ʗ���E��a�Wܾ�iD*L�پ�_�T�l׾;�"�qʾ
�/eq
Ⱦ���%�>�uE����>pz�w�7�>I��P=�>O�ʗ��>>�?�s��>��d�r?�5�i}1?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�������:�	               @       @     @R@     �d@      i@     @f@     �c@     �a@      a@     @^@     �[@     �W@     �S@     �R@      P@     @P@      N@      P@     �M@     �L@      E@     �F@     �C@     �A@      >@      (@      >@      1@      6@      (@      3@      6@      *@       @      4@      @      $@      *@      @      @      @       @       @      @      @      @      �?       @      @      @      �?      @      �?       @       @       @              �?       @       @              �?      �?       @              �?               @               @              �?      �?              �?              �?              �?              �?              �?              �?              �?      �?              �?      �?              �?      �?       @      �?      �?       @      �?       @      �?       @       @               @      @      @      @      �?      @       @      @      &@       @      @      (@      "@       @      &@      0@       @      0@      0@      2@      0@      4@      <@      A@      ;@      <@      :@     �@@      I@     �G@     �C@      P@     @Q@      P@     �Q@     @S@     �T@     �Q@     @X@     �Z@     �b@      b@      b@      d@     @f@      g@      \@      "@        
�
conv1/biases_1*�	   @z�^�    ��?      @@!  ��}�?)���v9�^?2�E��{��^��m9�H�[���bB�SY�k�1^�sO�IcD���L��T���C��!�A��7Kaa+�I�I�)�(��7Kaa+?��VlQ.?�qU���I?IcD���L?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?�N�W�m?;8�clp?uWy��r?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�������:�              �?      �?              �?              �?              �?              �?              �?              �?              �?              �?              �?      �?              �?      �?              �?       @      �?      @       @       @      �?              �?              @        
�
conv2/weights_1*�	   @�_��   ���?      �@!����@@)/O+�FE@2�I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��uE���⾮��%ᾙѩ�-߾K+�E��Ͼ['�?�;��n�����豪}0ڰ��`�}6D>��Ő�;F>5�"�g��>G&�$�>['�?��>K+�E���>jqs&\��>��~]�[�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?�������:�              3@     ��@     ؠ@     ��@     \�@     �@     �@     ��@     ��@     ؐ@     ��@     �@     x�@     �@     ��@     ؄@     ��@     �~@     �}@     �y@     �w@     �v@     �u@     Pr@     Pq@     �m@     �m@      m@     �h@     `f@     �d@     �a@      b@     �^@     �^@     @W@     �U@     �S@     @Q@     �K@      N@      H@     �B@      F@     �A@     �C@      E@      ?@      @@      6@      ?@      0@      1@      5@      5@      (@      6@      ,@      2@       @      "@      $@       @      @      @      @      @      @      @       @      @      @      @       @      @       @      @       @      �?              @              �?      �?      �?              �?      �?              �?              �?              �?              �?              �?              �?              �?              �?              �?              @      �?       @               @       @      �?              �?      @      @       @       @      @      @       @       @       @       @      @      @       @      @      @       @      "@      "@      @      @      ,@      (@      0@      $@      "@      ,@      1@      .@      ;@      >@      1@      <@      B@     �B@      E@      B@      E@      D@      M@     �O@     @R@     @Q@     �Q@     �T@     �X@      _@     �a@     �a@     �a@     @b@     �i@      h@     @n@     �p@     Pp@     �u@     pw@      v@     �v@     p|@     �~@     ��@     Ё@     ��@     `�@     (�@     �@     H�@     ��@     \�@     D�@     ��@     �@     ��@     8�@     ؞@     ��@     x�@     �E@      �?        
�	
conv2/biases_1*�		    �e��   ����?      P@!  �Zm��?)���LZ?2�#�+(�ŉ�eiS�m��-Ա�L��>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^�nK���LQ�k�1^�sO�IcD���L��qU���I�a�$��{E��T���C����#@�d�\D�X=��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�        �-���q=pz�w�7�>I��P=�>����?f�ʜ�7
?�T���C?a�$��{E?
����G?�qU���I?IcD���L?nK���LQ?�lDZrS?<DKc��T?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�������:�              �?      �?              �?               @      �?      �?              �?       @       @      �?      �?              �?       @       @              �?              �?              �?              �?               @      �?      �?              �?              �?              �?              �?              �?              �?      �?               @      �?               @      �?      �?      �?      �?              @      @      �?      �?              @       @       @      �?      �?               @              �?      �?        
�
conv3/weights_1*�	   ��诿   ��װ?      �@!��-��j*@)1�K�eoU@2�����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��a�Ϭ(���(��澢f�����uE���⾮��%��_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ�u`P+d����n�����豪}0ڰ��������|�~�>���]���>�*��ڽ>�[�=�k�>;�"�q�>['�?��>�iD*L��>E��a�W�>���%�>�uE����>�f����>��(���>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�������:�             ��@     �@     J�@     ��@     ��@     ��@     v�@     ��@     H�@     ��@     �@     ̓@     D�@     T�@     X�@     h�@     �@     �@     0�@     8�@     ��@     �@      z@     �x@     �y@     �u@     0t@     `p@     �n@      l@      i@     `i@     �c@     `b@     �c@     �b@      ^@     �Z@     @U@     �Y@     �R@     �U@     @T@      K@      J@      Q@     �F@      =@      <@      ;@     �A@      2@      :@      1@      4@      7@      &@      (@      1@      .@      2@      .@      $@      &@      @      &@      @      �?      @      @      @      @      @      @      @      @       @       @      @       @      @      �?       @       @      �?      �?      �?              �?              �?      �?               @              �?      �?              �?              �?              �?              �?              �?               @              �?              �?              �?              �?      �?              �?      �?       @      @      @              �?      @      @      @      @      @      @      @      @      @      @      @      @      @      @       @      .@      ,@      5@      4@      1@      .@      5@      0@      7@      <@     �B@      C@      D@      G@      G@     �J@     �K@     �G@     �I@     �P@     @S@      V@      Z@     �\@      [@      ]@     �a@      e@      d@     `g@     �l@      l@     �m@     pr@     Pq@      s@     �w@     pw@      ~@      @     �~@     x�@     ��@     ��@     X�@      �@     ��@     Ԑ@     h�@     t�@     ��@     ��@     ��@     Ĝ@     ��@     n�@     ��@     N�@     8�@     |�@     Ў@      *@        
�
conv3/biases_1*�	    "ዿ   �L}�?      `@!  }����?)�I,��Xc?2��7c_XY��#�+(�ŉ�����=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.�U�4@@�$��[^:��"�ji6�9���.���T7����5�i}1�})�l a��ߊ4F��        �-���q=I��P=�>��Zr[v�>��d�r?�5�i}1?I�I�)�(?�7Kaa+?��VlQ.?�u�w74?��%�V6?uܬ�@8?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?#�+(�ŉ?�7c_XY�?�������:�              �?               @       @              �?      @      �?       @              �?      �?      �?      �?      �?      @      @              @      �?              �?       @      @      @      �?              �?              �?              �?      �?       @              �?       @              �?              �?              �?              �?              @              �?              �?              �?      �?              @      �?              �?       @      �?              �?       @              �?              �?               @      @               @      @      @      �?       @      @      @       @      @      @      �?      @      @              @      �?       @       @               @        
�
conv4/weights_1*�	   ��w��   ����?      �@! z\jV�3�)}��te@2�Ӗ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'�������6�]���1��a˲���[��I��P=��pz�w�7��8K�ߝ�a�Ϭ(�8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>��Zr[v�>O�ʗ��>�FF�G ?��[�?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�������:�             �`@     (�@     �@     |�@     \�@     <�@     0�@     ��@      �@     �@     ��@     ��@     �@     �}@     �{@     �v@     0u@     Pt@     Pt@      q@     �o@     �i@     `i@      g@     `g@     �e@     �`@     @X@     @[@      X@     �R@     �U@      T@      R@      P@      I@     �F@     �I@     �D@     �A@     �C@      6@     �A@     �A@      5@      ?@      6@      :@      8@      ,@      &@      *@      "@       @      *@      @       @      @       @       @      @      @      @      @      @       @      �?      @      �?       @      �?       @      �?      �?      �?      �?              �?      @       @      �?      �?              �?              �?              �?              �?              �?              �?      �?              �?              �?              �?              �?       @              @               @      �?      @               @      �?       @      �?      �?      @      �?      @      @      @      �?      @      @      @      @      (@      *@       @      @      @       @      (@      @      4@      &@      ,@      "@      4@      8@      5@      >@      >@      >@      E@      @@      G@      K@     �M@     �Q@     �O@      N@      N@     �S@     �R@     �W@     �_@     �[@      a@     �c@     �e@     @e@     `i@     @i@     `k@      p@     �s@     �t@     �u@     pz@     py@     0|@     p�@     ��@     �@     8�@     ��@     ��@     ��@     ��@     ��@     `�@     �@      �@     �e@        
�
conv4/biases_1*�	   `mL��   �D�?      p@!�p�I^��?)Z��hVf?2��Rc�ݒ����&��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'�������6�]���1��a˲���[��>�?�s���O�ʗ�����Zr[v��I��P=��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(龢f�����uE���⾮��%ᾄiD*L�پ�_�T�l׾��>M|Kվ�[�=�k���*��ڽ��5�L�����]������Ő�;F��`�}6D�����W_>�p
T~�;�u 5�9��z��6��so쩾4�6NK��2�_"s�$1�7'_��+/��'v�V,����<�)�4��evk'���o�kJ%�4�e|�Z#���-�z�!�%����y�+pm��mm7&c��f׽r����tO����        �-���q=�tO���=�f׽r��=4�e|�Z#>��o�kJ%>4��evk'>7'_��+/>_"s�$1>6NK��2>p
T~�;>����W_>>��8"uH>6��>?�J>39W$:��>R%�����>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>['�?��>K+�E���>�ѩ�-�>���%�>�f����>��(���>a�Ϭ(�>8K�ߝ�>})�l a�>pz�w�7�>O�ʗ��>>�?�s��>�FF�G ?��[�?>h�'�?x?�x�?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?-Ա�L�?eiS�m�?�#�h/�?���&�?�Rc�ݒ?^�S���?�������:�              �?              �?      �?              �?      �?              �?              �?      @      @       @      @              �?              �?      @      �?              �?       @      �?       @      @       @      @       @               @      �?       @       @      �?       @       @              �?              �?              �?       @      �?      �?              �?       @       @      �?      �?              �?      �?              @              �?               @              �?              @       @              �?              �?      �?              �?      �?              �?              �?              �?              �?      �?      �?              �?              �?      �?              �?      �?      �?               @              �?              �?              7@              �?              �?      �?              �?      �?               @              �?              �?              �?              �?      �?               @              �?              �?              �?              �?              �?              �?               @              @              �?      �?      �?       @      �?              �?       @              �?      �?      �?      @      @       @       @      �?               @      @       @      @      @       @      �?       @      @      @              @      �?       @              @      �?              @      @      �?      @      @      @      @              �?      �?       @      @              �?              �?              �?        
�
conv5/weights_1*�	    u�¿   `g��?      �@! �?�$@): F_i�I@2�	�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���bȬ�0���VlQ.�+A�F�&�U�4@@�$��S�F !�ji6�9����ڋ��vV�R9������6�]�����Zr[v��I��P=��;9��R�>���?�ګ>��[�?1��a˲?��ڋ?�.�?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�������:�	             �j@     r@     r@     �m@     `l@     �l@      i@      c@     �a@     �b@     @a@      \@      Z@      X@     @X@     �R@     @T@     @S@     �Q@      I@     �G@     �@@     �C@      @@      A@      9@      B@      8@      ;@      3@      2@      (@      ,@      1@      *@       @      @      *@      @      @      "@      $@      @      @      @      @       @      @      @      �?              @      @      @      �?              @              �?              �?      �?      �?      �?              �?               @              �?              �?              �?              �?              �?              �?              �?              �?       @      @      �?       @       @               @      �?      �?      @      �?      �?      @       @      @      @       @      @      @      @      @      @       @      @      @      @       @      "@       @      &@      2@      &@      6@       @      1@      2@      2@      5@      7@      9@      B@      :@     �E@      I@      F@      D@      P@     �L@     �O@     @T@     �R@     @X@     @[@     @Z@     �`@      `@     �c@     �b@     `e@     �b@      i@      n@     �n@     @s@     �t@      n@      @        
�
conv5/biases_1*�	   @f�Z�   `O�K>      <@!  �*��>>)�t�]�<2���u}��\�4�j�6Z�p��Dp�@�����W_>�u 5�9��z��6��so쩾4�6NK��2�_"s�$1�7'_��+/�4��evk'���o�kJ%�RT��+��y�+pm��tO����f;H�\Q��'j��p���1��콱�.4N�=;3����=����%�=f;H�\Q�=nx6�X� >�`��>�J>2!K�R�>��f��p>�i
�k>%���>4�e|�Z#>��o�kJ%>4��evk'>���<�)>7'_��+/>_"s�$1>�so쩾4>�z��6>��Ő�;F>��8"uH>6��>?�J>������M>�������:�              �?              �?              �?              �?              @              �?              �?              �?              �?              �?              �?              �?              �?               @      @              �?              �?              �?               @              �?      �?      �?        Vf6��U      w8�	�J��'��A*٫

step  @A

lossP�}=
�
conv1/weights_1*�	   �q��   �s �?     `�@!  IJWNۿ)�7Nz�@2�	�{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�U�4@@�$��[^:��"��S�F !�ji6�9���vV�R9��T7����5�i}1���d�r�x?�x��>h�'��1��a˲���[���h���`�8K�ߝ�pz�w�7�>I��P=�>�.�?ji6�9�?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?�������:�	               @       @     @T@     �d@     �i@     �d@     @d@     �a@      a@     �^@     �[@      W@     �S@      R@     @P@     �P@      M@     @Q@     �N@     �N@     �@@      I@      B@      <@      =@      7@      5@      6@      7@      *@      0@      4@      .@      ,@      "@      $@      *@      @      @      @      @      @      @      �?      �?      @      @      @      �?      @       @      @      @       @      �?              @              �?      �?      @              @              �?      �?      �?              �?              �?      �?      �?              �?              �?              �?               @              �?              �?              @              �?      @              �?      �?      �?      �?      �?      @      @      @       @      �?      �?      @      @      @      $@      @      @      @      *@      "@      $@      "@      &@      *@      $@      "@      3@      2@      3@      6@      ;@      A@      =@      8@      :@      C@     �E@      F@      G@      N@     �Q@     �M@     @S@      R@     �T@     @R@      X@     @Z@      c@      a@     @b@      d@      f@     @g@     �\@      $@      �?        
�
conv1/biases_1*�	   ���^�   ��x�?      @@!  ��Ĝ�?)�{]#zc?2�E��{��^��m9�H�[���bB�SY�a�$��{E��T���C����#@�d�\D�X=���bȬ�0���VlQ.��7Kaa+?��VlQ.?a�$��{E?
����G?��bB�SY?�m9�H�[?E��{��^?�l�P�`?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�������:�              �?      �?              �?              �?              �?              �?              �?              �?              �?               @              �?              �?      �?              @              @      �?      @      �?              �?              �?       @        
�
conv2/weights_1*�	   �F뭿   �xm�?      �@! 	�z�SA@)`��֣KE@2�����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��})�l a��ߊ4F��a�Ϭ(���(����uE���⾮��%�����ž�XQ�þ�*��ڽ�G&�$��G&�$�>�*��ڽ>�ѩ�-�>���%�>�f����>��(���>a�Ϭ(�>8K�ߝ�>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?�������:�              �?      2@     ��@     ��@      �@     <�@     �@     �@     ��@     |�@     �@     Ȑ@     `�@      �@     (�@     `�@     �@     ��@     �~@     �|@     0z@     @w@     �w@     �v@     �r@     `o@     �n@     @l@     `n@      i@     �e@     �d@      c@     `a@      `@     @]@      Z@     @U@      R@      S@     �D@      Q@      H@      E@     �D@     �B@      :@     �G@     �F@      4@      3@      =@      6@      4@      ,@      5@      0@      0@      &@      "@      &@      *@      5@      &@      @      @       @      @      @      @      �?      @      @      @               @      @      @       @      �?       @      �?      �?              �?              �?               @              �?              �?              �?              �?              �?              �?              �?              �?              �?      @              @      @       @      @      @      �?      @       @       @      @       @      @      @      @      "@      @      "@      "@      @      $@      (@      .@      0@      "@      ,@      5@      ,@      5@      ;@      3@      F@      :@      A@     �B@      C@      I@      I@      I@     �O@      N@     �R@     �Q@     @V@     �X@     �\@     �a@     @b@     �a@     �b@     @i@     �g@     @m@     pq@     @p@     u@     �w@     pu@     �w@     �{@     `@     ȁ@     ��@     �@     ��@     `�@     `�@     �@     t�@     ��@     ��@     �@     �@     ��@     4�@     �@     |�@     ��@     �O@       @        
�	
conv2/biases_1*�		   @oԋ�   �#�?      P@!  ��'�?),�0�Z�_?2��7c_XY��#�+(�ŉ�eiS�m������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I��!�A����#@�d�\D�X=��u�w74���82���VlQ.��7Kaa+�I�I�)�(�+A�F�&�        �-���q=�iD*L��>E��a�W�>6�]��?����?
����G?�qU���I?IcD���L?k�1^�sO?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?�������:�              �?      �?              �?               @      �?      �?              �?       @              @      �?              �?       @      �?      �?              �?              �?      �?              �?      �?              �?              �?      �?      �?              �?              �?              �?              �?      �?      �?               @              �?              �?       @              �?       @               @      @       @      �?              �?       @      �?      @              �?               @              �?      �?        
�
conv3/weights_1*�	    ���   �F�?      �@!@����D0@)����ArU@2�����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�_�T�l׾��>M|Kվ��~]�[Ӿ['�?�;;�"�qʾ;9��R���5�L�����]������|�~����n����>�u`P+d�>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>�_�T�l�>�iD*L��>��(���>a�Ϭ(�>8K�ߝ�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�������:�             ��@     �@     V�@     ��@     ��@     ��@     p�@     ܜ@     x�@     ��@     ̕@     ��@     4�@     ��@     ��@     ��@     ��@     �@     8�@     ��@     ��@     @~@     �{@     �x@     �x@     �v@      t@     �p@      n@     �m@     `g@     �h@     �d@     �c@     �b@     @b@     @]@     �Z@     �W@     �U@      V@      S@      O@     @S@      G@      S@     �F@      B@      ?@      ;@      @@      9@      8@      .@      0@      1@      *@      &@      1@      &@      1@      *@      .@       @      "@      $@      "@      @      @      "@      @      @      @      @      @      @              @              @       @               @      �?      �?              �?              �?               @              �?      �?      �?              �?      �?              �?              �?              �?              �?              �?              �?              �?              �?              �?      �?              �?      �?       @      �?      �?       @      �?      @      �?      @       @      @      �?      @      @      @      @      @       @      @      @      @      @      @      @      @      7@      1@      1@      1@      2@      7@      8@      6@      3@      3@      >@      D@     �E@     �F@     �L@      F@      J@     �M@      M@     �P@     �S@      W@     �W@     @Z@     �Z@     @`@     �a@     �c@     `d@     @g@     �k@     @l@     �m@     �q@     pr@     Ps@     Pw@     �w@     �|@     �@      @     Ѐ@     �@     X�@     �@     ��@     ȍ@     ��@     �@     ��@     ؕ@     p�@     �@     ؜@     ��@     t�@     v�@     f�@     "�@     ��@     ��@      0@        
�
conv3/biases_1*�	   �	z��   `�ӎ?      `@!  ��j��?)�@�u=�g?2��#�h/���7c_XY�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0�+A�F�&�U�4@@�$�ji6�9���.����ڋ�        �-���q=I��P=�>��Zr[v�>>h�'�?x?�x�?��d�r?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?�7c_XY�?�#�h/�?�������:�              �?              �?      @              �?      @      �?       @              �?      �?      �?      �?      �?      @       @      �?       @       @               @              @       @      �?              �?      �?      �?              �?       @      �?              �?               @              �?      �?              @              �?              �?      �?              �?      �?              �?      �?              �?              �?              �?      �?      �?       @              �?               @              �?              �?              @              @       @      @      �?      @       @      @      �?      @      @      �?      @      @      �?       @       @       @       @               @        
�
conv4/weights_1*�	    :���   ``��?      �@! �ޣ3�)���te@2�Ӗ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'����[���FF�G �I��P=��pz�w�7���uE���⾮��%��_�T�l׾��>M|Kվ:�AC)8g>ڿ�ɓ�i>��>M|K�>�_�T�l�>a�Ϭ(�>8K�ߝ�>})�l a�>pz�w�7�>��Zr[v�>O�ʗ��>6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�������:�             �`@     �@     �@     p�@     |�@     ,�@     0�@     ��@     �@     (�@     ؂@     ��@     �@     �}@     |@     �v@     �u@     `t@     �s@      q@     0p@     �i@     �i@      g@     @g@      e@     �`@     �X@     @\@      W@     �S@     @U@     @S@      S@     �N@      J@     �F@      I@     �D@      A@      E@      6@      >@     �A@      8@      >@      5@      ;@      8@      1@      ,@      $@      "@      @      &@      @      &@      @      @      @      @      @      @       @      @      @      �?       @      �?      @       @       @      �?               @      �?              �?      �?       @              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?       @      �?       @      �?              @              �?       @       @       @      @      �?       @       @      @       @      @      �?      @      @       @      @      $@      (@      &@      @      @      "@      "@      @      2@      (@      0@      ,@      .@      :@      7@      :@      ?@      ;@      E@     �A@      C@      K@     �N@     �R@     @P@     �M@      N@     @S@     �R@     �W@     �^@     �\@      a@     �c@     �e@      e@      i@      j@      k@     �o@     0t@     `t@     `u@     �z@     �y@     �{@     ��@     ��@     ��@     P�@     ��@     @�@     x�@     ��@     ��@     l�@     �@     ��@     `f@        
�
conv4/biases_1*�	   �� ��   �i��?      p@!u3^Ch�?)�]}jtk?2�^�S�����Rc�ݒ��7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f���侄iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿ��~��¾�[�=�k���5�L�����]����p��Dp�@�����W_>�p
T~�;�u 5�9��z��6��so쩾4�_"s�$1�7'_��+/��'v�V,����<�)�4��evk'���o�kJ%�4�e|�Z#���-�z�!�y�+pm��mm7&c���1���=��]���        �-���q=�f׽r��=nx6�X� >��o�kJ%>4��evk'>6NK��2>�so쩾4>����W_>>p��Dp�@>/�p`B>6��>?�J>������M>R%�����>�u��gr�>�[�=�k�>��~���>�XQ��>jqs&\��>��~]�[�>���%�>�uE����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>pz�w�7�>I��P=�>�FF�G ?��[�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?eiS�m�?#�+(�ŉ?���&�?�Rc�ݒ?^�S���?�"�uԖ?�������:�              �?              �?      �?      �?              �?              �?               @       @      @      �?       @      �?      �?      �?               @       @              �?       @      @              @      �?      @      �?      �?              �?      �?       @       @      @              @               @               @      �?      �?      �?               @      �?       @              �?              �?      �?      �?              �?      �?      �?      �?              �?              �?              �?      �?       @      �?      �?              �?      �?              �?              �?              �?              �?              �?      �?      �?              �?               @               @      �?              �?      �?              �?              �?              7@              �?               @               @              �?      �?              �?              �?              �?      �?              �?              �?               @      �?       @              �?              �?              �?               @              @       @               @              �?              @               @      �?       @      �?      �?      @      �?      @              @       @      @      @      @      @      @       @      @      @      �?      @      �?       @      �?      @              @       @      @      @      @      @      @              �?      �?       @      @              �?              �?              �?        
�
conv5/weights_1*�	   �ÿ    ��?      �@!  ��%@)p!�I@2�	�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�uܬ�@8���%�V6��u�w74���bȬ�0���VlQ.�+A�F�&�U�4@@�$��S�F !�ji6�9����ڋ��vV�R9���d�r�x?�x�����%�>�uE����>�h���`�>�ߊ4F��>����?f�ʜ�7
?��ڋ?�.�?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�������:�	             �j@     r@      r@     �m@     �l@     �l@      i@     �b@     `a@      c@      a@     �[@     �Z@     �W@     �W@     �S@     �R@     �S@     @S@      H@     �F@     �A@      C@     �@@     �A@      9@      A@      8@      <@      1@      5@      &@      *@      2@      ,@       @      @      &@       @      @       @      &@       @      @      @      @       @      @      @       @      �?      @       @      @      @              �?              �?      �?               @       @              �?              �?               @              �?              �?              �?              �?              �?              �?               @      �?      @      �?       @      �?      @              �?      �?      �?       @       @              @       @      @       @      @       @      @      @      @      @      @      @      @      @       @       @       @      *@      *@      0@      3@      $@      ,@      4@      2@      6@      8@      8@     �B@      ;@     �D@     �H@     �F@      E@      O@     �L@      N@      U@      S@     �W@     @[@     �Z@     �`@      `@      c@     �b@     �e@     �b@     �h@     �n@     @n@     0s@     �t@     �n@      @        
�
conv5/biases_1*�	   �^�   `�cP>      <@!   �[C>)���S���<2�d�V�_���u}��\�/�p`B�p��Dp�@�p
T~�;�u 5�9��z��6��so쩾4�6NK��2�_"s�$1����<�)�4��evk'�y�+pm��mm7&c��f׽r����tO�����Qu�R"�PæҭUݽ��
"
�=���X>�=�9�e��=����%�=�`��>�mm7&c>�#���j>�J>�i
�k>%���>��-�z�!>��o�kJ%>4��evk'>���<�)>�'v�V,>�so쩾4>�z��6>u 5�9>��8"uH>6��>?�J>������M>28���FP>�
L�v�Q>�������:�              �?              �?              �?              �?      �?       @              �?              �?              �?              �?              �?              �?              �?              �?              @      �?              �?              �?              �?       @              �?      �?              �?        �U]$(V      �+	��*�'��A*��

step  PA

loss�Ѡ=
�
conv1/weights_1*�	   @�ò�    9Z�?     `�@!  �+�tҿ)�9!s��@2�	�{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !���d�r�x?�x��>h�'��f�ʜ�7
������>�?�s���O�ʗ�����Zr[v��I��P=��X$�z�>.��fc��>��(���>a�Ϭ(�>�ߊ4F��>})�l a�>1��a˲?6�]��?>h�'�?x?�x�?�T7��?�vV�R9?�.�?ji6�9�?�S�F !?�[^:��"?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?�������:�	              @      &@     @U@     �c@      j@      e@     �c@     �a@     �`@     �^@     @]@      T@     �T@      Q@     �P@     �R@      L@     @Q@     �N@      N@     �B@     �E@      ?@      A@      <@      >@      1@      2@      9@      0@      4@      ,@      0@      &@      $@      @      $@      $@      @      @      @      @      @      @       @      @       @      @      @      @      �?      @       @      �?       @      @               @      �?              �?       @              �?       @      �?      �?               @      �?              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?      �?      �?              �?       @      @              �?       @      @       @      �?       @      @      �?      @      @      @      @      @       @      @      *@      @      (@      "@      @       @      &@      (@      &@      0@      4@      3@      9@      :@      =@      ?@      =@      :@      C@     �D@     �D@     �H@     @Q@     �L@     �O@     �S@     �P@      U@     �R@     �W@     @[@     �a@     �a@     @b@     `d@     �d@     `g@     �^@      (@       @        
�
conv1/biases_1*�	    ��^�   ���?      @@!  �����?)\v.ԏGh?2�E��{��^��m9�H�[���bB�SY�uܬ�@8���%�V6���82���bȬ�0��7Kaa+?��VlQ.?�T���C?a�$��{E?E��{��^?�l�P�`?���%��b?5Ucv0ed?ߤ�(g%k?�N�W�m?hyO�s?&b՞
�u?*QH�x?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?���&�?�Rc�ݒ?^�S���?�"�uԖ?�������:�              �?      �?              �?               @              �?              �?              �?              �?              @              �?      �?              @              @      �?       @       @              �?      �?       @        
�
conv2/weights_1*�	   �^z��   ����?      �@!���\�B@)7�<j�QE@2�����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F����(��澢f�����uE���⾮��%�jqs&\�ѾK+�E��Ͼ['�?�;
�/eq
Ⱦ����ž�XQ�þ��~��¾5�"�g���0�6�/n���4[_>������m!#��39W$:��>R%�����>jqs&\��>��~]�[�>a�Ϭ(�>8K�ߝ�>�h���`�>})�l a�>pz�w�7�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?�������:�               @      9@      �@     ��@     �@     (�@     4�@     P�@     ��@     ,�@     <�@     Đ@     8�@     ȋ@     ��@     (�@     8�@     ��@      @     �|@     `z@     �v@     �w@     �v@     s@     �o@     @o@      m@      l@      k@     �c@     @e@     �b@     �a@      `@     �^@     �[@     �T@     �T@     @Q@      I@     �K@     �G@     �D@      F@     �A@      7@     �I@     �@@      @@      ;@      7@      .@      0@      0@      .@      2@      2@      ,@      $@      "@      .@      &@       @      @      *@      @      ,@      @      @       @      @      @      @      �?      @      @      @       @      @      �?      �?      �?              �?      @      �?               @              �?              �?      �?               @              �?              �?              �?              �?              �?              �?      �?               @               @              �?               @      �?      @      @      @      @      �?      @      @      @      @      @      @      $@      @      @      @      @      @      1@      $@      0@      @      0@      3@      0@      0@      :@      8@      D@      B@      @@     �C@      E@      H@      H@     �P@      J@     �K@     @S@     �T@     �T@     @W@     �[@     ``@     �b@     @b@     `c@      i@     �g@     �l@     @q@     �p@     �t@     �w@     @u@      x@     �{@     �@     H�@     �@     �@     8�@     ��@     8�@     x�@     0�@     ��@     d�@     �@     ��@     ��@     8�@     0�@     b�@     �@      S@      @        
�	
conv2/biases_1*�		   @;7��   @��?      P@!  vU0ñ?)�
פj�b?2��#�h/���7c_XY��#�+(�ŉ����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`��lDZrS�nK���LQ�k�1^�sO�IcD���L�a�$��{E��T���C��!�A��u�w74���82�I�I�)�(�+A�F�&�U�4@@�$��[^:��"�        �-���q=})�l a�>pz�w�7�>6�]��?����?�qU���I?IcD���L?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�������:�              �?      �?              �?               @      �?      �?              �?      �?       @      �?       @              �?       @      �?      �?              �?      �?      �?              �?      �?              �?               @              �?              �?              �?              �?               @              �?              �?      �?      �?      �?      �?      �?              �?       @               @      @       @      �?      �?      �?      �?      �?      @      �?               @              �?      �?        
�
conv3/weights_1*�	   �4��   �WW�?      �@! �|�/W3@)"�IuU@2�����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%�
�/eq
Ⱦ����ž�[�=�k�>��~���>;�"�q�>['�?��>E��a�W�>�ѩ�-�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�������:�             ��@     ��@     .�@     ��@     ��@     ġ@     Z�@     $�@     d�@     �@     �@     h�@     X�@     ��@     ��@     ��@     ��@     ��@     ��@     X�@     ��@     0@     p{@     px@     0y@     �u@     `t@     @q@     �o@     @l@     �g@     �h@     �c@     �d@     `d@      a@     @[@     @Y@     �X@     �T@      V@      U@     @R@      J@     �M@     �K@      F@     �E@     �@@      C@      C@      7@      4@      3@      3@      *@      ,@      .@      0@      1@      &@      1@      $@      @      (@      @      &@      @      $@      @      @      @      @      �?      @      @              @      @      @      @      �?       @       @               @      �?      �?              �?       @              �?      �?              �?              �?              �?              �?              �?      �?               @              �?              �?      �?      @      @      �?      @      @       @              �?       @      @      @      @      @      @       @      @      @      @      @      @       @      @      @      ,@      4@      2@      0@      1@      4@      3@      1@      8@      4@      @@     �@@     �F@      G@     �K@      D@     �K@     @P@      P@     @Q@     �Q@      W@     @[@     �Y@     �W@     @a@      b@     `b@     �c@     @g@     �l@     �j@     �o@     0q@     �r@     �s@     �w@     �w@     p|@     @@     �@     ��@     �@     8�@     ��@     8�@     `�@     �@     �@     P�@      �@     L�@     $�@     ��@     ̞@     j�@     z�@     l�@     �@     Z�@     ��@      7@        
�
conv3/biases_1*�	   �ކ��   `kk�?      `@!  X����?)�1HE��l?2����&���#�h/��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82�+A�F�&�U�4@@�$�ji6�9���.����ڋ�        �-���q=�_�T�l�>�iD*L��>��[�?1��a˲?>h�'�?x?�x�?��d�r?�5�i}1?�.�?ji6�9�?�7Kaa+?��VlQ.?��bȬ�0?uܬ�@8?��%>��:?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�#�h/�?���&�?�Rc�ݒ?�������:�              �?              �?       @      �?      �?      @      �?       @      �?               @      �?              @      �?       @       @      �?       @      �?      �?      @      @      �?              �?      �?      �?              �?              �?       @              �?              �?              �?      �?              @              �?              �?              �?              �?              �?              �?      �?               @              �?      �?      �?      �?      �?      �?      �?              �?      �?       @              �?      @              @      @       @       @      @      @       @       @      @      @      �?      @      @      �?       @      @      �?       @              �?      �?        
�
conv4/weights_1*�	   @����   �z�?      �@!  *M`W2�)��.V�ue@2�Ӗ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"�ji6�9���.����ڋ��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
���[���FF�G �>�?�s���I��P=��pz�w�7���iD*L�پ�_�T�l׾��(���>a�Ϭ(�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�������:�             �`@      �@     �@     x�@     l�@     8�@     �@     ��@     8�@     �@     ��@     ��@     �@     �}@     �{@     �v@     �u@     Pt@     t@     q@     0p@     �i@     �h@     �g@     �g@      e@     �`@     @X@     �\@      W@     @S@     @U@      S@     �R@      O@      I@     �F@     �J@     �B@     �C@      E@      6@      <@      ?@     �@@      9@      5@      :@      <@      .@      "@      (@      &@      $@       @      @      "@       @      $@      @      @      @      @      @      @      @      �?      @       @      @      @      �?               @      �?              @              �?      �?      �?              �?      �?              �?              �?              �?              �?              �?              �?              �?      �?      �?      �?      @      �?      �?              @      �?      �?       @      �?      �?      �?      @      �?      @      @       @      �?      @      @      @      @      @      @      &@      "@      "@      *@      @      @      @      $@      0@      *@      2@      .@      ,@      <@      2@      >@      :@     �@@      A@      D@     �@@     �M@      M@     @R@     @Q@     �P@      L@     �Q@     �R@     @W@      `@     �Z@     �a@     `c@      f@      e@     `h@     �j@     �j@     `o@     0t@     pt@     �u@     �z@      y@     �{@     Ѐ@     ��@     ؄@     8�@     ��@     �@     ��@     `�@     ��@     t�@     ��@     ��@      g@        
�
conv4/biases_1*�	    ᰕ�   ��ޘ?      p@!�#~%�?)v���p�p?2��"�uԖ�^�S�����#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� ��*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�>h�'��f�ʜ�7
��������[���FF�G �>�?�s���O�ʗ�����Zr[v��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(���E��a�Wܾ�iD*L�پ�XQ�þ��~��¾;9��R���5�L��/�p`B�p��Dp�@�����W_>�p
T~�;�u 5�9��z��6�6NK��2�_"s�$1�7'_��+/��'v�V,����<�)�4��evk'���o�kJ%�4�e|�Z#�y�+pm��mm7&c��b1�ĽK?�\��½        �-���q=nx6�X� >�`��>��o�kJ%>4��evk'>���<�)>6NK��2>�so쩾4>�z��6>p��Dp�@>/�p`B>�`�}6D>��Ő�;F>������M>28���FP>�u��gr�>�MZ��K�>��~���>�XQ��>�����>
�/eq
�>��>M|K�>�_�T�l�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>>�?�s��>�FF�G ?1��a˲?6�]��?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?#�+(�ŉ?�7c_XY�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?�������:�              �?              �?      �?      �?              �?              �?               @       @      @      �?       @               @      �?      �?      �?       @               @       @       @      �?      @              @               @               @       @       @      @      �?       @              �?       @      �?      �?      �?               @              �?      �?      �?      �?       @              �?      �?              �?      @              �?      �?              �?              @              @      �?              �?              �?              �?              �?               @              �?              �?              �?       @      �?      �?              �?      �?              �?              �?              7@              �?              �?      �?              �?      �?              �?              �?              �?              �?              �?              �?              �?              �?              �?      �?              �?              �?              �?              �?              �?              �?      �?      �?               @      �?      @      �?              �?      �?      �?               @      @              �?       @      @      @      �?       @              @      @      @      @      @      @      @      �?       @      @       @      @               @      �?      @              @       @      @       @      @      @      @      �?      �?      �?      @      @              �?              �?              �?        
�
conv5/weights_1*�	   �#ÿ    X��?      �@! ��#��&@)�p}��I@2�	�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���VlQ.��7Kaa+�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9����ڋ��vV�R9��T7����h���`�8K�ߝ��ߊ4F��>})�l a�>>h�'�?x?�x�?��ڋ?�.�?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�������:�	             �j@     r@     r@     �m@     `l@     �l@      i@     �b@      a@     �c@     �`@      \@      [@     �V@     �X@     �S@      R@      T@     �R@      G@     �G@      C@     �A@     �A@      A@      :@      @@      <@      :@      2@      3@      *@      (@      0@      ,@      "@      @      &@       @      @       @      (@      @      @      @      @      �?      @       @       @               @      @      @       @      �?              �?      �?              �?      �?      �?              �?               @      �?      �?      �?              �?      �?              �?              �?              �?              �?               @       @      @      �?      �?              �?              �?       @       @       @       @       @      �?       @      @      @              @       @      @      @      @      @      @       @      @      @      "@      @      "@      (@      0@      ,@      4@       @      *@      6@      0@      6@      9@      6@      D@      >@     �C@      H@      F@      F@     @P@      J@      O@      U@     �R@     �W@     @[@     �Z@     �`@      `@     �b@     �b@     �e@     �b@     @h@     �n@     @n@     0s@     `t@     �n@      @        
�
conv5/biases_1*�	   @_�`�   `H�R>      <@!  ����D>)����s��<2�w&���qa�d�V�_��`�}6D�/�p`B�����W_>�p
T~�;�u 5�9��z��6��so쩾4�6NK��2��'v�V,����<�)��mm7&c��`����f׽r����tO����_�H�}��������嚽�Qu�R"�=i@4[��=�K���=�9�e��=�mm7&c>y�+pm>�#���j>�J>�i
�k>%���>��-�z�!>4�e|�Z#>��o�kJ%>4��evk'>���<�)>�'v�V,>�so쩾4>�z��6>u 5�9>p
T~�;>����W_>>6��>?�J>������M>28���FP>�
L�v�Q>H��'ϱS>�������:�              �?              �?              �?              �?      �?       @              �?              �?              �?              �?              �?              �?              �?              �?              �?      @              �?              �?      �?              �?              �?      �?              �?      �?              �?        A.㜈V      ��%	��'��A*��

step  `A

loss�z�=
�
conv1/weights_1*�	   �`��   �_��?     `�@! �����)+�}�A�@2�	�{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�+A�F�&�U�4@@�$�ji6�9���.��x?�x��>h�'��f�ʜ�7
��FF�G �>�?�s����iD*L�پ�_�T�l׾���%�>�uE����>�FF�G ?��[�?1��a˲?6�]��?����?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?�������:�	              @      1@     �S@     @d@     `j@     �d@     �c@     �a@     `a@      ]@     @[@     �V@     @S@     �R@      O@      S@     �M@     @P@     �M@      L@     �F@     �E@      ?@      <@      A@      8@      6@      0@      5@      6@      4@      (@      &@      (@      (@      @      @      "@      @      @      @      @      "@      @       @       @      @      @       @      @      @       @       @      @      �?      �?              �?       @       @       @      @               @              �?              �?      �?              �?              �?              �?              �?      �?              �?               @              �?              �?              �?       @       @       @               @       @              @              @       @      @       @      @      @       @       @      �?      @       @      "@      @       @       @      *@      &@       @       @      *@      (@      $@      7@      .@      3@      7@      5@     �A@      ?@      ?@      6@      D@      F@      B@      J@      O@     �L@     @P@     @S@     �R@     �T@     @R@     @W@     �\@     �a@     �`@     @c@     �c@     �d@      f@     @`@      0@       @        
�
conv1/biases_1*�	   @��]�    ]+�?      @@!  ��:e�?)B��7m?2�E��{��^��m9�H�[��u�w74���82�ji6�9���.���5�i}1���d�r��7Kaa+?��VlQ.?d�\D�X=?���#@?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?�������:�               @              �?              �?              �?              �?              �?              �?              �?               @      �?              �?              �?              @              @               @       @              �?      �?       @        
�
conv2/weights_1*�	   �k��   @���?      �@! Pm���C@)j��9�WE@2�����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��8K�ߝ�a�Ϭ(���(��澢f���侮��%ᾙѩ�-߾E��a�Wܾ�_�T�l׾��>M|Kվ�u��gr��R%�������u`P+d�>0�6�/n�>
�/eq
�>;�"�q�>['�?��>K+�E���>�iD*L��>E��a�W�>�ѩ�-�>���%�>�ߊ4F��>})�l a�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?�������:�              @      =@     Ȍ@     ��@     ��@     P�@     �@     ��@      �@     p�@     @�@     ��@     8�@     �@      �@     H�@     ��@     ��@     @@     }@     0z@     w@      w@     �v@     �r@     Pp@     �n@     @m@     �l@     �h@     �e@     `e@     �b@     @_@      b@     �Z@      _@     @V@     �U@      J@      L@      N@      H@     �E@     �E@      C@      A@      E@      5@     �@@      ;@      4@      4@      8@      (@      4@      &@      0@      &@       @      *@      (@      ,@      (@      "@      "@      @      @       @      @      @      �?      @      @      �?              @       @      �?               @       @      @       @      �?              @              �?              �?              �?      �?              �?              �?              �?              �?              �?               @              �?              �?              �?      @       @      @       @       @      @      @      @      @      @      @      @       @      @      @      (@      @      @      @      @      @      @      &@      (@       @      6@      "@      3@      .@      ,@      3@      8@      3@      D@     �D@      C@      >@      H@     �I@      E@      K@     �P@     �M@      Q@     �S@      X@     �U@     �Z@     �_@     @c@     �b@     @c@     �i@     @g@     �k@     �p@     �q@     �s@     x@     �t@     0y@     �{@     `�@     ��@     `�@     (�@     �@     ȉ@     Љ@     (�@     ��@     ̑@     X�@     ��@      �@     p�@     p�@     $�@     Z�@     D�@      X@      @        
�	
conv2/biases_1*�		   ��J��    ��?      P@!  ���^�?)�����f?2����&���#�h/���7c_XY��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`��lDZrS�nK���LQ�k�1^�sO��qU���I�
����G�a�$��{E���%�V6��u�w74�I�I�)�(�+A�F�&�U�4@@�$��[^:��"���d�r�x?�x��K+�E��Ͼ['�?�;�FF�G ?��[�?
����G?�qU���I?IcD���L?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?�������:�              �?      �?              �?               @               @              �?      �?       @               @      �?      �?               @      �?      �?               @      �?              �?      �?              �?              �?              �?              �?              �?               @              �?      �?              �?              �?      �?              @              �?       @      �?              �?      @       @       @              �?      �?       @      @      �?               @      �?              �?        
�
conv3/weights_1*�	   `pS��    ���?      �@!�IɆ�E6@)>a�toxU@2���]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲��FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ뾢f�����uE���⾮��%ᾄiD*L�پ�_�T�l׾��>M|Kվ��~]�[ӾK+�E��Ͼ['�?�;
�/eq
Ⱦ����žd�V�_>w&���qa>�����>
�/eq
�>;�"�q�>['�?��>jqs&\��>��~]�[�>�_�T�l�>�iD*L��>���%�>�uE����>�f����>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�������:�              �?     (�@     ��@     �@     ��@     ��@     ֡@     H�@     �@     |�@     �@     �@     d�@     x�@     ��@     ��@     `�@     Ј@     ��@     ��@     �@     Ѐ@     �~@     �{@     �x@     py@     �u@     `s@     Pq@     �o@     �l@     @i@     �h@     �b@     @c@     �d@      d@      Z@     @X@     �V@     �U@     �T@     @U@     @Q@      O@     �N@     �H@      G@      D@     �A@      <@      @@      @@      3@      7@      :@      *@      1@      *@      1@      0@      0@      ,@      @      @      @      ,@       @      @      @       @      @      @      @      @      �?      @      �?       @               @              �?      �?      @               @      �?       @              �?      �?              �?      �?      �?              �?              �?              �?              �?              �?              �?              �?              �?      �?              �?      �?      @       @              �?      �?      @       @      �?       @       @      @      @      @      @      @      @       @      @      @      @      @      &@      "@      @      (@      .@      ,@      0@      *@      3@      9@       @      1@      2@      <@      @@      A@      F@     �C@     �K@      F@     �N@     �K@     �P@     �S@      R@     �U@     �X@     @^@     �W@     @`@     `b@     `a@     �d@     �g@      k@     �j@     @o@     �q@     �r@     `s@     0x@     �w@     �{@     @     �@     ��@     ��@     x�@     ��@     Њ@     Ȍ@     @�@     �@     T�@     $�@     h�@     �@     ��@     ��@     z�@     j�@     b�@     �@     b�@     ��@      ?@        
�
conv3/biases_1*�	   `t͑�    _`�?      `@!  �s���?)��/q?2��Rc�ݒ����&��eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�IcD���L��qU���I�a�$��{E��T���C����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82�I�I�)�(�+A�F�&��S�F !�ji6�9���.����ڋ�1��a˲���[��        �-���q=>�?�s��>�FF�G ?f�ʜ�7
?>h�'�?�T7��?�vV�R9?�[^:��"?U�4@@�$?I�I�)�(?�7Kaa+?��bȬ�0?��82?uܬ�@8?��%>��:?���#@?�!�A?�T���C?a�$��{E?
����G?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�������:�              �?              �?       @      �?      �?      @      �?       @      �?               @      �?              @      @      @              �?       @      @              @      �?      @              �?               @              �?       @              �?              �?              �?              �?              �?              �?              @              �?              �?              �?              �?              �?              �?              �?              �?      �?              @               @              �?      �?       @               @       @      �?       @      @       @      @       @      @       @      @      @       @      �?      @       @       @      �?      @      �?      �?      �?              �?      �?        
�
conv4/weights_1*�	   ����   �) �?      �@! �	ʕ�1�)��Rj�ve@2�Ӗ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
�6�]���1��a˲�I��P=��pz�w�7��8K�ߝ�a�Ϭ(��u`P+d����n��������%�>�uE����>�f����>��(���>I��P=�>��Zr[v�>O�ʗ��>6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�������:�             �`@     �@     �@     l�@     `�@     `�@     ،@     ��@     P�@     Ѕ@      �@     ��@     �@     0}@     �|@     �v@     �u@     `t@     �s@     @q@     p@      j@     �h@     @g@     `g@     �d@     �`@      Y@     �\@      X@     �R@     �T@     �R@     �R@      O@      H@      F@     �K@      D@     �B@      F@      7@      9@     �@@      A@      7@      6@      <@      7@      2@      (@      "@      "@      "@       @      @      *@      @      @      @       @      @      @      �?      @      @      �?      @      @      @      @      @       @               @              �?      �?      �?      �?               @      �?              �?              �?              �?              �?              �?              �?              �?      �?              �?      �?      �?      @      �?              @              �?              @      �?       @       @       @      @              @      @      @      �?      @      @      @      @      (@      &@      "@      @       @       @      "@      &@      2@      $@      2@      .@      .@      >@      2@      9@      <@      =@      D@      B@     �@@     �K@     �N@     @R@     @P@     �Q@      N@     �R@     @S@     �U@     ``@     �Z@     `a@      d@      e@     �e@      h@     �j@     `k@     @o@      t@     �t@     �u@     pz@     Py@     �{@     ��@     ȁ@     Ȅ@      �@     Ї@     �@     ��@     p�@     ��@     ��@     ��@     ��@     `g@        
�
conv4/biases_1*�	   `�\��   @V#�?      p@!��f߫�?)���s?2�}Y�4j���"�uԖ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6���82���bȬ�0���VlQ.��7Kaa+�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1�>h�'��f�ʜ�7
���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�a�Ϭ(���(��澙ѩ�-߾E��a�Wܾ�XQ�þ��~��¾���?�ګ�;9��R���`�}6D�/�p`B�p��Dp�@�����W_>�p
T~�;�u 5�9��so쩾4�6NK��2�_"s�$1�7'_��+/��'v�V,����<�)�4��evk'��mm7&c��`���        �-���q==��]���=��1���=�mm7&c>y�+pm>4��evk'>���<�)>�so쩾4>�z��6>u 5�9>/�p`B>�`�}6D>��8"uH>6��>?�J>28���FP>�
L�v�Q>�MZ��K�>��|�~�>�XQ��>�����>
�/eq
�>;�"�q�>�iD*L��>E��a�W�>�uE����>�f����>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>I��P=�>��Zr[v�>O�ʗ��>6�]��?����?f�ʜ�7
?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?�#�h/�?���&�?^�S���?�"�uԖ?}Y�4j�?��<�A��?�������:�              �?              �?      �?      �?              �?              �?              @       @      @      �?      �?      �?       @      �?               @      �?      �?      @       @      �?      �?       @      �?      @       @      �?               @       @       @      @               @              �?              @      �?       @              �?               @      �?      �?      �?      �?      �?       @              @               @              �?      �?               @      �?      �?       @              �?              �?              �?              �?               @              �?              �?               @      �?               @      �?      �?              �?              7@              �?              �?               @              �?      �?              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?      �?              �?      �?              �?      �?              �?      �?              �?              �?      @               @       @               @      �?       @               @       @              @       @       @      �?              @      @      @      @      @      @      @      �?      @      @      �?      @      �?      @              @              �?      @       @      @       @      @      @       @       @              �?      @      @              �?              �?              �?        
�
conv5/weights_1*�	   ` 6ÿ    ��?      �@! ��NY'@),^j'_�I@2�
yD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0�+A�F�&�U�4@@�$��[^:��"��S�F !��.����ڋ��vV�R9��5�i}1���d�r�})�l a�>pz�w�7�>��Zr[v�>O�ʗ��>��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?�S�F !?�[^:��"?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�������:�
              �?     @j@      r@     r@     �m@     �l@      m@     �h@     �b@     `a@      c@     �`@     @\@     @[@      W@     @X@      S@     @R@     �T@      R@      G@      G@      D@      A@      B@      @@      ;@     �@@      9@      :@      2@      4@      .@      *@      ,@      ,@       @      @      $@      $@      @      $@      &@      @      @      @      @       @      @      @      @              @      @      �?      �?              @              �?      �?      �?              @              �?              �?               @              �?      �?              �?              �?              �?              �?              �?              �?              �?               @      �?       @              �?               @      �?              �?      �?      �?      @       @              @       @      @      �?      @       @      @       @      @      @      @      @      @      @      $@       @       @      $@      2@      *@      6@      @      &@      7@      1@      6@      9@      7@      C@      >@     �D@      G@     �E@     �G@      P@     �I@     @P@     �T@     �R@     �W@      [@     �Z@     �`@     �_@     �b@     @c@     �e@     �b@     @h@     �n@     �n@     �r@     @t@     �o@      @        
�
conv5/biases_1*�	   �_c�    �9U>      <@!  �5�F>)��� �A�<2������0c�w&���qa���Ő�;F��`�}6D�p��Dp�@�����W_>�p
T~�;�u 5�9��z��6��so쩾4�7'_��+/��'v�V,��`���nx6�X� ��f׽r���H�����=PæҭU�=ݟ��uy�=�/�4��=��-��J�=�K���=y�+pm>RT��+�>Z�TA[�>�#���j>%���>��-�z�!>4�e|�Z#>��o�kJ%>4��evk'>���<�)>�'v�V,>7'_��+/>�z��6>u 5�9>p
T~�;>����W_>>p��Dp�@>/�p`B>������M>28���FP>�
L�v�Q>H��'ϱS>��x��U>�������:�              �?              �?              �?              �?      �?       @              �?              �?      �?              �?              �?              �?              �?              �?               @       @              �?      �?              �?              �?              �?              �?              �?      �?              �?        ��eHX      |�]�	�`��'��A*��

step  pA

loss���=
�
conv1/weights_1*�	   �����   `��?     `�@! �>�K���)&݋b�
@2�
�{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !��.����ڋ��vV�R9��T7�����d�r�x?�x��>h�'��f�ʜ�7
�6�]���1��a˲�a�Ϭ(���(�����(���>a�Ϭ(�>��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?�������:�
              @      7@     �S@     �d@      j@      e@      c@     `b@     @a@      ^@     �X@      V@      S@      Q@      R@     �R@     �M@     �P@     �Q@     �H@     �C@     �F@      ?@      ;@     �A@      7@      6@      1@      0@      3@      3@      1@      &@      "@      ,@      "@      @      @      @      @      @      @      "@      @       @      @      @      @      �?      @       @      @      @               @      �?              �?              �?              @              �?               @               @       @      �?              �?              �?              �?              �?              �?              �?              �?      �?      �?               @              �?              �?      �?      �?              �?      �?      �?      �?      �?       @      @      @      �?       @      @              �?      @       @      �?              @       @      @      @      &@      @      @      $@      @      &@      @      "@      *@      0@      0@      1@      *@      6@      .@      ?@      =@      A@      ?@      6@     �B@     �D@      E@     �L@      K@     �M@      O@      T@     �P@     @U@     �Q@     @Y@      \@     �`@      a@     @c@      d@     �c@     @f@      `@      2@      @        
�
conv1/biases_1*�	   ��3]�   �#�?      @@!  �~��?)�e�q?2�E��{��^��m9�H�[���bB�SY�uܬ�@8���%�V6��S�F !?�[^:��"?I�I�)�(?�7Kaa+?��VlQ.?��%�V6?uܬ�@8?5Ucv0ed?Tw��Nof?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?^�S���?�"�uԖ?}Y�4j�?��<�A��?�������:�              �?      �?              �?              �?              �?      �?              �?               @               @              �?      �?      �?              @      �?      @               @       @              �?      �?       @        
�
conv2/weights_1*�	    p���   �:	�?      �@!�|ס�E@)f���^E@2�����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ���(��澢f�����uE���⾮��%ᾙѩ�-߾���]������|�~��
�/eq
�>;�"�q�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>�f����>��(���>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?�������:�              @      @@     X�@     ��@     `�@     ��@     ��@     |�@      �@     ��@     <�@     p�@     H�@     �@     ؇@     ��@     X�@     �@     �@     p|@     �z@      w@      w@     �u@     �r@     �q@     `m@     �k@     `l@      j@      e@     @e@      c@      `@      a@     �]@     �\@     �S@     �S@     �T@      L@     �G@      I@      E@      H@      D@     �@@     �G@      7@      6@      ;@      6@      ;@      (@      0@      5@      ,@      9@      &@      *@      @              (@      *@      .@      @      @      @      @      @      @      @      �?      @      @      �?      �?       @      @      �?      �?       @      �?       @              @      �?      �?      �?              �?              �?      �?              �?               @              �?              �?              �?               @      �?      �?      �?              �?      �?      @       @              �?       @      �?       @      @      @      �?      @      @      @      @      @      @      @      @      @      @      @      0@      @      "@      $@      &@      9@      $@      0@      4@      8@      <@      <@      @@     �@@      D@      E@     �E@     �H@     �B@      M@     �L@      N@     @T@     @U@     @S@     �X@     @Y@     �^@     �b@     �c@      b@     �j@     @h@     `k@      o@     �r@     �r@     �x@     �u@     �x@     0|@     P�@     8�@     ��@     `�@     ��@     p�@     ȉ@     h�@     ��@     ԑ@     H�@     ��@     ̘@     ��@     d�@     $�@     D�@     ��@     �]@      @        
�	
conv2/biases_1*�		   ��x��   `Hj�?      P@!  WE(��?)�Q��i?2��Rc�ݒ����&���#�h/��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I���%�V6��u�w74���VlQ.��7Kaa+��[^:��"��S�F !�.��fc��>39W$:��>�uE����>�f����>x?�x�?��d�r?�5�i}1?�T7��?a�$��{E?
����G?�qU���I?IcD���L?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�������:�              �?      �?              �?       @               @              �?              �?       @       @      �?      �?               @      �?      �?              �?       @              �?      �?              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?      �?       @      �?              �?       @      �?              �?      @       @       @      �?              �?       @      @      �?               @      �?              �?        
�
conv3/weights_1*�	   �Mr��   `[Ա?      �@!����G.9@)��k+�{U@2���]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(����uE���⾮��%�E��a�Wܾ�iD*L�پjqs&\�ѾK+�E��Ͼ
�/eq
Ⱦ����žG&�$��5�"�g������]������|�~��T�L<�>��z!�?�>
�}���>X$�z�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�iD*L��>E��a�W�>�ѩ�-�>�f����>��(���>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�������:�              @     h�@     F�@      �@     ��@     x�@     ȡ@     J�@     @�@     ��@     Ȗ@      �@     4�@     \�@     T�@     8�@     ��@     8�@     ��@     ��@     ��@     ��@      @     {@     y@     �y@     @u@     u@      n@     �p@      l@     �j@     �h@     �b@      c@     �d@      d@     �[@     @W@     @V@     @V@     @U@      Q@     �R@     �N@     @P@     �M@     �C@      A@      >@      B@     �B@      :@      ;@      6@      7@      2@      $@      (@      2@       @      $@      .@      $@      (@      @      "@      @      @      @      @      @      @      �?      @      �?       @      @       @      @      �?      @       @               @              �?               @      �?              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?       @      �?              �?       @              �?              �?              �?      �?       @       @      �?               @      @               @      @      @       @      @      @      @      �?      @      @       @      $@      @      @      @      &@      .@      &@      (@      .@      1@      3@      7@      ,@      2@      3@      2@      8@     �B@      K@     �F@      J@     �H@      J@      P@      M@     �P@     @V@      U@     �X@     �]@     @Y@     ``@     �`@     �b@     `c@     @h@      k@     �j@     �n@     �r@     @r@     s@     `x@     0x@      |@     @     p@     ؀@     ��@     `�@     p�@     �@      �@     ̐@     T�@     P�@     �@     t�@     �@     Ȝ@     Ȟ@     ��@     Z�@     d�@     �@     ,�@     h�@      F@        
�
conv3/biases_1*�	   ����   �xZ�?      `@!  �B��?)�R�t %t?2�^�S�����Rc�ݒ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=�uܬ�@8���%�V6��u�w74��7Kaa+�I�I�)�(��[^:��"��S�F !��.����ڋ��vV�R9�        �-���q=>�?�s��>�FF�G ?f�ʜ�7
?>h�'�?�T7��?�vV�R9?U�4@@�$?+A�F�&?I�I�)�(?��82?�u�w74?uܬ�@8?��%>��:?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?�������:�              �?               @       @      �?      �?      @      �?       @               @      �?               @      @       @      �?      �?       @       @      �?      @      �?       @      �?              �?              �?              �?      �?      �?      �?              �?      �?              �?              �?              �?      �?              @              �?              �?              �?              �?      �?              �?              �?              �?      �?      �?      �?      �?      �?              �?      �?       @               @               @       @       @      �?      @              @      �?      @      �?      @      @       @      �?      @      @       @      �?       @      �?      �?      �?              �?      �?        
�
conv4/weights_1*�	   �����   ��:�?      �@! �����0�)���(�we@2�Ӗ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
���Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��E��a�Wܾ�iD*L�پjqs&\��>��~]�[�>�ѩ�-�>���%�>�uE����>�f����>8K�ߝ�>�h���`�>I��P=�>��Zr[v�>O�ʗ��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�������:�              a@     �@      �@     l�@     X�@     \�@     ��@     ��@     @�@     ��@     �@     ��@     �@     `}@     `|@     �v@     �u@     pt@     �s@     `q@     `o@     @j@     @i@     `g@     `g@     �d@     �`@     @Z@     @[@     @X@     �S@      T@     @R@     �S@     �K@     �I@     �E@     �K@     �D@     �B@     �G@      6@      <@      A@      <@      <@      6@      8@      :@      ,@      &@      $@      $@      @      (@      @      &@      @       @      @      @       @      @      @      @      �?      @      @      @      @      �?      @      �?              �?               @              @       @      �?      �?      �?              �?      �?              �?              �?              �?              �?              �?              �?              �?      �?              �?      �?               @              �?      @      �?              �?      �?       @              �?      @              �?      �?      �?       @              @      @       @      @      @      @      @      @      $@      $@      "@      $@      $@       @      $@      @      2@      &@      3@      .@      1@      =@      5@      8@      :@      >@     �D@     �@@      >@      L@      P@     �Q@      Q@     �P@     @Q@     �P@     �S@     �U@     �`@     �Z@     `a@      d@     @d@     @f@     `g@      k@     @k@     �o@     �s@     Pt@     �u@     �z@     y@     |@     ��@     Ё@     ��@     �@     ��@     �@     ��@     p�@     ��@     `�@     �@     ��@     `g@        
�
conv4/biases_1*�	   ���   �b�?      p@![qA/?�?)�w3SK�v?2�}Y�4j���"�uԖ��Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\�����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(龮��%ᾙѩ�-߾����ž�XQ�þ��������?�ګ���Ő�;F��`�}6D�/�p`B�p��Dp�@�p
T~�;�u 5�9��z��6��so쩾4�6NK��2�_"s�$1�7'_��+/��'v�V,����<�)��mm7&c��`���        �-���q=f;H�\Q�=�tO���=RT��+�>���">4��evk'>���<�)>�'v�V,>�so쩾4>�z��6>p
T~�;>����W_>>/�p`B>�`�}6D>6��>?�J>������M>�
L�v�Q>H��'ϱS>��|�~�>���]���>;9��R�>���?�ګ>�����>
�/eq
�>;�"�q�>['�?��>�ѩ�-�>���%�>�f����>��(���>8K�ߝ�>�h���`�>})�l a�>pz�w�7�>��Zr[v�>O�ʗ��>>�?�s��>����?f�ʜ�7
?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�#�h/�?���&�?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�������:�              �?              �?               @              �?              �?              @       @       @       @      �?      �?      �?       @               @      �?      �?       @      @       @              @       @      @      �?      �?               @              @      @      �?       @              �?               @      �?       @      �?              �?               @      �?      �?              �?               @       @              �?      �?              �?              �?       @      �?               @       @              �?      �?              �?              �?              �?              �?               @              �?              �?              �?       @               @      �?      �?              �?              7@              �?              �?              �?      �?              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?      �?              �?               @      �?              �?               @       @              �?       @      �?              @      �?              �?      �?              @      @      �?       @       @       @       @      @       @      @      @      @       @      �?      @      @      �?      @      �?      @      �?       @              �?      @       @      @       @      @      @       @       @               @      @       @              �?              �?              �?        
�
conv5/weights_1*�	   @�Iÿ   �oD�?      �@! x�}�(@)����I@2�
yD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�a�$��{E��T���C����#@�d�\D�X=���%>��:���%�V6��u�w74���82�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9����ڋ��vV�R9��5�i}1���d�r�6�]���1��a˲�O�ʗ��>>�?�s��>�FF�G ?��[�?6�]��?����?��d�r?�5�i}1?��ڋ?�.�?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�������:�
              �?     @j@      r@     �q@      n@      l@     `m@     �h@     �b@     �a@     �b@     �`@     �\@     �Z@      W@     @X@     �R@      R@     @U@     @Q@     �H@     �F@      C@     �B@     �@@     �A@      8@     �B@      7@      :@      3@      5@      ,@      *@      ,@      $@      (@      @      (@      "@      @      "@      "@      @      @      @      @      �?      @       @      �?      @      @      @       @      �?               @              �?      �?              �?       @              �?      �?      �?       @              �?              �?              �?              �?              �?              �?              �?              �?              �?      �?              �?               @              �?              �?              �?      �?      @       @       @       @      @      �?      @              @      @       @      @      @       @      @      @       @      @      *@       @       @      $@      1@      ,@      6@      @      (@      1@      5@      7@      8@      6@      C@      A@     �C@      G@      F@      G@     �O@      J@     @P@      T@      S@      X@     @Z@     �Z@      a@     �_@     �b@      c@     �e@     @c@      h@     �n@      o@     �r@     0t@     p@      @        
�
conv5/biases_1*�	    J�d�   @�zW>      <@!   &�H>)�ߵ�m��<2�cR�k�e������0c���8"uH���Ő�;F�/�p`B�p��Dp�@�����W_>�p
T~�;�u 5�9��z��6��so쩾4�_"s�$1�7'_��+/�nx6�X� ��f׽r���z�����=ݟ��uy�==��]���=��1���='j��p�=��-��J�=RT��+�>���">%���>��-�z�!>4�e|�Z#>��o�kJ%>4��evk'>���<�)>�'v�V,>7'_��+/>u 5�9>p
T~�;>����W_>>�`�}6D>��Ő�;F>������M>28���FP>�
L�v�Q>H��'ϱS>��x��U>Fixі�W>�������:�              �?              �?              �?              �?      �?      �?      �?              �?               @              �?              �?              �?               @              �?       @      �?              �?      �?      �?              �?      �?              �?              �?              �?              �?        ȋ:]�V      ���	\c�'��A*��

step  �A

loss�/�=
�
conv1/weights_1*�	   `Q���   @w��?     `�@! �ү���)v�H�B4@2�	� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.���vV�R9��T7����5�i}1���[���FF�G �E��a�W�>�ѩ�-�>>�?�s��>�FF�G ?1��a˲?6�]��?�5�i}1?�T7��?ji6�9�?�S�F !?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?�������:�	               @      @     �@@     @T@     �d@      i@     �f@     �a@     �b@     �a@     @]@     @X@      V@     @S@      S@     @R@      R@     �K@      O@      R@      H@     �F@     �E@      =@     �@@      9@      7@      3@      8@      2@      0@      $@      4@      0@       @      *@      @      $@      @      @      @      @      $@      @      @       @      @       @      @       @      @      @      @      @       @      �?      @      @      @       @      �?      �?      @       @       @       @      �?      �?       @      �?               @      �?              �?              �?              �?              �?              �?              �?               @      �?      �?      @       @      �?      �?      �?              �?       @      �?      @       @       @       @      @      @      @      @      @      �?      @      "@      $@      $@      @       @      @      $@      5@      *@      (@      ,@      .@      5@      5@      2@      A@     �B@      <@      9@      C@      E@      H@     �D@      N@     �L@     �P@     @R@     �R@     �S@     �Q@     @Y@     �^@      `@     @b@     �a@     �c@     �d@     @e@     @_@      7@      @        
�
conv1/biases_1*�	   ���^�   �x��?      @@!  ̀��?)U%$<r�s?2�E��{��^��m9�H�[���bB�SY�uܬ�@8���%�V6��7Kaa+?��VlQ.?��bȬ�0?�u�w74?��%�V6?a�$��{E?
����G?5Ucv0ed?Tw��Nof?P}���h?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�������:�              �?      �?              �?              �?      �?              �?              �?              �?      �?               @               @              �?              �?       @      @      @      �?      @              �?               @      �?        
�
conv2/weights_1*�	   @J!��   �Q�?      �@! 8��oKF@)�7�h�dE@2�����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=���ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾��>M|Kվ��~]�[Ӿ���]������|�~��E��a�W�>�ѩ�-�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�������:�              @     �A@     ��@     L�@     X�@     ��@     ��@     ��@     �@     ��@     8�@     l�@     ��@     ��@     ��@     H�@     �@     8�@     0@     �|@     @z@      w@     �v@     �u@     @s@     �q@     �m@     `j@      n@     `h@      f@     �d@      c@     �_@     �a@     �[@      ^@     @S@     @T@     @V@     �J@      J@     �F@      K@      B@      B@      @@     �F@      ?@      3@      <@      8@      ;@      5@      &@      0@      .@      .@      &@      &@      $@      @      ,@      &@      "@      @      @      "@      @      @       @       @       @      @      @       @              @      @       @              @      �?      �?              �?              �?               @               @       @              �?              �?              �?               @       @      �?      �?      �?      @      @      @      @      @      �?      @       @      @      @      @      @      @      @      @      "@      @      &@      "@       @      &@      "@      8@      $@      2@      0@      0@      2@      2@      A@     �A@     �D@      D@     �@@      E@     �J@     �D@     �N@      I@      O@      R@     @U@      V@      Y@     �X@      _@     `b@     �c@     `d@      i@     �g@     `k@     �m@     �r@      s@     `x@     pv@     px@     `{@     Ȁ@     0�@     ��@     H�@     0�@     �@     �@     (�@      �@     ��@     D�@     8�@     ��@     ��@     x�@     �@     P�@     x�@     �a@       @      �?        
�	
conv2/biases_1*�		   �,���   ��?      P@!  �p}��?)�ku�/9m?2��Rc�ݒ����&���#�h/��eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO���%�V6��u�w74���ڋ��vV�R9��XQ�þ��~��¾��>M|K�>�_�T�l�>�vV�R9?��ڋ?+A�F�&?I�I�)�(?�T���C?a�$��{E?�qU���I?IcD���L?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?�7c_XY�?�#�h/�?���&�?^�S���?�"�uԖ?�������:�              �?      �?              �?               @               @      �?              @      �?      �?       @              �?       @              �?              �?       @      �?      �?               @              �?              �?              �?              �?              �?              �?              �?              �?              �?              @              �?              �?       @      �?              �?              @       @      �?      �?      �?       @              @               @      �?              �?        
�
conv3/weights_1*�	   �����    3�?      �@!�&	n�;@)��y�&U@2���]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(龢f�����uE���⾮��%ᾙѩ�-߾['�?�;;�"�qʾ�XQ�þ��~��¾;9��R���5�L��;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�f����>��(���>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?�������:�              @     ��@     �@     �@     ¥@     \�@     ҡ@     J�@     <�@     ��@     Ȗ@     $�@     `�@     @�@     �@     ��@     @�@     h�@      �@     Ȅ@     ��@     x�@     �@     �z@     Py@     �y@     �t@     u@     �o@     �n@      m@     �k@      j@     �`@      c@     �d@     �c@     @\@     �W@      Y@      U@     �T@     @Q@     @T@      O@      N@      N@     �A@     �C@      ?@      :@      B@      <@      3@      7@      1@      6@      (@      3@      0@      *@      &@      $@      *@      @       @      @      @      @      @      @      @      @      @      �?      @      @      �?       @      �?               @      @      �?      �?       @      �?      �?              �?      �?              �?      �?      �?              �?              �?              �?               @              �?      �?              �?              �?              �?               @      �?       @       @      �?       @      @      �?      @       @      @      �?       @       @       @      @       @      "@      @      @      @       @       @      @       @      (@      @      $@      &@      *@      7@      1@      1@      ,@      2@      6@      1@      5@      ?@      A@     �E@     �D@     �L@      H@      N@      P@      L@     �R@     �Q@      Y@     @W@     �\@     @Z@     �`@     @a@     �`@     �c@     �i@     @j@     �i@     @o@     �r@     �r@      s@     �x@     @x@     �{@     `@     �~@      �@     ��@     ��@     `�@     P�@     �@     ��@     h�@     h�@     ��@     ��@     �@     ��@     Ğ@     ��@     D�@     v�@     �@     �@     ̑@      J@      @        
�
conv3/biases_1*�	   ��P��   �JK�?      `@!  4V��?)=r��Vw?2�^�S�����Rc�ݒ�#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=�uܬ�@8���%�V6���VlQ.��7Kaa+�U�4@@�$��[^:��"��S�F !��.����ڋ�        �-���q=��[�?1��a˲?����?f�ʜ�7
?��d�r?�5�i}1?�S�F !?�[^:��"?�7Kaa+?��VlQ.?�u�w74?��%�V6?uܬ�@8?��%>��:?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?�������:�              �?              �?       @       @      �?      @      �?       @              �?       @              �?      @      �?       @      �?      @              �?      @      �?       @       @      �?              �?              �?      �?      �?              �?               @              �?              �?      �?              �?              @              �?              �?              �?              �?              �?              �?              �?              �?              �?       @              �?      �?              �?      �?       @      �?      �?              �?       @      �?      @       @      @       @      @       @       @       @      @      @       @       @      @      @       @      �?       @      �?      �?      �?              �?      �?        
�
conv4/weights_1*�	   @����    T�?      �@! �KW�W0�)rظye@2�Ӗ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1�>h�'��f�ʜ�7
������6�]�����[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7���u`P+d����n�����['�?��>K+�E���>�ѩ�-�>���%�>8K�ߝ�>�h���`�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>��[�?1��a˲?6�]��?����?>h�'�?x?�x�?��d�r?�5�i}1?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�������:�             �a@     Ԙ@     �@     h�@     d�@     T�@     �@     ��@     �@      �@      �@     ��@     p@     �}@     0|@     �v@     �u@     `t@     0s@     pq@     �o@     �i@     �i@     @h@     �f@      d@     ``@     �Z@     @\@     �X@     @S@     �T@     �R@     �R@      L@     �H@      G@     �I@      E@     �B@     �G@      ;@      7@     �@@      >@      ;@      ;@      7@      3@      0@      "@      ,@      "@      "@      (@      @       @      @      "@      @      @      @       @      @      @      @      �?      @      @      @       @              �?              �?      �?      �?               @              �?               @              �?      �?      �?      �?      �?      �?              �?              �?              �?              �?              �?       @      �?              �?              �?               @      @       @              �?      �?      �?              �?       @      @      �?              �?       @       @       @       @      @      @      @      @      $@      *@      "@      @      "@      ,@      "@      @      0@      *@      .@      "@      ;@      :@      6@      9@      9@     �@@     �C@      A@      ?@      L@      L@     �R@     @R@      M@     �Q@      R@     �S@     �T@     �`@      Z@     �a@      d@     @d@     �e@     @h@     �j@     @k@     �o@      t@     0t@     Pv@     �z@     �x@      |@     Ȁ@     ��@     ��@     �@     ��@     �@     p�@     x�@     ��@     h�@     �@     ��@     `h@        
�
conv4/biases_1*�	   �B���    
��?      p@!�by_��?)�[�M�xz?2���<�A���}Y�4j���Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L��>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7�����d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a�8K�ߝ�a�Ϭ(���(��澢f���侮��%ᾙѩ�-߾
�/eq
Ⱦ����ž��������?�ګ���8"uH���Ő�;F�/�p`B�p��Dp�@�����W_>�p
T~�;�u 5�9��z��6��so쩾4�6NK��2�_"s�$1�7'_��+/��mm7&c��`���        �-���q=RT��+�>���">Z�TA[�>���<�)>�'v�V,>7'_��+/>�z��6>u 5�9>����W_>>p��Dp�@>�`�}6D>��Ő�;F>������M>28���FP>�
L�v�Q>H��'ϱS>���?�ګ>����>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>���%�>�uE����>��(���>a�Ϭ(�>�h���`�>�ߊ4F��>pz�w�7�>I��P=�>O�ʗ��>>�?�s��>�FF�G ?��[�?f�ʜ�7
?>h�'�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?���&�?�Rc�ݒ?}Y�4j�?��<�A��?�v��ab�?�/��?�������:�              �?              �?      �?      �?              �?      �?              �?      @      �?      @               @              @              �?      �?      @              @      �?      �?               @      �?      @      �?      �?               @              @      @      �?       @              �?      �?      �?      �?       @      �?              �?       @               @              �?              �?      �?              �?      �?              �?       @      �?      �?              �?      �?              @               @              �?              �?              �?              �?              �?               @              �?              �?      �?              �?      �?      �?      @              �?              7@              �?      �?              �?      �?              �?              �?              �?              �?              �?              �?              �?      �?              �?              �?              �?              �?              �?              �?              �?              �?              �?      �?      �?               @              �?      �?       @              �?       @      �?              @               @              �?              @      @      �?      �?      �?      @      @      @       @      @      @      @              @       @      �?      @      @      @              @              �?              @      @       @      @       @      @      �?       @              @       @       @              �?              �?              �?        
�
conv5/weights_1*�	   `!^ÿ   @`��?      �@! �r���(@)+|�УI@2�	yD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@���%�V6��u�w74���82���bȬ�0��7Kaa+�I�I�)�(�+A�F�&��[^:��"��S�F !���ڋ��vV�R9��5�i}1���d�r�����>豪}0ڰ>1��a˲?6�]��?��ڋ?�.�?ji6�9�?�[^:��"?U�4@@�$?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�������:�	              �?     @j@     r@     �q@     @n@      l@     @m@     �h@     �b@     �a@     �b@     �`@     @\@     @[@      W@     @X@     �R@     @R@      U@     @Q@     �H@      G@     �A@      D@      @@     �A@      8@     �B@      :@      :@      0@      2@      1@      (@      .@      &@      &@       @      "@      $@      @      $@       @      @      @      @      @      @      @      �?       @               @      @      �?       @              �?              �?      @              �?              �?              �?       @               @              �?              �?              �?              �?              �?       @              �?              �?      �?              �?      �?      �?      �?               @              �?      @      �?      �?      @      @      @       @      @       @       @      @      @      @      @      @      @      @      @      "@      &@      $@      1@      *@      5@      "@      &@      2@      4@      8@      6@      8@      C@      A@      C@      G@      G@      G@     �N@     �J@     �O@     @T@     @S@      X@     �Z@     �Z@     �`@      `@     �b@      c@     `e@     @c@      h@     �n@     �n@     �r@     �t@     �o@      &@        
�
conv5/biases_1*�	   @��f�   ���Y>      <@!   �D�J>)��@�̙�<2�:�AC)8g�cR�k�e�6��>?�J���8"uH�/�p`B�p��Dp�@�����W_>�p
T~�;�u 5�9��z��6�_"s�$1�7'_��+/��`���nx6�X� ��f׽r����tO����=��]���=��1���='j��p�=��-��J�=RT��+�>���">Z�TA[�>��-�z�!>4�e|�Z#>��o�kJ%>4��evk'>���<�)>�'v�V,>7'_��+/>_"s�$1>u 5�9>p
T~�;>����W_>>p��Dp�@>��Ő�;F>��8"uH>28���FP>�
L�v�Q>H��'ϱS>��x��U>Fixі�W>4�j�6Z>�������:�              �?              �?              �?               @               @              �?              �?              �?              �?      �?      �?              �?      �?               @      �?      �?               @              �?              �?              �?              �?              �?              �?              �?        	��d�W      �D}	��ˏ'��A*��

step  �A

loss]	L=
�
conv1/weights_1*�	    �i��   ���?     `�@! ��_<�)~�X�*_@2�	� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�ji6�9���.��x?�x��>h�'����~]�[Ӿjqs&\�Ѿ�f����>��(���>�FF�G ?��[�?1��a˲?6�]��?����?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?�[^:��"?U�4@@�$?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?�������:�	               @      @     �D@     @U@     @e@     �g@     �f@     �a@     @c@     �`@     �]@     �W@     @W@      T@     �Q@     �Q@     �R@     �J@     @R@      M@      K@      I@     �B@      @@      @@      6@      6@      5@      :@      .@      ,@      *@      ,@      ,@      (@      $@      .@      $@      @      @      @      @      @      @      @      @      @      @      @      @      @      @      @      �?      �?      @      @       @       @      �?              �?              �?              �?               @              �?              �?              �?              �?      �?              �?      �?      �?      �?      �?              �?              �?       @               @      @       @      �?       @      @      �?       @      �?      �?      �?      @       @       @      �?      @      @      �?      @      @      (@      $@      @       @      $@      @      .@      .@      0@      .@      4@      2@      0@      6@      A@      9@     �C@      >@      D@     �C@      E@      D@      M@      L@      S@      O@      R@      U@     �Q@     �Z@     �^@     @^@     `b@      b@      c@     �d@     �d@      `@      9@      @        
�
conv1/biases_1*�	    �`�   ��W�?      @@!  ��0h�?)��;!v?2��l�P�`�E��{��^�ܗ�SsW�<DKc��T���%>��:�uܬ�@8�U�4@@�$?+A�F�&?��VlQ.?��bȬ�0?�!�A?�T���C?nK���LQ?�lDZrS?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�������:�              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?      �?      �?      �?              �?              �?      @      @      @      �?      @              �?      �?       @        
�
conv2/weights_1*�	    �i��   �l��?      �@!��9uG@)|w�'tkE@2���]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ���(��澢f�����uE���⾮��%�E��a�Wܾ�iD*L�پjqs&\�ѾK+�E��Ͼ['�?�;;�"�qʾ
�/eq
Ⱦ����ž�i����v>E'�/��x>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>�iD*L��>E��a�W�>�uE����>�f����>��(���>a�Ϭ(�>�h���`�>�ߊ4F��>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�������:�              �?      @     �C@     ��@      �@     T�@     ��@     ��@     |�@     X�@     x�@     (�@     ��@     X�@     ȋ@      �@      �@     �@     (�@     �@     �{@      z@      x@     �v@     �s@     `t@     �r@     �l@      l@     �j@     �h@      f@     �e@     `a@     @a@     �a@     �\@     @[@     �U@     �T@     @Q@     �Q@     �K@     �M@     �D@     �D@      ?@      ;@     �H@      ;@      <@      5@      <@      :@      *@      3@      ,@      *@      ,@      .@      (@       @       @      "@      $@      "@      &@      @      @      @      @      @      @      @      @               @               @      �?      @       @      @       @      @              �?              �?      �?              �?              �?              �?              �?              �?              �?              �?              �?              �?               @              �?              �?               @              �?              �?              �?      �?              @       @              @      �?      @      @       @      �?      �?      @      @      @      "@      @      @      "@      @      $@      @      "@      &@      ,@      0@      *@      ,@      5@      ,@      0@      <@      6@      D@      B@      F@      B@     �D@     �F@     �H@      N@      O@     �L@     �R@      R@      W@      Y@     �Y@     �_@     `b@     �a@      e@      k@     �c@     `m@     �m@      r@     �s@     @w@      w@     Py@     P{@     Ѐ@     ��@     ؁@     x�@      �@     @�@     x�@     ��@     �@     ��@     4�@     P�@     ��@     ��@     ��@     ��@     f�@     x�@     �d@      *@      �?        
�	
conv2/biases_1*�		   �rʓ�    ��?      P@!  
��?)��U�p?2�^�S�����Rc�ݒ����&��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ���%>��:�uܬ�@8��u�w74���82�x?�x��>h�'��f�ʜ�7
�������f����>��(���>�S�F !?�[^:��"?�u�w74?��%�V6?�!�A?�T���C?a�$��{E?
����G?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�#�h/�?���&�?�Rc�ݒ?�"�uԖ?}Y�4j�?�������:�              �?      �?              �?               @               @      �?              @      �?      �?       @               @      �?              �?              �?      �?       @      �?              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?               @      �?      �?              �?               @      �?      �?              �?      �?      @      @      �?              �?       @              @               @      �?              �?        
�
conv3/weights_1*�	   �����   @�W�?      �@!@�`Ol�>@)���檂U@2���]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a�a�Ϭ(���(����uE���⾮��%ᾙѩ�-߾�iD*L�پ�_�T�l׾��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;5�"�g���0�6�/n����n�����豪}0ڰ�R%�����>�u��gr�>['�?��>K+�E���>jqs&\��>��~]�[�>�iD*L��>E��a�W�>���%�>�uE����>�f����>��(���>a�Ϭ(�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?�������:�              @     x�@     ��@     �@     ��@     b�@     �@     8�@     8�@     ��@     ��@     (�@     ��@     �@     H�@     �@     ��@     p�@     ��@     ؄@     �@     P�@     0~@     p|@      y@     �y@     Pu@     t@     �p@     �m@     �l@     @l@     �h@      c@      c@     �c@     �a@     �^@     �X@      W@      W@     �V@     @Q@     @S@     @Q@      H@      P@      C@     �C@     �A@      8@      D@      8@      2@      6@      "@      *@      3@      3@      4@      ,@      (@      *@      &@       @      "@      $@      $@      �?      @      @      @      @      @       @      �?       @      �?      �?      �?               @       @      �?              �?       @              �?              �?      �?              �?              �?              �?              �?              �?              �?              �?               @              �?              �?      �?      �?       @              �?       @       @      �?      @       @      @      @              �?      �?      @      @      @      @      @      @      @      @      @      &@      @      @      @      "@       @      $@      3@      0@      *@      5@      4@      ,@      1@      9@      7@      =@      A@      G@      E@     �F@      J@     �L@     @Q@     �L@      R@     �Q@     �U@     @Z@      ]@     @[@     �_@     �b@     �`@     �d@      g@      k@     �j@      n@     pr@     �r@     Ps@     �x@     �w@     �{@     `@     �~@      �@     P�@     ��@     @�@     �@     h�@     ��@     4�@     ��@     �@     \�@      �@     ��@     ��@     ��@     ,�@     ~�@     ,�@     ��@     �@      P@      @        
�
conv3/biases_1*�	    ����   ��5�?      `@!  �aNq�?)��|Bl�z?2��"�uԖ�^�S�����7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�uܬ�@8���%�V6���VlQ.��7Kaa+�I�I�)�(�+A�F�&��.����ڋ�        �-���q=��[�?1��a˲?6�]��?��d�r?�5�i}1?ji6�9�?�S�F !?�7Kaa+?��VlQ.?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?�qU���I?IcD���L?k�1^�sO?nK���LQ?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?�"�uԖ?}Y�4j�?��<�A��?�������:�              �?              �?      �?      @      �?       @       @       @              �?      �?      �?              @      @       @      �?      @              �?      @       @      �?       @       @              �?               @              �?              �?               @              �?      �?      �?              �?              @              �?      �?              �?              �?              �?              �?      �?               @              �?               @              �?              �?       @               @      �?              �?       @       @       @      @       @      @      @       @      �?      @      @       @       @      @       @      @      �?      �?       @      �?       @              �?      �?        
�
conv4/weights_1*�	   @����   ��m�?      �@! �vNf/�)����ze@2�Ӗ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.���vV�R9��T7����5�i}1�>h�'��f�ʜ�7
������1��a˲���[��I��P=��pz�w�7��;�"�q�>['�?��>�iD*L��>E��a�W�>�uE����>�f����>�h���`�>�ߊ4F��>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�������:�             �a@     ̘@     �@     T�@     l�@     L�@     �@     ��@     @�@     �@     �@     @�@     �@     �}@     0|@     w@      v@     0t@      s@     �q@      p@     @i@      j@      i@     �e@      d@      `@      [@     �\@     @X@      T@     �T@     �R@      R@      M@      H@     �G@     �H@     �B@      E@     �F@      9@      =@      ?@      ?@      <@      7@      =@      ,@      0@      "@      *@      (@      "@      *@      @      @      @      &@      @      (@      @      �?      @      @      @              @       @      @      �?      �?      �?      �?      �?              �?       @              �?      �?              �?              �?              �?              �?              �?              �?              �?       @       @               @              �?       @      @      �?              �?               @      �?       @              @       @      @              @       @      @      @      @      @      @      @      "@      *@      "@      "@      *@      @      "@       @      0@      (@      .@      &@      0@      ?@      8@      8@      :@      A@      C@      A@     �B@     �G@     �L@      S@     �R@     �J@      S@     �Q@      T@     @U@      `@     @Z@      b@      d@     @d@      f@     �g@     �j@     �j@      p@     �s@     �t@     `v@     Pz@     `x@     @|@     Ȁ@     ��@     ��@     �@     �@     ��@     ��@     p�@     ��@     d�@     (�@     Ж@     �h@        
�
conv4/biases_1*�	    HF��   �)�?      p@!�j���u�?)PZ1��;~?2��v��ab����<�A���^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=�uܬ�@8���%�V6��u�w74���82���bȬ�0��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������1��a˲���[���FF�G �O�ʗ�����Zr[v��I��P=��pz�w�7���h���`�8K�ߝ�a�Ϭ(��uE���⾮��%��_�T�l׾��>M|Kվ
�/eq
Ⱦ����ž豪}0ڰ������6��>?�J���8"uH��`�}6D�/�p`B�p��Dp�@�����W_>�p
T~�;��z��6��so쩾4�6NK��2�_"s�$1�7'_��+/��`���nx6�X� �        �-���q=Z�TA[�>�#���j>�J>�'v�V,>7'_��+/>�z��6>u 5�9>p��Dp�@>/�p`B>�`�}6D>��Ő�;F>�
L�v�Q>H��'ϱS>��x��U>;�"�q�>['�?��>K+�E���>jqs&\��>�uE����>�f����>��(���>a�Ϭ(�>�h���`�>�ߊ4F��>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>��[�?1��a˲?6�]��?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?�Rc�ݒ?^�S���?��<�A��?�v��ab�?�/��?�uS��a�?�������:�              �?              �?      �?      �?              �?      �?              �?       @      @      @               @              �?       @      �?      �?       @      �?      @      �?      �?              �?      �?      @       @      �?      �?       @              �?      @      �?       @      �?       @               @       @              �?              �?      �?      �?      �?      �?              �?      �?      �?               @              �?       @      �?      �?              �?      �?              �?       @      �?              �?      �?              �?              �?              �?              �?               @              �?              �?      �?              �?       @       @      �?              �?              7@              �?      �?               @              �?              �?              �?              �?      �?              �?               @              �?              �?              �?              �?              �?              �?      �?              �?              �?              �?      �?       @               @      �?               @               @              �?      �?      �?       @      �?      �?              @      @       @              �?      @      @      �?      @      �?       @      @      @              @      �?      �?      @      @       @              @              �?       @       @      @       @      @      @       @      �?      �?              @       @      �?              �?              �?              �?        
�
conv5/weights_1*�	   �Prÿ    ޻�?      �@! ػ}�k)@)Ŗ?낧I@2�	yD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�a�$��{E��T���C����#@�d�\D�X=���%>��:���%�V6��u�w74���82���bȬ�0���VlQ.�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9�����?f�ʜ�7
?��d�r?�5�i}1?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?�7Kaa+?��VlQ.?��82?�u�w74?��%�V6?uܬ�@8?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�������:�	              �?     `j@     �q@     �q@     �n@      l@     `m@     �h@     �b@     `a@      c@      a@     �[@     �[@     �V@     @X@     @S@     �Q@     @T@      R@     �G@      F@      C@     �C@     �@@     �A@      9@      B@      :@      <@      .@      4@      *@      ,@      ,@      (@      *@      @       @      "@      @      &@      "@      @      @      @      @       @      @      @      �?      �?      @      @       @       @              �?              �?      �?               @              �?      �?              @      �?              �?              �?              �?              �?              �?              �?              �?      �?              @              �?              �?              �?       @      �?      @       @       @       @      @      @       @      @              @      @      @       @      @      @      @      @      @       @      &@      (@      1@      (@      1@      $@      *@      3@      3@      6@      6@      6@     �E@     �@@     �D@      E@      H@      F@     �O@     �J@     �N@     @T@     �R@     �X@      Z@      [@     �`@      `@     `b@     @c@      e@     �c@     �g@     �n@     @n@     pr@     �t@     @o@      *@        
�
conv5/biases_1*�	   �*�h�   �7T\>      <@!  �)SN>)��G
E�<2�ڿ�ɓ�i�:�AC)8g�������M�6��>?�J��`�}6D�/�p`B�p��Dp�@�����W_>�p
T~�;�u 5�9�6NK��2�_"s�$1��`���nx6�X� ��tO����f;H�\Q���/�4��==��]���=��-��J�=�K���=�9�e��=����%�=y�+pm>RT��+�>���">Z�TA[�>��-�z�!>4�e|�Z#>��o�kJ%>4��evk'>���<�)>�'v�V,>7'_��+/>_"s�$1>u 5�9>p
T~�;>p��Dp�@>/�p`B>��8"uH>6��>?�J>�
L�v�Q>H��'ϱS>��x��U>Fixі�W>4�j�6Z>��u}��\>�������:�              �?              �?              �?               @               @              �?              �?              �?              �?              �?              �?              �?              �?               @      �?              �?               @      �?              �?              �?              �?              �?              �?              �?        �?ڄ�V      ���	��<�'��A*��

step  �A

loss�]�=
�
conv1/weights_1*�	   @�ϴ�   `�?�?     `�@!  �H|�)�,� V�@2�	� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"���ڋ��vV�R9�x?�x��>h�'��f�ʜ�7
�O�ʗ�����Zr[v�����%�>�uE����>6�]��?����?f�ʜ�7
?>h�'�?x?�x�?�5�i}1?�T7��?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?�������:�	               @      "@      I@     �T@     @f@     �f@      f@     �b@     �d@     �]@     �^@     �V@     �V@      U@     �Q@      R@     �R@     �O@      N@      M@     �L@     �F@      B@     �@@      >@      7@      6@      7@      9@      1@      &@      .@      (@      *@      .@      "@      .@      (@      "@      @      @      @      @      @      @      @      @      @      �?      �?      @      @      @              �?      �?       @      �?       @              �?       @      �?              �?               @               @               @      �?              �?              �?              �?      �?              �?              �?               @      �?               @              �?      �?              �?       @              �?      �?      �?      @      @       @       @      @      @      �?      �?       @       @      @       @      @      @      @      (@      @      @      $@      @      &@      4@      0@      5@      &@      5@      3@      7@      9@      =@     �E@      ?@     �E@     �A@      B@      F@      L@      K@     �R@     �Q@     �Q@     �S@      R@     �[@     �]@     @^@     @b@     �a@     �c@     `d@     �d@     �^@      <@      @        
�
conv1/biases_1*�	   ��`�   �W,�?      @@!  l�*��?)E�^A� y?2��l�P�`�E��{��^�ܗ�SsW�<DKc��T���%>��:�uܬ�@8�x?�x�?��d�r?��82?�u�w74?k�1^�sO?nK���LQ?ܗ�SsW?��bB�SY?5Ucv0ed?Tw��Nof?�N�W�m?;8�clp?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?}Y�4j�?��<�A��?�v��ab�?�/��?�������:�              �?              �?              �?              �?              �?              �?              �?              �?              �?               @              �?      �?              �?              �?       @      @       @       @      �?      @              �?      �?       @        
�
conv2/weights_1*�	   �����   �eٰ?      �@!@5}��H@)	��;�rE@2���]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[��>�?�s���O�ʗ�����Zr[v��I��P=��8K�ߝ�a�Ϭ(��uE���⾮��%ᾙѩ�-߾E��a�Wܾ�_�T�l׾��>M|Kվ��~]�[Ӿ豪}0ڰ���������?�ګ�G&�$�>�*��ڽ>;�"�q�>['�?��>K+�E���>�ѩ�-�>���%�>�uE����>�f����>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�������:�              �?       @      E@     ��@     �@     <�@     ��@     ��@     d�@     ��@     h�@     �@     \�@     ��@     `�@     ��@     ��@     ��@     ��@     0�@     P{@     @z@      x@     �v@      t@     `t@     �q@     �n@      k@     �k@     �h@     `e@      e@     �a@     �a@      b@     @Z@      \@     �X@     �R@     @R@      N@     @P@      F@     �K@     �E@     �B@      B@      C@      A@      :@      5@      5@      9@      (@      .@      ,@      1@      3@      ,@      @      "@       @       @      "@      @      @      (@      @      @      @      @       @      @      @       @      �?      �?      �?      @               @      �?      �?              �?              �?      �?      �?              �?       @              �?      �?              �?              �?      �?              �?              @              �?              �?      @              @      �?              �?              @      �?      @       @      �?      @      @      @      "@      @       @      @      @      (@      @       @       @      @      (@      .@      "@      ,@      2@      2@      7@      *@      5@      <@     �B@      B@      E@      C@     �E@      H@     �I@     �L@     �O@     �N@      O@      T@     �S@     @Y@      ]@     �^@     �c@      a@     �d@     @i@      f@     `l@     �n@     �p@     �t@     �v@     Pw@     �y@     p{@     ��@     ��@     h�@     ��@     @�@     �@     ��@     �@     �@     �@     �@     X�@     L�@     ��@     ��@     Ԟ@     >�@     ��@     `h@      .@      �?        
�	
conv2/biases_1*�		    �    WC�?      P@!  V����?)��
�Քr?2��"�uԖ�^�S�����Rc�ݒ�#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS��!�A����#@���82���bȬ�0��T7����5�i}1��ѩ�-߾E��a�Wܾ�ߊ4F��>})�l a�>+A�F�&?I�I�)�(?��%>��:?d�\D�X=?���#@?�!�A?a�$��{E?
����G?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�#�h/�?���&�?�Rc�ݒ?}Y�4j�?��<�A��?�������:�              �?      �?              �?               @       @              �?              �?      @              �?       @              �?       @              �?              �?       @               @              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?       @              �?              �?              @              �?              �?      @       @      @      �?               @      �?      @              �?       @              �?        
�
conv3/weights_1*�	    �̰�    ���?      �@!��h�r�@@)[�
EY�U@2���]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������1��a˲���[���FF�G �O�ʗ�����Zr[v��pz�w�7��})�l a��ߊ4F��a�Ϭ(���(����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�_�T�l׾��>M|Kվ����>豪}0ڰ>��~���>�XQ��>K+�E���>jqs&\��>��>M|K�>�_�T�l�>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?�������:�              @     ȃ@     ��@     �@     ��@     r�@     �@     *�@     \�@     ��@     �@     ĕ@     ܓ@     �@     ,�@     0�@     ��@     �@     (�@     ��@     0�@     8�@     p~@     p|@     �y@     0y@     u@     @t@     �p@     �n@     �l@     �k@     @j@     �a@      d@     �b@     @b@     @Z@     �Z@     @[@     @U@     �T@     @T@     �P@     @Q@     �L@      P@     �A@     �B@      A@      6@      D@      :@      6@      2@      3@      ,@      ,@      1@      0@       @      ,@      3@      &@      "@      "@      @      @      $@      "@       @       @       @      @      @      @      @              �?       @               @               @      �?              �?              �?              �?               @              �?              �?              �?              �?               @      �?      �?       @      @       @      @       @              �?      @              @      �?       @      @       @      @      @      @      @      @      @      @      @      @      "@      @      $@      5@      .@      ,@      .@      9@      1@      6@      2@      6@      ?@      >@     �D@      C@      E@      N@     �M@     �N@      N@     �Q@     �S@     �U@     �X@     @^@     �Y@     @a@     @a@      b@      c@     @h@     �k@     �i@      n@      r@     �r@     @s@     Py@     �w@     �{@      @     �@     ��@     H�@     ��@     h�@     ��@     ��@     ��@     �@     ��@     ��@     Ę@     ��@     ��@     ��@     ��@     �@     ��@     &�@     ��@     P�@     @U@      @        
�
conv3/biases_1*�	   `Qǖ�   �`�?      `@!  �iQ�?)9w���P~?2��"�uԖ�^�S�����7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T�nK���LQ�k�1^�sO��qU���I�
����G�a�$��{E��!�A����#@���%>��:�uܬ�@8���%�V6���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(��.����ڋ�        �-���q=O�ʗ��>>�?�s��>6�]��?����?��ڋ?�.�?�7Kaa+?��VlQ.?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�������:�              �?               @       @      �?      �?      @               @              �?      �?      �?              @      @       @              @      �?      �?       @       @       @       @      �?      �?              �?               @      �?              �?              �?      �?              �?      �?              �?              �?              @              �?               @              �?              �?              �?      �?      �?              �?      �?              �?      �?      �?              �?      �?      �?               @      �?      �?       @      �?       @       @      @       @      �?      @       @      �?      @      @       @      �?      @      @       @       @      �?       @               @              �?      �?        
�
conv4/weights_1*�	   �}���   `A��?      �@! �!�d.�)`T�S{e@2�Ӗ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.���T7����5�i}1���d�r�>h�'��f�ʜ�7
��FF�G �>�?�s���I��P=��pz�w�7��a�Ϭ(���(��澙ѩ�-߾E��a�WܾK+�E��Ͼ['�?�;��~]�[�>��>M|K�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?�������:�             �a@     Ș@     ��@     d�@     t�@     X�@     �@     ��@     8�@     �@     0�@     0�@     p@     �}@      |@      w@     Pv@     �s@     s@     �q@     Pp@     �h@     �j@     �h@     �e@     @d@     �^@     �[@     �]@     @X@     �S@      U@     �R@     �Q@      K@     �K@      C@      K@     �B@      E@     �D@      ;@      <@     �@@      >@      ;@      8@      >@      ,@      0@       @      *@      &@      (@      ,@      @      @      @       @      @      @      @       @       @      @      @      @       @      @      @      �?      �?              �?       @              �?      �?              �?              �?              �?              �?              �?              �?              �?               @      �?              �?              �?      �?      �?      @      �?              �?      �?       @              �?              @       @      @              @       @      @      @      @      @      @       @      @      (@      (@      $@      (@      "@      $@       @      ,@      0@      (@      "@      5@      7@      7@      ;@      <@      A@      C@      B@     �@@     �J@     �J@     �R@     �R@     �L@     �R@      R@     @T@     �T@     �_@     �[@     �a@     �c@      e@     �e@     `g@     `k@     �j@     0p@     �s@     Pt@     pv@      z@     �x@     @|@     ��@     ��@     ��@      �@     �@     ��@     ��@     ��@     ��@     h�@     �@     Ԗ@     �i@       @        
�
conv4/biases_1*�	   �&᝿   ����?      p@!r� �_�?)�����?2��v��ab����<�A���^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@���%>��:�uܬ�@8���%�V6��u�w74���82���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��6�]���1��a˲���[��>�?�s���O�ʗ�����Zr[v��I��P=��})�l a��ߊ4F��h���`f�����uE���⾮��%�;�"�qʾ
�/eq
Ⱦ豪}0ڰ������������M�6��>?�J��`�}6D�/�p`B�p��Dp�@�����W_>�u 5�9��z��6��so쩾4�6NK��2�_"s�$1��f׽r����tO����        �-���q=Z�TA[�>�#���j>�J>2!K�R�>�'v�V,>7'_��+/>_"s�$1>u 5�9>p
T~�;>p��Dp�@>/�p`B>��Ő�;F>��8"uH>H��'ϱS>��x��U>Fixі�W>['�?��>K+�E���>jqs&\��>��~]�[�>E��a�W�>�ѩ�-�>�f����>��(���>a�Ϭ(�>8K�ߝ�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>>�?�s��>�FF�G ?6�]��?����?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?^�S���?�"�uԖ?�v��ab�?�/��?�uS��a�?�������:�              �?              �?       @              �?              �?              @      @      �?       @      �?      �?              @               @      �?      �?      @      @      �?               @      @       @      �?      �?      �?       @              @      �?      �?       @       @              �?       @      �?      �?              �?      �?      �?      �?      �?              �?      �?      �?      �?               @      �?      @              �?      �?              �?       @      �?              �?      �?              �?      �?              �?              �?               @              �?      �?      �?              �?      �?       @       @              �?              7@              �?              �?              �?      �?              �?              �?              �?              �?      �?              �?              �?              �?              �?              �?              �?              �?              �?               @              �?      �?              �?              �?               @              �?              @               @               @      �?              �?       @      �?      �?      @       @       @              @      @       @      �?      @      @      @      @       @      @       @      �?      @      @      @      �?       @      �?       @              @      @              @      @      @       @       @              @      @      �?      �?              �?              �?      �?        
�
conv5/weights_1*�	   �߅ÿ   �[��?      �@! �I�8*@)/�!�Q�I@2�
yD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.�I�I�)�(�+A�F�&��[^:��"��S�F !���ڋ��vV�R9�x?�x��>h�'����Zr[v��I��P=��>�?�s��>�FF�G ?>h�'�?x?�x�?��ڋ?�.�?ji6�9�?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�������:�
              �?     `j@     �q@     �q@     �n@     @l@      m@     �h@     �b@     @a@     �b@     �a@     @[@     �[@     @V@     �X@     �R@     @S@      T@     �Q@     �G@      F@      C@     �C@      A@      =@      >@     �B@      9@      <@      .@      3@      (@      1@      (@      ,@      *@      @       @      @      @      $@      $@       @      @      @      @      �?      @      @      @      �?       @       @               @      �?               @              �?              �?               @      �?              �?               @              �?              �?              �?              �?              �?              �?              �?      �?              �?      �?       @              �?       @              �?      �?      �?              �?      �?      �?      @       @      �?       @      @      @      @      @      �?       @      @      @      @       @      @      @      @      @      $@      &@       @      6@      $@      3@       @      &@      2@      4@      ;@      3@      7@     �D@      @@      E@      E@      H@     �F@      P@     �J@     �M@     �S@     �R@     �Y@      Z@     �Z@     ``@     ``@     @b@     @c@     @e@     �c@     �g@      o@     @n@     �r@     �t@      o@      1@        
�
conv5/biases_1*�	   @GQj�   ``�^>      <@!  @*Q>)�e���<�<2�=�.^ol�ڿ�ɓ�i�28���FP�������M��`�}6D�/�p`B�p��Dp�@�����W_>�p
T~�;�u 5�9�6NK��2�_"s�$1��mm7&c��`���f;H�\Q������%��z�����=ݟ��uy�=�K���=�9�e��=f;H�\Q�=�tO���=�mm7&c>y�+pm>���">Z�TA[�>4�e|�Z#>��o�kJ%>4��evk'>���<�)>�'v�V,>7'_��+/>_"s�$1>6NK��2>u 5�9>p
T~�;>/�p`B>�`�}6D>6��>?�J>������M>�
L�v�Q>H��'ϱS>Fixі�W>4�j�6Z>��u}��\>d�V�_>�������:�              �?              �?              �?      �?      �?               @              �?              �?              �?              �?              �?              �?              �?              �?               @      �?              �?       @              �?              �?              �?              �?              �?              �?              �?        ��K!xW      �7�Z	���'��A*�

step  �A

loss.{�=
�
conv1/weights_1*�	   �?9��    �z�?     `�@! p��j;�)yVP��@2�	� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ���d�r�x?�x��>h�'��f�ʜ�7
������6�]����5�L�����]����['�?��>K+�E���>�f����>��(���>f�ʜ�7
?>h�'�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?�������:�	               @      (@     �J@     �V@     �e@      g@     `e@     �b@     �d@      \@     �^@      W@     �V@      W@     �O@     @S@      R@      O@      M@      N@     �J@      I@      B@      ?@      <@      9@      5@      =@      1@      3@       @      2@      .@      2@      &@      "@      &@      $@      "@       @      @      "@      @      @      @      @       @       @      @      @      �?      �?      @      @      @       @       @              �?              �?       @      �?      �?              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?               @      �?              �?       @      �?      @       @      @      �?      �?       @       @       @       @      @              @      �?      �?      @      @      @      @      @      @      @       @      @      "@      @      0@      0@       @      2@      3@      2@      .@      5@      7@      9@      =@      A@      F@     �@@     �D@     �B@     �D@      J@      P@      Q@     @Q@      R@     �S@     @R@     �\@     @]@     �\@     �b@      a@     �c@     �d@     @d@     �^@      @@      @        
�
conv1/biases_1*�	   ��@`�   �;��?      @@!  �3�?)�K鏚�{?2��l�P�`�E��{��^�<DKc��T��lDZrS���%>��:�uܬ�@8��5�i}1���d�r��u�w74?��%�V6?nK���LQ?�lDZrS?�m9�H�[?E��{��^?5Ucv0ed?Tw��Nof?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?}Y�4j�?��<�A��?�v��ab�?�/��?�������:�              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?      �?      �?              �?      �?              �?              @       @      �?      @      �?       @              �?      �?       @        
�
conv2/weights_1*�	   @���    ,�?      �@!��s4ԾI@)]�4��yE@2���]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%��_�T�l׾��>M|Kվ['�?�;;�"�qʾ
�/eq
Ⱦ�MZ��K���u��gr����x��U>Fixі�W>�XQ��>�����>�iD*L��>E��a�W�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�������:�              �?       @     �H@     ��@     ܟ@     �@     l�@     ��@     T�@     ��@     x�@     ��@     p�@     p�@     ��@     ��@     ��@      �@     ��@     �@     �z@     0{@     0x@     �u@     0u@     �s@     �q@     �n@     `m@     �j@     �g@     `d@     �e@     �b@     �_@     �a@      \@     @]@     �W@     �S@     �Q@     �P@      O@      K@     �H@     �F@      B@      E@      @@     �A@      ;@      5@      2@      8@      .@      7@      .@      (@      1@      &@      @      .@      @      &@      &@      @      @      @      "@      @      @      @      �?      @              �?      �?       @      @      �?      �?       @      �?       @              �?              �?      �?      �?      �?              �?              �?              �?       @              �?              �?              �?              �?               @      �?              �?              �?      �?       @               @              �?       @              �?      @      �?       @      @       @      @      @       @      @       @      @       @      @      @      @       @      &@      &@      .@      (@       @      0@      "@      0@      .@      .@      9@      :@      5@      <@     �C@      F@      G@     �D@      H@     �D@      P@      M@     �N@      T@      R@      T@     �Y@     �[@      _@     �c@     �b@     �d@     �f@     �g@     �j@      p@     Pp@     �s@     �w@     �v@     @z@     �{@     `�@     Ѐ@     8�@     Ѕ@     H�@     ��@     ��@     ��@     �@     đ@     @�@     L�@     8�@      �@     l�@     Ԟ@     6�@     ��@      j@      8@      @        
�	
conv2/biases_1*�		   @O��   @�͚?      P@!  ��u�?)�_]�t?2��"�uԖ�^�S�����Rc�ݒ��7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�Tw��Nof�5Ucv0ed�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�a�$��{E��T���C���82���bȬ�0��.����ڋ���Zr[v�>O�ʗ��>��[�?1��a˲?��VlQ.?��bȬ�0?�!�A?�T���C?a�$��{E?
����G?��bB�SY?�m9�H�[?E��{��^?�l�P�`?5Ucv0ed?Tw��Nof?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?}Y�4j�?��<�A��?�������:�              �?      �?              �?               @               @      �?              �?      �?       @      �?      �?      �?      �?       @              �?              �?       @              �?      �?              �?              �?              �?              �?              �?              �?               @              �?               @               @              �?              �?      �?       @              �?               @      @      @       @              �?      �?      �?      @              �?      �?      �?              �?        
�
conv3/weights_1*�	   �갿   ����?      �@!��b��B@)��!�U@2���]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`f�����uE���⾮��%�['�?�;;�"�qʾ�*��ڽ�G&�$��
�/eq
�>;�"�q�>K+�E���>jqs&\��>�iD*L��>E��a�W�>�f����>��(���>a�Ϭ(�>8K�ߝ�>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?�������:�              $@     ��@     ��@     �@     ��@     t�@     ��@     ,�@     8�@     К@     �@     ��@     Г@     �@     p�@     ��@     ��@     ��@     ��@      �@     @�@     8�@     �~@     P|@     �y@     Px@     @u@     �t@     �p@     @o@     �m@      k@     �i@     �b@     @b@     `b@      c@     @Z@      Y@     �]@      T@     �V@      U@      Q@      L@     �O@     �M@     �B@      B@     �@@      A@     �A@      <@      2@      3@      3@      1@      4@      .@      .@      *@      *@      1@      $@      .@       @      @      @       @      @      @      @      @      @      �?      �?      @               @              �?              �?      @               @      �?       @      �?              �?      �?              �?              �?              �?              �?              �?              �?              �?               @       @       @      �?      �?      @      �?              @      @      @      @      �?      @       @      @       @       @      &@      @      @      @      "@       @      (@      &@      ,@      ,@      $@      *@      9@      6@      5@      6@      <@      <@     �@@      C@     �B@      C@      I@     �M@      O@     @P@     �P@     @W@     �R@     @Z@     �[@     �[@      a@      c@     ``@      d@     �g@     �l@      i@     `n@      r@     �r@     �r@     �y@     0y@     �z@     �~@     �@     ��@     p�@     ��@     `�@      �@     ��@     Đ@     ԑ@      �@     X�@     ��@     L�@     p�@     ؞@     ��@     �@     ~�@     &�@     �@     ��@     @W@      $@        
�
conv3/biases_1*�	   �����   ����?      `@! ��q?�?)k�ug9�?2�}Y�4j���"�uԖ��#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� ��o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ��qU���I�
����G�a�$��{E��T���C��!�A���%>��:�uܬ�@8���%�V6��u�w74���bȬ�0���VlQ.��.����ڋ�        �-���q=I��P=�>��Zr[v�>f�ʜ�7
?>h�'�?x?�x�?��d�r?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�������:�              �?              �?       @       @      �?       @       @       @              �?       @              �?      @       @      �?       @       @      �?       @       @      �?       @      @              �?              �?               @      �?              �?              �?      �?      �?               @              �?              @              �?              �?               @              �?              �?              �?               @      �?              �?      �?      �?              �?       @              �?      �?      �?      �?       @       @      @      �?      @       @       @       @       @      @      @      @      @       @      @      @      �?      �?       @      �?      �?      �?              �?      �?        
�
conv4/weights_1*�	   `���   `}��?      �@! w�ّ�,�)8k3O)|e@2�Ӗ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1�>h�'��f�ʜ�7
�6�]���1��a˲���[��O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��['�?��>K+�E���>})�l a�>pz�w�7�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?�������:�             �a@     И@     �@     D�@     `�@     h�@     �@     ��@     @�@     �@     �@     `�@     0@     0~@     @|@     �v@     pv@     �s@     �r@     �q@     p@     @i@      j@     @i@     �e@     `d@      ^@     �Z@     �^@      Y@     �S@     @T@     �S@     �O@     �K@      L@     �D@     �H@      D@      D@     �D@      <@      ?@      ?@      <@      :@      ;@      6@      7@      .@      $@      &@      *@      &@      $@      @       @      @      "@      @       @      @      @      @      @      @       @      @       @      @      @      �?      �?      �?      �?              �?              �?               @              �?      �?              �?              �?              �?              �?              �?              �?              �?              �?       @      �?       @      �?              �?      �?      �?       @      �?      @               @               @       @      �?      �?      @      @      @      @      @      @       @      @      0@       @      (@       @      &@      @      1@      .@      *@      $@      7@      5@      8@      8@      ;@      C@     �B@      @@     �C@      H@      J@     �Q@     �S@      M@     �Q@      S@     �S@      V@     �^@      \@     `a@     �c@      e@     �e@     �f@     �k@     @k@     p@     �s@     0t@     �v@     �y@     �x@     @|@     ��@     ��@     �@     ��@     ��@     ؈@     ��@     ��@     ��@     d�@     �@     ̖@      j@       @        
�
conv4/biases_1*�	   `�w��   ��?      p@!�Ѿ@���?):[.d2�?2��/����v��ab���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
�6�]���1��a˲���[��>�?�s���O�ʗ�����Zr[v���ߊ4F��h���`�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾��>M|Kվ��~]�[Ӿ;�"�qʾ
�/eq
Ⱦ��n�����豪}0ڰ�28���FP�������M�6��>?�J���Ő�;F��`�}6D�/�p`B�p��Dp�@�u 5�9��z��6��so쩾4�6NK��2�_"s�$1��tO����f;H�\Q��        �-���q=�#���j>�J>2!K�R�>��R���>�'v�V,>7'_��+/>_"s�$1>p
T~�;>����W_>>/�p`B>�`�}6D>��Ő�;F>��8"uH>��x��U>Fixі�W>K+�E���>jqs&\��>��~]�[�>��(���>a�Ϭ(�>8K�ߝ�>�ߊ4F��>})�l a�>I��P=�>��Zr[v�>>�?�s��>�FF�G ?����?f�ʜ�7
?x?�x�?��d�r?�5�i}1?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?^�S���?�"�uԖ?�v��ab�?�/��?�uS��a�?`��a�8�?�������:�              �?              �?      �?      �?              �?      �?               @      @      �?      @               @               @      �?      �?       @      �?      @      @              �?              �?      @       @      �?      �?      �?      @              @      �?      �?      @       @              �?              �?       @      �?              �?      �?      �?       @      �?      �?      �?      �?      �?      �?              �?       @      �?      �?      �?              �?      �?               @       @              �?              �?              �?              �?              �?              �?              �?              �?      �?              �?               @               @      �?      �?       @              �?              7@              �?              �?              �?      �?              �?              �?              �?               @              �?      �?              �?      �?              �?              �?              �?              �?              �?      �?              �?              �?      �?      �?      �?               @       @              �?      �?      @              @      �?       @       @       @      �?      @              @       @      @       @      @      @      @       @      �?      @      �?      @      @      @              @               @       @       @      @      �?      @      @      @      �?      �?              @       @      �?              �?              �?              �?        
�
conv5/weights_1*�	    s�ÿ   �t.�?      �@! @F�'�*@)�˙�I@2�	yD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�a�$��{E��T���C�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9�8K�ߝ�a�Ϭ(��u`P+d�>0�6�/n�>x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�������:�	              �?      j@      r@     �q@     �n@     `l@     �l@     �h@     �b@     `a@     �b@     �a@     �Z@     �\@     �U@     �X@     �R@     �R@      T@      Q@      I@      F@     �B@      D@      A@      =@      >@      B@      ;@      =@      ,@      1@      $@      3@      *@      (@      ,@       @       @      @      @       @      $@      "@      @      @      @       @      @      �?      �?       @      �?      @              @              �?              �?      �?               @              �?      �?      �?      �?              �?              �?              �?              �?              �?              �?      �?              �?              �?              �?      �?              �?      �?      �?      �?       @               @               @      @       @      �?      @       @      �?      @      @      �?       @      @       @      @      @      @      @      @      @      $@       @      .@      0@      &@      4@      @      (@      .@      4@      >@      2@      9@     �B@      B@      D@      F@     �F@      I@      O@     �J@     �M@     �S@     �R@      Y@     �Z@     �Y@     �`@      a@      b@     @c@     @e@     �c@     �g@     �n@     @n@     �r@     `t@     �o@      0@      �?        
�
conv5/biases_1*�	   `��k�   ��`>      <@!   >�R>)���];�<2�=�.^ol�ڿ�ɓ�i�28���FP�������M���Ő�;F��`�}6D�/�p`B�p��Dp�@�����W_>�p
T~�;��so쩾4�6NK��2��mm7&c��`���f;H�\Q������%��i@4[��=z�����=�K���=�9�e��=�f׽r��=nx6�X� >�`��>���">Z�TA[�>4�e|�Z#>��o�kJ%>4��evk'>�'v�V,>7'_��+/>_"s�$1>6NK��2>�so쩾4>u 5�9>p
T~�;>/�p`B>�`�}6D>������M>28���FP>H��'ϱS>��x��U>Fixі�W>4�j�6Z>d�V�_>w&���qa>�������:�              �?              �?              �?      �?      �?               @              �?              �?              �?              �?              �?              �?      �?              �?               @      �?              �?       @              �?              �?              �?              �?              �?              �?              �?        tcD�XW      �A�	1D,�'��A*ɮ

step  �A

lossC��=
�
conv1/weights_1*�	    ����   �0��?     `�@! p�qO��)�'�I�@2�	� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !���ڋ��vV�R9��T7�����d�r�x?�x��>h�'��f�ʜ�7
�>�?�s���O�ʗ�����Zr[v��pz�w�7��})�l a�FF�G ?��[�?f�ʜ�7
?>h�'�?x?�x�?��d�r?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?��VlQ.?��bȬ�0?��82?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?�������:�	              @      *@     �L@      X@     �e@     `g@     �d@      c@     `d@     �\@     �^@     @U@     �X@      U@     �Q@     �S@      M@     @P@      O@     @P@      H@      F@      C@      >@      :@     �A@      6@      3@      3@      5@      6@      &@      ,@      .@      &@       @      ,@      "@      @      $@      @      @      @      @      @       @       @      �?       @      @      @      @      @       @      @       @      @               @              �?              @      �?      �?      �?      �?       @              �?      �?              �?              �?              �?      �?              �?              �?              �?              �?              �?      �?              �?              �?       @      �?               @      @              �?      @      �?      �?      �?      �?      @      �?      @      @      @      �?       @      @       @       @      @       @      .@       @      *@      @      .@      .@      (@      ,@      ,@      0@      .@      9@      A@      ,@      >@      B@     �E@     �A@      D@     �@@     �F@     �I@      P@     �P@      P@     �S@     @R@     �S@     @\@     �]@      [@     �b@     @a@     `c@     `d@     �d@     @]@     �C@       @      �?        
�
conv1/biases_1*�	    �0`�    Ao�?      @@!  8O�M�?)����f"?2��l�P�`�E��{��^��lDZrS�nK���LQ���%>��:�uܬ�@8���bȬ�0���VlQ.���%�V6?uܬ�@8?<DKc��T?ܗ�SsW?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?>	� �?����=��?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?��<�A��?�v��ab�?�/��?�uS��a�?�������:�              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?       @               @              �?      @       @       @       @      �?       @      �?              �?      �?       @        
�
conv2/weights_1*�	   ��E��   `�_�?      �@!�Ȝ(T�J@)�fD�F�E@2���]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澮��%ᾙѩ�-߾;�"�qʾ
�/eq
Ⱦ�[�=�k���*��ڽ�5�"�g���0�6�/n�����]���>�5�L�>豪}0ڰ>��n����>�XQ��>�����>
�/eq
�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�������:�              �?      "@     �L@     X�@     ��@      �@     ��@     ��@     (�@     ԕ@     ��@     ��@     H�@     (�@     `�@     ��@     ��@     Ȅ@     ��@     h�@      z@     p{@     pw@     �v@     �u@     �r@     �q@      n@     �m@      l@     `g@      d@     @e@     `a@     @a@      a@     @Z@     �_@     �U@      U@     @S@     �R@     �L@      M@     �H@     �E@      H@     �A@     �@@      =@      <@      7@      ,@      =@      ,@      *@      ,@      *@      2@      @      ,@       @      @      @      &@      *@       @      @      @       @      @       @      @      @       @               @      �?      @      �?      �?      @      @      �?       @      �?      �?              �?       @      �?      �?              �?              �?              �?              �?              �?              �?              �?      �?               @       @      �?      �?      @      �?      �?      �?      �?       @       @      �?      �?      @               @      @      @      @      �?      @      @      @      $@      "@      ,@      @      $@      &@      @      0@      3@      ,@      4@      0@      ,@      9@      3@      D@      <@     �D@      H@      D@      E@     �H@      N@     �N@      N@      T@     �T@      T@     @X@     @\@      ^@     `c@      b@     @e@     �g@     �f@     �j@     `p@     p@     �s@     �w@     Pw@      z@      |@      �@     0�@     ��@     ��@     x�@     ��@     8�@     ��@     �@     Б@     `�@     4�@     \�@     ��@     ��@     �@     
�@      �@     `l@      :@      @        
�	
conv2/biases_1*�		   @�2��    �E�?      P@!  � �r�?)h���C�v?2�}Y�4j���"�uԖ�^�S�����#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=������T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�Tw��Nof�5Ucv0ed��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�
����G�a�$��{E���bȬ�0���VlQ.��[^:��"��S�F !�����?f�ʜ�7
?x?�x�?��d�r?��bȬ�0?��82?�!�A?�T���C?a�$��{E?
����G?�qU���I?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?��<�A��?�v��ab�?�������:�              �?      �?              �?               @               @      �?              �?       @      �?       @               @               @              �?              �?       @              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?      �?               @              �?      �?              �?              �?       @      �?      �?               @              @      @      �?               @      �?       @       @              �?       @              �?        
�
conv3/weights_1*�	   �\��    �8�?      �@!@�VTsC@)�l�G �U@2���]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=���h���`�8K�ߝ�a�Ϭ(���(����uE���⾮��%ᾄiD*L�پ�_�T�l׾0�6�/n���u`P+d��豪}0ڰ>��n����>�����>
�/eq
�>jqs&\��>��~]�[�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?�������:�              &@     8�@     p�@     �@     r�@     |�@     �@     <�@     �@     ��@      �@     ��@     ȓ@     ��@     P�@     ��@     ��@     H�@     �@     ��@     X�@     X�@     P@     �{@     y@     `y@     �t@      u@      o@     `p@     @m@     �k@      j@      b@      c@     �b@     �`@      [@     �[@     �Z@     �T@     �V@     @S@     �P@      P@     �O@     �M@     �@@     �A@      H@      B@      B@      8@      5@      (@      9@      *@      3@      4@      "@      *@      &@      3@      "@       @       @      (@      @      @      @      @      @      @      @       @      @      @      �?       @      @      �?      �?              @       @              �?      �?      �?              �?               @              �?               @              �?              �?              �?       @              �?      �?      �?      �?              �?      �?              �?       @      �?               @       @       @       @       @      @      @      �?      @      @      @       @      @      @      @      @      @      @      @      @      @      *@      *@      0@      $@      ,@      "@      5@      1@      6@      2@      3@      >@      D@     �C@     �C@      C@     �H@     �G@     @P@     �Q@     �Q@     �S@     �R@      [@      ^@     �\@     �`@     �b@     �b@     �b@     �f@     �l@     `i@      n@     pr@     �q@     �s@     �x@     `y@     �z@     �}@     x�@     H�@      �@     І@      �@     (�@     ��@     Ԑ@     ��@     �@     <�@     ��@     H�@     l�@     ��@      @     �@     ��@     2�@     ި@     ��@     �Y@      2@        
�
conv3/biases_1*�	    M1��   ��Ԟ?      `@!  Ҁ
�?)��F���?2���<�A���}Y�4j���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=������T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A�d�\D�X=���%>��:�uܬ�@8���%�V6���82���bȬ�0���VlQ.��.����ڋ��FF�G �>�?�s���        �-���q=�uE����>�f����>>h�'�?x?�x�?�S�F !?�[^:��"?��bȬ�0?��82?�u�w74?uܬ�@8?��%>��:?d�\D�X=?���#@?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?��bB�SY?�m9�H�[?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�������:�              �?               @      @              �?      @       @              �?      �?      �?              @      @      �?       @       @      �?       @      �?       @       @      �?       @      �?              �?              �?      �?      �?              �?              �?               @              �?      �?              �?              �?              @              �?              �?              �?               @      �?              �?               @              �?               @              @              �?      �?      �?      @              @       @      @      @      �?      @       @       @      @      @      @      �?      @       @      @       @               @      �?      �?      �?              �?      �?        
�
conv4/weights_1*�	   @x���   @���?      �@! g(O�+�)��:}e@2�Ӗ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���I��P=��pz�w�7��})�l a��ߊ4F��[�=�k�>��~���>�XQ��>�uE����>�f����>8K�ߝ�>�h���`�>})�l a�>pz�w�7�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?6�]��?����?>h�'�?x?�x�?��d�r?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?�������:�              b@     Ԙ@     ��@     4�@     d�@     `�@     (�@     ��@      �@     �@     �@     `�@     @@     0~@     0|@     �v@     �v@     t@     �r@     �q@      p@      i@     �i@      j@      e@      e@     �\@     �Z@     �^@     @Z@     �S@     @T@     �R@     @P@      K@     �L@     �C@     �H@      F@     �B@     �D@      ?@      :@      A@      :@      8@      <@      ;@      5@      (@      .@      $@       @       @      3@      @      @      @      "@      @       @      @      �?      �?      @      @       @      @      @      @      �?      �?              �?               @              �?       @               @              �?      �?              �?              �?              �?              �?      �?              �?              �?              �?               @              �?               @              @      �?              �?       @              @       @       @      @              �?       @      @      @      @      @      @      @      @       @       @      0@      $@       @       @      $@      "@      3@      &@      ,@      (@      6@      5@      3@      9@     �@@      <@      G@      A@     �@@      I@     �H@     �Q@     @S@     �N@     �P@     �S@     �T@     �V@     �\@      \@     @a@     �d@      e@     �e@     �f@     �k@      k@      p@      t@      t@     �v@     �y@     y@     p|@     ��@     ��@      �@     Ȅ@      �@     ��@     ��@     ��@     ��@     h�@     ��@     Ж@     �j@       @        
�
conv4/biases_1*�	   `M���    O#�?      p@!�|?�7Q�?)gJue�?2��/����v��ab��}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@���%>��:�uܬ�@8���%�V6���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�6�]���1��a˲��FF�G �>�?�s���O�ʗ�����Zr[v���ߊ4F��h���`���(��澢f�����uE���⾮��%�['�?�;;�"�qʾ��n�����豪}0ڰ�28���FP�������M���8"uH���Ő�;F��`�}6D�/�p`B�p
T~�;�u 5�9��z��6��so쩾4�6NK��2�_"s�$1�f;H�\Q������%��        �-���q=�J>2!K�R�>��R���>�'v�V,>7'_��+/>_"s�$1>6NK��2>p
T~�;>����W_>>/�p`B>�`�}6D>��8"uH>6��>?�J>��x��U>Fixі�W>4�j�6Z>K+�E���>jqs&\��>��~]�[�>��>M|K�>E��a�W�>�ѩ�-�>a�Ϭ(�>8K�ߝ�>�h���`�>})�l a�>pz�w�7�>I��P=�>��Zr[v�>�FF�G ?��[�?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?�"�uԖ?}Y�4j�?�/��?�uS��a�?`��a�8�?�������:�              �?              �?               @              �?      �?              �?       @      @       @      �?      �?      �?               @      �?      @               @      @       @      �?              �?       @       @      �?      �?      �?      @               @      @       @       @      @              �?               @       @              �?      @      �?               @              �?               @      �?               @       @       @               @               @      �?       @              �?              �?              �?              �?              �?               @              �?               @              �?       @      �?      �?      �?              �?              7@              �?      �?              �?              �?              �?              �?              �?              �?      �?              �?              �?              �?              �?      �?              �?              �?              �?              �?              �?      �?              �?      �?      �?              �?      �?               @               @      �?              �?      �?      �?              @      �?       @      @      �?      �?       @       @       @       @      @      @      @       @      @       @              @      @      @      @       @       @      �?      �?      �?      @      @      �?      @      @      @       @      �?      �?       @      @      �?      �?              �?              �?      �?        
�
conv5/weights_1*�	   ���ÿ   ��e�?      �@! �_d�M+@)���I@2�	yD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I��T���C��!�A����#@���%�V6��u�w74���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !���ڋ��vV�R9�>h�'��f�ʜ�7
���d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��82?�u�w74?��%�V6?uܬ�@8?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�������:�	              �?      j@     r@     �q@     �n@     @l@     �l@     �h@      c@     �a@     �b@      b@     �Y@      ]@      V@     �X@     @S@      R@     @T@     �P@     �I@     �E@     �A@      E@      A@      >@      =@      B@      <@      ;@      .@      1@      $@      2@      0@      &@      *@      @      "@      @      @       @      "@       @       @      @      @      @       @      �?       @               @      @      �?      @              �?      �?              �?              �?      �?       @      �?              �?      �?              �?              �?              �?      �?              �?       @              �?              �?              �?              �?              �?              �?       @       @      �?      @      �?      �?      @      @      @      �?      @      �?      @      @      @       @      @      @      @      @       @      "@       @      0@      0@      $@      3@       @      $@      1@      3@      =@      3@      9@      B@     �A@      E@      F@     �G@     �H@     �N@     �J@      N@     @T@     @R@     �X@     @[@     �Y@     �_@     �`@     �b@      c@     �e@     @c@     �g@      o@     �n@     Pr@     �t@      o@      1@      �?        
�
conv5/biases_1*�	   `�m�   ���a>      <@!  ���FS>)��Z����<2�w`f���n�=�.^ol��
L�v�Q�28���FP���Ő�;F��`�}6D�/�p`B�p��Dp�@�����W_>�p
T~�;��z��6��so쩾4��mm7&c��`����9�e����K���PæҭU�=�Qu�R"�=����%�=f;H�\Q�=�f׽r��=nx6�X� >�`��>���">Z�TA[�>4�e|�Z#>��o�kJ%>4��evk'>���<�)>7'_��+/>_"s�$1>6NK��2>�so쩾4>u 5�9>p
T~�;>�`�}6D>��Ő�;F>28���FP>�
L�v�Q>H��'ϱS>��x��U>4�j�6Z>��u}��\>w&���qa>�����0c>�������:�              �?              �?              �?      �?      �?               @              �?              �?              �?              �?              �?              �?      �?              �?               @              �?               @      �?      �?              �?              �?              �?              �?              �?              �?        Z�BZ�V      	燀	�Rҕ'��A*٭

step  �A

loss$o=
�
conv1/weights_1*�	   ����    ���?     `�@!  ��\�)���+@2�	8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1��FF�G ?��[�?1��a˲?f�ʜ�7
?>h�'�?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?�������:�	               @      @      3@     �L@      Y@     �e@     `g@      d@     `c@      d@      \@     �^@     @W@     @W@     @T@     @S@     @R@     �O@     �L@     @Q@     @Q@      G@      A@      C@      @@     �B@      2@      ?@      4@      3@      8@      1@      ,@      ,@      *@      "@       @      *@      $@      &@      @      @      @      @       @       @      @       @      @      @       @       @      @      @       @      @      �?       @      @      �?      �?      �?      �?      �?      �?      �?              �?              �?      �?       @              �?      �?              �?      �?              �?              �?      �?       @      �?      �?      �?              �?      �?               @      �?       @      @       @      �?      @      �?      �?       @      �?       @       @      �?      �?       @      @      @              @      @      @       @      "@      &@      "@      $@      .@      &@      .@      (@      ,@      $@      0@      3@      =@      8@      2@      =@     �@@     �E@      C@      C@      @@      I@      H@     �P@      O@     �P@      R@     @Q@     �T@     �\@     �^@     �Z@     �b@      a@      c@     `d@     �c@     �^@      D@      @       @        
�
conv1/biases_1*�	   @� `�   `oE�?      @@!  ����?)��#
z0�?2��l�P�`�E��{��^�nK���LQ�k�1^�sO����#@�d�\D�X=���%>��:�uܬ�@8���%�V6?uܬ�@8?ܗ�SsW?��bB�SY?���%��b?5Ucv0ed?Tw��Nof?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?��<�A��?�v��ab�?�/��?�uS��a�?�������:�              �?              �?              �?              �?              �?              �?              �?      �?              �?      �?      �?      �?              �?      �?              �?              @               @      @      �?       @              �?      �?       @        
�
conv2/weights_1*�	   �ˎ��   `���?      �@!`���/�K@)yR��؈E@2���]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ���pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f���侮��%ᾙѩ�-߾�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ�u��gr��R%���������m!#���
�%W���*��ڽ>�[�=�k�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>8K�ߝ�>�h���`�>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�������:�               @      &@      M@     ��@     `�@      �@     x�@     X�@     \�@     ��@     Ē@     �@     P�@     P�@     �@     ��@     x�@     ��@     ��@     Ȁ@     pz@     �z@     Pw@     �v@     �u@     �r@     �p@     �n@     �m@      l@     �f@     �e@      d@     @a@     �`@     �`@     �Y@     @_@     @[@     @Q@     �U@     �P@     �Q@     �H@      F@     �H@      D@      C@      E@      ;@      =@      5@      1@      :@      5@      *@      *@      (@      1@      @      $@       @       @      @      &@       @       @      @       @      @      $@      @       @      @      @      @      �?       @       @      @       @      @       @              �?      �?              �?              �?      �?              �?              �?              �?              �?              �?              �?              �?               @      �?               @              �?              �?      �?      @       @               @      @      @       @      @       @       @       @       @      �?       @      @      @      @       @      @      @      @      $@      "@      @      "@      $@      .@      *@      .@      2@      0@      2@      2@      :@      5@      ?@      9@     �A@      G@     �G@     �F@      F@     �O@     @P@      M@     �T@     �R@      U@     @X@     �\@     �_@     �a@     �c@     @d@      h@     �e@     �k@     pp@     @p@     `s@     �v@     `x@     �y@     �|@     �@      �@     `�@     (�@     ��@     p�@     �@     �@     �@     ̑@     P�@     H�@     l�@     Ț@     P�@     �@     ��@     T�@     �n@     �@@      @        
�	
conv2/biases_1*�		   @6O��   `��?      P@!  �79ؽ?)����Y%y?2�}Y�4j���"�uԖ�^�S�����#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�P}���h�Tw��Nof��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�
����G�a�$��{E���VlQ.��7Kaa+�U�4@@�$��[^:��"���d�r?�5�i}1?�vV�R9?��ڋ?��82?�u�w74?d�\D�X=?���#@?�qU���I?IcD���L?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?��<�A��?�v��ab�?�������:�              �?      �?              �?               @               @      �?              �?              @      �?      �?      �?      �?       @              �?               @      �?              �?              �?              �?              �?              �?              �?              �?              �?              �?               @               @              �?              �?              �?              �?      �?       @              �?      �?      �?       @      @      @               @              �?      @              �?       @              �?        
�
conv3/weights_1*�	   �O$��   ����?      �@! ��]��D@)�'ӻ�U@2���]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%��u`P+d����n�����R%������39W$:���
�/eq
�>;�"�q�>jqs&\��>��~]�[�>�iD*L��>E��a�W�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?�������:�              *@     0�@     R�@     �@     d�@     ~�@     �@     D�@     ��@     ��@     (�@     ̕@     ��@     �@     H�@     �@     `�@     Ȉ@     `�@     �@     ��@     p�@     0@     `|@     Py@     �x@     0u@     t@      p@     �p@     @l@     @l@     �g@     �d@      c@     �a@      b@     �\@     �W@     �Y@     �U@     �U@     �S@      S@     �M@      J@      Q@      >@      F@      C@      A@      >@      @@      2@      0@      9@      &@      2@      1@      0@      *@      4@      .@      0@      "@      $@      @      @      @       @      @      @      @      @      @      �?      @       @      @      �?      �?      �?               @       @      �?              �?              �?              @       @      �?              �?              �?              �?              �?              �?              �?      �?      �?      �?      �?      �?      �?              �?      @      �?      @      �?      @      @      @      @       @      @      @      @      @      $@      "@       @      &@       @      @       @      *@      ,@      4@      3@      2@      .@      6@      $@      4@      ;@      =@      D@     �F@     �F@     �H@      F@     �I@     �P@      S@     �U@     �R@     �Y@     �^@     �\@     `a@     �a@      c@      c@      g@     @k@     �j@      m@     Pr@     �r@     ps@     �x@     �y@      {@     @~@     �@     H�@     P�@     H�@     X�@      �@     ��@     �@     ��@     $�@     �@     �@     ,�@     ��@     Ğ@     ֠@     ܢ@     p�@     (�@     ܨ@     �@     �^@      4@        
�
conv3/biases_1*�	   @�a��   ��Y�?      `@!  ����?)�F�*"�?2���<�A���}Y�4j�����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=������T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T�IcD���L��qU���I�
����G�a�$��{E��T���C�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���bȬ�0���VlQ.��.����ڋ��vV�R9��T7����ߊ4F��h���`�        �-���q=x?�x�?��d�r?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�������:�              �?              �?       @       @      �?      @               @              �?       @              �?      @       @      �?       @       @      @      �?      �?       @      �?      �?       @      �?      �?              �?      �?      �?      �?              �?      �?      �?      �?              �?              �?              �?              �?              @              �?              �?       @              �?      �?              �?              �?              �?      �?      �?               @      �?              �?              �?      @      �?              @       @      @       @      @      �?      @       @      @      @      @       @       @      @      �?      �?       @      �?               @              �?      �?        
�
conv4/weights_1*�	   ����   ����?      �@!@cN��o*�)����H~e@2�Ӗ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��T7����5�i}1���d�r�x?�x��f�ʜ�7
������6�]���1��a˲���[���FF�G �I��P=��pz�w�7��a�Ϭ(���(��澢f������������?�ګ�����>豪}0ڰ>�uE����>�f����>��Zr[v�>O�ʗ��>�FF�G ?��[�?1��a˲?6�]��?����?>h�'�?x?�x�?��d�r?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?�������:�             �a@     И@      �@     �@     p�@     t�@     (�@     ��@     @�@      �@     �@     H�@     @@     `~@     |@     �v@     �v@     pt@     Pr@     �q@     @p@     �h@     �j@     @j@     �d@     �d@     �\@      Z@     �_@     �Z@     �S@     �T@     @R@      O@     �L@      M@     �B@     �H@     �F@      B@     �F@      9@      8@     �B@      <@      8@      8@      :@      7@      (@      *@      *@      *@       @      (@      @      $@      @      @      @      @      @       @      @      @       @      @      �?       @      @      @      �?      @              �?      �?              �?      �?      �?               @              �?              �?              �?              �?      �?              �?              �?              �?              �?               @      �?      �?      �?               @      �?               @       @       @      �?       @       @      @      �?       @              @      @      @      @      @      @      @      @      @      @      (@      $@      *@      @      *@       @      5@      &@      (@      0@      6@      2@      3@      ?@      :@     �A@      E@     �@@      ?@      J@     �G@      R@     @R@      P@     �P@     �S@     �U@     �U@     �]@      [@     `a@     �d@      e@     �e@     �f@     �k@      k@      p@     0t@      t@     �v@     `y@     �x@     }@     ��@     ��@     (�@     ؄@     �@     Ȉ@     p�@     ��@     ��@     p�@     �@     ��@     @k@      @        
�
conv4/biases_1*�	   ��L��   ��1�?      p@!^2���?)���^��?2��uS��a���/���}Y�4j���"�uԖ�^�S�������&���#�h/���7c_XY��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r������6�]���1��a˲��FF�G �>�?�s���O�ʗ���})�l a��ߊ4F����(��澢f�����uE���⾮��%�['�?�;;�"�qʾ�u`P+d����n������
L�v�Q�28���FP�������M���8"uH���Ő�;F��`�}6D�/�p`B�p
T~�;�u 5�9��z��6��so쩾4�6NK��2�_"s�$1���1���=��]���        �-���q=�J>2!K�R�>7'_��+/>_"s�$1>6NK��2>����W_>>p��Dp�@>/�p`B>�`�}6D>��8"uH>6��>?�J>Fixі�W>4�j�6Z>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>8K�ߝ�>�h���`�>})�l a�>pz�w�7�>O�ʗ��>>�?�s��>�FF�G ?��[�?6�]��?����?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?��ڋ?�.�?ji6�9�?�S�F !?U�4@@�$?+A�F�&?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?�uS��a�?`��a�8�?�/�*>�?�������:�              �?              �?       @              �?      �?               @      @      @      �?               @               @              @      �?       @       @      @              �?               @       @       @              �?       @       @      �?      @       @      �?      @       @      �?              �?       @      �?      �?              @              �?      �?      �?              �?      @       @              �?               @      �?               @      �?               @       @              �?              �?              �?              �?              �?              �?      �?              �?      �?      �?              @      �?      �?              �?              �?              7@               @              �?      �?              �?              �?              �?               @              �?              �?               @              �?              �?              �?              �?              �?      �?              �?              �?               @               @               @       @              �?      �?      �?              @              @      �?      @      �?              @      �?       @      @      @      @      @      @      @      �?      @              @      �?      @       @      @              �?      @       @      @      �?      @       @      @      �?              �?      @       @              �?              �?              �?      �?        
�
conv5/weights_1*�	   ���ÿ   `���?      �@! ��0I�+@)�|H㚶I@2�	yD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E���%�V6��u�w74���bȬ�0���VlQ.�I�I�)�(�+A�F�&��[^:��"��S�F !�ji6�9����ڋ��vV�R9��5�i}1���d�r�x?�x���[�=�k���*��ڽ�>h�'�?x?�x�?�5�i}1?�T7��?��ڋ?�.�?U�4@@�$?+A�F�&?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�������:�	              �?      j@     @r@     pq@     �n@     `l@     `l@     �h@     @c@     �a@     �b@     �a@     @Z@     �\@      V@      X@     �S@     @R@     �S@     �P@      J@      E@     �A@     �E@     �@@      @@      :@      C@      <@      =@      ,@      ,@      $@      4@      *@      ,@      ,@       @      @      @      @      "@       @      (@      @      @      @      @      @       @       @               @      @               @              �?              �?              �?              @              �?      �?               @              �?      �?              �?              �?              �?              �?              �?              �?      �?              �?              �?              �?               @               @      @              �?       @      �?      @       @      @      @      @      @      @       @      @      @      @      @      @      *@      @      .@      0@      &@      2@       @      &@      0@      4@      9@      6@      9@     �B@      A@      D@     �G@     �G@      I@      N@      J@      O@     @T@      R@     �X@      [@     @Y@      `@     �`@     �b@     `c@      e@     �c@      h@     �n@     �n@     `r@     �t@     `o@      1@      �?        
�
conv5/biases_1*�	   �#)o�    %�b>      <@!  @O"�T>)_���<2�ہkVl�p�w`f���n�H��'ϱS��
L�v�Q���8"uH���Ő�;F��`�}6D�/�p`B�p��Dp�@�����W_>��z��6��so쩾4��mm7&c��`�����-��J�'j��p���
"
�=���X>�=f;H�\Q�=�tO���=�f׽r��=�`��>�mm7&c>RT��+�>���">4�e|�Z#>��o�kJ%>4��evk'>���<�)>7'_��+/>_"s�$1>6NK��2>�so쩾4>u 5�9>p
T~�;>�`�}6D>��Ő�;F>�
L�v�Q>H��'ϱS>��x��U>Fixі�W>4�j�6Z>��u}��\>w&���qa>�����0c>�������:�              �?              �?              �?      �?      �?               @              �?              �?              �?              �?              �?      �?              �?              �?               @              �?              �?       @      �?              �?              �?              �?              �?              �?              �?        Ey�ˈW      �D}	ߦ^�'��A*��

step  �A

loss�z�=
�
conv1/weights_1*�	    [T��   �d/�?     `�@! �6c��)9�]bwO@2�
8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�+A�F�&�U�4@@�$��S�F !�ji6�9���.����d�r�x?�x��>h�'��f�ʜ�7
������6�]����h���`�8K�ߝ뾞[�=�k�>��~���>8K�ߝ�>�h���`�>I��P=�>��Zr[v�>>�?�s��>�FF�G ?f�ʜ�7
?>h�'�?�5�i}1?�T7��?�vV�R9?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?�������:�
               @      @      6@     �O@     �Y@     �d@     @g@     `d@     �c@     �b@     �\@      _@      W@      W@     �V@     @R@      Q@     @P@     �L@     �R@     �M@     �G@     �A@      B@     �B@      ?@      8@      :@      @@      .@      7@      2@      *@      *@       @      *@      $@      @      (@      &@      @      @      @       @      @      @      @      @      @      @      @      �?      @      @      @      @               @       @       @       @      �?      �?      �?               @              �?      �?               @              �?              �?              �?              �?              �?              �?              �?              �?              �?      �?              �?              �?      @      �?      @               @      �?      �?              @      �?               @      �?              �?       @      @      @       @      @      @      @       @      @      @      @       @      0@      @      &@      (@      1@      1@      (@      "@      3@      5@      ;@      6@      4@      7@     �B@      C@     �F@     �@@     �B@     �E@     �L@      L@     @P@     @P@     �Q@     @R@     �T@      ]@      `@     �Y@     �a@     �a@     �b@      d@     �c@     �]@     �G@      @       @        
�
conv1/biases_1*�	   �`%^�    >�?      @@!  ��$��?)�V]���?2�E��{��^��m9�H�[�nK���LQ�k�1^�sO��T���C��!�A�uܬ�@8���%�V6�uܬ�@8?��%>��:?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?&b՞
�u?*QH�x?o��5sz?���T}?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?�v��ab�?�/��?�uS��a�?`��a�8�?�������:�              �?              �?              �?              �?              �?              �?               @              �?      �?       @              �?      �?      �?              @      �?      �?      �?      @               @              �?       @      �?        
�
conv2/weights_1*�	   �_ٱ�   `��?      �@!�v��M@)���!��E@2���]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��})�l a��ߊ4F��h���`���(��澢f�����uE���⾮��%��_�T�l׾��>M|KվK+�E��Ͼ['�?�;G&�$��5�"�g������?�ګ�;9��R���u��gr�>�MZ��K�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�������:�              @      (@      O@     X�@     0�@     ��@     T�@     \�@     T�@     ��@     ̒@     0�@      �@      �@     H�@     ��@     X�@     ��@     Ȃ@     ��@     �z@      z@     0x@     �v@     Pu@      s@     pq@     @m@     `o@     �k@      f@      e@     �d@     �`@     �`@     �`@     @[@     �]@     �X@      Y@     @R@     @P@     �P@     �L@      C@      G@     �D@      @@     �I@      =@     �B@      :@      0@      =@      0@      &@      $@      &@      $@      $@      $@      @      @      "@      &@      @      $@      @      @      @      @      @      @      @       @      @      @      �?       @       @      @      @      @      �?      �?              �?      �?              �?              �?              �?              �?              �?              �?              �?              �?              �?      �?      �?      �?              �?      �?      �?      �?      @      �?      �?      �?       @       @      @       @      @      @      �?       @       @      "@      @      @      @      "@      (@      @      &@       @      *@      &@      ,@      &@      ,@      0@      ,@      8@      A@      =@      <@      ?@      @@     �A@      G@      D@      N@     �K@     �I@     �Q@     �T@      T@      R@      [@     �[@     �]@     �b@     `c@     �d@      g@      g@     @k@     @p@     �p@     0s@      v@     0y@     Py@     �|@     �@     �@     X�@     @�@     ��@     ��@     Ȉ@     ��@     ȏ@     �@     0�@     ,�@     l�@     К@     <�@     �@     ̟@     H�@      q@      C@      @        
�

conv2/biases_1*�		   ��h��   �=)�?      P@!  ��ed�?)+�̧��{?2���<�A���}Y�4j���"�uԖ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�P}���h�Tw��Nof����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�<DKc��T��lDZrS�
����G�a�$��{E��7Kaa+�I�I�)�(�U�4@@�$��[^:��"��T7��?�vV�R9?�.�?ji6�9�?�u�w74?��%�V6?uܬ�@8?��%>��:?IcD���L?k�1^�sO?nK���LQ?ܗ�SsW?��bB�SY?�m9�H�[?�l�P�`?���%��b?5Ucv0ed?P}���h?ߤ�(g%k?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?�v��ab�?�/��?�������:�              �?      �?              �?              �?      �?      �?       @              �?              �?       @       @              �?      �?       @              �?              �?       @              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?      �?              �?      �?              �?      �?              �?              �?      �?       @      �?      �?      �?              @      @      �?               @      �?       @       @               @      �?              �?        
�
conv3/weights_1*�	   ��@��    Tγ?      �@! ��P<F@)Q����U@2���]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(����uE���⾮��%��_�T�l׾��>M|Kվ�[�=�k���*��ڽ�豪}0ڰ��������|�~�>���]���>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?�������:�              2@     0�@     .�@     ��@     \�@     ��@     ء@     N�@     �@     ��@     H�@     ��@     ��@     ԑ@     d�@      �@     ��@     0�@     h�@     �@     �@     ��@     �~@      }@     �x@     �x@      u@     @u@      o@     �o@     �m@      k@     �h@     @d@     �a@     �b@     �a@     �\@     @Z@     �Y@     �U@     @S@     �T@     �S@     �J@     �K@     @P@      D@      @@     �C@      ?@      G@      7@      7@      *@      3@      .@      3@      4@      2@      $@      *@      "@      *@      @      "@      @      &@      @      @      @      @      @       @      �?       @      @       @      �?      �?      �?      �?      @      @              �?      @      �?      �?      @              �?               @              �?              �?              �?              �?              �?              �?              �?               @              �?      �?              �?      �?              �?       @              �?      �?      �?      @      �?      �?      @      �?      @      @      @      @       @       @      @      @              @      @      "@      @       @      "@      $@      4@      &@      3@      1@      <@      3@      9@      0@      4@     �A@      @@      >@     �C@     �E@      K@     �J@     �D@      R@     �Q@     @S@     @V@     @Y@      Z@     @^@     �a@      b@      b@      c@     �h@     �i@     `j@      n@     `r@     pr@     �s@     �x@     �x@     {@     �~@     (�@      �@     ��@     Ѕ@     p�@     �@     ��@      �@     d�@     4�@     �@     @�@     @�@     D�@     �@     ֠@     ޢ@     j�@     &�@     ֨@     �@     @a@      7@      �?        
�
conv3/biases_1*�	   `����   ��E�?      `@!  [�G��?)ĵV9iO�?2���<�A���}Y�4j�����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\��>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T�IcD���L��qU���I�
����G�a�$��{E����#@�d�\D�X=�uܬ�@8���%�V6���VlQ.��7Kaa+��[^:��"��S�F !���ڋ��vV�R9�6�]���1��a˲�        �-���q=��d�r?�5�i}1?I�I�)�(?�7Kaa+?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?�!�A?�T���C?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?�������:�              �?               @      @      �?              @       @              �?      �?      �?              @      @      �?       @      �?       @       @      �?      @      �?      �?      �?               @      �?              �?      �?       @               @               @              �?              �?              �?              �?              @              �?              �?              �?              �?       @              �?              �?              �?               @              �?       @              �?              @       @              @      �?      @      @      �?       @      @      �?      @      @      @      �?      @      �?      @       @               @      �?      �?      �?              �?      �?        
�
conv4/weights_1*�	   @���    ��?      �@!@ƹǍA)�)�]�`e@2�Ӗ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !���ڋ��vV�R9��5�i}1���d�r�>h�'��f�ʜ�7
�������FF�G �>�?�s���I��P=��pz�w�7��a�Ϭ(���(�����>M|Kվ��~]�[Ӿjqs&\�Ѿ�����~�f^��`{���>M|K�>�_�T�l�>��Zr[v�>O�ʗ��>�FF�G ?��[�?1��a˲?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?�������:�             �a@     ��@     �@     �@     h�@     ��@     �@     x�@     H�@     Ѕ@     H�@     0�@     �@      ~@     P|@     �v@     �v@     pt@     `r@     �q@     pp@     �h@     `j@      j@      e@      d@     @]@      Z@     �^@     �[@      S@      V@     @Q@      M@     �P@     �J@     �C@     �F@      G@      B@     �F@      ;@      :@     �@@      <@      9@      9@      8@      4@      1@       @      2@      @       @      2@      @      "@      @      @      @      @      @      @       @      @      @       @      @       @      @      �?      �?       @               @              �?               @      �?              �?              �?               @              �?      �?              �?              �?               @               @      �?              �?               @      �?       @      @      @      @      �?              @      �?      @      �?              @      @      @      @       @      @       @      @      @      @      @      &@      $@      "@      &@      @      ,@      3@      ,@      &@      0@      3@      4@      3@      >@      ?@      C@      @@      C@      >@      J@     �I@     @P@      S@     �N@      Q@      S@      V@     @W@     @\@     �[@     `a@     �d@     �d@     �e@      g@      k@     `k@     p@     @t@     �s@      w@      y@     �x@     }@     ��@     x�@     @�@     ��@     �@     ��@     p�@     ��@     ��@     h�@     �@     ��@      l@      @        
�
conv4/biases_1*�	   ����   �=�?      p@!T[چS?�?)&k-��?2��uS��a���/�����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/��eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x�������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ���})�l a��ߊ4F��a�Ϭ(���(��澢f����K+�E��Ͼ['�?�;�u`P+d����n�����H��'ϱS��
L�v�Q�28���FP�6��>?�J���8"uH���Ő�;F��`�}6D�����W_>�p
T~�;�u 5�9��z��6��so쩾4�6NK��2�_"s�$1�z�����i@4[��        �-���q=���">Z�TA[�>2!K�R�>��R���>_"s�$1>6NK��2>�so쩾4>����W_>>p��Dp�@>�`�}6D>��Ő�;F>6��>?�J>������M>4�j�6Z>��u}��\>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>8K�ߝ�>�h���`�>pz�w�7�>I��P=�>O�ʗ��>>�?�s��>�FF�G ?��[�?x?�x�?��d�r?�5�i}1?�T7��?�.�?ji6�9�?�S�F !?�[^:��"?+A�F�&?I�I�)�(?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?}Y�4j�?��<�A��?�uS��a�?`��a�8�?�/�*>�?�������:�              �?              �?               @              �?      �?               @       @       @       @      �?               @               @              @              @      @      �?      �?              �?       @       @      �?               @      @               @      @      �?      �?      @      �?               @      �?      @              �?               @               @       @              �?      �?      �?       @      �?               @      �?              �?               @              �?               @      �?              �?              �?      �?              �?              �?              �?      �?              �?               @               @       @              �?              �?              �?              7@              �?              �?              �?      �?              �?              �?              �?               @              �?              �?               @              �?              �?              �?               @               @              �?      �?      �?               @               @       @      �?      �?              �?               @      @              @      @       @      �?       @       @       @      �?      @      @      @      @      @       @              @       @      @      @       @      @      �?      �?              @       @       @      @      @      @       @      �?      �?       @      @      �?      �?              �?              �?      �?        
�
conv5/weights_1*�	   �L�ÿ   @N��?      �@! x.�Ph,@)�84�p�I@2�	yD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9���(���>a�Ϭ(�>6�]��?����?�T7��?�vV�R9?��ڋ?�.�?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�������:�	              �?     �i@     Pr@     pq@     �n@      m@     �k@     �h@     �c@     `a@     �b@      b@     @Z@     �\@     �U@     @X@     �S@     @R@     �S@     �P@     �J@     �C@      B@     �E@     �@@      @@      :@      B@      >@      >@      ,@      ,@      $@      .@      0@      .@      0@      @      @      @      @       @      "@      (@      @      @      @      @      @      �?      @      �?      @      @      �?      @              �?      �?              �?              �?      �?              �?               @              �?              �?              �?              �?              �?              �?              �?              �?              �?      �?              �?      �?      �?              �?               @      @       @               @      @      @              @      �?      @      @      @       @      @      @      @      @      @      $@      "@      ,@      3@       @      4@      @      (@      1@      3@      8@      7@      7@      D@      @@     �D@      H@      G@     �I@      M@     �J@     �O@     �S@      R@     �Y@     �Z@     �Y@      `@     �`@     `b@     @c@     `e@     �c@     �g@     �n@     �n@     r@     �t@     �o@      1@      �?        
�
conv5/biases_1*�	    �:p�   `* d>      <@!  P�WV>)��Ɵ�U=2�ہkVl�p�w`f���n�H��'ϱS��
L�v�Q���8"uH���Ő�;F��`�}6D�/�p`B�p��Dp�@�����W_>��z��6��so쩾4�y�+pm��mm7&c���1���=��]����!p/�^�=��.4N�=�K���=�9�e��=�tO���=�f׽r��=�mm7&c>y�+pm>RT��+�>���">4�e|�Z#>��o�kJ%>4��evk'>���<�)>_"s�$1>6NK��2>�so쩾4>p
T~�;>����W_>>��Ő�;F>��8"uH>H��'ϱS>��x��U>Fixі�W>��u}��\>d�V�_>�����0c>cR�k�e>�������:�              �?              �?              �?      �?      �?               @              �?              �?              �?              �?              �?              �?              �?              �?               @              �?              @      �?              �?              �?              �?      �?              �?              �?        ?W���V      	ѹ�	�N͘'��A*��

step  �A

loss��T=
�
conv1/weights_1*�	   �����   �,Z�?     `�@! 0O���)�̵~��@2�	8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��[^:��"��S�F !��vV�R9��T7���>h�'��f�ʜ�7
�O�ʗ�����Zr[v��E��a�Wܾ�iD*L�پjqs&\�ѾK+�E��ϾX$�z��
�}����x?�x�?��d�r?�5�i}1?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?�������:�	               @      @      <@      P@     @Z@     �d@     �f@     `d@     �d@     �a@     @\@      `@     �W@      U@     @X@      R@     �O@      R@     �J@     @R@      M@      G@      C@      E@     �@@      :@      >@      9@      =@      5@      3@      ,@      .@      &@      (@      *@       @      "@      @      @      $@      @      @      @      @      @      @      @      @       @      @      @      �?      @      @      @       @      �?              �?      �?      �?       @               @              �?              �?              �?              �?              �?              �?              �?      �?              �?              �?      �?      @      �?       @       @      @      �?              �?      @       @      @      �?              @       @              �?      @      �?      �?      @       @      @      @      @      $@      "@      "@      &@      5@      .@      0@       @      2@      7@      0@      5@      4@      4@      8@     �C@      ;@      G@     �E@      A@      J@     �I@     �I@      N@      R@     �Q@     @R@     �T@     �^@     �_@     @Y@     �`@      a@     @d@     `c@      c@     �\@     �I@      @       @        
�
conv1/biases_1*�	   �4�Y�   `c��?      @@!  P�^�?)�_aH��?2��m9�H�[���bB�SY�nK���LQ�k�1^�sO��qU���I�
����G���%�V6��u�w74���%>��:?d�\D�X=?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?�v��ab�?�/��?�uS��a�?`��a�8�?�������:�              �?              �?              �?              �?              �?              �?               @               @      �?      �?              �?               @               @      @      �?      �?      �?      @       @              �?      �?       @        
�
conv2/weights_1*�	    O��   ��'�?      �@!P�g�:N@)��X���E@2��{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=���ߊ4F��h���`�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾K+�E��Ͼ['�?�;G&�$��5�"�g���G&�$�>�*��ڽ>�[�=�k�>��~���>�ѩ�-�>���%�>�uE����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?�������:�              �?      @      *@     �Q@     H�@     ̞@     �@     x�@     $�@     (�@     ��@     �@     ��@     0�@     �@     8�@     ��@     p�@     8�@      �@     ��@     @z@     �z@     �w@     `w@     Pu@     �s@     �p@      m@     �n@     �j@     �g@     @d@     �b@     �a@     �a@     �`@     �[@     �\@     �Y@      T@     �V@     �Q@     �P@      E@     �I@      F@     �D@     �A@      F@      ?@      @@      *@      8@      >@      2@      0@      *@      .@      0@      *@      @      @      @      @      @      @       @       @      @      @      @      @      �?      @      @      @      �?       @      @       @       @      �?      �?      �?      �?              �?               @      �?              �?      �?              �?              �?              �?              �?              �?      �?              �?               @              �?      �?       @       @       @      @      �?      �?      @              @       @      �?      @      @      @      @       @      @      @      @      @      @      @      "@      .@      (@      $@      ,@      (@      &@      ,@      0@      0@      4@      A@      3@      ?@      <@      >@      H@      G@     �D@      M@     �J@      K@     �Q@     @S@      U@     @R@     �[@      Y@      _@     `b@     �c@      f@     `g@      e@      k@     �p@     Pp@      s@     �u@     0z@     px@     p}@     P~@     (�@     h�@     @�@     ��@     ��@     Ј@     x�@     $�@     ��@     ��@     ܔ@     x�@     ܚ@     ,�@     �@     �@     D�@     �r@      E@      @      �?        
�	
conv2/biases_1*�		   ���   �J�?      P@!  P��w�?)|��7!�}?2���<�A���}Y�4j���"�uԖ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`��m9�H�[���bB�SY�<DKc��T��lDZrS��qU���I�
����G��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��S�F !?�[^:��"?��bȬ�0?��82?��%�V6?uܬ�@8?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?ߤ�(g%k?�N�W�m?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?�v��ab�?�/��?�������:�              �?      �?              �?               @               @      �?              �?               @       @      �?      �?               @      �?              �?              �?       @              �?              �?              �?              �?              �?               @              �?              �?              �?      �?              �?              �?              �?              �?              �?               @      �?      �?      �?       @              �?      @      @               @               @      @              �?       @              �?        
�
conv3/weights_1*�	   �X]��   @��?      �@!�Ƅ��LG@)�$�)�U@2���]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��})�l a��ߊ4F��h���`�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ��~��¾�[�=�k���*��ڽ�G&�$����z!�?��T�L<��K���7��[#=�؏���u`P+d�>0�6�/n�>��~]�[�>��>M|K�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?�������:�              6@      �@     �@     �@     J�@     ��@     �@     H�@     �@     Ԛ@     `�@     x�@     ܓ@     ��@     ��@     Ѝ@     ��@      �@     p�@     ��@     �@     Ѐ@     p~@     }@     �y@     �w@     pu@     �u@      n@     p@     �m@     �k@     `h@     �b@     �c@     �a@     `c@     �Y@      Z@     �Z@     �V@     @R@     @U@     �Q@     �L@     �O@     �O@      C@     �B@     �B@      ;@     �@@      >@      :@      8@      *@      .@      :@      ,@      (@      (@      *@      "@      &@      &@      @      @       @      @      @       @      @       @      @      @      �?      @       @      �?      �?              @               @              @      �?              �?      @              �?              �?              �?              �?              �?              �?              �?              �?               @              �?      �?               @      �?       @      �?      @      �?      �?      @      @       @      @      @       @       @      @      .@      @      @      @      @      @      @      @      "@      $@      @      (@      ,@      &@      $@      4@      5@      5@      8@      8@      <@      9@      D@      @@     �@@     �J@      L@     �G@      H@     @Q@     �S@     �S@     �U@      Y@     �X@     @]@     �`@     @c@     `b@     �c@      g@     �k@      i@     `m@      s@     �r@     �r@     @x@     �y@     pz@     �~@     X�@     �@     ��@     ��@     P�@     �@     ��@     ��@     l�@     P�@     ؔ@     ��@     <�@     ��@      �@     Ƞ@     ��@     N�@     6�@     Ĩ@     P�@     `b@      9@       @        
�
conv3/biases_1*�	    K���   ��/�?      `@!  �,x��?)u�����?2��v��ab����<�A����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=�uܬ�@8���%�V6���VlQ.��7Kaa+���ڋ��vV�R9���d�r�x?�x��        �-���q=�5�i}1?�T7��?�[^:��"?U�4@@�$?��bȬ�0?��82?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�T���C?a�$��{E?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?�������:�              �?              �?      �?      @      �?      @      �?       @              �?               @              �?      @       @      @              �?      @      �?      �?       @      �?      �?      �?      �?       @              �?      �?      �?      �?              �?       @              �?               @              �?              �?              @              �?              �?               @              �?      �?              �?              �?              �?      �?      �?      �?      �?      �?              �?      �?      @      �?              @       @       @      @      @              @       @      @      @      @       @       @      @       @      �?       @      �?               @              �?      �?        
�
conv4/weights_1*�	    ���   ��#�?      �@! ��N!(�)�*6�r�e@2�Ӗ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"�ji6�9���.����ڋ��vV�R9��5�i}1���d�r�x?�x��f�ʜ�7
������6�]���1��a˲���[��I��P=��pz�w�7��
�/eq
Ⱦ����ž0�6�/n���u`P+d��0�6�/n�>5�"�g��>�����>
�/eq
�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?�������:�              b@     ��@     �@     $�@     h�@     ��@     �@     ��@     H�@     ��@     H�@     8�@     p@      ~@     @|@     �v@     �v@     �t@     @r@     �q@     pp@     @i@     �i@     �i@     @e@      d@     �^@      Y@     �]@     �\@     �R@     @U@     @R@      O@      N@     �K@      C@      H@      F@     �C@     �B@      <@      @@      >@      <@      =@      5@      9@      6@      $@      .@      $@      $@      $@      0@      @      $@       @      @      @       @      "@       @      @      @      @       @      @      @      @               @              �?              �?              �?      �?              �?      �?               @              �?              �?              �?              �?              �?               @      �?       @              �?              �?       @               @      �?      �?      �?      �?      �?       @      �?               @              @      �?      �?      �?      @      @      @       @      @      @      @      $@      @      "@      &@       @      @      @      &@      @      5@      0@      1@      ,@      2@      8@      4@      7@      C@      A@     �A@      ?@     �@@      K@     �I@      P@     @S@      M@     �Q@     �R@      U@     @X@     �\@     �\@     �`@     �d@     �d@     @e@      g@     �j@     @l@      o@     �t@      t@     �v@     y@      y@     }@     ��@     x�@     @�@     ��@     �@     ��@     0�@      �@     ��@     h�@     �@     ��@     �l@      @        
�
conv4/biases_1*�	   �Wբ�   `�F�?      p@!��J��?)���̒�?2�`��a�8���uS��a����<�A���}Y�4j���"�uԖ��Rc�ݒ����&���#�h/��eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.���vV�R9��T7����5�i}1�f�ʜ�7
������6�]���1��a˲��FF�G �>�?�s���pz�w�7��})�l a�a�Ϭ(���(���K+�E��Ͼ['�?�;0�6�/n���u`P+d��H��'ϱS��
L�v�Q�28���FP�6��>?�J���8"uH���Ő�;F�����W_>�p
T~�;�u 5�9��z��6�6NK��2�_"s�$1����6���Į#���        �-���q=RT��+�>���">��R���>Łt�=	>7'_��+/>_"s�$1>6NK��2>�so쩾4>����W_>>p��Dp�@>�`�}6D>��Ő�;F>6��>?�J>������M>4�j�6Z>��u}��\>d�V�_>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>�h���`�>�ߊ4F��>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>��[�?1��a˲?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?��<�A��?�v��ab�?`��a�8�?�/�*>�?�g���w�?�������:�              �?              �?       @              �?      �?               @      @       @       @               @               @               @       @       @      �?      @              �?               @       @       @              @      @      �?       @       @      �?      @      �?               @       @      �?      �?              �?      @              �?      �?      �?               @              @      �?               @      �?               @      �?      �?              @              �?               @              �?              �?              �?      �?              �?       @              @      �?      �?              �?              �?              7@              �?              �?              �?              �?              �?              �?              �?              �?      �?              �?              �?               @              �?              �?              �?              �?      �?              �?              �?      �?      �?      �?              �?      �?               @       @      �?      �?              �?      �?              @              @       @       @       @      �?      @      �?       @      @      @      @      @      @       @      �?       @      �?      @              @       @      @              �?      @      @       @      �?      @       @      @      �?              �?      @       @              �?              �?              �?      �?        
�
conv5/weights_1*�	   ���ÿ   ����?      �@! ̸S{�,@)t��(�I@2�	yD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0��7Kaa+�I�I�)�(�+A�F�&��[^:��"��S�F !���ڋ��vV�R9������6�]���>�?�s��>�FF�G ?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?�S�F !?�[^:��"?�7Kaa+?��VlQ.?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�������:�	              @     �i@      r@     pq@     �n@     �l@      l@     �h@     `c@     `a@     �b@      b@      [@     �[@     @V@     �W@     �R@     @S@     �S@     �P@      K@      C@     �B@      E@     �A@      @@      :@     �@@     �@@      <@      .@      .@      (@      .@      ,@      (@      1@       @      @      @      @      $@      "@      &@       @      "@      @      @      @       @       @      �?      �?      @       @      @              �?      �?              �?              �?              �?              �?      �?               @              �?              �?              �?              �?              �?       @              �?              �?              �?              �?      �?      �?              �?              @      @              �?      @      �?      �?      @      @              @      @      @      @       @      @      "@      @       @      @      &@      (@      4@       @      6@      @      "@      3@      5@      9@      6@      4@      D@      A@     �C@     �G@      H@      J@      M@     �J@      P@     �S@     @R@     �Y@     @Z@      Z@     �_@     �`@     `b@     �b@     �e@     `c@      h@     �n@     �n@      r@     �t@     �o@      2@      �?        
�
conv5/biases_1*�	   �*�p�   ��e>      <@!  �9Q�W>)�H�F��=2�ہkVl�p�w`f���n���x��U�H��'ϱS�6��>?�J���8"uH���Ő�;F��`�}6D�/�p`B�p��Dp�@�u 5�9��z��6�RT��+��y�+pm��/�4��ݟ��uy�K?�\���=�b1��=�/�4��==��]���=�f׽r��=nx6�X� >y�+pm>RT��+�>���">4�e|�Z#>��o�kJ%>4��evk'>���<�)>�'v�V,>_"s�$1>6NK��2>�so쩾4>p
T~�;>����W_>>��8"uH>6��>?�J>H��'ϱS>��x��U>Fixі�W>4�j�6Z>d�V�_>w&���qa>�����0c>cR�k�e>�������:�              �?              �?              �?               @               @              �?              �?              �?              �?              �?              �?              �?      �?              �?      �?              �?              �?      @              �?              �?              �?              �?              �?              �?        ��H�V      �ž2	*�B�'��A*��

step  �A

loss��=
�
conv1/weights_1*�	   `e��   `���?     `�@! �Q���)�gى��@2�	8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ�>h�'��f�ʜ�7
�������ߊ4F��h���`f�����uE����1��a˲?6�]��?f�ʜ�7
?>h�'�?x?�x�?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?��VlQ.?��bȬ�0?��82?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?�������:�	               @      "@      >@     �P@     @[@     �c@     �g@     `d@     �c@     �a@      \@     �`@     �W@      V@      X@     �P@      Q@     �Q@      L@     �Q@      K@      H@      E@      D@      >@      :@      <@      <@      ?@      5@      1@      0@      &@      1@      .@       @      @       @      @      &@       @      @      @      @      *@      @       @      @      @       @      @       @      @       @      @       @      �?      �?               @              �?       @              �?      �?              �?              @      �?              �?      �?              �?              �?              �?              �?       @               @              �?              �?              @      �?       @               @      �?               @      �?      �?      �?       @       @      @              �?      @      @      @      @      @      @      @      @      &@      $@      *@      ,@      ,@      (@      2@      .@      3@      2@      0@      3@      3@      0@      >@     �E@      =@     �C@     �D@     �D@      G@      J@     �G@     �L@     �S@     �N@     �S@      W@     @]@     @]@     �\@      _@      b@     �c@     �b@      c@      ]@      K@       @      @        
�
conv1/biases_1*�	   �ɷW�    "�?      @@!  x*�O�?)FЮ7w�?2���bB�SY�ܗ�SsW�nK���LQ�k�1^�sO�IcD���L��u�w74���82���%>��:?d�\D�X=?5Ucv0ed?Tw��Nof?P}���h?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�/��?�uS��a�?`��a�8�?�/�*>�?�������:�              �?              �?      �?              �?              �?              �?       @              �?       @              �?      �?              �?      �?      �?      @              �?      �?      @               @              �?       @      �?        
�
conv2/weights_1*�	   @ge��   @�k�?      �@!�~�9�XO@)f�i�ޠE@2��{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��8K�ߝ�a�Ϭ(龙ѩ�-߾E��a�Wܾ�iD*L�پjqs&\�ѾK+�E��Ͼ����ž�XQ�þu��6
�>T�L<�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>�����>
�/eq
�>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?�������:�              �?      @      .@     �Q@     H�@     x�@     (�@     ��@     ܘ@     8�@     �@     ��@     ��@     D�@     ��@     8�@     @�@     �@      �@     �@     Ȁ@     z@     @{@     w@     `w@     `u@     �s@     @q@     �l@     �m@      l@     `g@      f@      b@     �a@     @]@      b@      ^@     �Y@      Z@     �S@     �X@      Q@     �P@      F@     �H@      H@     �B@     �@@     �@@      D@      @@      4@      6@      :@      3@      (@      5@      (@      ,@      1@       @      @      @      @      @       @      "@      @      @      @      @      �?       @      @      �?      @      @      @      �?       @               @       @      �?       @      �?       @       @              �?              �?      �?              �?              �?              �?              �?              �?               @              �?              �?              �?      �?      �?       @       @       @      @      @      �?      @       @      @       @      @      @      @      @       @      @      @      @      @      "@      "@      @      .@      "@      *@      ,@      ,@       @      5@      2@      5@      :@      5@      A@      ;@      ?@      C@     �E@      P@      K@      N@     �L@      M@     �R@     �T@     �T@      Z@     �Y@     �`@     �a@     `b@     �e@     �g@     `f@     �j@     @p@     �p@     �r@     v@     Py@     �x@     P}@     0~@     ��@     h�@     Ȅ@     `�@     8�@     @�@     ��@     $�@     ̑@     ��@     Ĕ@     ��@     К@     �@     Ğ@     �@     d�@     �s@      G@      @      �?        
�	
conv2/biases_1*�		   `����   �9 �?      P@!  �+�?�?)t���>�?2���<�A���}Y�4j���Rc�ݒ����&���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`��m9�H�[���bB�SY�<DKc��T��lDZrS��qU���I�
����G��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$?+A�F�&?I�I�)�(?uܬ�@8?��%>��:?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?ߤ�(g%k?�N�W�m?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?�/��?�uS��a�?�������:�               @              �?               @      �?       @              �?              �?      �?      @              �?              @              �?               @      �?              �?              �?              �?              �?      �?               @      �?              �?              �?      �?      �?              �?              �?              �?              �?              �?      �?       @      �?      �?      �?              @      @      �?      �?      �?      �?       @       @      �?      �?      �?              �?        
�
conv3/weights_1*�	   �oy��    Y^�?      �@! ��Cu�H@)�S�q�U@2���]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾�iD*L�پ�_�T�l׾�[�=�k���*��ڽ���|�~���MZ��K����n����>�u`P+d�>�*��ڽ>�[�=�k�>['�?��>K+�E���>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?�������:�              8@      �@      �@     �@     8�@     z�@     �@     H�@     �@     Ț@     p�@     l�@     �@     ��@     p�@     ��@     ��@     ��@     ��@     ��@     Ё@     �@      ~@     �}@     �x@     �x@     Pu@     �u@     �o@      o@     �l@     �l@     �h@     `b@     �a@     @b@     �b@      ]@     �Y@     �Y@     �U@     �R@     @S@     �S@      M@      K@      L@     �G@     �@@      D@      ?@     �A@      ;@      4@      7@      >@      .@      0@      &@      *@      &@      0@      ,@      "@      @      @      @      @      @      @      @      @      @      @      @      @      @      �?      @      �?       @      �?      @      �?       @      @              �?              �?               @              @       @      �?              �?              �?              �?              �?              �?               @              �?      @      �?              �?              �?      �?              @      �?      �?      �?              @      �?      @      �?      �?              @       @       @      @      @       @       @      @      �?       @       @      @      @      &@      @       @      @      .@      (@      ,@      3@      0@      *@      7@      2@      =@     �@@     �@@      D@      E@     �G@     �J@      J@      M@      K@     @V@      T@      V@     �W@     @Z@     @\@     �`@     @b@     @a@     �d@     �g@     �l@     �g@     �m@     �r@     �r@     �s@     �w@     �y@     `z@     �@     �@     �@     Є@     ��@     `�@     ȉ@     ��@     ,�@     t�@     ,�@     �@     ��@     H�@     �@      �@     ��@      �@     @�@     6�@     Ш@     `�@     �d@      9@      @        
�
conv3/biases_1*�	   @����   ���?      `@!  �O��?)�\��/�?2��v��ab����<�A����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=�uܬ�@8���%�V6���82���bȬ�0��7Kaa+�I�I�)�(���ڋ��vV�R9�        �-���q=�T7��?�vV�R9?��ڋ?�7Kaa+?��VlQ.?��bȬ�0?�u�w74?��%�V6?uܬ�@8?��%>��:?�T���C?a�$��{E?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�������:�              �?              �?      @      �?      �?      @       @              �?      �?      �?              @      @              @      �?       @       @       @       @      �?      �?              �?      �?      �?      �?              �?       @              �?       @              �?              �?              �?              �?               @              @              �?      �?              �?      �?              �?              �?               @               @      �?      �?               @              �?              @       @              @      @       @      @       @      �?      @       @       @      @      @              @              @       @               @      �?      �?      �?              �?      �?        
�
conv4/weights_1*�	    �/��   ��D�?      �@!@�Dd�&�)�	�y��e@2�Ӗ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r������6�]���I��P=��pz�w�7���f�����uE���⾙ѩ�-߾E��a�Wܾ�*��ڽ�G&�$���ѩ�-�>���%�>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>��[�?1��a˲?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?�������:�             @b@     ��@     �@     0�@     L�@     ��@     �@     ��@     P�@     ��@     H�@     (�@     �@      ~@     |@     �v@     `v@     �t@     @r@     pq@     �p@     �i@     �i@     �i@     �d@     �d@      ^@      Y@     �]@      \@     �S@     �U@     �Q@     �N@     �L@      M@      D@      J@      D@     �B@     �B@      <@     �@@      =@      =@      ;@      7@      <@      3@      ,@      @      .@      "@      "@      (@      @       @      @      @      @      $@       @      @       @      @      @       @       @      @      @      �?      @      �?      �?      �?              �?       @      �?      �?               @              �?              �?              �?              �?              �?              �?      �?              �?              �?              �?      �?      @      �?              �?      �?      �?               @      @       @      @      @       @      �?       @       @      @       @      @      @      @       @      (@      @      *@       @      @      @      (@      "@      ,@      1@      1@      2@      2@      7@      5@      ;@      A@     �B@      @@      ?@     �@@     �J@      L@     �O@     �Q@     �N@      R@      S@      T@     @Y@     �[@     @]@     ``@     `d@     �e@     �d@     �f@     @k@      l@     �n@     �t@     0t@     pv@     0y@      y@      }@     ��@     h�@     X�@     ��@     �@      �@     (�@     �@     ��@     h�@     �@     ��@     `m@      @        
�
conv4/biases_1*�	    ����    ^M�?      p@!!uL7���?)k��|j(�?2�`��a�8���uS��a����<�A���}Y�4j���"�uԖ��Rc�ݒ����&��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G��T���C��!�A����#@�d�\D�X=���%>��:���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.���vV�R9��T7����5�i}1�f�ʜ�7
������6�]���1��a˲��FF�G �>�?�s���O�ʗ���pz�w�7��})�l a�a�Ϭ(���(��澢f����K+�E��Ͼ['�?�;0�6�/n���u`P+d����x��U�H��'ϱS��
L�v�Q�6��>?�J���8"uH���Ő�;F�p��Dp�@�����W_>�p
T~�;�u 5�9��so쩾4�6NK��2�        �-���q=K?�\���=�b1��=�`��>�mm7&c>Łt�=	>��f��p>_"s�$1>6NK��2>�so쩾4>�z��6>p��Dp�@>/�p`B>�`�}6D>��Ő�;F>6��>?�J>������M>4�j�6Z>��u}��\>d�V�_>��>M|K�>�_�T�l�>�iD*L��>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��[�?1��a˲?>h�'�?x?�x�?�5�i}1?�T7��?�vV�R9?ji6�9�?�S�F !?�[^:��"?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?��<�A��?�v��ab�?`��a�8�?�/�*>�?�g���w�?�������:�              �?              �?       @               @               @      �?      @       @      �?               @               @      �?       @       @       @      @      �?              �?               @       @       @              �?      @               @       @      �?      @      @              �?      �?      �?      @              �?               @              �?      �?      �?      �?       @       @       @      �?              �?      �?               @      �?      �?               @      �?              �?              �?      �?              �?              �?              �?      �?              �?       @               @       @      �?              �?              7@              �?              �?              �?              �?              �?              �?              �?              �?              �?      �?              �?      �?              �?       @              �?              �?              �?              �?      �?              �?       @              �?       @              @       @      �?              �?      �?               @      �?              @      @      �?              �?      @      �?      �?      @      @      @      @      @       @              @      @       @       @      @      @              �?      �?      @      @      �?      @      @      @       @      �?      �?      @      @      �?      �?              �?              �?      �?        
�
conv5/weights_1*�	   ��Ŀ    d0�?      �@! ��U�a-@)����I@2�	yD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��!�A����#@���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(��[^:��"��S�F !�ji6�9����ڋ��vV�R9��T7���pz�w�7��})�l a�})�l a�>pz�w�7�>O�ʗ��>>�?�s��>��ڋ?�.�?ji6�9�?+A�F�&?I�I�)�(?�7Kaa+?��82?�u�w74?��%�V6?uܬ�@8?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�������:�	              @     �i@      r@     �q@     �n@     �l@     �k@     �h@     �c@     �a@     �b@     @b@     �Z@     �[@     �V@     @W@      S@     @S@     �S@      O@     �L@     �C@      B@      E@      B@      >@      9@      A@      A@      >@      (@      0@      (@      .@      *@      &@      1@      @      "@      @       @      @      $@      (@      @      @      @      @      @      @       @               @      @      �?       @              �?              �?               @      �?      �?              �?      �?              �?      �?              �?      �?              �?              �?              �?               @       @              �?      �?              �?              �?               @       @      @       @      �?       @       @      @       @      �?      @      �?      @      @      "@      @      @      @      @      @      "@       @      @      *@      4@      $@      5@      @      "@      3@      4@      ;@      7@      1@     �D@     �A@      B@     �H@     �H@     �H@     �N@      J@     @P@     �S@      R@     �Y@     �Z@      Z@     @_@     �`@     @b@      c@     �e@     @c@     `h@     �n@      n@     Pr@     �t@     �o@      3@      �?        
�
conv5/biases_1*�	   �N�q�   �	1f>      <@!  `&VY>)J3NA�=2�BvŐ�r�ہkVl�p���x��U�H��'ϱS�6��>?�J���8"uH���Ő�;F��`�}6D�/�p`B�p��Dp�@�u 5�9��z��6�RT��+��y�+pm�z�����i@4[��5%���=�Bb�!�=i@4[��=z�����=nx6�X� >�`��>RT��+�>���">4�e|�Z#>��o�kJ%>4��evk'>���<�)>�'v�V,>_"s�$1>6NK��2>�so쩾4>�z��6>����W_>>p��Dp�@>��8"uH>6��>?�J>��x��U>Fixі�W>4�j�6Z>d�V�_>w&���qa>cR�k�e>:�AC)8g>�������:�              �?              �?              �?      �?      �?               @              �?              �?              �?              �?              �?              �?               @              �?      �?              �?              �?       @      �?              �?              �?              �?      �?              �?              �?        ����W      s�	c쯛'��A*��

step  �A

loss<��=
�
conv1/weights_1*�	   ��f��    �ڴ?     `�@!  ����)�� �"�@2�	8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1�x?�x��>h�'��f�ʜ�7
���(��澢f������>M|Kվ��~]�[Ӿ�ѩ�-�>���%�>�ߊ4F��>})�l a�>����?f�ʜ�7
?x?�x�?��d�r?�.�?ji6�9�?�S�F !?�[^:��"?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?�������:�	               @      &@      B@     �P@     �[@     �c@      h@      d@     `c@     �a@      ^@     �_@     �X@     �U@     @V@      Q@     �P@     @R@     �M@     @P@      M@     �G@      F@      @@      A@      ;@      ?@      ;@      :@      8@      4@      ,@      (@      2@      &@      @      @       @      "@      @      "@      @      "@      @      @      @      @      @      @              "@      @      @      �?       @              @       @       @       @               @      �?      �?              �?       @              �?              �?      �?              �?      �?              �?              �?              �?              �?               @               @              �?              �?               @      �?      �?      �?      �?              �?       @      �?      @       @       @      �?       @              @      @      @      @      @      @      &@      "@      ,@      *@      @      (@      1@      ,@      9@      0@      .@      ,@      6@      0@      5@      @@     �@@     �A@      @@     �C@     �F@     �H@      I@     �I@      K@      R@     @P@     @U@     �V@     @]@     �\@     �\@     �]@     `b@     @c@     `c@     @b@     @]@     �M@       @      @        
�
conv1/biases_1*�	    {�U�    ��?      @@!  hS��?)~�>���?2�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L���82���bȬ�0�d�\D�X=?���#@?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�/��?�uS��a�?`��a�8�?�/�*>�?�������:�              �?              �?              �?              �?              �?              �?              �?      �?              �?       @      �?              �?               @              @      �?      �?      �?       @       @       @              �?      �?       @        
�
conv2/weights_1*�	    >���   `���?      �@!@�9W5P@)Cz��4�E@2��{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a�a�Ϭ(���(���E��a�Wܾ�iD*L�پ['�?�;;�"�qʾ�XQ�þ��~��¾��������?�ګ�0�6�/n�>5�"�g��>K+�E���>jqs&\��>��~]�[�>�uE����>�f����>��(���>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?�������:�              �?      @      1@     @S@     h�@     $�@     �@     ��@     ��@     $�@     <�@     ��@     L�@     ��@     ��@     0�@     �@     �@     �@     ��@     ��@     @z@     @{@     �v@     �w@     �u@     @s@     Pq@      m@     `k@     �l@      h@     `e@      c@      b@      _@     �_@     @\@     �^@     @X@      U@     �T@     @T@      O@      F@      H@     �H@      G@      C@     �A@      1@      A@      ;@      6@      6@      6@      &@      1@      "@      ,@      ,@      $@       @       @      $@       @      "@      $@      @      @      @      @      @      @      @      @              @      @      @      @              @       @      �?      @      �?      �?              �?               @              �?              �?              �?              �?              �?      �?              �?      �?              �?       @       @      �?       @       @      @      �?      �?      @      @       @       @      @              @      @      @      @      @      $@      "@      @      @      "@      ,@      "@      (@      1@      3@      0@      5@      1@      @@      1@      :@      9@      =@      F@     �B@     �L@      O@     @Q@      H@      L@     �U@      U@      S@     �X@     �^@     �\@      d@     �`@     �d@     �h@     �e@     �j@     @o@     0q@     s@     �u@     �y@     y@     |@     �@     �@     ؁@     ��@     h�@     �@      �@      �@     \�@     ��@     ��@     Ȕ@     ��@     �@     �@     ̞@     �@     p�@     �t@      H@      (@      �?        
�	
conv2/biases_1*�		    ����    d��?      P@!   �.�?)����X��?2��v��ab����<�A���}Y�4j���Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��qU���I�
����G��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$������>
�/eq
�>I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��%>��:?d�\D�X=?<DKc��T?ܗ�SsW?�m9�H�[?E��{��^?���%��b?5Ucv0ed?Tw��Nof?P}���h?�N�W�m?;8�clp?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?�/��?�uS��a�?�������:�              �?      �?              �?               @              �?       @              �?              �?       @       @      �?              �?       @              �?              �?       @              �?              �?              �?              �?              �?              �?              �?              �?              �?               @               @              �?              �?              �?               @      �?              @              �?      �?      @       @               @               @      @      �?      �?      �?              �?        
�
conv3/weights_1*�	   �>���   �_��?      �@!��3�E�I@)�d8�ǢU@2���]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾�iD*L�پ�_�T�l׾��~]�[Ӿjqs&\�Ѿ�XQ�þ��~��¾�u��gr�>�MZ��K�>����>豪}0ڰ>�[�=�k�>��~���>�XQ��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>���%�>�uE����>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?�������:�              <@     (�@     �@     �@     *�@     ��@     ܡ@     D�@     ��@     �@     l�@     0�@     0�@     ��@     ��@     X�@     ��@     8�@     ��@     h�@     ��@     (�@     �~@      |@     z@      x@     �u@     `u@     p@     �n@     @m@     @l@     �h@     �b@     @`@      d@     �b@     �Z@      Z@     @[@     �S@     @T@      S@     @Q@     �L@     �K@     @P@      A@     �A@     �@@      B@      A@      ;@      5@      2@      9@      3@      2@      &@      1@      1@      *@      &@      @      @      @      @      $@      @       @       @      @      @      @       @       @      @      �?       @               @       @       @      �?      �?      �?              @       @      �?      �?      �?      �?              �?       @              �?              �?              �?              �?              �?              �?      �?              �?              �?      �?       @      �?              �?              �?       @      �?       @      @      @       @       @               @      @              @      @      @      @      @      @      "@      @      @      @      @      @      *@      @      $@      7@      ,@      ,@      .@      0@      *@      6@      4@      4@     �@@      B@     �A@     �D@      N@      F@      L@     �L@      P@     �Q@     �V@     �T@      Y@     @Z@     �\@      a@     @a@      a@      e@     @h@     @k@     �i@     �n@     pr@     �q@     �s@     �w@     �y@     `z@     �@     �@     H�@     x�@     ��@     H�@     ��@     Ѝ@     �@     ��@     ��@     (�@     d�@     p�@     ̛@      �@     ��@     �@     D�@     8�@     ��@     ��@      g@      <@      @        
�
conv3/biases_1*�	   @���   ����?      `@!  �J`��?)i�Ǎ`R�?2��/����v��ab���Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�uܬ�@8���%�V6��u�w74�I�I�)�(�+A�F�&�ji6�9���.����ڋ��vV�R9�        �-���q=>h�'�?x?�x�?�vV�R9?��ڋ?+A�F�&?I�I�)�(?�7Kaa+?��82?�u�w74?��%>��:?d�\D�X=?a�$��{E?
����G?�qU���I?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�������:�              �?               @      @      �?       @       @       @              �?       @              �?      @      @      @               @      �?      @               @      �?      �?      �?               @              �?              �?       @      �?      �?      �?      �?              �?      �?              �?              �?              �?              @              �?              �?              �?      �?              �?              �?              �?      �?               @               @              �?      �?              �?              �?       @       @      @       @      @      @      @      �?      �?      @      @      @      @       @       @       @      @      �?      �?       @      �?               @              �?      �?        
�
conv4/weights_1*�	   ��B��   `f�?      �@! Q1ce�%�)Z��蓼e@2�Ӗ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲�I��P=��pz�w�7��})�l a��ߊ4F��jqs&\�ѾK+�E��Ͼ�XQ�þ��~��¾���%�>�uE����>8K�ߝ�>�h���`�>�ߊ4F��>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�vV�R9?��ڋ?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?�������:�             �b@     ��@     �@     P�@     4�@     ��@     �@     ��@     `�@     ��@     P�@     @�@     p@     �}@     0|@     �v@     `v@     �t@     0r@     Pq@     �p@     �i@      j@      i@      e@     �d@      _@     �W@     @]@     �\@     �S@     @U@     @Q@     �O@      L@      N@     �C@      I@      D@      E@      =@     �A@      ?@      >@      ;@      9@      <@      7@      5@      .@      "@      "@      &@      $@      .@      @      "@      @      @      @       @      "@      @      @       @      @       @      @      @       @      �?      �?      �?              �?      �?              �?      �?              �?      �?      �?              �?              �?              �?              �?              �?              �?      �?              �?              �?              �?      �?       @       @       @      �?       @               @              �?      �?       @      @      �?      �?      @      @       @      @      @       @      @      @      @      @      @      1@       @      @      &@      @      @      2@      &@      .@      3@      1@      7@      @@      6@      B@     �@@      @@      @@      A@     �J@     �L@     �N@     @Q@      N@     @R@     �R@     �T@     @X@     �\@      ]@     �`@     �d@      f@      d@      g@      k@     �k@     �n@      u@      t@     `v@      y@     �x@     @}@     x�@     ��@     H�@     ��@      �@     �@     �@     ��@     ��@     x�@     ��@     ��@     �m@      @        
�
conv4/biases_1*�	   �;V��   �@Q�?      p@!~�h�؞�?)p9�W�?2��/�*>��`��a�8���v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9����ڋ��vV�R9��T7���>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��pz�w�7��})�l a�8K�ߝ�a�Ϭ(龮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پjqs&\�ѾK+�E��Ͼ0�6�/n���u`P+d����x��U�H��'ϱS��
L�v�Q�������M�6��>?�J���8"uH���Ő�;F�p��Dp�@�����W_>�p
T~�;�u 5�9��so쩾4�6NK��2�        �-���q=�/�4��==��]���=�mm7&c>y�+pm>Łt�=	>��f��p>_"s�$1>6NK��2>�so쩾4>�z��6>p��Dp�@>/�p`B>��Ő�;F>��8"uH>������M>28���FP>��u}��\>d�V�_>w&���qa>�
�%W�>���m!#�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ߊ4F��>})�l a�>I��P=�>��Zr[v�>1��a˲?6�]��?>h�'�?x?�x�?��d�r?�5�i}1?�vV�R9?��ڋ?ji6�9�?�S�F !?�[^:��"?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?�v��ab�?�/��?�/�*>�?�g���w�?���g��?�������:�              �?              �?      �?      �?              �?      �?               @      @       @      �?      �?       @               @              �?       @       @       @      @              �?               @       @       @              �?      �?      @      �?      �?      @      �?      @      �?              �?      �?       @               @      �?               @              �?               @      �?      �?      �?      @      �?              �?      �?              �?      �?      �?              �?      �?      �?              �?              �?              �?              �?              �?              �?              �?              �?      �?              �?      �?      �?              @      �?      �?              �?              7@              �?              �?              �?              �?              �?              �?              �?              �?              �?      �?              �?              �?              �?               @              �?              �?              �?              �?              �?              �?       @              �?              �?       @      @      @              �?              �?       @               @      @      @      �?      �?      @      @      �?      @      @      @      @      @       @              �?      @      @              @      @       @      �?               @      @       @       @      @      @      @      �?               @      @      �?              �?              �?              �?      �?        
�
conv5/weights_1*�	   `-Ŀ   @d_�?      �@! .%o�-@)<�O��I@2�	yD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(��[^:��"��S�F !��.����ڋ��vV�R9���d�r�x?�x���4[_>��>
�}���>�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?+A�F�&?I�I�)�(?�7Kaa+?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�������:�	              @     �i@     0r@     �q@     �n@      m@     �k@     �h@     �c@     �a@     �b@     @b@     �Y@     @\@     �V@     �V@     @T@     @S@     @S@     �N@     �L@      D@      B@      D@     �A@      ?@      ;@      B@     �@@      :@      ,@      2@      *@      *@      (@      *@      (@      $@      "@      @       @       @       @       @      @      @      @      @      @      @      �?       @       @      @              �?      �?      �?      �?              �?               @              �?      �?              �?              �?              �?      �?              �?              �?              �?      @              �?              �?      �?              �?      �?              �?               @      �?      @      @      �?      @       @       @       @              @      @      �?      @      @      @              @      @      @      "@       @      @      ,@      .@      ,@      6@      @      @      7@      1@      8@      <@      2@      C@      B@     �@@     �J@      I@     �G@     �N@      L@      N@     @T@     @Q@     @Y@     @[@     �Y@     @_@     �`@      b@      c@      f@     `c@      h@     �n@     �m@     pr@     �t@     �o@      4@      �?        
�
conv5/biases_1*�	   �pQr�   ��Jg>      <@!  �43Z>)��g,��=2�BvŐ�r�ہkVl�p�Fixі�W���x��U�6��>?�J���8"uH���Ő�;F��`�}6D�/�p`B�p
T~�;�u 5�9����"�RT��+��PæҭUݽH����ڽ|_�@V5�=<QGEԬ=�8�4L��=nx6�X� >�`��>RT��+�>���">Z�TA[�>4�e|�Z#>��o�kJ%>4��evk'>���<�)>�'v�V,>_"s�$1>6NK��2>�so쩾4>�z��6>����W_>>p��Dp�@>6��>?�J>������M>��x��U>Fixі�W>4�j�6Z>w&���qa>�����0c>:�AC)8g>ڿ�ɓ�i>�������:�              �?              �?              �?       @               @              �?              �?              �?              �?      �?              �?              �?      �?              �?      �?              �?              �?              @              �?              �?              �?      �?              �?              �?        ���XW      �A�	�� �'��A*ɮ

step  �A

lossm��=
�
conv1/weights_1*�	   ` ���   @��?     `�@! ��� �)8�
�t(@2�	8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9����d�r�x?�x��f�ʜ�7
��������[�?1��a˲?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?�������:�	               @      0@      B@     �R@      \@      d@      f@     `d@     @c@     �a@     ``@     �\@     �Y@     �V@      T@     @R@     @R@     �P@      M@     @P@     �O@     �B@     �D@     �E@      :@     �@@      @@      9@      4@      9@      ,@      6@      (@      (@      "@      @      "@      "@       @      "@      @      (@      @      @      $@      (@      @      �?      @      @      @      �?      �?       @       @      @      �?       @      �?       @              @       @               @      �?      �?              �?      �?              �?              �?              �?               @              �?               @      �?       @       @              �?              �?              �?      �?               @      �?      �?              �?      �?      �?      �?      �?      �?       @      @      �?      @      @      @       @      @      @      "@       @      $@      $@      ,@      @      ,@      0@      &@      0@      4@      ,@      2@      2@      7@      :@      9@     �B@      8@     �A@     �D@      H@      G@      L@     �F@      K@      Q@     �P@     �T@     �X@     �]@     �\@     �[@      ]@     `b@     `a@     �d@     �a@     �]@      O@      $@      @        
�
conv1/biases_1*�	   �U�   �$ڦ?      @@!  L4���?).��;���?2�ܗ�SsW�<DKc��T��lDZrS�IcD���L��qU���I�I�I�)�(�+A�F�&�d�\D�X=?���#@?5Ucv0ed?Tw��Nof?ߤ�(g%k?�N�W�m?;8�clp?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�uS��a�?`��a�8�?�/�*>�?�g���w�?�������:�              �?      �?              �?              �?              �?              �?              �?      �?              �?               @      �?      �?               @              @       @               @              @               @              �?       @      �?        
�
conv2/weights_1*�	   �����   `���?      �@! ����P@)������E@2��{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澮��%ᾙѩ�-߾E��a�Wܾ0�6�/n���u`P+d��u��6
�>T�L<�>��~���>�XQ��>��~]�[�>��>M|K�>�f����>��(���>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?�������:�              �?      @      0@     �S@     ��@     ܝ@     ��@     ��@     d�@     D�@     �@     ؒ@     @�@     ��@     ��@     ��@     `�@     �@     �@     ��@     ��@     z@     �z@     �w@     �w@     `u@     �s@     Pp@      m@      l@     �l@     �h@      c@     `c@      c@      `@     �]@      ^@      \@     �Z@     �R@      V@     �R@     �N@     �J@     �C@     �I@      G@     �C@      @@      =@      9@      >@      3@      :@      .@      .@      1@      (@      (@      "@      "@      (@      *@      $@      &@       @      @      @      @      @      @      @      @       @      @       @      @      @       @      @      @      @      �?              �?      �?      �?      �?              �?              �?               @       @              �?              �?              �?              �?              �?              �?              �?       @       @       @              �?      @      �?       @      @      @       @       @      @      @      @      @       @      @       @      @      (@      @      &@      $@      (@      *@      (@      &@      ,@      .@      5@      1@      :@      ;@      3@      9@      =@      ?@     �B@      F@     �K@      I@      I@      K@     @S@     �T@     �T@      S@     @[@      ]@     �^@     �a@     @c@      d@      h@     @f@      j@     `o@      q@      s@     �u@      z@     �x@     �{@      �@     �@     �@     ��@     �@     (�@     Ј@     �@     X�@     ԑ@     ��@     Ԕ@     ��@     |�@     X�@     ��@     �@     4�@     �v@     �K@      ,@      �?        
�	
conv2/biases_1*�		   �E���   �s�?      P@!  �����?)��dԱ�?2��v��ab����<�A����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\��>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��qU���I�
����G��7Kaa+�I�I�)�(�U�4@@�$��[^:��"��S�F !�ji6�9����VlQ.?��bȬ�0?�u�w74?��%�V6?d�\D�X=?���#@?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�uS��a�?`��a�8�?�������:�               @              �?               @      �?       @              �?               @      �?      @              �?      �?      �?              �?               @      �?              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?      �?      �?              �?              �?              �?               @              �?      @              �?              @      @      �?       @               @      �?       @       @              �?              �?        
�
conv3/weights_1*�	   �ư��   `]�?      �@!���0�J@)ȶ$ �U@2���]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��})�l a��ߊ4F����(��澢f�����uE���⾮��%ᾙѩ�-߾�_�T�l׾��>M|Kվ��~]�[ӾK+�E��Ͼ['�?�;;�"�qʾ
�/eq
Ⱦ�u`P+d����n�����T�L<�>��z!�?�>��n����>�u`P+d�>��~]�[�>��>M|K�>�iD*L��>E��a�W�>�ѩ�-�>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?�������:�              >@     P�@     Ȧ@     ئ@     ,�@     ��@     ҡ@     L�@     ؜@     �@     t�@     �@     h�@     ��@     ��@     (�@     (�@      �@     ��@      �@      �@     ��@     �~@     �|@     `y@     �x@     v@      t@      p@      o@     �n@     @l@      h@     `b@     @b@      c@     @b@      \@     @[@     �X@      W@      R@     �R@     �R@      J@     �J@     �K@      F@      =@     �@@      :@      D@      B@      .@      4@      <@      *@      1@      *@      ,@      .@      0@      *@      &@      @      @      @      "@      @      @      @      @      @      @      @      @      @      @       @      �?      �?      @      @               @              �?              @      �?      �?      �?              @      �?              �?              �?              �?              �?              �?              �?              �?       @              �?      �?              �?       @      �?       @      �?              @      �?      @       @               @      @       @      @       @       @      @      $@      @      @      @      @       @       @       @      @      $@      (@      ,@      5@      1@      ,@      *@      ,@      8@      .@      7@     �A@     �A@      D@      C@     �J@      I@     �N@      N@     @Q@     �Q@      U@      U@     �V@     �[@     @]@     `b@     �]@     �c@     @d@     @g@     �k@      i@     �o@     �q@     pr@     Pt@     �w@     �y@     �z@     �@     �@     ؀@     p�@      �@     ��@     ��@     ��@      �@     ��@     �@     h�@     <�@     l�@     �@     �@     ��@     �@     D�@     6�@     ��@     ��@     �h@      >@      "@        
�
conv3/biases_1*�	   �7��   @0�?      `@!  ��΂�?)��]�h�?2��/����v��ab��^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�<DKc��T��lDZrS�IcD���L��qU���I�
����G�a�$��{E��!�A����#@�d�\D�X=�uܬ�@8���%�V6�U�4@@�$��[^:��"��S�F !���ڋ��vV�R9�        �-���q=�ߊ4F��>})�l a�>��ڋ?�.�?�S�F !?�[^:��"?U�4@@�$?��bȬ�0?��82?d�\D�X=?���#@?
����G?�qU���I?IcD���L?k�1^�sO?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�������:�              �?              �?       @       @      �?      @      �?      �?              �?      �?      �?               @      @      �?       @      �?      �?       @      @      �?      �?      �?      �?      �?      �?      �?              �?               @       @      �?              �?      �?               @              �?      �?              �?              @              �?              �?              �?      �?              �?              �?              �?              �?              �?      �?      �?      �?               @              �?              @       @              @      @      @       @       @      �?      @       @      �?       @      @              @      �?      @       @      �?      �?      �?      �?      �?      �?              �?        
�
conv4/weights_1*�	   ��U��   @>��?      �@!�o���$�)݀�Ĺ�e@2�Ӗ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'�������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ���I��P=��pz�w�7��
�/eq
Ⱦ����ž���%�>�uE����>�f����>�h���`�>�ߊ4F��>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?�������:�             �b@     ��@     �@     X�@     $�@     ��@     (�@     ��@     X�@     ��@     h�@     @�@     `@     �}@     @|@     w@      v@      u@      r@     @q@     �p@     @j@     �i@     �h@     `e@      e@     �^@     �X@      ]@      ]@     @R@      W@      O@     �P@      L@     �L@     �E@     �I@     �C@      D@      >@      A@      ?@     �@@      8@      8@      =@      7@      5@      ,@      @      *@      &@      *@      &@      @      "@      @      @      @      &@      @      @      @      �?       @      @      @      @      @              �?      �?              �?      �?              �?      �?       @              �?              �?              �?              �?      �?              �?              �?              �?      �?              �?              �?               @      �?               @              �?               @              @       @      �?      �?      �?      �?      �?               @       @               @       @      �?      @      @      @      @       @       @      "@      @      (@      ,@      "@       @       @      "@      2@      $@      ,@      .@      4@      6@      >@      ?@      ?@     �A@      @@      >@      B@      J@     �K@      N@     �R@      N@     @P@     �S@      U@      X@     �\@      ^@     �_@     `e@     �e@     @d@     �f@     @k@     �k@     �n@     u@     Pt@     �u@     py@     �x@     `}@     ��@     h�@     `�@     ��@     �@     �@     P�@     ��@     ��@     ��@     �@     ��@     �n@      @        
�
conv4/biases_1*�	    ���    �R�?      p@!
T9n�e�?)N2��+M�?2��/�*>��`��a�8���v��ab����<�A���}Y�4j��^�S�����Rc�ݒ��7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !���ڋ��vV�R9��T7���>h�'��f�ʜ�7
��������[���FF�G �>�?�s���I��P=��pz�w�7��8K�ߝ�a�Ϭ(龙ѩ�-߾E��a�Wܾjqs&\�ѾK+�E��Ͼ5�"�g���0�6�/n��Fixі�W���x��U�H��'ϱS�������M�6��>?�J���8"uH�/�p`B�p��Dp�@�����W_>�p
T~�;�_"s�$1�7'_��+/�        �-���q=i@4[��=z�����=����%�=f;H�\Q�=��f��p>�i
�k>_"s�$1>6NK��2>�z��6>u 5�9>/�p`B>�`�}6D>��Ő�;F>��8"uH>������M>28���FP>��u}��\>d�V�_>w&���qa>�����0c>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>�FF�G ?��[�?1��a˲?6�]��?>h�'�?x?�x�?�5�i}1?�T7��?�vV�R9?��ڋ?�S�F !?�[^:��"?U�4@@�$?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?}Y�4j�?��<�A��?�v��ab�?�/��?�/�*>�?�g���w�?���g��?�������:�              �?              �?       @               @              �?       @      @       @      �?               @               @      �?       @      �?      @       @       @              �?               @       @       @              �?      @       @       @      �?       @       @      @              �?               @      �?      @               @              �?      �?      �?      �?      @      �?       @              �?      �?              �?       @               @      �?               @              �?              �?              �?              �?              �?      �?              �?       @              �?      @      �?              �?              7@              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?      �?              �?              �?              �?              �?              �?              �?              @      �?              �?               @      @      @               @      �?      �?      �?      @      �?       @       @       @       @      @      �?      @      @      @      @      @      @       @              @      @      �?       @      @      @              �?      �?      @      @              @       @      @      �?              �?      @       @              �?              �?              �?      �?        
�
conv5/weights_1*�	   �4Ŀ   `E��?      �@! �R�B.@)7�B�u�I@2�	yD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G��!�A����#@�d�\D�X=���%>��:���%�V6��u�w74���82��7Kaa+�I�I�)�(��[^:��"��S�F !���ڋ��vV�R9��T7���})�l a��ߊ4F���5�i}1?�T7��?��ڋ?�.�?ji6�9�?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�������:�	              @     �i@     r@     �q@     �n@      m@      k@      i@     �c@      b@     �a@     �b@     �Y@     �\@      W@     �V@     �S@     @S@      S@     �N@      M@      D@     �A@     �D@      A@      >@      >@      ?@     �A@      ;@      ,@      3@      *@      *@      ,@      $@      *@      @      *@      @       @      @       @      "@      @      @      @      �?      @       @      @              �?      @      �?      @      �?              �?              �?               @      �?              �?              �?              �?      �?              �?              �?              �?      �?              �?              �?              �?               @              �?               @      �?      @      @       @       @      @      �?       @      �?      @      @       @       @      @      @      @      @      @      @      "@      "@      @      (@      0@      ,@      5@      @      @      8@      ,@      8@      ;@      6@     �B@      B@      @@     �I@      J@      G@      O@      L@      N@     @T@     @Q@      Y@     �[@     �Y@      _@      a@      b@     �b@      f@     �b@     �h@      n@     �m@     �r@     �t@     �o@      4@       @        
�
conv5/biases_1*�	    �s�   @�@h>      <@!  $v�Z>)\���.�=2��H5�8�t�BvŐ�r�Fixі�W���x��U�6��>?�J���8"uH���Ő�;F��`�}6D�/�p`B�p
T~�;�u 5�9����"�RT��+����1���=��]���;3���н��.4Nν�-���q�        �mm7&c>y�+pm>RT��+�>���">Z�TA[�>4�e|�Z#>��o�kJ%>4��evk'>�'v�V,>7'_��+/>_"s�$1>6NK��2>�so쩾4>�z��6>u 5�9>����W_>>p��Dp�@>6��>?�J>������M>Fixі�W>4�j�6Z>��u}��\>w&���qa>�����0c>:�AC)8g>ڿ�ɓ�i>�������:�              �?              �?              �?       @               @              �?              �?              �?              �?              �?              �?      �?              �?              �?      �?              �?              �?               @      �?              �?              �?              �?      �?              �?              �?        �[~�X      ~���	%?��'��A*��

step  �A

loss�v=
�
conv1/weights_1*�	   �0��    6`�?     `�@! p؅YK!�)vZ+a@2�	%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&��[^:��"��S�F !��vV�R9��T7���O�ʗ�����Zr[v��
�/eq
�>;�"�q�>pz�w�7�>I��P=�>�FF�G ?��[�?6�]��?����?>h�'�?x?�x�?�5�i}1?�T7��?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?�������:�	               @      @      2@     �C@     @R@     @\@     �d@      e@      e@      c@      a@     @`@     �]@     �Y@      T@     @T@     @R@     �R@     �P@     �O@     �O@     �J@      E@     �E@     �E@      :@      E@      9@      7@      8@      0@      ,@      3@      *@      "@      *@       @      0@      @      (@      @      &@      (@      "@      "@      @       @       @              @       @      @      @      @      �?      @      �?       @              @       @      �?              @              �?      �?               @              �?              �?              �?              �?              �?               @              �?               @              �?              �?              @              �?      �?      @       @      �?      �?      �?      �?       @      @      �?      �?      �?      @       @      @      �?      @       @      @      @      @       @      "@      &@      $@      *@      $@      *@       @      .@      4@      ,@      6@      7@      1@     �A@      2@      @@      ;@      C@     �B@      G@      H@      M@      G@     �J@      Q@     �N@      V@      Z@      ]@     @\@     �Z@      ]@      b@     �a@      d@     �a@     �^@     �O@      ,@      @        
�
conv1/biases_1*�	   @��X�   �٧?      @@!  $�"��?)v�Yc1��?2���bB�SY�ܗ�SsW�<DKc��T��lDZrS�
����G�a�$��{E��[^:��"��S�F !����#@?�!�A?5Ucv0ed?Tw��Nof?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?�������:�              �?              �?              �?              �?              �?              �?              �?              �?              �?      �?      �?      �?      �?      �?      �?      �?      @              �?      �?       @       @      �?      �?              �?       @      �?        
�
conv2/weights_1*�	    }E��    �G�?      �@!�zg	.Q@)�.��E@2��{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��})�l a��ߊ4F��8K�ߝ�a�Ϭ(龄iD*L�پ�_�T�l׾��>M|Kվ
�/eq
Ⱦ����ž�XQ�þ��n�����豪}0ڰ����m!#���
�%W��K+�E���>jqs&\��>�iD*L��>E��a�W�>�ѩ�-�>�f����>��(���>a�Ϭ(�>8K�ߝ�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?�������:�               @      @      0@     �T@     ��@     ؝@     ��@     ��@     h�@     ,�@     $�@     ؒ@     8�@     ��@     x�@     �@     h�@     �@     ��@     �@     ��@     �z@     pz@     0w@     �w@      v@     �s@     �o@     �m@     �j@     �m@     @g@     @e@     �b@     �a@     �b@     �^@     �Y@     @[@     �Y@     �T@      T@      T@     �L@      L@      H@      E@     �D@      E@     �D@      6@      :@      7@      9@      7@      1@      2@      0@      .@      *@      "@      (@      &@      *@      @      @       @      @      @      "@      @      @      @       @       @      @      �?       @      �?      @      @      @       @      �?       @       @               @              �?              �?      �?              �?      �?              �?              �?              �?              �?      �?              �?      �?      �?              �?      �?       @      @              �?       @      �?              �?       @      @       @      �?      @       @      @       @       @      @      @      "@      @      �?      @      @      &@      "@      &@      *@      0@      2@      $@      3@      ;@      $@      @@      9@      ;@      @@      A@     �@@      D@      I@      G@     �K@     �J@     �Q@     �U@      V@      W@     @Y@     �\@     �^@     �a@      b@     �f@      f@     @g@     @j@     �m@     @q@     �s@      u@     �y@     Px@      |@     x�@     ��@     h�@     x�@     �@     Ȉ@     (�@     (�@     <�@     ��@     ��@     �@     ��@     x�@     x�@     ̞@     ��@     �@     �w@      O@      0@      @        
�	
conv2/biases_1*�		   �߽��   @|,�?      P@!  �����?)O�h|M�?2��/����v��ab����<�A���^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��qU���I�
����G���bȬ�0���VlQ.��7Kaa+�I�I�)�(��[^:��"��S�F !���bȬ�0?��82?��%�V6?uܬ�@8?���#@?�!�A?<DKc��T?ܗ�SsW?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?P}���h?ߤ�(g%k?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�uS��a�?`��a�8�?�������:�              �?      �?              �?              �?      �?      �?       @              �?              �?      �?      @      �?              �?      �?      �?              �?      �?       @              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?      �?      �?      �?              �?              �?               @      �?              @      �?              �?      @       @      �?      �?               @      @       @              �?              �?        
�
conv3/weights_1*�	   @̱�   �?0�?      �@! `��L@)V�rҏ�U@2���]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(����uE���⾮��%���>M|Kվ��~]�[Ӿjqs&\�Ѿ�[�=�k���*��ڽ�G&�$��5�"�g�����n�����豪}0ڰ�;9��R�>���?�ګ>G&�$�>�*��ڽ>�[�=�k�>K+�E���>jqs&\��>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?�������:�             �@@     @�@     ��@     Ʀ@     2�@     ��@     С@     6�@     ��@     ��@     ��@     �@     H�@     t�@     ܐ@     �@     Ћ@     ��@     @�@      �@     P�@     ��@     @~@     �|@     py@     �x@     �u@     @t@     �o@      p@     �n@     �j@      i@     �b@      b@     �b@     �b@      [@     �\@     @Z@     @T@     �S@      S@      T@     �G@     �G@      O@      @@      >@     �A@      <@     �@@      C@      ,@      7@      6@      1@      3@      .@      (@      &@      &@       @      $@      "@      "@       @      (@      @      @      @      @      @      @      @      @      @      @       @      �?      @      �?       @       @      �?      �?      @              �?      �?               @               @              �?      �?              �?              �?              �?              �?              �?      �?              �?              �?              �?      @      �?              �?      �?       @       @              �?       @      �?       @      �?      @              @      �?       @      @       @      "@      @      @      @      @      @      "@      "@      $@       @      &@      (@      (@      1@      .@      0@      0@      3@      *@      5@     �C@      >@     �E@      ;@      L@     �I@     �Q@      P@     �S@     @R@     �T@      R@     �W@     �Y@      _@     �`@      a@     �a@     @f@     �e@     �k@     �j@     �n@      r@     0r@     �t@     �v@     �y@     `z@     �@     8�@     �@     �@      �@     P�@     Љ@     ��@     �@     ��@     ؓ@     ��@     �@     l�@     �@     ��@     ~�@     :�@     .�@     6�@     ��@     ̓@     �j@     �B@      "@        
�
conv3/biases_1*�	   �~���   ��ǥ?      `@!  ��mh�?)�\��ꯑ?2��/����v��ab��^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�<DKc��T��lDZrS�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74�+A�F�&�U�4@@�$��[^:��"��S�F !���ڋ��vV�R9�O�ʗ�����Zr[v��        �-���q=��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?��VlQ.?��bȬ�0?d�\D�X=?���#@?
����G?�qU���I?k�1^�sO?nK���LQ?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�������:�              �?              �?      @      �?      @       @       @              �?      �?      �?      �?      @      @       @      �?       @              @              �?       @      �?              �?      �?      �?              �?               @      �?      �?      �?              �?              �?      �?              �?              �?              �?              �?              �?              @              �?              �?      �?              �?              �?              �?              �?               @               @               @              �?               @      �?       @      @      @      @       @      @      �?      �?      @       @      @      @      �?      @      �?      @       @               @      �?      �?      �?              �?      �?        
�
conv4/weights_1*�	   @]h��   થ�?      �@! `�^�#�)?,��΄e@2�Ӗ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.���vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[��O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��K+�E��Ͼ['�?�;jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�ѩ�-�>���%�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?�������:�             �b@     t�@     $�@     T�@     ,�@     ��@     �@     ��@     p�@     p�@     h�@     h�@     0@     �}@     P|@      w@     �u@     @u@      r@     �p@     �p@     `j@      j@      h@     �e@     �d@     @^@      Y@     �]@     @\@     �S@     @U@     �P@     �N@     �M@     �L@      G@      I@     �E@      ?@      A@      =@      A@     �A@      7@      :@      ;@      9@      4@      *@      $@      &@      $@      $@      $@      "@      @      @       @      @      @       @      @      @       @      @      �?      @      @      @      �?      �?              @       @              �?      �?              �?              �?      �?              �?      �?              �?              �?              �?               @              �?              �?              �?              �?              �?      �?              �?              �?              �?              @      �?      �?      �?              �?              �?               @      @      �?      �?      �?      @              @      @      @      @      @      @      @       @      &@      "@      *@      (@       @      @      ,@      (@      2@      1@      *@      ;@      =@      9@     �C@      @@      A@      =@     �A@     �I@      L@      P@     �P@      P@     �P@     @T@      T@     �Y@     �\@     �\@      `@      e@     �e@     @d@     �f@      k@     �k@     `o@     �t@     pt@     �u@     Py@      y@     p}@     ��@     P�@     x�@     ��@     @�@     �@      �@     ��@     ��@     ��@     ��@     p�@      o@       @        
�
conv4/biases_1*�	    wϥ�   �@Q�?      p@!f�|4�(�?)��Y#���?2��/�*>��`��a�8���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ��7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !��.����ڋ��vV�R9��T7���>h�'��f�ʜ�7
��������[���FF�G �I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(���jqs&\�ѾK+�E��Ͼ5�"�g���0�6�/n��Fixі�W���x��U�H��'ϱS�28���FP�������M�6��>?�J���8"uH�/�p`B�p��Dp�@�����W_>�p
T~�;�_"s�$1�7'_��+/�        �-���q=�|86	�=��
"
�=ݟ��uy�=�/�4��=��f��p>�i
�k>_"s�$1>6NK��2>�z��6>u 5�9>/�p`B>�`�}6D>��8"uH>6��>?�J>������M>28���FP>d�V�_>w&���qa>�����0c>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>1��a˲?6�]��?f�ʜ�7
?>h�'�?�5�i}1?�T7��?��ڋ?�.�?�S�F !?�[^:��"?U�4@@�$?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�g���w�?���g��?I���?�������:�              �?              �?               @              �?      �?               @       @      @      �?      �?      �?      �?      �?      �?      �?       @      �?      @       @      �?      �?               @               @       @      �?              @       @      �?       @      �?      @       @      �?      �?      �?      �?      �?       @              �?      �?              �?       @      �?       @      �?       @      �?              �?              �?               @      �?               @              �?      �?      �?              �?              �?              �?              �?              �?      �?              �?      �?      �?              @      �?      �?              �?              7@              �?              �?              �?              �?              �?              �?              �?              �?              �?      �?              �?              �?              �?      �?              �?              �?              �?              �?              �?              �?               @      �?              �?      �?              �?      @      @              �?      �?       @       @               @       @       @      @      �?       @      @      �?       @      "@      �?      @      @      �?      �?      �?      @       @       @       @      @       @      �?               @      @       @      @      @       @      @      �?              @      @      �?              �?              �?              �?      �?        
�
conv5/weights_1*�	   ��JĿ   �5��?      �@!  �z2�.@)�49�$�I@2�	yD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G��T���C��!�A�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74��7Kaa+�I�I�)�(��[^:��"��S�F !���ڋ��vV�R9��5�i}1���d�r���[���FF�G �x?�x�?��d�r?��ڋ?�.�?ji6�9�?�S�F !?I�I�)�(?�7Kaa+?��VlQ.?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�������:�	              @     �i@      r@     �q@      n@     �m@      k@     �h@      d@      b@     �a@     �b@     �Y@      \@     �W@     @V@     �S@     @S@     @S@     �N@      M@     �D@      A@     �D@     �@@      ?@      >@      >@      A@      >@      ,@      3@      &@      .@      (@      *@      $@       @      (@       @      @      @      $@      @      @      @      @      @      @      @       @      �?      �?      @       @      �?       @              �?              �?              �?       @              �?              �?              �?              �?              �?              �?              �?              �?              �?      �?              �?               @              �?              �?      �?      @      @      @       @       @       @       @      �?      @      @      �?       @      @      �?      @      @       @      @      @      "@       @      &@      2@      (@      5@      @      "@      6@      ,@      6@      ;@      8@     �B@     �A@      @@      H@      K@     �G@     �N@      N@      N@     @S@     �Q@     �X@     �\@     @Y@     �^@     @a@     �a@     �b@      g@     �b@     �h@     �m@     �n@     pr@     �t@     �o@      5@      @        
�
conv5/biases_1*�	   `�s�   @\Ci>      <@!  �z�\>)�����x
=2��H5�8�t�BvŐ�r�4�j�6Z�Fixі�W�������M�6��>?�J���8"uH���Ő�;F��`�}6D�p
T~�;�u 5�9����"�RT��+������%���9�e���<QGEԬ�|_�@V5���>�i�E��_�H�}���y�+pm>RT��+�>Z�TA[�>�#���j>��o�kJ%>4��evk'>�'v�V,>7'_��+/>_"s�$1>�so쩾4>�z��6>u 5�9>p��Dp�@>/�p`B>������M>28���FP>Fixі�W>4�j�6Z>��u}��\>�����0c>cR�k�e>:�AC)8g>ڿ�ɓ�i>�������:�              �?              �?              �?      �?      �?       @              �?              �?              �?              �?              �?               @              �?               @              �?      �?              �?       @              �?              �?              �?      �?              �?              �?        ӣ�f�W      ���	{���'��A*��

step  �A

loss:k�=
�
conv1/weights_1*�	   �Dg��   ����?     `�@! � �S"�){��pl�@2�	%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��vV�R9��T7���f�ʜ�7
������6�]���1��a˲���[���FF�G �I��P=��pz�w�7��;�"�qʾ
�/eq
Ⱦpz�w�7�>I��P=�>>�?�s��>�FF�G ?1��a˲?6�]��?x?�x�?��d�r?�.�?ji6�9�?�[^:��"?U�4@@�$?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?�������:�	               @      @      3@      E@     @S@      ^@     `c@     �c@     `f@      c@     �`@      `@      _@     @X@     �T@     @T@      R@     �R@      N@      S@      L@     �J@      E@     �E@      E@      <@     �B@     �@@      8@      1@      .@      0@       @      (@      1@      *@      .@      *@      $@      .@      1@      (@      @      @      @       @      @      @      @      @      @      @      @      @      �?       @      @       @      �?      �?               @       @       @              �?              �?              �?              �?              �?              �?              �?              �?               @              �?              �?               @              @              �?               @      �?      �?              �?              �?      �?      @      @       @      @              @       @      @      @      �?      @      $@      "@       @      @       @      "@      (@      @      &@      &@      *@      5@      2@      9@      9@      0@     �@@      5@      5@      ;@      D@      @@      I@      L@      I@     �H@     �I@     @P@     @Q@      U@      [@      ]@     �[@     �Y@     �]@     �a@     �a@     `d@      a@     �^@     �O@      1@       @      �?        
�
conv1/biases_1*�	   ���[�   ���?      @@!  to�I�?)�`���?2�E��{��^��m9�H�[�<DKc��T��lDZrS����#@�d�\D�X=��5�i}1���d�r����#@?�!�A?5Ucv0ed?Tw��Nof?ߤ�(g%k?�N�W�m?uWy��r?hyO�s?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?`��a�8�?�/�*>�?�g���w�?���g��?�������:�              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?       @      �?               @              @      �?      �?      �?      �?      @               @               @      �?      �?        
�
conv2/weights_1*�	   �쉳�    ���?      �@!����T�Q@)�N���E@2��{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?�������:�               @      @      1@     �W@     ��@     ��@     ��@     ��@     ��@     �@     (�@     ��@     8�@     �@     P�@     `�@     Ȉ@     �@     �@     �@     H�@     `|@     �y@     �w@     �w@     @u@     �s@     pp@      m@     �l@     �k@     �f@     �c@     �d@     �b@     �b@     @]@     @\@     �X@      Z@     @R@     �Q@      T@     �P@     �I@     �I@     �I@     �H@      A@      @@      5@      <@      9@      7@      5@      (@      5@      6@      &@      $@      2@      "@      $@      *@      ,@      "@       @      $@      @       @      @      @      @      @       @      @      �?       @              @      �?      @              �?       @       @      �?      �?       @      �?              �?       @              �?              �?              �?              �?               @              �?              �?               @       @              @      �?      �?              �?      @      �?      @       @      �?      @      @      @      @       @      @      @      @      $@      $@      @      "@      5@      *@      &@      *@      *@      2@      7@      (@      >@      >@      ;@      :@     �A@     �@@      H@      H@      H@     �G@      P@      O@     @T@     @W@      W@     @[@     @Y@      a@     �b@      b@     �f@     `e@     �f@     `k@     �k@     @r@     �r@     �u@     �x@     �x@     @|@     0�@     Ȁ@     0�@     ��@     ��@     ��@     P�@     ��@     <�@     ̑@     h�@     @�@     X�@     `�@     ��@     ��@     ��@     Ȗ@     �y@     �P@      1@      @        
�	
conv2/biases_1*�		   ��Ɵ�   ���?      P@!  ��B�?)��}B��?2��/����v��ab����<�A���^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��qU���I�
����G���%>��:�uܬ�@8�I�I�)�(�+A�F�&��S�F !�ji6�9����82?�u�w74?��%>��:?d�\D�X=?���#@?�!�A?<DKc��T?ܗ�SsW?E��{��^?�l�P�`?���%��b?5Ucv0ed?P}���h?ߤ�(g%k?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�uS��a�?`��a�8�?�������:�              �?      �?              �?               @              �?       @      �?              �?       @       @      �?      �?      �?              �?              �?       @      �?              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?               @               @              �?              �?               @              �?      @              �?      �?      @      �?      @      �?               @       @       @      �?      �?              �?        
�
conv3/weights_1*�	    籿   �bs�?      �@!�/��6M@)F����U@2���]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ��>M|Kվ��~]�[Ӿ['�?�;;�"�qʾ����ž�XQ�þ���?�ګ>����>豪}0ڰ>��n����>['�?��>K+�E���>��>M|K�>�_�T�l�>�ѩ�-�>���%�>�uE����>�f����>a�Ϭ(�>8K�ߝ�>})�l a�>pz�w�7�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?�������:�              C@     0�@     ��@     ��@     (�@     ��@     С@     0�@     �@     �@     l�@     @�@     0�@     ��@     ��@     @�@     ��@     ��@     @�@     �@     8�@     ��@     @     �{@      y@     �x@      v@     �t@     �p@     �m@     �o@     �j@      h@     �b@     �a@      d@     `b@      [@     �[@     @Z@     �V@     �T@     @Q@      R@      P@      F@     �O@      ?@      9@      B@      7@      A@      >@      .@      ;@      7@      6@      0@      (@      *@      *@      ,@      &@      @      @      @       @      "@      @      @      @      @      @      @      @      @      @       @      @      �?      @      @              @       @              �?      �?      �?              �?              �?      �?      �?               @              �?              �?              �?              �?              �?              �?              �?              �?              �?              @              �?              @              @      @      @      @       @       @       @      �?      @      @      @       @      @      @      @       @      @      @      *@      @      *@      &@      4@      *@      ,@      $@      1@      6@      .@      9@     �A@      8@     �E@      B@      K@     �I@     �N@     �P@      Q@     @S@     �V@     �T@     �V@      Y@     �_@      a@     @`@     `b@      f@     @f@     �j@     @j@     @p@     �q@     �r@     Ps@     Pw@      y@     �z@     h�@     ��@     ��@     @�@      �@     �@     ��@     ��@     8�@     ��@     ��@     ��@     �@     d�@     T�@     ܞ@     ��@     �@     Z�@     �@     ��@     ��@     �l@     �D@      $@        
�
conv3/biases_1*�	   �V0��   ���?      `@!  b�H�?):x�B��?2��uS��a���/����"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�a�$��{E��T���C��!�A�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74��7Kaa+�I�I�)�(�ji6�9���.����ڋ��vV�R9��T7����5�i}1�        �-���q=�T7��?�vV�R9?�.�?ji6�9�?�S�F !?I�I�)�(?�7Kaa+?���#@?�!�A?�qU���I?IcD���L?nK���LQ?�lDZrS?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?���%��b?5Ucv0ed?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?�������:�              �?              �?      �?      @      �?      @       @      �?              �?               @              �?      @       @       @      �?      �?      @       @      �?      �?      �?      �?              �?       @              �?              �?      �?       @              �?      �?              �?      �?              �?              �?              �?              �?              �?              @              �?              �?      �?              �?              �?              �?              �?              �?      �?       @               @              �?      �?       @      �?      @       @      @      @      @               @      @      @      @      @       @       @       @      �?      @      �?       @      �?              �?      �?      �?              �?        
�
conv4/weights_1*�	   @�z��    ���?      �@! s���"�)Щ��e@2��?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ���I��P=��pz�w�7���iD*L�پ�_�T�l׾jqs&\�ѾK+�E��Ͼ�u`P+d����n�����jqs&\��>��~]�[�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?�5�i}1?�T7��?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?�������:�              �?     �b@     h�@     (�@     D�@     <�@     t�@     �@     ��@     `�@     p�@     ��@     0�@     �~@     �}@     `|@     �v@     @v@     u@     @r@     �p@     �p@     �i@      k@     �g@      f@     `d@     @]@      Z@      ^@      [@     �U@      T@     �P@      N@      O@     �J@      H@      J@      F@      >@      ?@      A@      >@     �A@      9@      ?@      7@      8@      6@      .@      @      &@      $@      &@      "@      @      @      $@      @      @      @       @      @       @       @      @      �?      @      @      @      @       @       @      �?       @              �?              �?               @      �?      �?               @              �?               @              �?              �?              �?              �?              �?      �?              �?      �?      �?               @      �?      �?       @              �?              �?       @       @              �?       @      �?      @               @       @      @      @       @      @      @      @      @      @      $@      .@       @      $@       @      $@      *@      ,@      *@      1@      7@      5@      <@      8@     �B@     �@@      B@      >@      A@     �J@      L@      O@      Q@     @P@     @P@      T@      T@      Z@     �\@     �\@     �_@     @e@     �e@     �d@      f@     `k@      l@     `o@     �t@     �t@     pu@     Py@     `y@     `}@     x�@     @�@     p�@     x�@     h�@     �@     �@     ��@     ��@     ��@     ܓ@     ��@     �o@      "@        
�
conv4/biases_1*�	   �A���   `kM�?      p@!P�Q~]��?)�p��.�?2��g���w���/�*>���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ��#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��.����ڋ��vV�R9�x?�x��>h�'��f�ʜ�7
�1��a˲���[���FF�G �I��P=��pz�w�7���ߊ4F��h���`�8K�ߝ��uE���⾮��%���~]�[Ӿjqs&\�Ѿ5�"�g���0�6�/n��4�j�6Z�Fixі�W���x��U�H��'ϱS�28���FP�������M�6��>?�J���8"uH��`�}6D�/�p`B�p��Dp�@�����W_>�p
T~�;�6NK��2�_"s�$1�'j��p���1���        �-���q=�9�e��=����%�=�i
�k>%���>6NK��2>�so쩾4>�z��6>u 5�9>/�p`B>�`�}6D>��8"uH>6��>?�J>28���FP>�
L�v�Q>d�V�_>w&���qa>�����0c>cR�k�e>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>6�]��?����?f�ʜ�7
?>h�'�?�5�i}1?�T7��?��ڋ?�.�?�S�F !?�[^:��"?U�4@@�$?�7Kaa+?��VlQ.?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�/��?�uS��a�?�g���w�?���g��?I���?�������:�              �?              �?      �?      �?              �?      �?              �?      �?      @       @              �?       @               @      �?              @      �?      @      @              �?               @       @       @      �?              �?      @      @      �?       @       @      @      �?      �?              �?       @      �?      �?               @              �?              @      �?       @              @              �?      �?              �?       @               @      �?               @              �?      �?              �?              �?              �?              �?              �?              �?      �?      �?              �?      @              �?              �?              �?              7@              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?      @              �?              �?      @      �?              �?      �?       @              �?      �?       @      @       @              @      �?      @      �?      @      @      @      @      @       @      �?      �?      @      @      �?      @      @       @      �?               @      @       @       @      @       @      @      �?              �?      @      �?              �?              �?              �?      �?        
�
conv5/weights_1*�	   ��aĿ   ���?      �@! �A</@)�����I@2�	yD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G��T���C��!�A����#@���%>��:�uܬ�@8���%�V6��u�w74��7Kaa+�I�I�)�(��[^:��"��S�F !���ڋ��vV�R9�>h�'��f�ʜ�7
������6�]��?����?��ڋ?�.�?ji6�9�?�S�F !?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�������:�	              @     �i@     �q@     �q@      n@     �m@      k@     �h@      d@      b@     �a@     �b@     �Y@      \@     �W@     �V@     @S@     �R@     �S@      O@      M@     �D@     �A@     �D@      >@      ?@      @@      >@     �A@      ;@      ,@      3@      (@      0@      (@      (@      *@      @      "@       @      "@       @      $@      @      @      @      @      @      @      �?      @               @      @      �?      �?       @              �?      �?              �?      �?       @              �?              �?              �?              �?      �?              �?              �?              �?              �?      �?      �?      �?       @              �?              @      �?       @      @      �?      �?      �?       @      @      @      @      @       @       @      @      @      @      @      @      @      @      $@      @      ,@      .@      ,@      4@      @      "@      3@      0@      5@      ;@      :@     �B@     �A@      ?@     �G@      J@      J@      M@     �M@      O@     �S@     �Q@     @X@     @]@     �X@     @_@     �`@     �a@     �b@     �f@      c@     �h@     �m@     `n@     pr@     �t@      o@      8@      @        
�
conv5/biases_1*�	   �rit�   ��Mj>      <@!  б�p]>)�&Q�0�=2��H5�8�t�BvŐ�r�4�j�6Z�Fixі�W�������M�6��>?�J���8"uH���Ő�;F��`�}6D�����W_>�p
T~�;����"�RT��+��nx6�X� ��f׽r���G�L������6������/�=�Į#��=RT��+�>���">�#���j>�J>��o�kJ%>4��evk'>�'v�V,>7'_��+/>_"s�$1>�z��6>u 5�9>p
T~�;>����W_>>p��Dp�@>������M>28���FP>4�j�6Z>��u}��\>�����0c>cR�k�e>ڿ�ɓ�i>=�.^ol>�������:�              �?              �?              �?       @               @              �?              �?              �?              �?              �?               @              �?               @              �?      �?               @      �?              �?              �?               @              �?              �?        >����V      	燀	ZGj�'��A*٭

step  �A

loss��>
�
conv1/weights_1*�	   @����    o
�?     `�@! ��.S#�)�(`�@2�	%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(��[^:��"��S�F !��5�i}1���d�r�x?�x��1��a˲���[���ߊ4F��h���`��ߊ4F��>})�l a�>�FF�G ?��[�?�5�i}1?�T7��?��ڋ?�.�?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?�������:�	               @      @      3@      G@     �S@     @^@     `c@     �c@     @f@      c@      a@     �_@     �^@     �Z@     @Q@     �V@     @Q@      S@     �Q@      N@     �L@      L@      F@     �E@     �C@      A@     �@@      @@      4@      2@      1@      (@      $@      (@      (@      .@      5@      0@      0@      (@      $@      @      @      @      @      "@       @      @      @      @      @      �?       @      @       @      �?      @      �?              �?      �?      �?      �?              �?      �?              �?              �?      �?              �?              �?              �?              �?               @              �?              �?              @      �?      �?      �?      �?       @               @              �?       @      �?       @      �?       @       @      "@              @      @      ,@      @      @      �?       @      $@       @       @      @      $@      ,@      &@      1@      .@      6@      8@      5@      2@      ?@      7@      5@      <@      C@     �@@      J@     �J@     �I@     �H@      I@      N@     @R@     �V@     �X@      ^@     �Z@     �Y@     �]@      b@      a@     `d@      a@      ]@     �O@      9@      @      �?        
�
conv1/biases_1*�	   @��^�    ���?      @@! ��=��?)��U
��?2��l�P�`�E��{��^�ܗ�SsW�<DKc��T�+A�F�&�U�4@@�$�8K�ߝ�a�Ϭ(��!�A?�T���C?5Ucv0ed?Tw��Nof?ߤ�(g%k?�N�W�m?hyO�s?&b՞
�u?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?�������:�              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?               @      �?       @              @      �?               @              @      �?      �?      �?               @      �?      �?        
�
conv2/weights_1*�	   �˳�   `,�?      �@!�SbuR@)n2��E@2�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��})�l a��ߊ4F��8K�ߝ�a�Ϭ(�G&�$��5�"�g���Fixі�W>4�j�6Z>���?�ګ>����>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?�������:�              �?      �?      @      2@     �Y@     P�@     ��@     D�@     ��@     x�@     �@     0�@     �@     0�@     �@     �@      �@     ؈@     0�@     �@      �@     �@     �}@     �x@     �x@     @w@     �u@     �s@     �o@     �n@     `l@      k@     �g@     @c@     �c@     @d@     �b@     �\@     @]@     @Y@     @X@     �R@     �R@     �O@      M@      M@      L@     �H@      E@     �G@      ?@      6@      6@      8@      :@      8@      4@      $@      3@      8@      3@      *@      &@      @      $@      @       @      @      @      $@       @      @      @       @      @       @       @      �?       @       @       @       @      �?       @      �?       @              �?               @              �?              �?              �?              �?      �?      �?      �?               @              �?      �?               @              �?      �?              �?       @      @      �?      �?      @      �?       @      �?       @      @      �?       @      @      �?      @      @      @      @       @      @      @      @      @      4@       @      &@      $@      (@      ,@      0@      &@      2@      4@      6@      7@      9@      :@      :@      B@      @@      I@      J@     �I@     �G@      O@     @Q@     @R@     �W@     �V@     �^@     �X@     �`@     �b@      b@      e@      f@      h@     �j@      m@     @q@     �r@     Pv@     �w@     �z@     �|@     p@     Ѐ@     `�@     ��@     H�@     0�@     �@      �@     �@     �@     D�@     (�@     t�@     4�@     h�@     �@     ��@     ��@     �z@      S@      4@      @      �?        
�	
conv2/biases_1*�		   ��f��   ���?      P@!  ����?)���;P/�?2��/����v��ab��^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��qU���I�
����G��!�A����#@�U�4@@�$��[^:��"�ji6�9���.���u�w74?��%�V6?d�\D�X=?���#@?�!�A?�T���C?�lDZrS?<DKc��T?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?`��a�8�?�/�*>�?�������:�               @              �?               @      �?       @              �?              �?      @       @              �?      �?              �?              �?      �?       @              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?      �?      �?      �?              �?              �?               @      �?       @      �?      �?               @      @       @       @               @      �?      @      �?              �?              �?        
�
conv3/weights_1*�	   ����   `߶�?      �@!�0��XN@)A��U@2��{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(����uE���⾮��%ᾄiD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿ�XQ�þ��~��¾�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>�uE����>�f����>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?�������:�              �?      D@     P�@     ��@     ��@     �@     ��@     �@      �@     �@     Ț@     |�@     P�@     $�@     ��@     ؐ@     Ȍ@     �@     H�@     ��@     Ѓ@      �@     Ȁ@     p~@     `{@     �z@     �x@     �u@     �t@     �p@     `o@     �o@      j@     �h@     �b@      b@     �a@     @c@      ^@      \@     �X@     �V@      V@      Q@     �S@      J@      F@      O@      ?@     �A@      C@      :@      :@      =@      5@      5@      ;@      2@      ,@      5@      *@      $@      $@      $@      @      @      @       @      @      @      @      @       @      @      @      @      @      @      @       @      @      �?       @      @              �?       @              �?       @       @              �?              �?               @              �?              �?              �?              �?              �?               @      @      �?      @               @       @      @       @      @       @      @       @      @      @      @      @       @      @      &@      @      @       @      @       @      @      @      *@      $@      *@      2@      (@      :@      2@      *@      $@      >@      9@     �@@      C@      >@     �M@      J@     @Q@      O@     @R@     �P@     @T@     @V@      X@      [@     @^@     �a@     �`@     �a@     @e@     @f@     �l@      i@     �o@     �r@     r@     �s@     Pw@     @y@     pz@     p�@     h�@     ��@     X�@     0�@     ��@     ��@     ��@      �@     Б@     ��@     ȕ@     �@     l�@     t�@     Ԟ@     ��@     ��@     ��@     �@     ��@     8�@     �l@      F@      *@      �?        
�
conv3/biases_1*�	   �ü��   ����?      `@!  ��-�?)��8[�?2��uS��a���/����"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���VlQ.��7Kaa+�U�4@@�$��[^:��"��.����ڋ��vV�R9�        �-���q=�5�i}1?�T7��?�vV�R9?��ڋ?ji6�9�?�S�F !?+A�F�&?I�I�)�(?�!�A?�T���C?IcD���L?k�1^�sO?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?�������:�              �?              �?       @       @      �?      @      �?      �?              �?      �?      �?      �?       @      @       @      �?      �?      �?      @      �?      �?              @              �?               @              �?      �?               @      �?               @              �?              �?              �?              �?              �?              �?      �?              @              �?              �?              �?              �?              �?              �?              �?              �?      �?      �?      �?               @              �?      �?      �?       @       @      @      @      @      @               @       @       @       @      @      @              @      �?      @       @               @      �?      �?      �?              �?      �?        
�
conv4/weights_1*�	    ���   ����?      �@! KRԬ�!�)z1!K��e@2��?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���Zr[v��I��P=��pz�w�7��8K�ߝ�a�Ϭ(�E��a�Wܾ�iD*L�پ�_�T�l׾0�6�/n���u`P+d����Zr[v�>O�ʗ��>1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?�������:�              �?     �b@     \�@     $�@     P�@     8�@     ��@     �@     x�@     ��@     H�@     ��@     �@      @     �}@     �|@     �v@     @v@     u@     pr@     �p@     �p@     �i@     �k@      g@     `f@     �d@     �\@     @Z@     �^@     �Y@     �V@      T@     �N@     @Q@     �K@      L@      H@      I@     �F@      =@     �@@      >@      @@      A@      9@     �A@      7@      6@      6@      *@      @      (@      (@      "@      &@      @      @      @      &@      @      @      @      @      @       @      @      �?      @      @      @       @       @      �?      �?               @      �?      �?      �?       @      �?      �?       @               @      �?              �?      @              �?              �?      �?              �?              �?              �?       @       @               @      �?              �?      �?              �?      �?              �?      �?      @      @              �?      �?       @      @      @      @      @      @       @      @      @      ,@       @      @      (@      @      "@      1@      (@      ,@      2@      6@      6@      6@      <@     �B@      =@      C@      A@      A@      I@      M@      N@      R@     @P@      Q@     �Q@     @U@     �Y@     �]@     �\@     @_@      e@     �e@     �d@     �e@     �k@     �k@      o@     �t@     �t@     pu@     Py@     py@      }@     ��@     @�@     x�@     ��@     p�@     Ј@     (�@     h�@     ��@     ��@     ̓@     ��@     0p@      "@        
�
conv4/biases_1*�	   �0A��   �G�?      p@!4M��?)S-"����?2��g���w���/�*>���/����v��ab����<�A����"�uԖ�^�S�����#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !��.����ڋ��vV�R9�x?�x��>h�'��f�ʜ�7
�1��a˲���[���FF�G �>�?�s�����Zr[v��I��P=���h���`�8K�ߝ���~]�[Ӿjqs&\�ѾG&�$��5�"�g���4�j�6Z�Fixі�W���x��U�28���FP�������M�6��>?�J���8"uH��`�}6D�/�p`B�p��Dp�@�����W_>��z��6��so쩾4��mm7&c��`���        �-���q=�`��>�mm7&c>�i
�k>%���>_"s�$1>6NK��2>u 5�9>p
T~�;>�`�}6D>��Ő�;F>6��>?�J>������M>28���FP>�
L�v�Q>d�V�_>w&���qa>�����0c>cR�k�e>�iD*L��>E��a�W�>�ѩ�-�>���%�>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>6�]��?����?f�ʜ�7
?>h�'�?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?�S�F !?�[^:��"?U�4@@�$?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?�Rc�ݒ?^�S���?�"�uԖ?��<�A��?�v��ab�?�/��?�uS��a�?�g���w�?���g��?I���?�������:�              �?              �?       @               @               @      �?      @       @      �?               @      �?      �?      �?       @      �?       @       @      @              �?      �?      �?       @      @              @      �?       @       @       @      @      @      �?              �?      �?      �?       @              �?      �?              �?       @       @      �?      �?       @      �?      �?              �?      �?               @      �?              �?      �?      �?              �?               @              �?              �?              �?      �?              �?      �?      �?               @       @      �?              �?              �?              7@              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?      �?      �?              �?      �?              �?              �?              �?              �?      �?              �?              �?       @              �?              �?      @      @               @      �?      �?      �?      @      �?       @      �?       @      @       @      �?       @      @      "@      @      @      @      �?      �?      @      @      �?       @       @      @      �?      �?      �?       @      @              @      @      @      �?              @      @              �?              �?              �?      �?        
�
conv5/weights_1*�	    zĿ    A�?      �@! H�UY/@):����I@2�	yD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���VlQ.��7Kaa+�I�I�)�(�+A�F�&��[^:��"��S�F !���ڋ��vV�R9���d�r�x?�x��1��a˲���[����Zr[v�>O�ʗ��>��ڋ?�.�?�S�F !?�[^:��"?U�4@@�$?I�I�)�(?�7Kaa+?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?���#@?�!�A?�T���C?a�$��{E?
����G?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�������:�	              @     �i@     �q@     �q@     �m@     �n@     `j@     �h@     @d@      b@     �a@     �b@      Y@     �\@      W@      W@     �R@     �R@     �S@      O@      N@      D@     �A@     �D@      =@     �@@      =@     �@@     �@@      9@      0@      4@      "@      1@      ,@      ,@      &@      @      &@      @      @      &@      $@      @      @      @       @       @      @      �?       @      �?      �?       @      @       @      �?              �?      �?       @      �?      �?              �?      �?      �?              �?               @              �?              �?              �?              �?              �?      �?              �?              �?               @      �?              @      @       @      @              @       @       @      �?      @      @      @       @      @       @       @      &@      @      @      @      $@      &@      (@      0@      (@      3@      $@      &@      1@      0@      5@      :@      <@     �A@      @@      A@      G@      K@      I@      M@     �M@      P@     �S@     @Q@     �X@     �\@     �X@     �_@     �`@     �a@     �b@      f@     �c@     �h@      m@      o@     `r@     �t@     @o@      9@      @        
�
conv5/biases_1*�	    �u�    b_k>      <@!  peC_>)�
�#��=2��i����v��H5�8�t