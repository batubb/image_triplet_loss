       ЃK"	  РZрЊзAbrain.Event:2ниMбЯ      fn@	@мнZрЊзA"Т

anchor_inputPlaceholder*
dtype0*1
_output_shapes
:џџџџџџџџџ ч*&
shape:џџџџџџџџџ ч

positive_inputPlaceholder*&
shape:џџџџџџџџџ ч*
dtype0*1
_output_shapes
:џџџџџџџџџ ч

negative_inputPlaceholder*
dtype0*1
_output_shapes
:џџџџџџџџџ ч*&
shape:џџџџџџџџџ ч
Љ
.conv1/weights/Initializer/random_uniform/shapeConst*%
valueB"             * 
_class
loc:@conv1/weights*
dtype0*
_output_shapes
:

,conv1/weights/Initializer/random_uniform/minConst*
valueB
 *ЌErН* 
_class
loc:@conv1/weights*
dtype0*
_output_shapes
: 

,conv1/weights/Initializer/random_uniform/maxConst*
valueB
 *ЌEr=* 
_class
loc:@conv1/weights*
dtype0*
_output_shapes
: 
№
6conv1/weights/Initializer/random_uniform/RandomUniformRandomUniform.conv1/weights/Initializer/random_uniform/shape*
dtype0*&
_output_shapes
: *

seed *
T0* 
_class
loc:@conv1/weights*
seed2 
в
,conv1/weights/Initializer/random_uniform/subSub,conv1/weights/Initializer/random_uniform/max,conv1/weights/Initializer/random_uniform/min*
T0* 
_class
loc:@conv1/weights*
_output_shapes
: 
ь
,conv1/weights/Initializer/random_uniform/mulMul6conv1/weights/Initializer/random_uniform/RandomUniform,conv1/weights/Initializer/random_uniform/sub*
T0* 
_class
loc:@conv1/weights*&
_output_shapes
: 
о
(conv1/weights/Initializer/random_uniformAdd,conv1/weights/Initializer/random_uniform/mul,conv1/weights/Initializer/random_uniform/min*
T0* 
_class
loc:@conv1/weights*&
_output_shapes
: 
Г
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
г
conv1/weights/AssignAssignconv1/weights(conv1/weights/Initializer/random_uniform*
T0* 
_class
loc:@conv1/weights*
validate_shape(*&
_output_shapes
: *
use_locking(

conv1/weights/readIdentityconv1/weights*&
_output_shapes
: *
T0* 
_class
loc:@conv1/weights

conv1/biases/Initializer/zerosConst*
dtype0*
_output_shapes
: *
valueB *    *
_class
loc:@conv1/biases

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
К
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
ш
model/conv1/Conv2DConv2Danchor_inputconv1/weights/read*
paddingSAME*1
_output_shapes
:џџџџџџџџџ ч *
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(

model/conv1/BiasAddBiasAddmodel/conv1/Conv2Dconv1/biases/read*
T0*
data_formatNHWC*1
_output_shapes
:џџџџџџџџџ ч 
o
model/conv1/conv1/ReluRelumodel/conv1/BiasAdd*
T0*1
_output_shapes
:џџџџџџџџџ ч 
Ю
model/conv1/MaxPool2D/MaxPoolMaxPoolmodel/conv1/conv1/Relu*
data_formatNHWC*
strides
*
ksize
*
paddingSAME*1
_output_shapes
:џџџџџџџџџД *
T0
Љ
.conv2/weights/Initializer/random_uniform/shapeConst*%
valueB"          @   * 
_class
loc:@conv2/weights*
dtype0*
_output_shapes
:

,conv2/weights/Initializer/random_uniform/minConst*
valueB
 *ЭЬLН* 
_class
loc:@conv2/weights*
dtype0*
_output_shapes
: 

,conv2/weights/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *
valueB
 *ЭЬL=* 
_class
loc:@conv2/weights
№
6conv2/weights/Initializer/random_uniform/RandomUniformRandomUniform.conv2/weights/Initializer/random_uniform/shape*
T0* 
_class
loc:@conv2/weights*
seed2 *
dtype0*&
_output_shapes
: @*

seed 
в
,conv2/weights/Initializer/random_uniform/subSub,conv2/weights/Initializer/random_uniform/max,conv2/weights/Initializer/random_uniform/min*
T0* 
_class
loc:@conv2/weights*
_output_shapes
: 
ь
,conv2/weights/Initializer/random_uniform/mulMul6conv2/weights/Initializer/random_uniform/RandomUniform,conv2/weights/Initializer/random_uniform/sub*
T0* 
_class
loc:@conv2/weights*&
_output_shapes
: @
о
(conv2/weights/Initializer/random_uniformAdd,conv2/weights/Initializer/random_uniform/mul,conv2/weights/Initializer/random_uniform/min*
T0* 
_class
loc:@conv2/weights*&
_output_shapes
: @
Г
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
г
conv2/weights/AssignAssignconv2/weights(conv2/weights/Initializer/random_uniform*
use_locking(*
T0* 
_class
loc:@conv2/weights*
validate_shape(*&
_output_shapes
: @

conv2/weights/readIdentityconv2/weights*&
_output_shapes
: @*
T0* 
_class
loc:@conv2/weights

conv2/biases/Initializer/zerosConst*
valueB@*    *
_class
loc:@conv2/biases*
dtype0*
_output_shapes
:@

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
К
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
љ
model/conv2/Conv2DConv2Dmodel/conv1/MaxPool2D/MaxPoolconv2/weights/read*
paddingSAME*1
_output_shapes
:џџџџџџџџџД@*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(

model/conv2/BiasAddBiasAddmodel/conv2/Conv2Dconv2/biases/read*
data_formatNHWC*1
_output_shapes
:џџџџџџџџџД@*
T0
o
model/conv2/conv2/ReluRelumodel/conv2/BiasAdd*
T0*1
_output_shapes
:џџџџџџџџџД@
Ю
model/conv2/MaxPool2D/MaxPoolMaxPoolmodel/conv2/conv2/Relu*1
_output_shapes
:џџџџџџџџџ@*
T0*
data_formatNHWC*
strides
*
ksize
*
paddingSAME
Љ
.conv3/weights/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*%
valueB"      @      * 
_class
loc:@conv3/weights

,conv3/weights/Initializer/random_uniform/minConst*
valueB
 *я[qН* 
_class
loc:@conv3/weights*
dtype0*
_output_shapes
: 

,conv3/weights/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *
valueB
 *я[q=* 
_class
loc:@conv3/weights
ё
6conv3/weights/Initializer/random_uniform/RandomUniformRandomUniform.conv3/weights/Initializer/random_uniform/shape*
T0* 
_class
loc:@conv3/weights*
seed2 *
dtype0*'
_output_shapes
:@*

seed 
в
,conv3/weights/Initializer/random_uniform/subSub,conv3/weights/Initializer/random_uniform/max,conv3/weights/Initializer/random_uniform/min*
T0* 
_class
loc:@conv3/weights*
_output_shapes
: 
э
,conv3/weights/Initializer/random_uniform/mulMul6conv3/weights/Initializer/random_uniform/RandomUniform,conv3/weights/Initializer/random_uniform/sub*
T0* 
_class
loc:@conv3/weights*'
_output_shapes
:@
п
(conv3/weights/Initializer/random_uniformAdd,conv3/weights/Initializer/random_uniform/mul,conv3/weights/Initializer/random_uniform/min*
T0* 
_class
loc:@conv3/weights*'
_output_shapes
:@
Е
conv3/weights
VariableV2*
shared_name * 
_class
loc:@conv3/weights*
	container *
shape:@*
dtype0*'
_output_shapes
:@
д
conv3/weights/AssignAssignconv3/weights(conv3/weights/Initializer/random_uniform*
use_locking(*
T0* 
_class
loc:@conv3/weights*
validate_shape(*'
_output_shapes
:@

conv3/weights/readIdentityconv3/weights*
T0* 
_class
loc:@conv3/weights*'
_output_shapes
:@

conv3/biases/Initializer/zerosConst*
dtype0*
_output_shapes	
:*
valueB*    *
_class
loc:@conv3/biases

conv3/biases
VariableV2*
shared_name *
_class
loc:@conv3/biases*
	container *
shape:*
dtype0*
_output_shapes	
:
Л
conv3/biases/AssignAssignconv3/biasesconv3/biases/Initializer/zeros*
use_locking(*
T0*
_class
loc:@conv3/biases*
validate_shape(*
_output_shapes	
:
r
conv3/biases/readIdentityconv3/biases*
T0*
_class
loc:@conv3/biases*
_output_shapes	
:
j
model/conv3/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
њ
model/conv3/Conv2DConv2Dmodel/conv2/MaxPool2D/MaxPoolconv3/weights/read*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*2
_output_shapes 
:џџџџџџџџџ*
	dilations
*
T0

model/conv3/BiasAddBiasAddmodel/conv3/Conv2Dconv3/biases/read*
T0*
data_formatNHWC*2
_output_shapes 
:џџџџџџџџџ
p
model/conv3/conv3/ReluRelumodel/conv3/BiasAdd*
T0*2
_output_shapes 
:џџџџџџџџџ
Ю
model/conv3/MaxPool2D/MaxPoolMaxPoolmodel/conv3/conv3/Relu*1
_output_shapes
:џџџџџџџџџD*
T0*
data_formatNHWC*
strides
*
ksize
*
paddingSAME
Љ
.conv4/weights/Initializer/random_uniform/shapeConst*%
valueB"            * 
_class
loc:@conv4/weights*
dtype0*
_output_shapes
:

,conv4/weights/Initializer/random_uniform/minConst*
valueB
 *   О* 
_class
loc:@conv4/weights*
dtype0*
_output_shapes
: 

,conv4/weights/Initializer/random_uniform/maxConst*
valueB
 *   >* 
_class
loc:@conv4/weights*
dtype0*
_output_shapes
: 
ђ
6conv4/weights/Initializer/random_uniform/RandomUniformRandomUniform.conv4/weights/Initializer/random_uniform/shape*

seed *
T0* 
_class
loc:@conv4/weights*
seed2 *
dtype0*(
_output_shapes
:
в
,conv4/weights/Initializer/random_uniform/subSub,conv4/weights/Initializer/random_uniform/max,conv4/weights/Initializer/random_uniform/min*
T0* 
_class
loc:@conv4/weights*
_output_shapes
: 
ю
,conv4/weights/Initializer/random_uniform/mulMul6conv4/weights/Initializer/random_uniform/RandomUniform,conv4/weights/Initializer/random_uniform/sub*
T0* 
_class
loc:@conv4/weights*(
_output_shapes
:
р
(conv4/weights/Initializer/random_uniformAdd,conv4/weights/Initializer/random_uniform/mul,conv4/weights/Initializer/random_uniform/min*
T0* 
_class
loc:@conv4/weights*(
_output_shapes
:
З
conv4/weights
VariableV2*
dtype0*(
_output_shapes
:*
shared_name * 
_class
loc:@conv4/weights*
	container *
shape:
е
conv4/weights/AssignAssignconv4/weights(conv4/weights/Initializer/random_uniform*
use_locking(*
T0* 
_class
loc:@conv4/weights*
validate_shape(*(
_output_shapes
:

conv4/weights/readIdentityconv4/weights*
T0* 
_class
loc:@conv4/weights*(
_output_shapes
:

conv4/biases/Initializer/zerosConst*
valueB*    *
_class
loc:@conv4/biases*
dtype0*
_output_shapes	
:

conv4/biases
VariableV2*
shared_name *
_class
loc:@conv4/biases*
	container *
shape:*
dtype0*
_output_shapes	
:
Л
conv4/biases/AssignAssignconv4/biasesconv4/biases/Initializer/zeros*
T0*
_class
loc:@conv4/biases*
validate_shape(*
_output_shapes	
:*
use_locking(
r
conv4/biases/readIdentityconv4/biases*
T0*
_class
loc:@conv4/biases*
_output_shapes	
:
j
model/conv4/dilation_rateConst*
dtype0*
_output_shapes
:*
valueB"      
љ
model/conv4/Conv2DConv2Dmodel/conv3/MaxPool2D/MaxPoolconv4/weights/read*
paddingSAME*1
_output_shapes
:џџџџџџџџџD*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(

model/conv4/BiasAddBiasAddmodel/conv4/Conv2Dconv4/biases/read*
T0*
data_formatNHWC*1
_output_shapes
:џџџџџџџџџD
o
model/conv4/conv4/ReluRelumodel/conv4/BiasAdd*
T0*1
_output_shapes
:џџџџџџџџџD
Э
model/conv4/MaxPool2D/MaxPoolMaxPoolmodel/conv4/conv4/Relu*0
_output_shapes
:џџџџџџџџџ"G*
T0*
data_formatNHWC*
strides
*
ksize
*
paddingSAME
Љ
.conv5/weights/Initializer/random_uniform/shapeConst*%
valueB"            * 
_class
loc:@conv5/weights*
dtype0*
_output_shapes
:

,conv5/weights/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *
valueB
 *СжО* 
_class
loc:@conv5/weights

,conv5/weights/Initializer/random_uniform/maxConst*
valueB
 *Сж>* 
_class
loc:@conv5/weights*
dtype0*
_output_shapes
: 
ё
6conv5/weights/Initializer/random_uniform/RandomUniformRandomUniform.conv5/weights/Initializer/random_uniform/shape*
dtype0*'
_output_shapes
:*

seed *
T0* 
_class
loc:@conv5/weights*
seed2 
в
,conv5/weights/Initializer/random_uniform/subSub,conv5/weights/Initializer/random_uniform/max,conv5/weights/Initializer/random_uniform/min*
T0* 
_class
loc:@conv5/weights*
_output_shapes
: 
э
,conv5/weights/Initializer/random_uniform/mulMul6conv5/weights/Initializer/random_uniform/RandomUniform,conv5/weights/Initializer/random_uniform/sub*
T0* 
_class
loc:@conv5/weights*'
_output_shapes
:
п
(conv5/weights/Initializer/random_uniformAdd,conv5/weights/Initializer/random_uniform/mul,conv5/weights/Initializer/random_uniform/min*
T0* 
_class
loc:@conv5/weights*'
_output_shapes
:
Е
conv5/weights
VariableV2*
shared_name * 
_class
loc:@conv5/weights*
	container *
shape:*
dtype0*'
_output_shapes
:
д
conv5/weights/AssignAssignconv5/weights(conv5/weights/Initializer/random_uniform*
use_locking(*
T0* 
_class
loc:@conv5/weights*
validate_shape(*'
_output_shapes
:

conv5/weights/readIdentityconv5/weights*'
_output_shapes
:*
T0* 
_class
loc:@conv5/weights

conv5/biases/Initializer/zerosConst*
valueB*    *
_class
loc:@conv5/biases*
dtype0*
_output_shapes
:

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
К
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
ї
model/conv5/Conv2DConv2Dmodel/conv4/MaxPool2D/MaxPoolconv5/weights/read*
paddingSAME*/
_output_shapes
:џџџџџџџџџ"G*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(

model/conv5/BiasAddBiasAddmodel/conv5/Conv2Dconv5/biases/read*
T0*
data_formatNHWC*/
_output_shapes
:џџџџџџџџџ"G
Щ
model/conv5/MaxPool2D/MaxPoolMaxPoolmodel/conv5/BiasAdd*
T0*
data_formatNHWC*
strides
*
ksize
*
paddingSAME*/
_output_shapes
:џџџџџџџџџ$
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
ч
#model/Flatten/flatten/strided_sliceStridedSlicemodel/Flatten/flatten/Shape)model/Flatten/flatten/strided_slice/stack+model/Flatten/flatten/strided_slice/stack_1+model/Flatten/flatten/strided_slice/stack_2*
Index0*
T0*
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
џџџџџџџџџ
Б
#model/Flatten/flatten/Reshape/shapePack#model/Flatten/flatten/strided_slice%model/Flatten/flatten/Reshape/shape/1*
N*
_output_shapes
:*
T0*

axis 
Ў
model/Flatten/flatten/ReshapeReshapemodel/conv5/MaxPool2D/MaxPool#model/Flatten/flatten/Reshape/shape*)
_output_shapes
:џџџџџџџџџ№*
T0*
Tshape0
l
model_1/conv1/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
ь
model_1/conv1/Conv2DConv2Dpositive_inputconv1/weights/read*
paddingSAME*1
_output_shapes
:џџџџџџџџџ ч *
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(

model_1/conv1/BiasAddBiasAddmodel_1/conv1/Conv2Dconv1/biases/read*
T0*
data_formatNHWC*1
_output_shapes
:џџџџџџџџџ ч 
s
model_1/conv1/conv1/ReluRelumodel_1/conv1/BiasAdd*1
_output_shapes
:џџџџџџџџџ ч *
T0
в
model_1/conv1/MaxPool2D/MaxPoolMaxPoolmodel_1/conv1/conv1/Relu*
ksize
*
paddingSAME*1
_output_shapes
:џџџџџџџџџД *
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
§
model_1/conv2/Conv2DConv2Dmodel_1/conv1/MaxPool2D/MaxPoolconv2/weights/read*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*1
_output_shapes
:џџџџџџџџџД@

model_1/conv2/BiasAddBiasAddmodel_1/conv2/Conv2Dconv2/biases/read*
T0*
data_formatNHWC*1
_output_shapes
:џџџџџџџџџД@
s
model_1/conv2/conv2/ReluRelumodel_1/conv2/BiasAdd*1
_output_shapes
:џџџџџџџџџД@*
T0
в
model_1/conv2/MaxPool2D/MaxPoolMaxPoolmodel_1/conv2/conv2/Relu*
ksize
*
paddingSAME*1
_output_shapes
:џџџџџџџџџ@*
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
ў
model_1/conv3/Conv2DConv2Dmodel_1/conv2/MaxPool2D/MaxPoolconv3/weights/read*
paddingSAME*2
_output_shapes 
:џџџџџџџџџ*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(

model_1/conv3/BiasAddBiasAddmodel_1/conv3/Conv2Dconv3/biases/read*
T0*
data_formatNHWC*2
_output_shapes 
:џџџџџџџџџ
t
model_1/conv3/conv3/ReluRelumodel_1/conv3/BiasAdd*2
_output_shapes 
:џџџџџџџџџ*
T0
в
model_1/conv3/MaxPool2D/MaxPoolMaxPoolmodel_1/conv3/conv3/Relu*
T0*
data_formatNHWC*
strides
*
ksize
*
paddingSAME*1
_output_shapes
:џџџџџџџџџD
l
model_1/conv4/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
§
model_1/conv4/Conv2DConv2Dmodel_1/conv3/MaxPool2D/MaxPoolconv4/weights/read*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*1
_output_shapes
:џџџџџџџџџD*
	dilations


model_1/conv4/BiasAddBiasAddmodel_1/conv4/Conv2Dconv4/biases/read*
T0*
data_formatNHWC*1
_output_shapes
:џџџџџџџџџD
s
model_1/conv4/conv4/ReluRelumodel_1/conv4/BiasAdd*
T0*1
_output_shapes
:џџџџџџџџџD
б
model_1/conv4/MaxPool2D/MaxPoolMaxPoolmodel_1/conv4/conv4/Relu*
ksize
*
paddingSAME*0
_output_shapes
:џџџџџџџџџ"G*
T0*
data_formatNHWC*
strides

l
model_1/conv5/dilation_rateConst*
dtype0*
_output_shapes
:*
valueB"      
ћ
model_1/conv5/Conv2DConv2Dmodel_1/conv4/MaxPool2D/MaxPoolconv5/weights/read*/
_output_shapes
:џџџџџџџџџ"G*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME

model_1/conv5/BiasAddBiasAddmodel_1/conv5/Conv2Dconv5/biases/read*
data_formatNHWC*/
_output_shapes
:џџџџџџџџџ"G*
T0
Э
model_1/conv5/MaxPool2D/MaxPoolMaxPoolmodel_1/conv5/BiasAdd*/
_output_shapes
:џџџџџџџџџ$*
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
ё
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
'model_1/Flatten/flatten/Reshape/shape/1Const*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 
З
%model_1/Flatten/flatten/Reshape/shapePack%model_1/Flatten/flatten/strided_slice'model_1/Flatten/flatten/Reshape/shape/1*
T0*

axis *
N*
_output_shapes
:
Д
model_1/Flatten/flatten/ReshapeReshapemodel_1/conv5/MaxPool2D/MaxPool%model_1/Flatten/flatten/Reshape/shape*
T0*
Tshape0*)
_output_shapes
:џџџџџџџџџ№
l
model_2/conv1/dilation_rateConst*
dtype0*
_output_shapes
:*
valueB"      
ь
model_2/conv1/Conv2DConv2Dnegative_inputconv1/weights/read*1
_output_shapes
:џџџџџџџџџ ч *
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME

model_2/conv1/BiasAddBiasAddmodel_2/conv1/Conv2Dconv1/biases/read*
T0*
data_formatNHWC*1
_output_shapes
:џџџџџџџџџ ч 
s
model_2/conv1/conv1/ReluRelumodel_2/conv1/BiasAdd*
T0*1
_output_shapes
:џџџџџџџџџ ч 
в
model_2/conv1/MaxPool2D/MaxPoolMaxPoolmodel_2/conv1/conv1/Relu*
data_formatNHWC*
strides
*
ksize
*
paddingSAME*1
_output_shapes
:џџџџџџџџџД *
T0
l
model_2/conv2/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
§
model_2/conv2/Conv2DConv2Dmodel_2/conv1/MaxPool2D/MaxPoolconv2/weights/read*1
_output_shapes
:џџџџџџџџџД@*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME

model_2/conv2/BiasAddBiasAddmodel_2/conv2/Conv2Dconv2/biases/read*
T0*
data_formatNHWC*1
_output_shapes
:џџџџџџџџџД@
s
model_2/conv2/conv2/ReluRelumodel_2/conv2/BiasAdd*
T0*1
_output_shapes
:џџџџџџџџџД@
в
model_2/conv2/MaxPool2D/MaxPoolMaxPoolmodel_2/conv2/conv2/Relu*
data_formatNHWC*
strides
*
ksize
*
paddingSAME*1
_output_shapes
:џџџџџџџџџ@*
T0
l
model_2/conv3/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
ў
model_2/conv3/Conv2DConv2Dmodel_2/conv2/MaxPool2D/MaxPoolconv3/weights/read*2
_output_shapes 
:џџџџџџџџџ*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME

model_2/conv3/BiasAddBiasAddmodel_2/conv3/Conv2Dconv3/biases/read*
T0*
data_formatNHWC*2
_output_shapes 
:џџџџџџџџџ
t
model_2/conv3/conv3/ReluRelumodel_2/conv3/BiasAdd*
T0*2
_output_shapes 
:џџџџџџџџџ
в
model_2/conv3/MaxPool2D/MaxPoolMaxPoolmodel_2/conv3/conv3/Relu*
T0*
data_formatNHWC*
strides
*
ksize
*
paddingSAME*1
_output_shapes
:џџџџџџџџџD
l
model_2/conv4/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
§
model_2/conv4/Conv2DConv2Dmodel_2/conv3/MaxPool2D/MaxPoolconv4/weights/read*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*1
_output_shapes
:џџџџџџџџџD*
	dilations
*
T0

model_2/conv4/BiasAddBiasAddmodel_2/conv4/Conv2Dconv4/biases/read*
data_formatNHWC*1
_output_shapes
:џџџџџџџџџD*
T0
s
model_2/conv4/conv4/ReluRelumodel_2/conv4/BiasAdd*1
_output_shapes
:џџџџџџџџџD*
T0
б
model_2/conv4/MaxPool2D/MaxPoolMaxPoolmodel_2/conv4/conv4/Relu*
data_formatNHWC*
strides
*
ksize
*
paddingSAME*0
_output_shapes
:џџџџџџџџџ"G*
T0
l
model_2/conv5/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
ћ
model_2/conv5/Conv2DConv2Dmodel_2/conv4/MaxPool2D/MaxPoolconv5/weights/read*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME*/
_output_shapes
:џџџџџџџџџ"G*
	dilations


model_2/conv5/BiasAddBiasAddmodel_2/conv5/Conv2Dconv5/biases/read*
T0*
data_formatNHWC*/
_output_shapes
:џџџџџџџџџ"G
Э
model_2/conv5/MaxPool2D/MaxPoolMaxPoolmodel_2/conv5/BiasAdd*
ksize
*
paddingSAME*/
_output_shapes
:џџџџџџџџџ$*
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
ё
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
'model_2/Flatten/flatten/Reshape/shape/1Const*
dtype0*
_output_shapes
: *
valueB :
џџџџџџџџџ
З
%model_2/Flatten/flatten/Reshape/shapePack%model_2/Flatten/flatten/strided_slice'model_2/Flatten/flatten/Reshape/shape/1*
T0*

axis *
N*
_output_shapes
:
Д
model_2/Flatten/flatten/ReshapeReshapemodel_2/conv5/MaxPool2D/MaxPool%model_2/Flatten/flatten/Reshape/shape*
T0*
Tshape0*)
_output_shapes
:џџџџџџџџџ№
~
subSubmodel/Flatten/flatten/Reshapemodel_1/Flatten/flatten/Reshape*
T0*)
_output_shapes
:џџџџџџџџџ№
J
Pow/yConst*
valueB
 *   @*
dtype0*
_output_shapes
: 
J
PowPowsubPow/y*
T0*)
_output_shapes
:џџџџџџџџџ№
W
Sum/reduction_indicesConst*
dtype0*
_output_shapes
: *
value	B :
u
SumSumPowSum/reduction_indices*
T0*'
_output_shapes
:џџџџџџџџџ*
	keep_dims(*

Tidx0
C
SqrtSqrtSum*
T0*'
_output_shapes
:џџџџџџџџџ

sub_1Submodel/Flatten/flatten/Reshapemodel_2/Flatten/flatten/Reshape*
T0*)
_output_shapes
:џџџџџџџџџ№
L
Pow_1/yConst*
valueB
 *   @*
dtype0*
_output_shapes
: 
P
Pow_1Powsub_1Pow_1/y*
T0*)
_output_shapes
:џџџџџџџџџ№
Y
Sum_1/reduction_indicesConst*
value	B :*
dtype0*
_output_shapes
: 
{
Sum_1SumPow_1Sum_1/reduction_indices*
	keep_dims(*

Tidx0*
T0*'
_output_shapes
:џџџџџџџџџ
G
Sqrt_1SqrtSum_1*
T0*'
_output_shapes
:џџџџџџџџџ
L
sub_2SubSqrtSqrt_1*
T0*'
_output_shapes
:џџџџџџџџџ
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
:џџџџџџџџџ
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
:џџџџџџџџџ
V
ConstConst*
valueB"       *
dtype0*
_output_shapes
:
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
Ђ
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
 *  ?
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

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

gradients/Mean_grad/TileTilegradients/Mean_grad/Reshapegradients/Mean_grad/Shape*'
_output_shapes
:џџџџџџџџџ*

Tmultiples0*
T0
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

gradients/Mean_grad/ProdProdgradients/Mean_grad/Shape_1gradients/Mean_grad/Const*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
e
gradients/Mean_grad/Const_1Const*
dtype0*
_output_shapes
:*
valueB: 

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

gradients/Mean_grad/MaximumMaximumgradients/Mean_grad/Prod_1gradients/Mean_grad/Maximum/y*
T0*
_output_shapes
: 

gradients/Mean_grad/floordivFloorDivgradients/Mean_grad/Prodgradients/Mean_grad/Maximum*
_output_shapes
: *
T0
~
gradients/Mean_grad/CastCastgradients/Mean_grad/floordiv*

SrcT0*
Truncate( *
_output_shapes
: *

DstT0

gradients/Mean_grad/truedivRealDivgradients/Mean_grad/Tilegradients/Mean_grad/Cast*
T0*'
_output_shapes
:џџџџџџџџџ
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
Ќ
gradients/Maximum_grad/zerosFillgradients/Maximum_grad/Shape_2"gradients/Maximum_grad/zeros/Const*
T0*

index_type0*'
_output_shapes
:џџџџџџџџџ
u
#gradients/Maximum_grad/GreaterEqualGreaterEqualadd	Maximum/y*
T0*'
_output_shapes
:џџџџџџџџџ
Р
,gradients/Maximum_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/Maximum_grad/Shapegradients/Maximum_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0
Й
gradients/Maximum_grad/SelectSelect#gradients/Maximum_grad/GreaterEqualgradients/Mean_grad/truedivgradients/Maximum_grad/zeros*'
_output_shapes
:џџџџџџџџџ*
T0
Л
gradients/Maximum_grad/Select_1Select#gradients/Maximum_grad/GreaterEqualgradients/Maximum_grad/zerosgradients/Mean_grad/truediv*
T0*'
_output_shapes
:џџџџџџџџџ
Ў
gradients/Maximum_grad/SumSumgradients/Maximum_grad/Select,gradients/Maximum_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Ѓ
gradients/Maximum_grad/ReshapeReshapegradients/Maximum_grad/Sumgradients/Maximum_grad/Shape*'
_output_shapes
:џџџџџџџџџ*
T0*
Tshape0
Д
gradients/Maximum_grad/Sum_1Sumgradients/Maximum_grad/Select_1.gradients/Maximum_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0

 gradients/Maximum_grad/Reshape_1Reshapegradients/Maximum_grad/Sum_1gradients/Maximum_grad/Shape_1*
_output_shapes
: *
T0*
Tshape0
s
'gradients/Maximum_grad/tuple/group_depsNoOp^gradients/Maximum_grad/Reshape!^gradients/Maximum_grad/Reshape_1
ъ
/gradients/Maximum_grad/tuple/control_dependencyIdentitygradients/Maximum_grad/Reshape(^gradients/Maximum_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/Maximum_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
п
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
Д
(gradients/add_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_grad/Shapegradients/add_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
И
gradients/add_grad/SumSum/gradients/Maximum_grad/tuple/control_dependency(gradients/add_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0

gradients/add_grad/ReshapeReshapegradients/add_grad/Sumgradients/add_grad/Shape*'
_output_shapes
:џџџџџџџџџ*
T0*
Tshape0
М
gradients/add_grad/Sum_1Sum/gradients/Maximum_grad/tuple/control_dependency*gradients/add_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0

gradients/add_grad/Reshape_1Reshapegradients/add_grad/Sum_1gradients/add_grad/Shape_1*
_output_shapes
: *
T0*
Tshape0
g
#gradients/add_grad/tuple/group_depsNoOp^gradients/add_grad/Reshape^gradients/add_grad/Reshape_1
к
+gradients/add_grad/tuple/control_dependencyIdentitygradients/add_grad/Reshape$^gradients/add_grad/tuple/group_deps*
T0*-
_class#
!loc:@gradients/add_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
Я
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
К
*gradients/sub_2_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/sub_2_grad/Shapegradients/sub_2_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
И
gradients/sub_2_grad/SumSum+gradients/add_grad/tuple/control_dependency*gradients/sub_2_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:

gradients/sub_2_grad/ReshapeReshapegradients/sub_2_grad/Sumgradients/sub_2_grad/Shape*'
_output_shapes
:џџџџџџџџџ*
T0*
Tshape0
М
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
Ё
gradients/sub_2_grad/Reshape_1Reshapegradients/sub_2_grad/Neggradients/sub_2_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
m
%gradients/sub_2_grad/tuple/group_depsNoOp^gradients/sub_2_grad/Reshape^gradients/sub_2_grad/Reshape_1
т
-gradients/sub_2_grad/tuple/control_dependencyIdentitygradients/sub_2_grad/Reshape&^gradients/sub_2_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/sub_2_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
ш
/gradients/sub_2_grad/tuple/control_dependency_1Identitygradients/sub_2_grad/Reshape_1&^gradients/sub_2_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/sub_2_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџ

gradients/Sqrt_grad/SqrtGradSqrtGradSqrt-gradients/sub_2_grad/tuple/control_dependency*'
_output_shapes
:џџџџџџџџџ*
T0

gradients/Sqrt_1_grad/SqrtGradSqrtGradSqrt_1/gradients/sub_2_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:џџџџџџџџџ
[
gradients/Sum_grad/ShapeShapePow*
_output_shapes
:*
T0*
out_type0

gradients/Sum_grad/SizeConst*
dtype0*
_output_shapes
: *
value	B :*+
_class!
loc:@gradients/Sum_grad/Shape

gradients/Sum_grad/addAddSum/reduction_indicesgradients/Sum_grad/Size*
T0*+
_class!
loc:@gradients/Sum_grad/Shape*
_output_shapes
: 
Ё
gradients/Sum_grad/modFloorModgradients/Sum_grad/addgradients/Sum_grad/Size*
T0*+
_class!
loc:@gradients/Sum_grad/Shape*
_output_shapes
: 

gradients/Sum_grad/Shape_1Const*
valueB *+
_class!
loc:@gradients/Sum_grad/Shape*
dtype0*
_output_shapes
: 

gradients/Sum_grad/range/startConst*
value	B : *+
_class!
loc:@gradients/Sum_grad/Shape*
dtype0*
_output_shapes
: 

gradients/Sum_grad/range/deltaConst*
value	B :*+
_class!
loc:@gradients/Sum_grad/Shape*
dtype0*
_output_shapes
: 
Я
gradients/Sum_grad/rangeRangegradients/Sum_grad/range/startgradients/Sum_grad/Sizegradients/Sum_grad/range/delta*

Tidx0*+
_class!
loc:@gradients/Sum_grad/Shape*
_output_shapes
:

gradients/Sum_grad/Fill/valueConst*
value	B :*+
_class!
loc:@gradients/Sum_grad/Shape*
dtype0*
_output_shapes
: 
К
gradients/Sum_grad/FillFillgradients/Sum_grad/Shape_1gradients/Sum_grad/Fill/value*
T0*

index_type0*+
_class!
loc:@gradients/Sum_grad/Shape*
_output_shapes
: 
ё
 gradients/Sum_grad/DynamicStitchDynamicStitchgradients/Sum_grad/rangegradients/Sum_grad/modgradients/Sum_grad/Shapegradients/Sum_grad/Fill*
T0*+
_class!
loc:@gradients/Sum_grad/Shape*
N*
_output_shapes
:

gradients/Sum_grad/Maximum/yConst*
value	B :*+
_class!
loc:@gradients/Sum_grad/Shape*
dtype0*
_output_shapes
: 
З
gradients/Sum_grad/MaximumMaximum gradients/Sum_grad/DynamicStitchgradients/Sum_grad/Maximum/y*
T0*+
_class!
loc:@gradients/Sum_grad/Shape*
_output_shapes
:
Џ
gradients/Sum_grad/floordivFloorDivgradients/Sum_grad/Shapegradients/Sum_grad/Maximum*
_output_shapes
:*
T0*+
_class!
loc:@gradients/Sum_grad/Shape
Ў
gradients/Sum_grad/ReshapeReshapegradients/Sqrt_grad/SqrtGrad gradients/Sum_grad/DynamicStitch*
T0*
Tshape0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ

gradients/Sum_grad/TileTilegradients/Sum_grad/Reshapegradients/Sum_grad/floordiv*

Tmultiples0*
T0*)
_output_shapes
:џџџџџџџџџ№
_
gradients/Sum_1_grad/ShapeShapePow_1*
T0*
out_type0*
_output_shapes
:

gradients/Sum_1_grad/SizeConst*
value	B :*-
_class#
!loc:@gradients/Sum_1_grad/Shape*
dtype0*
_output_shapes
: 
Ѓ
gradients/Sum_1_grad/addAddSum_1/reduction_indicesgradients/Sum_1_grad/Size*
T0*-
_class#
!loc:@gradients/Sum_1_grad/Shape*
_output_shapes
: 
Љ
gradients/Sum_1_grad/modFloorModgradients/Sum_1_grad/addgradients/Sum_1_grad/Size*
T0*-
_class#
!loc:@gradients/Sum_1_grad/Shape*
_output_shapes
: 

gradients/Sum_1_grad/Shape_1Const*
valueB *-
_class#
!loc:@gradients/Sum_1_grad/Shape*
dtype0*
_output_shapes
: 

 gradients/Sum_1_grad/range/startConst*
dtype0*
_output_shapes
: *
value	B : *-
_class#
!loc:@gradients/Sum_1_grad/Shape

 gradients/Sum_1_grad/range/deltaConst*
dtype0*
_output_shapes
: *
value	B :*-
_class#
!loc:@gradients/Sum_1_grad/Shape
й
gradients/Sum_1_grad/rangeRange gradients/Sum_1_grad/range/startgradients/Sum_1_grad/Size gradients/Sum_1_grad/range/delta*-
_class#
!loc:@gradients/Sum_1_grad/Shape*
_output_shapes
:*

Tidx0

gradients/Sum_1_grad/Fill/valueConst*
value	B :*-
_class#
!loc:@gradients/Sum_1_grad/Shape*
dtype0*
_output_shapes
: 
Т
gradients/Sum_1_grad/FillFillgradients/Sum_1_grad/Shape_1gradients/Sum_1_grad/Fill/value*
T0*

index_type0*-
_class#
!loc:@gradients/Sum_1_grad/Shape*
_output_shapes
: 
§
"gradients/Sum_1_grad/DynamicStitchDynamicStitchgradients/Sum_1_grad/rangegradients/Sum_1_grad/modgradients/Sum_1_grad/Shapegradients/Sum_1_grad/Fill*
N*
_output_shapes
:*
T0*-
_class#
!loc:@gradients/Sum_1_grad/Shape

gradients/Sum_1_grad/Maximum/yConst*
value	B :*-
_class#
!loc:@gradients/Sum_1_grad/Shape*
dtype0*
_output_shapes
: 
П
gradients/Sum_1_grad/MaximumMaximum"gradients/Sum_1_grad/DynamicStitchgradients/Sum_1_grad/Maximum/y*
T0*-
_class#
!loc:@gradients/Sum_1_grad/Shape*
_output_shapes
:
З
gradients/Sum_1_grad/floordivFloorDivgradients/Sum_1_grad/Shapegradients/Sum_1_grad/Maximum*
_output_shapes
:*
T0*-
_class#
!loc:@gradients/Sum_1_grad/Shape
Д
gradients/Sum_1_grad/ReshapeReshapegradients/Sqrt_1_grad/SqrtGrad"gradients/Sum_1_grad/DynamicStitch*
T0*
Tshape0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
Є
gradients/Sum_1_grad/TileTilegradients/Sum_1_grad/Reshapegradients/Sum_1_grad/floordiv*
T0*)
_output_shapes
:џџџџџџџџџ№*

Tmultiples0
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
Д
(gradients/Pow_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/Pow_grad/Shapegradients/Pow_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
q
gradients/Pow_grad/mulMulgradients/Sum_grad/TilePow/y*
T0*)
_output_shapes
:џџџџџџџџџ№
]
gradients/Pow_grad/sub/yConst*
dtype0*
_output_shapes
: *
valueB
 *  ?
_
gradients/Pow_grad/subSubPow/ygradients/Pow_grad/sub/y*
T0*
_output_shapes
: 
n
gradients/Pow_grad/PowPowsubgradients/Pow_grad/sub*)
_output_shapes
:џџџџџџџџџ№*
T0

gradients/Pow_grad/mul_1Mulgradients/Pow_grad/mulgradients/Pow_grad/Pow*
T0*)
_output_shapes
:џџџџџџџџџ№
Ё
gradients/Pow_grad/SumSumgradients/Pow_grad/mul_1(gradients/Pow_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:

gradients/Pow_grad/ReshapeReshapegradients/Pow_grad/Sumgradients/Pow_grad/Shape*
T0*
Tshape0*)
_output_shapes
:џџџџџџџџџ№
a
gradients/Pow_grad/Greater/yConst*
valueB
 *    *
dtype0*
_output_shapes
: 
|
gradients/Pow_grad/GreaterGreatersubgradients/Pow_grad/Greater/y*
T0*)
_output_shapes
:џџџџџџџџџ№
e
"gradients/Pow_grad/ones_like/ShapeShapesub*
_output_shapes
:*
T0*
out_type0
g
"gradients/Pow_grad/ones_like/ConstConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
В
gradients/Pow_grad/ones_likeFill"gradients/Pow_grad/ones_like/Shape"gradients/Pow_grad/ones_like/Const*
T0*

index_type0*)
_output_shapes
:џџџџџџџџџ№

gradients/Pow_grad/SelectSelectgradients/Pow_grad/Greatersubgradients/Pow_grad/ones_like*
T0*)
_output_shapes
:џџџџџџџџџ№
l
gradients/Pow_grad/LogLoggradients/Pow_grad/Select*
T0*)
_output_shapes
:џџџџџџџџџ№
c
gradients/Pow_grad/zeros_like	ZerosLikesub*
T0*)
_output_shapes
:џџџџџџџџџ№
Ќ
gradients/Pow_grad/Select_1Selectgradients/Pow_grad/Greatergradients/Pow_grad/Loggradients/Pow_grad/zeros_like*
T0*)
_output_shapes
:џџџџџџџџџ№
q
gradients/Pow_grad/mul_2Mulgradients/Sum_grad/TilePow*)
_output_shapes
:џџџџџџџџџ№*
T0

gradients/Pow_grad/mul_3Mulgradients/Pow_grad/mul_2gradients/Pow_grad/Select_1*
T0*)
_output_shapes
:џџџџџџџџџ№
Ѕ
gradients/Pow_grad/Sum_1Sumgradients/Pow_grad/mul_3*gradients/Pow_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:

gradients/Pow_grad/Reshape_1Reshapegradients/Pow_grad/Sum_1gradients/Pow_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
g
#gradients/Pow_grad/tuple/group_depsNoOp^gradients/Pow_grad/Reshape^gradients/Pow_grad/Reshape_1
м
+gradients/Pow_grad/tuple/control_dependencyIdentitygradients/Pow_grad/Reshape$^gradients/Pow_grad/tuple/group_deps*
T0*-
_class#
!loc:@gradients/Pow_grad/Reshape*)
_output_shapes
:џџџџџџџџџ№
Я
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
gradients/Pow_1_grad/Shape_1Const*
dtype0*
_output_shapes
: *
valueB 
К
*gradients/Pow_1_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/Pow_1_grad/Shapegradients/Pow_1_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
w
gradients/Pow_1_grad/mulMulgradients/Sum_1_grad/TilePow_1/y*
T0*)
_output_shapes
:џџџџџџџџџ№
_
gradients/Pow_1_grad/sub/yConst*
dtype0*
_output_shapes
: *
valueB
 *  ?
e
gradients/Pow_1_grad/subSubPow_1/ygradients/Pow_1_grad/sub/y*
_output_shapes
: *
T0
t
gradients/Pow_1_grad/PowPowsub_1gradients/Pow_1_grad/sub*
T0*)
_output_shapes
:џџџџџџџџџ№

gradients/Pow_1_grad/mul_1Mulgradients/Pow_1_grad/mulgradients/Pow_1_grad/Pow*
T0*)
_output_shapes
:џџџџџџџџџ№
Ї
gradients/Pow_1_grad/SumSumgradients/Pow_1_grad/mul_1*gradients/Pow_1_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:

gradients/Pow_1_grad/ReshapeReshapegradients/Pow_1_grad/Sumgradients/Pow_1_grad/Shape*
T0*
Tshape0*)
_output_shapes
:џџџџџџџџџ№
c
gradients/Pow_1_grad/Greater/yConst*
valueB
 *    *
dtype0*
_output_shapes
: 

gradients/Pow_1_grad/GreaterGreatersub_1gradients/Pow_1_grad/Greater/y*
T0*)
_output_shapes
:џџџџџџџџџ№
i
$gradients/Pow_1_grad/ones_like/ShapeShapesub_1*
T0*
out_type0*
_output_shapes
:
i
$gradients/Pow_1_grad/ones_like/ConstConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
И
gradients/Pow_1_grad/ones_likeFill$gradients/Pow_1_grad/ones_like/Shape$gradients/Pow_1_grad/ones_like/Const*
T0*

index_type0*)
_output_shapes
:џџџџџџџџџ№

gradients/Pow_1_grad/SelectSelectgradients/Pow_1_grad/Greatersub_1gradients/Pow_1_grad/ones_like*
T0*)
_output_shapes
:џџџџџџџџџ№
p
gradients/Pow_1_grad/LogLoggradients/Pow_1_grad/Select*)
_output_shapes
:џџџџџџџџџ№*
T0
g
gradients/Pow_1_grad/zeros_like	ZerosLikesub_1*
T0*)
_output_shapes
:џџџџџџџџџ№
Д
gradients/Pow_1_grad/Select_1Selectgradients/Pow_1_grad/Greatergradients/Pow_1_grad/Loggradients/Pow_1_grad/zeros_like*)
_output_shapes
:џџџџџџџџџ№*
T0
w
gradients/Pow_1_grad/mul_2Mulgradients/Sum_1_grad/TilePow_1*)
_output_shapes
:џџџџџџџџџ№*
T0

gradients/Pow_1_grad/mul_3Mulgradients/Pow_1_grad/mul_2gradients/Pow_1_grad/Select_1*
T0*)
_output_shapes
:џџџџџџџџџ№
Ћ
gradients/Pow_1_grad/Sum_1Sumgradients/Pow_1_grad/mul_3,gradients/Pow_1_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0

gradients/Pow_1_grad/Reshape_1Reshapegradients/Pow_1_grad/Sum_1gradients/Pow_1_grad/Shape_1*
_output_shapes
: *
T0*
Tshape0
m
%gradients/Pow_1_grad/tuple/group_depsNoOp^gradients/Pow_1_grad/Reshape^gradients/Pow_1_grad/Reshape_1
ф
-gradients/Pow_1_grad/tuple/control_dependencyIdentitygradients/Pow_1_grad/Reshape&^gradients/Pow_1_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/Pow_1_grad/Reshape*)
_output_shapes
:џџџџџџџџџ№
з
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
Д
(gradients/sub_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/sub_grad/Shapegradients/sub_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Д
gradients/sub_grad/SumSum+gradients/Pow_grad/tuple/control_dependency(gradients/sub_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0

gradients/sub_grad/ReshapeReshapegradients/sub_grad/Sumgradients/sub_grad/Shape*
T0*
Tshape0*)
_output_shapes
:џџџџџџџџџ№
И
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

gradients/sub_grad/Reshape_1Reshapegradients/sub_grad/Neggradients/sub_grad/Shape_1*
T0*
Tshape0*)
_output_shapes
:џџџџџџџџџ№
g
#gradients/sub_grad/tuple/group_depsNoOp^gradients/sub_grad/Reshape^gradients/sub_grad/Reshape_1
м
+gradients/sub_grad/tuple/control_dependencyIdentitygradients/sub_grad/Reshape$^gradients/sub_grad/tuple/group_deps*
T0*-
_class#
!loc:@gradients/sub_grad/Reshape*)
_output_shapes
:џџџџџџџџџ№
т
-gradients/sub_grad/tuple/control_dependency_1Identitygradients/sub_grad/Reshape_1$^gradients/sub_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/sub_grad/Reshape_1*)
_output_shapes
:џџџџџџџџџ№
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
К
*gradients/sub_1_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/sub_1_grad/Shapegradients/sub_1_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0
К
gradients/sub_1_grad/SumSum-gradients/Pow_1_grad/tuple/control_dependency*gradients/sub_1_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0

gradients/sub_1_grad/ReshapeReshapegradients/sub_1_grad/Sumgradients/sub_1_grad/Shape*)
_output_shapes
:џџџџџџџџџ№*
T0*
Tshape0
О
gradients/sub_1_grad/Sum_1Sum-gradients/Pow_1_grad/tuple/control_dependency,gradients/sub_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
^
gradients/sub_1_grad/NegNeggradients/sub_1_grad/Sum_1*
T0*
_output_shapes
:
Ѓ
gradients/sub_1_grad/Reshape_1Reshapegradients/sub_1_grad/Neggradients/sub_1_grad/Shape_1*)
_output_shapes
:џџџџџџџџџ№*
T0*
Tshape0
m
%gradients/sub_1_grad/tuple/group_depsNoOp^gradients/sub_1_grad/Reshape^gradients/sub_1_grad/Reshape_1
ф
-gradients/sub_1_grad/tuple/control_dependencyIdentitygradients/sub_1_grad/Reshape&^gradients/sub_1_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/sub_1_grad/Reshape*)
_output_shapes
:џџџџџџџџџ№
ъ
/gradients/sub_1_grad/tuple/control_dependency_1Identitygradients/sub_1_grad/Reshape_1&^gradients/sub_1_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/sub_1_grad/Reshape_1*)
_output_shapes
:џџџџџџџџџ№

4gradients/model_1/Flatten/flatten/Reshape_grad/ShapeShapemodel_1/conv5/MaxPool2D/MaxPool*
T0*
out_type0*
_output_shapes
:
ю
6gradients/model_1/Flatten/flatten/Reshape_grad/ReshapeReshape-gradients/sub_grad/tuple/control_dependency_14gradients/model_1/Flatten/flatten/Reshape_grad/Shape*
T0*
Tshape0*/
_output_shapes
:џџџџџџџџџ$
о
gradients/AddNAddN+gradients/sub_grad/tuple/control_dependency-gradients/sub_1_grad/tuple/control_dependency*
T0*-
_class#
!loc:@gradients/sub_grad/Reshape*
N*)
_output_shapes
:џџџџџџџџџ№

2gradients/model/Flatten/flatten/Reshape_grad/ShapeShapemodel/conv5/MaxPool2D/MaxPool*
T0*
out_type0*
_output_shapes
:
Ы
4gradients/model/Flatten/flatten/Reshape_grad/ReshapeReshapegradients/AddN2gradients/model/Flatten/flatten/Reshape_grad/Shape*
T0*
Tshape0*/
_output_shapes
:џџџџџџџџџ$

4gradients/model_2/Flatten/flatten/Reshape_grad/ShapeShapemodel_2/conv5/MaxPool2D/MaxPool*
T0*
out_type0*
_output_shapes
:
№
6gradients/model_2/Flatten/flatten/Reshape_grad/ReshapeReshape/gradients/sub_1_grad/tuple/control_dependency_14gradients/model_2/Flatten/flatten/Reshape_grad/Shape*
T0*
Tshape0*/
_output_shapes
:џџџџџџџџџ$
Х
:gradients/model_1/conv5/MaxPool2D/MaxPool_grad/MaxPoolGradMaxPoolGradmodel_1/conv5/BiasAddmodel_1/conv5/MaxPool2D/MaxPool6gradients/model_1/Flatten/flatten/Reshape_grad/Reshape*
ksize
*
paddingSAME*/
_output_shapes
:џџџџџџџџџ"G*
T0*
data_formatNHWC*
strides

Н
8gradients/model/conv5/MaxPool2D/MaxPool_grad/MaxPoolGradMaxPoolGradmodel/conv5/BiasAddmodel/conv5/MaxPool2D/MaxPool4gradients/model/Flatten/flatten/Reshape_grad/Reshape*
T0*
data_formatNHWC*
strides
*
ksize
*
paddingSAME*/
_output_shapes
:џџџџџџџџџ"G
Х
:gradients/model_2/conv5/MaxPool2D/MaxPool_grad/MaxPoolGradMaxPoolGradmodel_2/conv5/BiasAddmodel_2/conv5/MaxPool2D/MaxPool6gradients/model_2/Flatten/flatten/Reshape_grad/Reshape*
T0*
data_formatNHWC*
strides
*
ksize
*
paddingSAME*/
_output_shapes
:џџџџџџџџџ"G
З
0gradients/model_1/conv5/BiasAdd_grad/BiasAddGradBiasAddGrad:gradients/model_1/conv5/MaxPool2D/MaxPool_grad/MaxPoolGrad*
T0*
data_formatNHWC*
_output_shapes
:
­
5gradients/model_1/conv5/BiasAdd_grad/tuple/group_depsNoOp1^gradients/model_1/conv5/BiasAdd_grad/BiasAddGrad;^gradients/model_1/conv5/MaxPool2D/MaxPool_grad/MaxPoolGrad
Ц
=gradients/model_1/conv5/BiasAdd_grad/tuple/control_dependencyIdentity:gradients/model_1/conv5/MaxPool2D/MaxPool_grad/MaxPoolGrad6^gradients/model_1/conv5/BiasAdd_grad/tuple/group_deps*
T0*M
_classC
A?loc:@gradients/model_1/conv5/MaxPool2D/MaxPool_grad/MaxPoolGrad*/
_output_shapes
:џџџџџџџџџ"G

?gradients/model_1/conv5/BiasAdd_grad/tuple/control_dependency_1Identity0gradients/model_1/conv5/BiasAdd_grad/BiasAddGrad6^gradients/model_1/conv5/BiasAdd_grad/tuple/group_deps*
_output_shapes
:*
T0*C
_class9
75loc:@gradients/model_1/conv5/BiasAdd_grad/BiasAddGrad
Г
.gradients/model/conv5/BiasAdd_grad/BiasAddGradBiasAddGrad8gradients/model/conv5/MaxPool2D/MaxPool_grad/MaxPoolGrad*
data_formatNHWC*
_output_shapes
:*
T0
Ї
3gradients/model/conv5/BiasAdd_grad/tuple/group_depsNoOp/^gradients/model/conv5/BiasAdd_grad/BiasAddGrad9^gradients/model/conv5/MaxPool2D/MaxPool_grad/MaxPoolGrad
О
;gradients/model/conv5/BiasAdd_grad/tuple/control_dependencyIdentity8gradients/model/conv5/MaxPool2D/MaxPool_grad/MaxPoolGrad4^gradients/model/conv5/BiasAdd_grad/tuple/group_deps*
T0*K
_classA
?=loc:@gradients/model/conv5/MaxPool2D/MaxPool_grad/MaxPoolGrad*/
_output_shapes
:џџџџџџџџџ"G

=gradients/model/conv5/BiasAdd_grad/tuple/control_dependency_1Identity.gradients/model/conv5/BiasAdd_grad/BiasAddGrad4^gradients/model/conv5/BiasAdd_grad/tuple/group_deps*
_output_shapes
:*
T0*A
_class7
53loc:@gradients/model/conv5/BiasAdd_grad/BiasAddGrad
З
0gradients/model_2/conv5/BiasAdd_grad/BiasAddGradBiasAddGrad:gradients/model_2/conv5/MaxPool2D/MaxPool_grad/MaxPoolGrad*
data_formatNHWC*
_output_shapes
:*
T0
­
5gradients/model_2/conv5/BiasAdd_grad/tuple/group_depsNoOp1^gradients/model_2/conv5/BiasAdd_grad/BiasAddGrad;^gradients/model_2/conv5/MaxPool2D/MaxPool_grad/MaxPoolGrad
Ц
=gradients/model_2/conv5/BiasAdd_grad/tuple/control_dependencyIdentity:gradients/model_2/conv5/MaxPool2D/MaxPool_grad/MaxPoolGrad6^gradients/model_2/conv5/BiasAdd_grad/tuple/group_deps*
T0*M
_classC
A?loc:@gradients/model_2/conv5/MaxPool2D/MaxPool_grad/MaxPoolGrad*/
_output_shapes
:џџџџџџџџџ"G

?gradients/model_2/conv5/BiasAdd_grad/tuple/control_dependency_1Identity0gradients/model_2/conv5/BiasAdd_grad/BiasAddGrad6^gradients/model_2/conv5/BiasAdd_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/model_2/conv5/BiasAdd_grad/BiasAddGrad*
_output_shapes
:
­
*gradients/model_1/conv5/Conv2D_grad/ShapeNShapeNmodel_1/conv4/MaxPool2D/MaxPoolconv5/weights/read*
N* 
_output_shapes
::*
T0*
out_type0
і
7gradients/model_1/conv5/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput*gradients/model_1/conv5/Conv2D_grad/ShapeNconv5/weights/read=gradients/model_1/conv5/BiasAdd_grad/tuple/control_dependency*
paddingSAME*0
_output_shapes
:џџџџџџџџџ"G*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
ў
8gradients/model_1/conv5/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFiltermodel_1/conv4/MaxPool2D/MaxPool,gradients/model_1/conv5/Conv2D_grad/ShapeN:1=gradients/model_1/conv5/BiasAdd_grad/tuple/control_dependency*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*'
_output_shapes
:*
	dilations
*
T0
Б
4gradients/model_1/conv5/Conv2D_grad/tuple/group_depsNoOp9^gradients/model_1/conv5/Conv2D_grad/Conv2DBackpropFilter8^gradients/model_1/conv5/Conv2D_grad/Conv2DBackpropInput
П
<gradients/model_1/conv5/Conv2D_grad/tuple/control_dependencyIdentity7gradients/model_1/conv5/Conv2D_grad/Conv2DBackpropInput5^gradients/model_1/conv5/Conv2D_grad/tuple/group_deps*
T0*J
_class@
><loc:@gradients/model_1/conv5/Conv2D_grad/Conv2DBackpropInput*0
_output_shapes
:џџџџџџџџџ"G
К
>gradients/model_1/conv5/Conv2D_grad/tuple/control_dependency_1Identity8gradients/model_1/conv5/Conv2D_grad/Conv2DBackpropFilter5^gradients/model_1/conv5/Conv2D_grad/tuple/group_deps*
T0*K
_classA
?=loc:@gradients/model_1/conv5/Conv2D_grad/Conv2DBackpropFilter*'
_output_shapes
:
Љ
(gradients/model/conv5/Conv2D_grad/ShapeNShapeNmodel/conv4/MaxPool2D/MaxPoolconv5/weights/read*
N* 
_output_shapes
::*
T0*
out_type0
№
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
:џџџџџџџџџ"G
і
6gradients/model/conv5/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFiltermodel/conv4/MaxPool2D/MaxPool*gradients/model/conv5/Conv2D_grad/ShapeN:1;gradients/model/conv5/BiasAdd_grad/tuple/control_dependency*
paddingSAME*'
_output_shapes
:*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
Ћ
2gradients/model/conv5/Conv2D_grad/tuple/group_depsNoOp7^gradients/model/conv5/Conv2D_grad/Conv2DBackpropFilter6^gradients/model/conv5/Conv2D_grad/Conv2DBackpropInput
З
:gradients/model/conv5/Conv2D_grad/tuple/control_dependencyIdentity5gradients/model/conv5/Conv2D_grad/Conv2DBackpropInput3^gradients/model/conv5/Conv2D_grad/tuple/group_deps*
T0*H
_class>
<:loc:@gradients/model/conv5/Conv2D_grad/Conv2DBackpropInput*0
_output_shapes
:џџџџџџџџџ"G
В
<gradients/model/conv5/Conv2D_grad/tuple/control_dependency_1Identity6gradients/model/conv5/Conv2D_grad/Conv2DBackpropFilter3^gradients/model/conv5/Conv2D_grad/tuple/group_deps*
T0*I
_class?
=;loc:@gradients/model/conv5/Conv2D_grad/Conv2DBackpropFilter*'
_output_shapes
:
­
*gradients/model_2/conv5/Conv2D_grad/ShapeNShapeNmodel_2/conv4/MaxPool2D/MaxPoolconv5/weights/read*
T0*
out_type0*
N* 
_output_shapes
::
і
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
:џџџџџџџџџ"G
ў
8gradients/model_2/conv5/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFiltermodel_2/conv4/MaxPool2D/MaxPool,gradients/model_2/conv5/Conv2D_grad/ShapeN:1=gradients/model_2/conv5/BiasAdd_grad/tuple/control_dependency*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*'
_output_shapes
:*
	dilations

Б
4gradients/model_2/conv5/Conv2D_grad/tuple/group_depsNoOp9^gradients/model_2/conv5/Conv2D_grad/Conv2DBackpropFilter8^gradients/model_2/conv5/Conv2D_grad/Conv2DBackpropInput
П
<gradients/model_2/conv5/Conv2D_grad/tuple/control_dependencyIdentity7gradients/model_2/conv5/Conv2D_grad/Conv2DBackpropInput5^gradients/model_2/conv5/Conv2D_grad/tuple/group_deps*
T0*J
_class@
><loc:@gradients/model_2/conv5/Conv2D_grad/Conv2DBackpropInput*0
_output_shapes
:џџџџџџџџџ"G
К
>gradients/model_2/conv5/Conv2D_grad/tuple/control_dependency_1Identity8gradients/model_2/conv5/Conv2D_grad/Conv2DBackpropFilter5^gradients/model_2/conv5/Conv2D_grad/tuple/group_deps*
T0*K
_classA
?=loc:@gradients/model_2/conv5/Conv2D_grad/Conv2DBackpropFilter*'
_output_shapes
:
Ь
gradients/AddN_1AddN?gradients/model_1/conv5/BiasAdd_grad/tuple/control_dependency_1=gradients/model/conv5/BiasAdd_grad/tuple/control_dependency_1?gradients/model_2/conv5/BiasAdd_grad/tuple/control_dependency_1*
T0*C
_class9
75loc:@gradients/model_1/conv5/BiasAdd_grad/BiasAddGrad*
N*
_output_shapes
:
а
:gradients/model_1/conv4/MaxPool2D/MaxPool_grad/MaxPoolGradMaxPoolGradmodel_1/conv4/conv4/Relumodel_1/conv4/MaxPool2D/MaxPool<gradients/model_1/conv5/Conv2D_grad/tuple/control_dependency*
T0*
data_formatNHWC*
strides
*
ksize
*
paddingSAME*1
_output_shapes
:џџџџџџџџџD
Ш
8gradients/model/conv4/MaxPool2D/MaxPool_grad/MaxPoolGradMaxPoolGradmodel/conv4/conv4/Relumodel/conv4/MaxPool2D/MaxPool:gradients/model/conv5/Conv2D_grad/tuple/control_dependency*
data_formatNHWC*
strides
*
ksize
*
paddingSAME*1
_output_shapes
:џџџџџџџџџD*
T0
а
:gradients/model_2/conv4/MaxPool2D/MaxPool_grad/MaxPoolGradMaxPoolGradmodel_2/conv4/conv4/Relumodel_2/conv4/MaxPool2D/MaxPool<gradients/model_2/conv5/Conv2D_grad/tuple/control_dependency*
T0*
data_formatNHWC*
strides
*
ksize
*
paddingSAME*1
_output_shapes
:џџџџџџџџџD
о
gradients/AddN_2AddN>gradients/model_1/conv5/Conv2D_grad/tuple/control_dependency_1<gradients/model/conv5/Conv2D_grad/tuple/control_dependency_1>gradients/model_2/conv5/Conv2D_grad/tuple/control_dependency_1*
N*'
_output_shapes
:*
T0*K
_classA
?=loc:@gradients/model_1/conv5/Conv2D_grad/Conv2DBackpropFilter
Ю
0gradients/model_1/conv4/conv4/Relu_grad/ReluGradReluGrad:gradients/model_1/conv4/MaxPool2D/MaxPool_grad/MaxPoolGradmodel_1/conv4/conv4/Relu*
T0*1
_output_shapes
:џџџџџџџџџD
Ш
.gradients/model/conv4/conv4/Relu_grad/ReluGradReluGrad8gradients/model/conv4/MaxPool2D/MaxPool_grad/MaxPoolGradmodel/conv4/conv4/Relu*
T0*1
_output_shapes
:џџџџџџџџџD
Ю
0gradients/model_2/conv4/conv4/Relu_grad/ReluGradReluGrad:gradients/model_2/conv4/MaxPool2D/MaxPool_grad/MaxPoolGradmodel_2/conv4/conv4/Relu*1
_output_shapes
:џџџџџџџџџD*
T0
Ў
0gradients/model_1/conv4/BiasAdd_grad/BiasAddGradBiasAddGrad0gradients/model_1/conv4/conv4/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes	
:
Ѓ
5gradients/model_1/conv4/BiasAdd_grad/tuple/group_depsNoOp1^gradients/model_1/conv4/BiasAdd_grad/BiasAddGrad1^gradients/model_1/conv4/conv4/Relu_grad/ReluGrad
Д
=gradients/model_1/conv4/BiasAdd_grad/tuple/control_dependencyIdentity0gradients/model_1/conv4/conv4/Relu_grad/ReluGrad6^gradients/model_1/conv4/BiasAdd_grad/tuple/group_deps*1
_output_shapes
:џџџџџџџџџD*
T0*C
_class9
75loc:@gradients/model_1/conv4/conv4/Relu_grad/ReluGrad
 
?gradients/model_1/conv4/BiasAdd_grad/tuple/control_dependency_1Identity0gradients/model_1/conv4/BiasAdd_grad/BiasAddGrad6^gradients/model_1/conv4/BiasAdd_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/model_1/conv4/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:
Њ
.gradients/model/conv4/BiasAdd_grad/BiasAddGradBiasAddGrad.gradients/model/conv4/conv4/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes	
:

3gradients/model/conv4/BiasAdd_grad/tuple/group_depsNoOp/^gradients/model/conv4/BiasAdd_grad/BiasAddGrad/^gradients/model/conv4/conv4/Relu_grad/ReluGrad
Ќ
;gradients/model/conv4/BiasAdd_grad/tuple/control_dependencyIdentity.gradients/model/conv4/conv4/Relu_grad/ReluGrad4^gradients/model/conv4/BiasAdd_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients/model/conv4/conv4/Relu_grad/ReluGrad*1
_output_shapes
:џџџџџџџџџD

=gradients/model/conv4/BiasAdd_grad/tuple/control_dependency_1Identity.gradients/model/conv4/BiasAdd_grad/BiasAddGrad4^gradients/model/conv4/BiasAdd_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients/model/conv4/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:
Ў
0gradients/model_2/conv4/BiasAdd_grad/BiasAddGradBiasAddGrad0gradients/model_2/conv4/conv4/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes	
:
Ѓ
5gradients/model_2/conv4/BiasAdd_grad/tuple/group_depsNoOp1^gradients/model_2/conv4/BiasAdd_grad/BiasAddGrad1^gradients/model_2/conv4/conv4/Relu_grad/ReluGrad
Д
=gradients/model_2/conv4/BiasAdd_grad/tuple/control_dependencyIdentity0gradients/model_2/conv4/conv4/Relu_grad/ReluGrad6^gradients/model_2/conv4/BiasAdd_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/model_2/conv4/conv4/Relu_grad/ReluGrad*1
_output_shapes
:џџџџџџџџџD
 
?gradients/model_2/conv4/BiasAdd_grad/tuple/control_dependency_1Identity0gradients/model_2/conv4/BiasAdd_grad/BiasAddGrad6^gradients/model_2/conv4/BiasAdd_grad/tuple/group_deps*
_output_shapes	
:*
T0*C
_class9
75loc:@gradients/model_2/conv4/BiasAdd_grad/BiasAddGrad
­
*gradients/model_1/conv4/Conv2D_grad/ShapeNShapeNmodel_1/conv3/MaxPool2D/MaxPoolconv4/weights/read*
N* 
_output_shapes
::*
T0*
out_type0
ї
7gradients/model_1/conv4/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput*gradients/model_1/conv4/Conv2D_grad/ShapeNconv4/weights/read=gradients/model_1/conv4/BiasAdd_grad/tuple/control_dependency*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*1
_output_shapes
:џџџџџџџџџD*
	dilations

џ
8gradients/model_1/conv4/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFiltermodel_1/conv3/MaxPool2D/MaxPool,gradients/model_1/conv4/Conv2D_grad/ShapeN:1=gradients/model_1/conv4/BiasAdd_grad/tuple/control_dependency*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*(
_output_shapes
:*
	dilations

Б
4gradients/model_1/conv4/Conv2D_grad/tuple/group_depsNoOp9^gradients/model_1/conv4/Conv2D_grad/Conv2DBackpropFilter8^gradients/model_1/conv4/Conv2D_grad/Conv2DBackpropInput
Р
<gradients/model_1/conv4/Conv2D_grad/tuple/control_dependencyIdentity7gradients/model_1/conv4/Conv2D_grad/Conv2DBackpropInput5^gradients/model_1/conv4/Conv2D_grad/tuple/group_deps*1
_output_shapes
:џџџџџџџџџD*
T0*J
_class@
><loc:@gradients/model_1/conv4/Conv2D_grad/Conv2DBackpropInput
Л
>gradients/model_1/conv4/Conv2D_grad/tuple/control_dependency_1Identity8gradients/model_1/conv4/Conv2D_grad/Conv2DBackpropFilter5^gradients/model_1/conv4/Conv2D_grad/tuple/group_deps*
T0*K
_classA
?=loc:@gradients/model_1/conv4/Conv2D_grad/Conv2DBackpropFilter*(
_output_shapes
:
Љ
(gradients/model/conv4/Conv2D_grad/ShapeNShapeNmodel/conv3/MaxPool2D/MaxPoolconv4/weights/read*
N* 
_output_shapes
::*
T0*
out_type0
ё
5gradients/model/conv4/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput(gradients/model/conv4/Conv2D_grad/ShapeNconv4/weights/read;gradients/model/conv4/BiasAdd_grad/tuple/control_dependency*1
_output_shapes
:џџџџџџџџџD*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME
ї
6gradients/model/conv4/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFiltermodel/conv3/MaxPool2D/MaxPool*gradients/model/conv4/Conv2D_grad/ShapeN:1;gradients/model/conv4/BiasAdd_grad/tuple/control_dependency*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*(
_output_shapes
:*
	dilations

Ћ
2gradients/model/conv4/Conv2D_grad/tuple/group_depsNoOp7^gradients/model/conv4/Conv2D_grad/Conv2DBackpropFilter6^gradients/model/conv4/Conv2D_grad/Conv2DBackpropInput
И
:gradients/model/conv4/Conv2D_grad/tuple/control_dependencyIdentity5gradients/model/conv4/Conv2D_grad/Conv2DBackpropInput3^gradients/model/conv4/Conv2D_grad/tuple/group_deps*
T0*H
_class>
<:loc:@gradients/model/conv4/Conv2D_grad/Conv2DBackpropInput*1
_output_shapes
:џџџџџџџџџD
Г
<gradients/model/conv4/Conv2D_grad/tuple/control_dependency_1Identity6gradients/model/conv4/Conv2D_grad/Conv2DBackpropFilter3^gradients/model/conv4/Conv2D_grad/tuple/group_deps*
T0*I
_class?
=;loc:@gradients/model/conv4/Conv2D_grad/Conv2DBackpropFilter*(
_output_shapes
:
­
*gradients/model_2/conv4/Conv2D_grad/ShapeNShapeNmodel_2/conv3/MaxPool2D/MaxPoolconv4/weights/read*
T0*
out_type0*
N* 
_output_shapes
::
ї
7gradients/model_2/conv4/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput*gradients/model_2/conv4/Conv2D_grad/ShapeNconv4/weights/read=gradients/model_2/conv4/BiasAdd_grad/tuple/control_dependency*1
_output_shapes
:џџџџџџџџџD*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME
џ
8gradients/model_2/conv4/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFiltermodel_2/conv3/MaxPool2D/MaxPool,gradients/model_2/conv4/Conv2D_grad/ShapeN:1=gradients/model_2/conv4/BiasAdd_grad/tuple/control_dependency*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME*(
_output_shapes
:*
	dilations
*
T0
Б
4gradients/model_2/conv4/Conv2D_grad/tuple/group_depsNoOp9^gradients/model_2/conv4/Conv2D_grad/Conv2DBackpropFilter8^gradients/model_2/conv4/Conv2D_grad/Conv2DBackpropInput
Р
<gradients/model_2/conv4/Conv2D_grad/tuple/control_dependencyIdentity7gradients/model_2/conv4/Conv2D_grad/Conv2DBackpropInput5^gradients/model_2/conv4/Conv2D_grad/tuple/group_deps*
T0*J
_class@
><loc:@gradients/model_2/conv4/Conv2D_grad/Conv2DBackpropInput*1
_output_shapes
:џџџџџџџџџD
Л
>gradients/model_2/conv4/Conv2D_grad/tuple/control_dependency_1Identity8gradients/model_2/conv4/Conv2D_grad/Conv2DBackpropFilter5^gradients/model_2/conv4/Conv2D_grad/tuple/group_deps*(
_output_shapes
:*
T0*K
_classA
?=loc:@gradients/model_2/conv4/Conv2D_grad/Conv2DBackpropFilter
Э
gradients/AddN_3AddN?gradients/model_1/conv4/BiasAdd_grad/tuple/control_dependency_1=gradients/model/conv4/BiasAdd_grad/tuple/control_dependency_1?gradients/model_2/conv4/BiasAdd_grad/tuple/control_dependency_1*
T0*C
_class9
75loc:@gradients/model_1/conv4/BiasAdd_grad/BiasAddGrad*
N*
_output_shapes	
:
б
:gradients/model_1/conv3/MaxPool2D/MaxPool_grad/MaxPoolGradMaxPoolGradmodel_1/conv3/conv3/Relumodel_1/conv3/MaxPool2D/MaxPool<gradients/model_1/conv4/Conv2D_grad/tuple/control_dependency*
data_formatNHWC*
strides
*
ksize
*
paddingSAME*2
_output_shapes 
:џџџџџџџџџ*
T0
Щ
8gradients/model/conv3/MaxPool2D/MaxPool_grad/MaxPoolGradMaxPoolGradmodel/conv3/conv3/Relumodel/conv3/MaxPool2D/MaxPool:gradients/model/conv4/Conv2D_grad/tuple/control_dependency*
ksize
*
paddingSAME*2
_output_shapes 
:џџџџџџџџџ*
T0*
data_formatNHWC*
strides

б
:gradients/model_2/conv3/MaxPool2D/MaxPool_grad/MaxPoolGradMaxPoolGradmodel_2/conv3/conv3/Relumodel_2/conv3/MaxPool2D/MaxPool<gradients/model_2/conv4/Conv2D_grad/tuple/control_dependency*
T0*
data_formatNHWC*
strides
*
ksize
*
paddingSAME*2
_output_shapes 
:џџџџџџџџџ
п
gradients/AddN_4AddN>gradients/model_1/conv4/Conv2D_grad/tuple/control_dependency_1<gradients/model/conv4/Conv2D_grad/tuple/control_dependency_1>gradients/model_2/conv4/Conv2D_grad/tuple/control_dependency_1*
T0*K
_classA
?=loc:@gradients/model_1/conv4/Conv2D_grad/Conv2DBackpropFilter*
N*(
_output_shapes
:
Я
0gradients/model_1/conv3/conv3/Relu_grad/ReluGradReluGrad:gradients/model_1/conv3/MaxPool2D/MaxPool_grad/MaxPoolGradmodel_1/conv3/conv3/Relu*
T0*2
_output_shapes 
:џџџџџџџџџ
Щ
.gradients/model/conv3/conv3/Relu_grad/ReluGradReluGrad8gradients/model/conv3/MaxPool2D/MaxPool_grad/MaxPoolGradmodel/conv3/conv3/Relu*
T0*2
_output_shapes 
:џџџџџџџџџ
Я
0gradients/model_2/conv3/conv3/Relu_grad/ReluGradReluGrad:gradients/model_2/conv3/MaxPool2D/MaxPool_grad/MaxPoolGradmodel_2/conv3/conv3/Relu*
T0*2
_output_shapes 
:џџџџџџџџџ
Ў
0gradients/model_1/conv3/BiasAdd_grad/BiasAddGradBiasAddGrad0gradients/model_1/conv3/conv3/Relu_grad/ReluGrad*
data_formatNHWC*
_output_shapes	
:*
T0
Ѓ
5gradients/model_1/conv3/BiasAdd_grad/tuple/group_depsNoOp1^gradients/model_1/conv3/BiasAdd_grad/BiasAddGrad1^gradients/model_1/conv3/conv3/Relu_grad/ReluGrad
Е
=gradients/model_1/conv3/BiasAdd_grad/tuple/control_dependencyIdentity0gradients/model_1/conv3/conv3/Relu_grad/ReluGrad6^gradients/model_1/conv3/BiasAdd_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/model_1/conv3/conv3/Relu_grad/ReluGrad*2
_output_shapes 
:џџџџџџџџџ
 
?gradients/model_1/conv3/BiasAdd_grad/tuple/control_dependency_1Identity0gradients/model_1/conv3/BiasAdd_grad/BiasAddGrad6^gradients/model_1/conv3/BiasAdd_grad/tuple/group_deps*
_output_shapes	
:*
T0*C
_class9
75loc:@gradients/model_1/conv3/BiasAdd_grad/BiasAddGrad
Њ
.gradients/model/conv3/BiasAdd_grad/BiasAddGradBiasAddGrad.gradients/model/conv3/conv3/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes	
:

3gradients/model/conv3/BiasAdd_grad/tuple/group_depsNoOp/^gradients/model/conv3/BiasAdd_grad/BiasAddGrad/^gradients/model/conv3/conv3/Relu_grad/ReluGrad
­
;gradients/model/conv3/BiasAdd_grad/tuple/control_dependencyIdentity.gradients/model/conv3/conv3/Relu_grad/ReluGrad4^gradients/model/conv3/BiasAdd_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients/model/conv3/conv3/Relu_grad/ReluGrad*2
_output_shapes 
:џџџџџџџџџ

=gradients/model/conv3/BiasAdd_grad/tuple/control_dependency_1Identity.gradients/model/conv3/BiasAdd_grad/BiasAddGrad4^gradients/model/conv3/BiasAdd_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients/model/conv3/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:
Ў
0gradients/model_2/conv3/BiasAdd_grad/BiasAddGradBiasAddGrad0gradients/model_2/conv3/conv3/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes	
:
Ѓ
5gradients/model_2/conv3/BiasAdd_grad/tuple/group_depsNoOp1^gradients/model_2/conv3/BiasAdd_grad/BiasAddGrad1^gradients/model_2/conv3/conv3/Relu_grad/ReluGrad
Е
=gradients/model_2/conv3/BiasAdd_grad/tuple/control_dependencyIdentity0gradients/model_2/conv3/conv3/Relu_grad/ReluGrad6^gradients/model_2/conv3/BiasAdd_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/model_2/conv3/conv3/Relu_grad/ReluGrad*2
_output_shapes 
:џџџџџџџџџ
 
?gradients/model_2/conv3/BiasAdd_grad/tuple/control_dependency_1Identity0gradients/model_2/conv3/BiasAdd_grad/BiasAddGrad6^gradients/model_2/conv3/BiasAdd_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/model_2/conv3/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:
­
*gradients/model_1/conv3/Conv2D_grad/ShapeNShapeNmodel_1/conv2/MaxPool2D/MaxPoolconv3/weights/read*
T0*
out_type0*
N* 
_output_shapes
::
ї
7gradients/model_1/conv3/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput*gradients/model_1/conv3/Conv2D_grad/ShapeNconv3/weights/read=gradients/model_1/conv3/BiasAdd_grad/tuple/control_dependency*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*1
_output_shapes
:џџџџџџџџџ@*
	dilations
*
T0
ў
8gradients/model_1/conv3/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFiltermodel_1/conv2/MaxPool2D/MaxPool,gradients/model_1/conv3/Conv2D_grad/ShapeN:1=gradients/model_1/conv3/BiasAdd_grad/tuple/control_dependency*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*'
_output_shapes
:@*
	dilations

Б
4gradients/model_1/conv3/Conv2D_grad/tuple/group_depsNoOp9^gradients/model_1/conv3/Conv2D_grad/Conv2DBackpropFilter8^gradients/model_1/conv3/Conv2D_grad/Conv2DBackpropInput
Р
<gradients/model_1/conv3/Conv2D_grad/tuple/control_dependencyIdentity7gradients/model_1/conv3/Conv2D_grad/Conv2DBackpropInput5^gradients/model_1/conv3/Conv2D_grad/tuple/group_deps*
T0*J
_class@
><loc:@gradients/model_1/conv3/Conv2D_grad/Conv2DBackpropInput*1
_output_shapes
:џџџџџџџџџ@
К
>gradients/model_1/conv3/Conv2D_grad/tuple/control_dependency_1Identity8gradients/model_1/conv3/Conv2D_grad/Conv2DBackpropFilter5^gradients/model_1/conv3/Conv2D_grad/tuple/group_deps*'
_output_shapes
:@*
T0*K
_classA
?=loc:@gradients/model_1/conv3/Conv2D_grad/Conv2DBackpropFilter
Љ
(gradients/model/conv3/Conv2D_grad/ShapeNShapeNmodel/conv2/MaxPool2D/MaxPoolconv3/weights/read*
T0*
out_type0*
N* 
_output_shapes
::
ё
5gradients/model/conv3/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput(gradients/model/conv3/Conv2D_grad/ShapeNconv3/weights/read;gradients/model/conv3/BiasAdd_grad/tuple/control_dependency*
paddingSAME*1
_output_shapes
:џџџџџџџџџ@*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
і
6gradients/model/conv3/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFiltermodel/conv2/MaxPool2D/MaxPool*gradients/model/conv3/Conv2D_grad/ShapeN:1;gradients/model/conv3/BiasAdd_grad/tuple/control_dependency*
paddingSAME*'
_output_shapes
:@*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
Ћ
2gradients/model/conv3/Conv2D_grad/tuple/group_depsNoOp7^gradients/model/conv3/Conv2D_grad/Conv2DBackpropFilter6^gradients/model/conv3/Conv2D_grad/Conv2DBackpropInput
И
:gradients/model/conv3/Conv2D_grad/tuple/control_dependencyIdentity5gradients/model/conv3/Conv2D_grad/Conv2DBackpropInput3^gradients/model/conv3/Conv2D_grad/tuple/group_deps*1
_output_shapes
:џџџџџџџџџ@*
T0*H
_class>
<:loc:@gradients/model/conv3/Conv2D_grad/Conv2DBackpropInput
В
<gradients/model/conv3/Conv2D_grad/tuple/control_dependency_1Identity6gradients/model/conv3/Conv2D_grad/Conv2DBackpropFilter3^gradients/model/conv3/Conv2D_grad/tuple/group_deps*
T0*I
_class?
=;loc:@gradients/model/conv3/Conv2D_grad/Conv2DBackpropFilter*'
_output_shapes
:@
­
*gradients/model_2/conv3/Conv2D_grad/ShapeNShapeNmodel_2/conv2/MaxPool2D/MaxPoolconv3/weights/read*
T0*
out_type0*
N* 
_output_shapes
::
ї
7gradients/model_2/conv3/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput*gradients/model_2/conv3/Conv2D_grad/ShapeNconv3/weights/read=gradients/model_2/conv3/BiasAdd_grad/tuple/control_dependency*
paddingSAME*1
_output_shapes
:џџџџџџџџџ@*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
ў
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
:@
Б
4gradients/model_2/conv3/Conv2D_grad/tuple/group_depsNoOp9^gradients/model_2/conv3/Conv2D_grad/Conv2DBackpropFilter8^gradients/model_2/conv3/Conv2D_grad/Conv2DBackpropInput
Р
<gradients/model_2/conv3/Conv2D_grad/tuple/control_dependencyIdentity7gradients/model_2/conv3/Conv2D_grad/Conv2DBackpropInput5^gradients/model_2/conv3/Conv2D_grad/tuple/group_deps*
T0*J
_class@
><loc:@gradients/model_2/conv3/Conv2D_grad/Conv2DBackpropInput*1
_output_shapes
:џџџџџџџџџ@
К
>gradients/model_2/conv3/Conv2D_grad/tuple/control_dependency_1Identity8gradients/model_2/conv3/Conv2D_grad/Conv2DBackpropFilter5^gradients/model_2/conv3/Conv2D_grad/tuple/group_deps*
T0*K
_classA
?=loc:@gradients/model_2/conv3/Conv2D_grad/Conv2DBackpropFilter*'
_output_shapes
:@
Э
gradients/AddN_5AddN?gradients/model_1/conv3/BiasAdd_grad/tuple/control_dependency_1=gradients/model/conv3/BiasAdd_grad/tuple/control_dependency_1?gradients/model_2/conv3/BiasAdd_grad/tuple/control_dependency_1*
T0*C
_class9
75loc:@gradients/model_1/conv3/BiasAdd_grad/BiasAddGrad*
N*
_output_shapes	
:
а
:gradients/model_1/conv2/MaxPool2D/MaxPool_grad/MaxPoolGradMaxPoolGradmodel_1/conv2/conv2/Relumodel_1/conv2/MaxPool2D/MaxPool<gradients/model_1/conv3/Conv2D_grad/tuple/control_dependency*
data_formatNHWC*
strides
*
ksize
*
paddingSAME*1
_output_shapes
:џџџџџџџџџД@*
T0
Ш
8gradients/model/conv2/MaxPool2D/MaxPool_grad/MaxPoolGradMaxPoolGradmodel/conv2/conv2/Relumodel/conv2/MaxPool2D/MaxPool:gradients/model/conv3/Conv2D_grad/tuple/control_dependency*
data_formatNHWC*
strides
*
ksize
*
paddingSAME*1
_output_shapes
:џџџџџџџџџД@*
T0
а
:gradients/model_2/conv2/MaxPool2D/MaxPool_grad/MaxPoolGradMaxPoolGradmodel_2/conv2/conv2/Relumodel_2/conv2/MaxPool2D/MaxPool<gradients/model_2/conv3/Conv2D_grad/tuple/control_dependency*
ksize
*
paddingSAME*1
_output_shapes
:џџџџџџџџџД@*
T0*
data_formatNHWC*
strides

о
gradients/AddN_6AddN>gradients/model_1/conv3/Conv2D_grad/tuple/control_dependency_1<gradients/model/conv3/Conv2D_grad/tuple/control_dependency_1>gradients/model_2/conv3/Conv2D_grad/tuple/control_dependency_1*
T0*K
_classA
?=loc:@gradients/model_1/conv3/Conv2D_grad/Conv2DBackpropFilter*
N*'
_output_shapes
:@
Ю
0gradients/model_1/conv2/conv2/Relu_grad/ReluGradReluGrad:gradients/model_1/conv2/MaxPool2D/MaxPool_grad/MaxPoolGradmodel_1/conv2/conv2/Relu*1
_output_shapes
:џџџџџџџџџД@*
T0
Ш
.gradients/model/conv2/conv2/Relu_grad/ReluGradReluGrad8gradients/model/conv2/MaxPool2D/MaxPool_grad/MaxPoolGradmodel/conv2/conv2/Relu*
T0*1
_output_shapes
:џџџџџџџџџД@
Ю
0gradients/model_2/conv2/conv2/Relu_grad/ReluGradReluGrad:gradients/model_2/conv2/MaxPool2D/MaxPool_grad/MaxPoolGradmodel_2/conv2/conv2/Relu*
T0*1
_output_shapes
:џџџџџџџџџД@
­
0gradients/model_1/conv2/BiasAdd_grad/BiasAddGradBiasAddGrad0gradients/model_1/conv2/conv2/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes
:@
Ѓ
5gradients/model_1/conv2/BiasAdd_grad/tuple/group_depsNoOp1^gradients/model_1/conv2/BiasAdd_grad/BiasAddGrad1^gradients/model_1/conv2/conv2/Relu_grad/ReluGrad
Д
=gradients/model_1/conv2/BiasAdd_grad/tuple/control_dependencyIdentity0gradients/model_1/conv2/conv2/Relu_grad/ReluGrad6^gradients/model_1/conv2/BiasAdd_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/model_1/conv2/conv2/Relu_grad/ReluGrad*1
_output_shapes
:џџџџџџџџџД@

?gradients/model_1/conv2/BiasAdd_grad/tuple/control_dependency_1Identity0gradients/model_1/conv2/BiasAdd_grad/BiasAddGrad6^gradients/model_1/conv2/BiasAdd_grad/tuple/group_deps*
_output_shapes
:@*
T0*C
_class9
75loc:@gradients/model_1/conv2/BiasAdd_grad/BiasAddGrad
Љ
.gradients/model/conv2/BiasAdd_grad/BiasAddGradBiasAddGrad.gradients/model/conv2/conv2/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes
:@

3gradients/model/conv2/BiasAdd_grad/tuple/group_depsNoOp/^gradients/model/conv2/BiasAdd_grad/BiasAddGrad/^gradients/model/conv2/conv2/Relu_grad/ReluGrad
Ќ
;gradients/model/conv2/BiasAdd_grad/tuple/control_dependencyIdentity.gradients/model/conv2/conv2/Relu_grad/ReluGrad4^gradients/model/conv2/BiasAdd_grad/tuple/group_deps*1
_output_shapes
:џџџџџџџџџД@*
T0*A
_class7
53loc:@gradients/model/conv2/conv2/Relu_grad/ReluGrad

=gradients/model/conv2/BiasAdd_grad/tuple/control_dependency_1Identity.gradients/model/conv2/BiasAdd_grad/BiasAddGrad4^gradients/model/conv2/BiasAdd_grad/tuple/group_deps*
_output_shapes
:@*
T0*A
_class7
53loc:@gradients/model/conv2/BiasAdd_grad/BiasAddGrad
­
0gradients/model_2/conv2/BiasAdd_grad/BiasAddGradBiasAddGrad0gradients/model_2/conv2/conv2/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes
:@
Ѓ
5gradients/model_2/conv2/BiasAdd_grad/tuple/group_depsNoOp1^gradients/model_2/conv2/BiasAdd_grad/BiasAddGrad1^gradients/model_2/conv2/conv2/Relu_grad/ReluGrad
Д
=gradients/model_2/conv2/BiasAdd_grad/tuple/control_dependencyIdentity0gradients/model_2/conv2/conv2/Relu_grad/ReluGrad6^gradients/model_2/conv2/BiasAdd_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/model_2/conv2/conv2/Relu_grad/ReluGrad*1
_output_shapes
:џџџџџџџџџД@

?gradients/model_2/conv2/BiasAdd_grad/tuple/control_dependency_1Identity0gradients/model_2/conv2/BiasAdd_grad/BiasAddGrad6^gradients/model_2/conv2/BiasAdd_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/model_2/conv2/BiasAdd_grad/BiasAddGrad*
_output_shapes
:@
­
*gradients/model_1/conv2/Conv2D_grad/ShapeNShapeNmodel_1/conv1/MaxPool2D/MaxPoolconv2/weights/read*
T0*
out_type0*
N* 
_output_shapes
::
ї
7gradients/model_1/conv2/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput*gradients/model_1/conv2/Conv2D_grad/ShapeNconv2/weights/read=gradients/model_1/conv2/BiasAdd_grad/tuple/control_dependency*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*1
_output_shapes
:џџџџџџџџџД *
	dilations
*
T0
§
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
Б
4gradients/model_1/conv2/Conv2D_grad/tuple/group_depsNoOp9^gradients/model_1/conv2/Conv2D_grad/Conv2DBackpropFilter8^gradients/model_1/conv2/Conv2D_grad/Conv2DBackpropInput
Р
<gradients/model_1/conv2/Conv2D_grad/tuple/control_dependencyIdentity7gradients/model_1/conv2/Conv2D_grad/Conv2DBackpropInput5^gradients/model_1/conv2/Conv2D_grad/tuple/group_deps*
T0*J
_class@
><loc:@gradients/model_1/conv2/Conv2D_grad/Conv2DBackpropInput*1
_output_shapes
:џџџџџџџџџД 
Й
>gradients/model_1/conv2/Conv2D_grad/tuple/control_dependency_1Identity8gradients/model_1/conv2/Conv2D_grad/Conv2DBackpropFilter5^gradients/model_1/conv2/Conv2D_grad/tuple/group_deps*
T0*K
_classA
?=loc:@gradients/model_1/conv2/Conv2D_grad/Conv2DBackpropFilter*&
_output_shapes
: @
Љ
(gradients/model/conv2/Conv2D_grad/ShapeNShapeNmodel/conv1/MaxPool2D/MaxPoolconv2/weights/read*
T0*
out_type0*
N* 
_output_shapes
::
ё
5gradients/model/conv2/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput(gradients/model/conv2/Conv2D_grad/ShapeNconv2/weights/read;gradients/model/conv2/BiasAdd_grad/tuple/control_dependency*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*1
_output_shapes
:џџџџџџџџџД *
	dilations

ѕ
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
Ћ
2gradients/model/conv2/Conv2D_grad/tuple/group_depsNoOp7^gradients/model/conv2/Conv2D_grad/Conv2DBackpropFilter6^gradients/model/conv2/Conv2D_grad/Conv2DBackpropInput
И
:gradients/model/conv2/Conv2D_grad/tuple/control_dependencyIdentity5gradients/model/conv2/Conv2D_grad/Conv2DBackpropInput3^gradients/model/conv2/Conv2D_grad/tuple/group_deps*
T0*H
_class>
<:loc:@gradients/model/conv2/Conv2D_grad/Conv2DBackpropInput*1
_output_shapes
:џџџџџџџџџД 
Б
<gradients/model/conv2/Conv2D_grad/tuple/control_dependency_1Identity6gradients/model/conv2/Conv2D_grad/Conv2DBackpropFilter3^gradients/model/conv2/Conv2D_grad/tuple/group_deps*
T0*I
_class?
=;loc:@gradients/model/conv2/Conv2D_grad/Conv2DBackpropFilter*&
_output_shapes
: @
­
*gradients/model_2/conv2/Conv2D_grad/ShapeNShapeNmodel_2/conv1/MaxPool2D/MaxPoolconv2/weights/read*
N* 
_output_shapes
::*
T0*
out_type0
ї
7gradients/model_2/conv2/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput*gradients/model_2/conv2/Conv2D_grad/ShapeNconv2/weights/read=gradients/model_2/conv2/BiasAdd_grad/tuple/control_dependency*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*1
_output_shapes
:џџџџџџџџџД 
§
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
Б
4gradients/model_2/conv2/Conv2D_grad/tuple/group_depsNoOp9^gradients/model_2/conv2/Conv2D_grad/Conv2DBackpropFilter8^gradients/model_2/conv2/Conv2D_grad/Conv2DBackpropInput
Р
<gradients/model_2/conv2/Conv2D_grad/tuple/control_dependencyIdentity7gradients/model_2/conv2/Conv2D_grad/Conv2DBackpropInput5^gradients/model_2/conv2/Conv2D_grad/tuple/group_deps*
T0*J
_class@
><loc:@gradients/model_2/conv2/Conv2D_grad/Conv2DBackpropInput*1
_output_shapes
:џџџџџџџџџД 
Й
>gradients/model_2/conv2/Conv2D_grad/tuple/control_dependency_1Identity8gradients/model_2/conv2/Conv2D_grad/Conv2DBackpropFilter5^gradients/model_2/conv2/Conv2D_grad/tuple/group_deps*&
_output_shapes
: @*
T0*K
_classA
?=loc:@gradients/model_2/conv2/Conv2D_grad/Conv2DBackpropFilter
Ь
gradients/AddN_7AddN?gradients/model_1/conv2/BiasAdd_grad/tuple/control_dependency_1=gradients/model/conv2/BiasAdd_grad/tuple/control_dependency_1?gradients/model_2/conv2/BiasAdd_grad/tuple/control_dependency_1*
T0*C
_class9
75loc:@gradients/model_1/conv2/BiasAdd_grad/BiasAddGrad*
N*
_output_shapes
:@
а
:gradients/model_1/conv1/MaxPool2D/MaxPool_grad/MaxPoolGradMaxPoolGradmodel_1/conv1/conv1/Relumodel_1/conv1/MaxPool2D/MaxPool<gradients/model_1/conv2/Conv2D_grad/tuple/control_dependency*
ksize
*
paddingSAME*1
_output_shapes
:џџџџџџџџџ ч *
T0*
data_formatNHWC*
strides

Ш
8gradients/model/conv1/MaxPool2D/MaxPool_grad/MaxPoolGradMaxPoolGradmodel/conv1/conv1/Relumodel/conv1/MaxPool2D/MaxPool:gradients/model/conv2/Conv2D_grad/tuple/control_dependency*
T0*
data_formatNHWC*
strides
*
ksize
*
paddingSAME*1
_output_shapes
:џџџџџџџџџ ч 
а
:gradients/model_2/conv1/MaxPool2D/MaxPool_grad/MaxPoolGradMaxPoolGradmodel_2/conv1/conv1/Relumodel_2/conv1/MaxPool2D/MaxPool<gradients/model_2/conv2/Conv2D_grad/tuple/control_dependency*1
_output_shapes
:џџџџџџџџџ ч *
T0*
data_formatNHWC*
strides
*
ksize
*
paddingSAME
н
gradients/AddN_8AddN>gradients/model_1/conv2/Conv2D_grad/tuple/control_dependency_1<gradients/model/conv2/Conv2D_grad/tuple/control_dependency_1>gradients/model_2/conv2/Conv2D_grad/tuple/control_dependency_1*
T0*K
_classA
?=loc:@gradients/model_1/conv2/Conv2D_grad/Conv2DBackpropFilter*
N*&
_output_shapes
: @
Ю
0gradients/model_1/conv1/conv1/Relu_grad/ReluGradReluGrad:gradients/model_1/conv1/MaxPool2D/MaxPool_grad/MaxPoolGradmodel_1/conv1/conv1/Relu*
T0*1
_output_shapes
:џџџџџџџџџ ч 
Ш
.gradients/model/conv1/conv1/Relu_grad/ReluGradReluGrad8gradients/model/conv1/MaxPool2D/MaxPool_grad/MaxPoolGradmodel/conv1/conv1/Relu*
T0*1
_output_shapes
:џџџџџџџџџ ч 
Ю
0gradients/model_2/conv1/conv1/Relu_grad/ReluGradReluGrad:gradients/model_2/conv1/MaxPool2D/MaxPool_grad/MaxPoolGradmodel_2/conv1/conv1/Relu*1
_output_shapes
:џџџџџџџџџ ч *
T0
­
0gradients/model_1/conv1/BiasAdd_grad/BiasAddGradBiasAddGrad0gradients/model_1/conv1/conv1/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes
: 
Ѓ
5gradients/model_1/conv1/BiasAdd_grad/tuple/group_depsNoOp1^gradients/model_1/conv1/BiasAdd_grad/BiasAddGrad1^gradients/model_1/conv1/conv1/Relu_grad/ReluGrad
Д
=gradients/model_1/conv1/BiasAdd_grad/tuple/control_dependencyIdentity0gradients/model_1/conv1/conv1/Relu_grad/ReluGrad6^gradients/model_1/conv1/BiasAdd_grad/tuple/group_deps*1
_output_shapes
:џџџџџџџџџ ч *
T0*C
_class9
75loc:@gradients/model_1/conv1/conv1/Relu_grad/ReluGrad

?gradients/model_1/conv1/BiasAdd_grad/tuple/control_dependency_1Identity0gradients/model_1/conv1/BiasAdd_grad/BiasAddGrad6^gradients/model_1/conv1/BiasAdd_grad/tuple/group_deps*
_output_shapes
: *
T0*C
_class9
75loc:@gradients/model_1/conv1/BiasAdd_grad/BiasAddGrad
Љ
.gradients/model/conv1/BiasAdd_grad/BiasAddGradBiasAddGrad.gradients/model/conv1/conv1/Relu_grad/ReluGrad*
data_formatNHWC*
_output_shapes
: *
T0

3gradients/model/conv1/BiasAdd_grad/tuple/group_depsNoOp/^gradients/model/conv1/BiasAdd_grad/BiasAddGrad/^gradients/model/conv1/conv1/Relu_grad/ReluGrad
Ќ
;gradients/model/conv1/BiasAdd_grad/tuple/control_dependencyIdentity.gradients/model/conv1/conv1/Relu_grad/ReluGrad4^gradients/model/conv1/BiasAdd_grad/tuple/group_deps*1
_output_shapes
:џџџџџџџџџ ч *
T0*A
_class7
53loc:@gradients/model/conv1/conv1/Relu_grad/ReluGrad

=gradients/model/conv1/BiasAdd_grad/tuple/control_dependency_1Identity.gradients/model/conv1/BiasAdd_grad/BiasAddGrad4^gradients/model/conv1/BiasAdd_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients/model/conv1/BiasAdd_grad/BiasAddGrad*
_output_shapes
: 
­
0gradients/model_2/conv1/BiasAdd_grad/BiasAddGradBiasAddGrad0gradients/model_2/conv1/conv1/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes
: 
Ѓ
5gradients/model_2/conv1/BiasAdd_grad/tuple/group_depsNoOp1^gradients/model_2/conv1/BiasAdd_grad/BiasAddGrad1^gradients/model_2/conv1/conv1/Relu_grad/ReluGrad
Д
=gradients/model_2/conv1/BiasAdd_grad/tuple/control_dependencyIdentity0gradients/model_2/conv1/conv1/Relu_grad/ReluGrad6^gradients/model_2/conv1/BiasAdd_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/model_2/conv1/conv1/Relu_grad/ReluGrad*1
_output_shapes
:џџџџџџџџџ ч 

?gradients/model_2/conv1/BiasAdd_grad/tuple/control_dependency_1Identity0gradients/model_2/conv1/BiasAdd_grad/BiasAddGrad6^gradients/model_2/conv1/BiasAdd_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/model_2/conv1/BiasAdd_grad/BiasAddGrad*
_output_shapes
: 

*gradients/model_1/conv1/Conv2D_grad/ShapeNShapeNpositive_inputconv1/weights/read*
N* 
_output_shapes
::*
T0*
out_type0
ї
7gradients/model_1/conv1/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput*gradients/model_1/conv1/Conv2D_grad/ShapeNconv1/weights/read=gradients/model_1/conv1/BiasAdd_grad/tuple/control_dependency*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*1
_output_shapes
:џџџџџџџџџ ч
ь
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
Б
4gradients/model_1/conv1/Conv2D_grad/tuple/group_depsNoOp9^gradients/model_1/conv1/Conv2D_grad/Conv2DBackpropFilter8^gradients/model_1/conv1/Conv2D_grad/Conv2DBackpropInput
Р
<gradients/model_1/conv1/Conv2D_grad/tuple/control_dependencyIdentity7gradients/model_1/conv1/Conv2D_grad/Conv2DBackpropInput5^gradients/model_1/conv1/Conv2D_grad/tuple/group_deps*
T0*J
_class@
><loc:@gradients/model_1/conv1/Conv2D_grad/Conv2DBackpropInput*1
_output_shapes
:џџџџџџџџџ ч
Й
>gradients/model_1/conv1/Conv2D_grad/tuple/control_dependency_1Identity8gradients/model_1/conv1/Conv2D_grad/Conv2DBackpropFilter5^gradients/model_1/conv1/Conv2D_grad/tuple/group_deps*
T0*K
_classA
?=loc:@gradients/model_1/conv1/Conv2D_grad/Conv2DBackpropFilter*&
_output_shapes
: 

(gradients/model/conv1/Conv2D_grad/ShapeNShapeNanchor_inputconv1/weights/read*
T0*
out_type0*
N* 
_output_shapes
::
ё
5gradients/model/conv1/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput(gradients/model/conv1/Conv2D_grad/ShapeNconv1/weights/read;gradients/model/conv1/BiasAdd_grad/tuple/control_dependency*1
_output_shapes
:џџџџџџџџџ ч*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME
ф
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
Ћ
2gradients/model/conv1/Conv2D_grad/tuple/group_depsNoOp7^gradients/model/conv1/Conv2D_grad/Conv2DBackpropFilter6^gradients/model/conv1/Conv2D_grad/Conv2DBackpropInput
И
:gradients/model/conv1/Conv2D_grad/tuple/control_dependencyIdentity5gradients/model/conv1/Conv2D_grad/Conv2DBackpropInput3^gradients/model/conv1/Conv2D_grad/tuple/group_deps*1
_output_shapes
:џџџџџџџџџ ч*
T0*H
_class>
<:loc:@gradients/model/conv1/Conv2D_grad/Conv2DBackpropInput
Б
<gradients/model/conv1/Conv2D_grad/tuple/control_dependency_1Identity6gradients/model/conv1/Conv2D_grad/Conv2DBackpropFilter3^gradients/model/conv1/Conv2D_grad/tuple/group_deps*
T0*I
_class?
=;loc:@gradients/model/conv1/Conv2D_grad/Conv2DBackpropFilter*&
_output_shapes
: 

*gradients/model_2/conv1/Conv2D_grad/ShapeNShapeNnegative_inputconv1/weights/read*
N* 
_output_shapes
::*
T0*
out_type0
ї
7gradients/model_2/conv1/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput*gradients/model_2/conv1/Conv2D_grad/ShapeNconv1/weights/read=gradients/model_2/conv1/BiasAdd_grad/tuple/control_dependency*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*1
_output_shapes
:џџџџџџџџџ ч*
	dilations

ь
8gradients/model_2/conv1/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilternegative_input,gradients/model_2/conv1/Conv2D_grad/ShapeN:1=gradients/model_2/conv1/BiasAdd_grad/tuple/control_dependency*
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
Б
4gradients/model_2/conv1/Conv2D_grad/tuple/group_depsNoOp9^gradients/model_2/conv1/Conv2D_grad/Conv2DBackpropFilter8^gradients/model_2/conv1/Conv2D_grad/Conv2DBackpropInput
Р
<gradients/model_2/conv1/Conv2D_grad/tuple/control_dependencyIdentity7gradients/model_2/conv1/Conv2D_grad/Conv2DBackpropInput5^gradients/model_2/conv1/Conv2D_grad/tuple/group_deps*
T0*J
_class@
><loc:@gradients/model_2/conv1/Conv2D_grad/Conv2DBackpropInput*1
_output_shapes
:џџџџџџџџџ ч
Й
>gradients/model_2/conv1/Conv2D_grad/tuple/control_dependency_1Identity8gradients/model_2/conv1/Conv2D_grad/Conv2DBackpropFilter5^gradients/model_2/conv1/Conv2D_grad/tuple/group_deps*
T0*K
_classA
?=loc:@gradients/model_2/conv1/Conv2D_grad/Conv2DBackpropFilter*&
_output_shapes
: 
Ь
gradients/AddN_9AddN?gradients/model_1/conv1/BiasAdd_grad/tuple/control_dependency_1=gradients/model/conv1/BiasAdd_grad/tuple/control_dependency_1?gradients/model_2/conv1/BiasAdd_grad/tuple/control_dependency_1*
T0*C
_class9
75loc:@gradients/model_1/conv1/BiasAdd_grad/BiasAddGrad*
N*
_output_shapes
: 
о
gradients/AddN_10AddN>gradients/model_1/conv1/Conv2D_grad/tuple/control_dependency_1<gradients/model/conv1/Conv2D_grad/tuple/control_dependency_1>gradients/model_2/conv1/Conv2D_grad/tuple/control_dependency_1*
N*&
_output_shapes
: *
T0*K
_classA
?=loc:@gradients/model_1/conv1/Conv2D_grad/Conv2DBackpropFilter
Г
8conv1/weights/Momentum/Initializer/zeros/shape_as_tensorConst* 
_class
loc:@conv1/weights*%
valueB"             *
dtype0*
_output_shapes
:

.conv1/weights/Momentum/Initializer/zeros/ConstConst* 
_class
loc:@conv1/weights*
valueB
 *    *
dtype0*
_output_shapes
: 
џ
(conv1/weights/Momentum/Initializer/zerosFill8conv1/weights/Momentum/Initializer/zeros/shape_as_tensor.conv1/weights/Momentum/Initializer/zeros/Const*
T0* 
_class
loc:@conv1/weights*

index_type0*&
_output_shapes
: 
М
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
х
conv1/weights/Momentum/AssignAssignconv1/weights/Momentum(conv1/weights/Momentum/Initializer/zeros*
validate_shape(*&
_output_shapes
: *
use_locking(*
T0* 
_class
loc:@conv1/weights

conv1/weights/Momentum/readIdentityconv1/weights/Momentum*
T0* 
_class
loc:@conv1/weights*&
_output_shapes
: 

'conv1/biases/Momentum/Initializer/zerosConst*
_class
loc:@conv1/biases*
valueB *    *
dtype0*
_output_shapes
: 
Ђ
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
е
conv1/biases/Momentum/AssignAssignconv1/biases/Momentum'conv1/biases/Momentum/Initializer/zeros*
use_locking(*
T0*
_class
loc:@conv1/biases*
validate_shape(*
_output_shapes
: 

conv1/biases/Momentum/readIdentityconv1/biases/Momentum*
T0*
_class
loc:@conv1/biases*
_output_shapes
: 
Г
8conv2/weights/Momentum/Initializer/zeros/shape_as_tensorConst* 
_class
loc:@conv2/weights*%
valueB"          @   *
dtype0*
_output_shapes
:

.conv2/weights/Momentum/Initializer/zeros/ConstConst* 
_class
loc:@conv2/weights*
valueB
 *    *
dtype0*
_output_shapes
: 
џ
(conv2/weights/Momentum/Initializer/zerosFill8conv2/weights/Momentum/Initializer/zeros/shape_as_tensor.conv2/weights/Momentum/Initializer/zeros/Const*
T0* 
_class
loc:@conv2/weights*

index_type0*&
_output_shapes
: @
М
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
х
conv2/weights/Momentum/AssignAssignconv2/weights/Momentum(conv2/weights/Momentum/Initializer/zeros*
use_locking(*
T0* 
_class
loc:@conv2/weights*
validate_shape(*&
_output_shapes
: @

conv2/weights/Momentum/readIdentityconv2/weights/Momentum*
T0* 
_class
loc:@conv2/weights*&
_output_shapes
: @

'conv2/biases/Momentum/Initializer/zerosConst*
_class
loc:@conv2/biases*
valueB@*    *
dtype0*
_output_shapes
:@
Ђ
conv2/biases/Momentum
VariableV2*
shared_name *
_class
loc:@conv2/biases*
	container *
shape:@*
dtype0*
_output_shapes
:@
е
conv2/biases/Momentum/AssignAssignconv2/biases/Momentum'conv2/biases/Momentum/Initializer/zeros*
T0*
_class
loc:@conv2/biases*
validate_shape(*
_output_shapes
:@*
use_locking(

conv2/biases/Momentum/readIdentityconv2/biases/Momentum*
T0*
_class
loc:@conv2/biases*
_output_shapes
:@
Г
8conv3/weights/Momentum/Initializer/zeros/shape_as_tensorConst* 
_class
loc:@conv3/weights*%
valueB"      @      *
dtype0*
_output_shapes
:

.conv3/weights/Momentum/Initializer/zeros/ConstConst* 
_class
loc:@conv3/weights*
valueB
 *    *
dtype0*
_output_shapes
: 

(conv3/weights/Momentum/Initializer/zerosFill8conv3/weights/Momentum/Initializer/zeros/shape_as_tensor.conv3/weights/Momentum/Initializer/zeros/Const*
T0* 
_class
loc:@conv3/weights*

index_type0*'
_output_shapes
:@
О
conv3/weights/Momentum
VariableV2*
dtype0*'
_output_shapes
:@*
shared_name * 
_class
loc:@conv3/weights*
	container *
shape:@
ц
conv3/weights/Momentum/AssignAssignconv3/weights/Momentum(conv3/weights/Momentum/Initializer/zeros*
use_locking(*
T0* 
_class
loc:@conv3/weights*
validate_shape(*'
_output_shapes
:@

conv3/weights/Momentum/readIdentityconv3/weights/Momentum*
T0* 
_class
loc:@conv3/weights*'
_output_shapes
:@

'conv3/biases/Momentum/Initializer/zerosConst*
dtype0*
_output_shapes	
:*
_class
loc:@conv3/biases*
valueB*    
Є
conv3/biases/Momentum
VariableV2*
_class
loc:@conv3/biases*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name 
ж
conv3/biases/Momentum/AssignAssignconv3/biases/Momentum'conv3/biases/Momentum/Initializer/zeros*
use_locking(*
T0*
_class
loc:@conv3/biases*
validate_shape(*
_output_shapes	
:

conv3/biases/Momentum/readIdentityconv3/biases/Momentum*
_output_shapes	
:*
T0*
_class
loc:@conv3/biases
Г
8conv4/weights/Momentum/Initializer/zeros/shape_as_tensorConst* 
_class
loc:@conv4/weights*%
valueB"            *
dtype0*
_output_shapes
:

.conv4/weights/Momentum/Initializer/zeros/ConstConst* 
_class
loc:@conv4/weights*
valueB
 *    *
dtype0*
_output_shapes
: 

(conv4/weights/Momentum/Initializer/zerosFill8conv4/weights/Momentum/Initializer/zeros/shape_as_tensor.conv4/weights/Momentum/Initializer/zeros/Const*
T0* 
_class
loc:@conv4/weights*

index_type0*(
_output_shapes
:
Р
conv4/weights/Momentum
VariableV2*
dtype0*(
_output_shapes
:*
shared_name * 
_class
loc:@conv4/weights*
	container *
shape:
ч
conv4/weights/Momentum/AssignAssignconv4/weights/Momentum(conv4/weights/Momentum/Initializer/zeros*
use_locking(*
T0* 
_class
loc:@conv4/weights*
validate_shape(*(
_output_shapes
:

conv4/weights/Momentum/readIdentityconv4/weights/Momentum*
T0* 
_class
loc:@conv4/weights*(
_output_shapes
:

'conv4/biases/Momentum/Initializer/zerosConst*
_class
loc:@conv4/biases*
valueB*    *
dtype0*
_output_shapes	
:
Є
conv4/biases/Momentum
VariableV2*
shared_name *
_class
loc:@conv4/biases*
	container *
shape:*
dtype0*
_output_shapes	
:
ж
conv4/biases/Momentum/AssignAssignconv4/biases/Momentum'conv4/biases/Momentum/Initializer/zeros*
use_locking(*
T0*
_class
loc:@conv4/biases*
validate_shape(*
_output_shapes	
:

conv4/biases/Momentum/readIdentityconv4/biases/Momentum*
T0*
_class
loc:@conv4/biases*
_output_shapes	
:
Г
8conv5/weights/Momentum/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:* 
_class
loc:@conv5/weights*%
valueB"            

.conv5/weights/Momentum/Initializer/zeros/ConstConst* 
_class
loc:@conv5/weights*
valueB
 *    *
dtype0*
_output_shapes
: 

(conv5/weights/Momentum/Initializer/zerosFill8conv5/weights/Momentum/Initializer/zeros/shape_as_tensor.conv5/weights/Momentum/Initializer/zeros/Const*
T0* 
_class
loc:@conv5/weights*

index_type0*'
_output_shapes
:
О
conv5/weights/Momentum
VariableV2* 
_class
loc:@conv5/weights*
	container *
shape:*
dtype0*'
_output_shapes
:*
shared_name 
ц
conv5/weights/Momentum/AssignAssignconv5/weights/Momentum(conv5/weights/Momentum/Initializer/zeros*
use_locking(*
T0* 
_class
loc:@conv5/weights*
validate_shape(*'
_output_shapes
:

conv5/weights/Momentum/readIdentityconv5/weights/Momentum*'
_output_shapes
:*
T0* 
_class
loc:@conv5/weights

'conv5/biases/Momentum/Initializer/zerosConst*
_class
loc:@conv5/biases*
valueB*    *
dtype0*
_output_shapes
:
Ђ
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
е
conv5/biases/Momentum/AssignAssignconv5/biases/Momentum'conv5/biases/Momentum/Initializer/zeros*
T0*
_class
loc:@conv5/biases*
validate_shape(*
_output_shapes
:*
use_locking(

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
з#<*
dtype0*
_output_shapes
: 
V
Momentum/momentumConst*
valueB
 *Єp}?*
dtype0*
_output_shapes
: 

+Momentum/update_conv1/weights/ApplyMomentumApplyMomentumconv1/weightsconv1/weights/MomentumMomentum/learning_rategradients/AddN_10Momentum/momentum*
use_nesterov(*&
_output_shapes
: *
use_locking( *
T0* 
_class
loc:@conv1/weights

*Momentum/update_conv1/biases/ApplyMomentumApplyMomentumconv1/biasesconv1/biases/MomentumMomentum/learning_rategradients/AddN_9Momentum/momentum*
use_nesterov(*
_output_shapes
: *
use_locking( *
T0*
_class
loc:@conv1/biases

+Momentum/update_conv2/weights/ApplyMomentumApplyMomentumconv2/weightsconv2/weights/MomentumMomentum/learning_rategradients/AddN_8Momentum/momentum*
use_nesterov(*&
_output_shapes
: @*
use_locking( *
T0* 
_class
loc:@conv2/weights

*Momentum/update_conv2/biases/ApplyMomentumApplyMomentumconv2/biasesconv2/biases/MomentumMomentum/learning_rategradients/AddN_7Momentum/momentum*
T0*
_class
loc:@conv2/biases*
use_nesterov(*
_output_shapes
:@*
use_locking( 

+Momentum/update_conv3/weights/ApplyMomentumApplyMomentumconv3/weightsconv3/weights/MomentumMomentum/learning_rategradients/AddN_6Momentum/momentum*
use_locking( *
T0* 
_class
loc:@conv3/weights*
use_nesterov(*'
_output_shapes
:@

*Momentum/update_conv3/biases/ApplyMomentumApplyMomentumconv3/biasesconv3/biases/MomentumMomentum/learning_rategradients/AddN_5Momentum/momentum*
use_locking( *
T0*
_class
loc:@conv3/biases*
use_nesterov(*
_output_shapes	
:

+Momentum/update_conv4/weights/ApplyMomentumApplyMomentumconv4/weightsconv4/weights/MomentumMomentum/learning_rategradients/AddN_4Momentum/momentum*
use_nesterov(*(
_output_shapes
:*
use_locking( *
T0* 
_class
loc:@conv4/weights

*Momentum/update_conv4/biases/ApplyMomentumApplyMomentumconv4/biasesconv4/biases/MomentumMomentum/learning_rategradients/AddN_3Momentum/momentum*
use_nesterov(*
_output_shapes	
:*
use_locking( *
T0*
_class
loc:@conv4/biases

+Momentum/update_conv5/weights/ApplyMomentumApplyMomentumconv5/weightsconv5/weights/MomentumMomentum/learning_rategradients/AddN_2Momentum/momentum*
T0* 
_class
loc:@conv5/weights*
use_nesterov(*'
_output_shapes
:*
use_locking( 

*Momentum/update_conv5/biases/ApplyMomentumApplyMomentumconv5/biasesconv5/biases/MomentumMomentum/learning_rategradients/AddN_1Momentum/momentum*
T0*
_class
loc:@conv5/biases*
use_nesterov(*
_output_shapes
:*
use_locking( 
о
Momentum/updateNoOp+^Momentum/update_conv1/biases/ApplyMomentum,^Momentum/update_conv1/weights/ApplyMomentum+^Momentum/update_conv2/biases/ApplyMomentum,^Momentum/update_conv2/weights/ApplyMomentum+^Momentum/update_conv3/biases/ApplyMomentum,^Momentum/update_conv3/weights/ApplyMomentum+^Momentum/update_conv4/biases/ApplyMomentum,^Momentum/update_conv4/weights/ApplyMomentum+^Momentum/update_conv5/biases/ApplyMomentum,^Momentum/update_conv5/weights/ApplyMomentum

Momentum/valueConst^Momentum/update*
dtype0*
_output_shapes
: *
value	B :*
_class
loc:@Variable

Momentum	AssignAddVariableMomentum/value*
T0*
_class
loc:@Variable*
_output_shapes
: *
use_locking( 
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
ш
save/SaveV2/tensor_namesConst*
valueBBVariableBconv1/biasesBconv1/biases/MomentumBconv1/weightsBconv1/weights/MomentumBconv2/biasesBconv2/biases/MomentumBconv2/weightsBconv2/weights/MomentumBconv3/biasesBconv3/biases/MomentumBconv3/weightsBconv3/weights/MomentumBconv4/biasesBconv4/biases/MomentumBconv4/weightsBconv4/weights/MomentumBconv5/biasesBconv5/biases/MomentumBconv5/weightsBconv5/weights/Momentum*
dtype0*
_output_shapes
:

save/SaveV2/shape_and_slicesConst*=
value4B2B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:

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
њ
save/RestoreV2/tensor_namesConst"/device:CPU:0*
valueBBVariableBconv1/biasesBconv1/biases/MomentumBconv1/weightsBconv1/weights/MomentumBconv2/biasesBconv2/biases/MomentumBconv2/weightsBconv2/weights/MomentumBconv3/biasesBconv3/biases/MomentumBconv3/weightsBconv3/weights/MomentumBconv4/biasesBconv4/biases/MomentumBconv4/weightsBconv4/weights/MomentumBconv5/biasesBconv5/biases/MomentumBconv5/weightsBconv5/weights/Momentum*
dtype0*
_output_shapes
:

save/RestoreV2/shape_and_slicesConst"/device:CPU:0*=
value4B2B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:

save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices"/device:CPU:0*#
dtypes
2*h
_output_shapesV
T:::::::::::::::::::::

save/AssignAssignVariablesave/RestoreV2*
use_locking(*
T0*
_class
loc:@Variable*
validate_shape(*
_output_shapes
: 
І
save/Assign_1Assignconv1/biasessave/RestoreV2:1*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*
_class
loc:@conv1/biases
Џ
save/Assign_2Assignconv1/biases/Momentumsave/RestoreV2:2*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*
_class
loc:@conv1/biases
Д
save/Assign_3Assignconv1/weightssave/RestoreV2:3*
T0* 
_class
loc:@conv1/weights*
validate_shape(*&
_output_shapes
: *
use_locking(
Н
save/Assign_4Assignconv1/weights/Momentumsave/RestoreV2:4*
use_locking(*
T0* 
_class
loc:@conv1/weights*
validate_shape(*&
_output_shapes
: 
І
save/Assign_5Assignconv2/biasessave/RestoreV2:5*
T0*
_class
loc:@conv2/biases*
validate_shape(*
_output_shapes
:@*
use_locking(
Џ
save/Assign_6Assignconv2/biases/Momentumsave/RestoreV2:6*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*
_class
loc:@conv2/biases
Д
save/Assign_7Assignconv2/weightssave/RestoreV2:7*
T0* 
_class
loc:@conv2/weights*
validate_shape(*&
_output_shapes
: @*
use_locking(
Н
save/Assign_8Assignconv2/weights/Momentumsave/RestoreV2:8*
use_locking(*
T0* 
_class
loc:@conv2/weights*
validate_shape(*&
_output_shapes
: @
Ї
save/Assign_9Assignconv3/biasessave/RestoreV2:9*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*
_class
loc:@conv3/biases
В
save/Assign_10Assignconv3/biases/Momentumsave/RestoreV2:10*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*
_class
loc:@conv3/biases
З
save/Assign_11Assignconv3/weightssave/RestoreV2:11*
use_locking(*
T0* 
_class
loc:@conv3/weights*
validate_shape(*'
_output_shapes
:@
Р
save/Assign_12Assignconv3/weights/Momentumsave/RestoreV2:12*
use_locking(*
T0* 
_class
loc:@conv3/weights*
validate_shape(*'
_output_shapes
:@
Љ
save/Assign_13Assignconv4/biasessave/RestoreV2:13*
use_locking(*
T0*
_class
loc:@conv4/biases*
validate_shape(*
_output_shapes	
:
В
save/Assign_14Assignconv4/biases/Momentumsave/RestoreV2:14*
use_locking(*
T0*
_class
loc:@conv4/biases*
validate_shape(*
_output_shapes	
:
И
save/Assign_15Assignconv4/weightssave/RestoreV2:15*
use_locking(*
T0* 
_class
loc:@conv4/weights*
validate_shape(*(
_output_shapes
:
С
save/Assign_16Assignconv4/weights/Momentumsave/RestoreV2:16*
T0* 
_class
loc:@conv4/weights*
validate_shape(*(
_output_shapes
:*
use_locking(
Ј
save/Assign_17Assignconv5/biasessave/RestoreV2:17*
T0*
_class
loc:@conv5/biases*
validate_shape(*
_output_shapes
:*
use_locking(
Б
save/Assign_18Assignconv5/biases/Momentumsave/RestoreV2:18*
T0*
_class
loc:@conv5/biases*
validate_shape(*
_output_shapes
:*
use_locking(
З
save/Assign_19Assignconv5/weightssave/RestoreV2:19*
validate_shape(*'
_output_shapes
:*
use_locking(*
T0* 
_class
loc:@conv5/weights
Р
save/Assign_20Assignconv5/weights/Momentumsave/RestoreV2:20*
validate_shape(*'
_output_shapes
:*
use_locking(*
T0* 
_class
loc:@conv5/weights
ё
save/restore_allNoOp^save/Assign^save/Assign_1^save/Assign_10^save/Assign_11^save/Assign_12^save/Assign_13^save/Assign_14^save/Assign_15^save/Assign_16^save/Assign_17^save/Assign_18^save/Assign_19^save/Assign_2^save/Assign_20^save/Assign_3^save/Assign_4^save/Assign_5^save/Assign_6^save/Assign_7^save/Assign_8^save/Assign_9
К
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
є
Merge/MergeSummaryMergeSummarysteplossconv1/weights_1conv1/biases_1conv2/weights_1conv2/biases_1conv3/weights_1conv3/biases_1conv4/weights_1conv4/biases_1conv5/weights_1conv5/biases_1*
N*
_output_shapes
: "ЉхТhЯ)     < Rм	
0рZрЊзAJТг
З,,
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
2	
А
ApplyMomentum
var"T
accum"T
lr"T	
grad"T
momentum"T
out"T" 
Ttype:
2	"
use_lockingbool( "
use_nesterovbool( 
x
Assign
ref"T

value"T

output_ref"T"	
Ttype"
validate_shapebool("
use_lockingbool(
s
	AssignAdd
ref"T

value"T

output_ref"T" 
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
ь
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

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

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
д
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
ю
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

2	

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
2	
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

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
2	
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
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
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
і
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

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
ref"dtype"
shapeshape"
dtypetype"
	containerstring "
shared_namestring 
&
	ZerosLike
x"T
y"T"	
Ttype*1.13.12b'v1.13.0-rc2-5-g6612da8951'Т

anchor_inputPlaceholder*
dtype0*1
_output_shapes
:џџџџџџџџџ ч*&
shape:џџџџџџџџџ ч

positive_inputPlaceholder*
dtype0*1
_output_shapes
:џџџџџџџџџ ч*&
shape:џџџџџџџџџ ч

negative_inputPlaceholder*
dtype0*1
_output_shapes
:џџџџџџџџџ ч*&
shape:џџџџџџџџџ ч
Љ
.conv1/weights/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:* 
_class
loc:@conv1/weights*%
valueB"             

,conv1/weights/Initializer/random_uniform/minConst* 
_class
loc:@conv1/weights*
valueB
 *ЌErН*
dtype0*
_output_shapes
: 

,conv1/weights/Initializer/random_uniform/maxConst* 
_class
loc:@conv1/weights*
valueB
 *ЌEr=*
dtype0*
_output_shapes
: 
№
6conv1/weights/Initializer/random_uniform/RandomUniformRandomUniform.conv1/weights/Initializer/random_uniform/shape*
dtype0*&
_output_shapes
: *

seed *
T0* 
_class
loc:@conv1/weights*
seed2 
в
,conv1/weights/Initializer/random_uniform/subSub,conv1/weights/Initializer/random_uniform/max,conv1/weights/Initializer/random_uniform/min*
T0* 
_class
loc:@conv1/weights*
_output_shapes
: 
ь
,conv1/weights/Initializer/random_uniform/mulMul6conv1/weights/Initializer/random_uniform/RandomUniform,conv1/weights/Initializer/random_uniform/sub*&
_output_shapes
: *
T0* 
_class
loc:@conv1/weights
о
(conv1/weights/Initializer/random_uniformAdd,conv1/weights/Initializer/random_uniform/mul,conv1/weights/Initializer/random_uniform/min*
T0* 
_class
loc:@conv1/weights*&
_output_shapes
: 
Г
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
г
conv1/weights/AssignAssignconv1/weights(conv1/weights/Initializer/random_uniform* 
_class
loc:@conv1/weights*
validate_shape(*&
_output_shapes
: *
use_locking(*
T0

conv1/weights/readIdentityconv1/weights*&
_output_shapes
: *
T0* 
_class
loc:@conv1/weights

conv1/biases/Initializer/zerosConst*
_class
loc:@conv1/biases*
valueB *    *
dtype0*
_output_shapes
: 

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
К
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
model/conv1/dilation_rateConst*
_output_shapes
:*
valueB"      *
dtype0
ш
model/conv1/Conv2DConv2Danchor_inputconv1/weights/read*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME*1
_output_shapes
:џџџџџџџџџ ч *
	dilations
*
T0

model/conv1/BiasAddBiasAddmodel/conv1/Conv2Dconv1/biases/read*
data_formatNHWC*1
_output_shapes
:џџџџџџџџџ ч *
T0
o
model/conv1/conv1/ReluRelumodel/conv1/BiasAdd*
T0*1
_output_shapes
:џџџџџџџџџ ч 
Ю
model/conv1/MaxPool2D/MaxPoolMaxPoolmodel/conv1/conv1/Relu*
ksize
*
paddingSAME*1
_output_shapes
:џџџџџџџџџД *
T0*
data_formatNHWC*
strides

Љ
.conv2/weights/Initializer/random_uniform/shapeConst* 
_class
loc:@conv2/weights*%
valueB"          @   *
dtype0*
_output_shapes
:

,conv2/weights/Initializer/random_uniform/minConst*
_output_shapes
: * 
_class
loc:@conv2/weights*
valueB
 *ЭЬLН*
dtype0

,conv2/weights/Initializer/random_uniform/maxConst* 
_class
loc:@conv2/weights*
valueB
 *ЭЬL=*
dtype0*
_output_shapes
: 
№
6conv2/weights/Initializer/random_uniform/RandomUniformRandomUniform.conv2/weights/Initializer/random_uniform/shape*

seed *
T0* 
_class
loc:@conv2/weights*
seed2 *
dtype0*&
_output_shapes
: @
в
,conv2/weights/Initializer/random_uniform/subSub,conv2/weights/Initializer/random_uniform/max,conv2/weights/Initializer/random_uniform/min* 
_class
loc:@conv2/weights*
_output_shapes
: *
T0
ь
,conv2/weights/Initializer/random_uniform/mulMul6conv2/weights/Initializer/random_uniform/RandomUniform,conv2/weights/Initializer/random_uniform/sub* 
_class
loc:@conv2/weights*&
_output_shapes
: @*
T0
о
(conv2/weights/Initializer/random_uniformAdd,conv2/weights/Initializer/random_uniform/mul,conv2/weights/Initializer/random_uniform/min*
T0* 
_class
loc:@conv2/weights*&
_output_shapes
: @
Г
conv2/weights
VariableV2*&
_output_shapes
: @*
shared_name * 
_class
loc:@conv2/weights*
	container *
shape: @*
dtype0
г
conv2/weights/AssignAssignconv2/weights(conv2/weights/Initializer/random_uniform*
use_locking(*
T0* 
_class
loc:@conv2/weights*
validate_shape(*&
_output_shapes
: @

conv2/weights/readIdentityconv2/weights*&
_output_shapes
: @*
T0* 
_class
loc:@conv2/weights

conv2/biases/Initializer/zerosConst*
_class
loc:@conv2/biases*
valueB@*    *
dtype0*
_output_shapes
:@

conv2/biases
VariableV2*
shape:@*
dtype0*
_output_shapes
:@*
shared_name *
_class
loc:@conv2/biases*
	container 
К
conv2/biases/AssignAssignconv2/biasesconv2/biases/Initializer/zeros*
use_locking(*
T0*
_class
loc:@conv2/biases*
validate_shape(*
_output_shapes
:@
q
conv2/biases/readIdentityconv2/biases*
_class
loc:@conv2/biases*
_output_shapes
:@*
T0
j
model/conv2/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
љ
model/conv2/Conv2DConv2Dmodel/conv1/MaxPool2D/MaxPoolconv2/weights/read*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*1
_output_shapes
:џџџџџџџџџД@

model/conv2/BiasAddBiasAddmodel/conv2/Conv2Dconv2/biases/read*
T0*
data_formatNHWC*1
_output_shapes
:џџџџџџџџџД@
o
model/conv2/conv2/ReluRelumodel/conv2/BiasAdd*
T0*1
_output_shapes
:џџџџџџџџџД@
Ю
model/conv2/MaxPool2D/MaxPoolMaxPoolmodel/conv2/conv2/Relu*1
_output_shapes
:џџџџџџџџџ@*
T0*
data_formatNHWC*
strides
*
ksize
*
paddingSAME
Љ
.conv3/weights/Initializer/random_uniform/shapeConst*
_output_shapes
:* 
_class
loc:@conv3/weights*%
valueB"      @      *
dtype0

,conv3/weights/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: * 
_class
loc:@conv3/weights*
valueB
 *я[qН

,conv3/weights/Initializer/random_uniform/maxConst* 
_class
loc:@conv3/weights*
valueB
 *я[q=*
dtype0*
_output_shapes
: 
ё
6conv3/weights/Initializer/random_uniform/RandomUniformRandomUniform.conv3/weights/Initializer/random_uniform/shape*
T0* 
_class
loc:@conv3/weights*
seed2 *
dtype0*'
_output_shapes
:@*

seed 
в
,conv3/weights/Initializer/random_uniform/subSub,conv3/weights/Initializer/random_uniform/max,conv3/weights/Initializer/random_uniform/min*
T0* 
_class
loc:@conv3/weights*
_output_shapes
: 
э
,conv3/weights/Initializer/random_uniform/mulMul6conv3/weights/Initializer/random_uniform/RandomUniform,conv3/weights/Initializer/random_uniform/sub*
T0* 
_class
loc:@conv3/weights*'
_output_shapes
:@
п
(conv3/weights/Initializer/random_uniformAdd,conv3/weights/Initializer/random_uniform/mul,conv3/weights/Initializer/random_uniform/min*
T0* 
_class
loc:@conv3/weights*'
_output_shapes
:@
Е
conv3/weights
VariableV2*
dtype0*'
_output_shapes
:@*
shared_name * 
_class
loc:@conv3/weights*
	container *
shape:@
д
conv3/weights/AssignAssignconv3/weights(conv3/weights/Initializer/random_uniform*
use_locking(*
T0* 
_class
loc:@conv3/weights*
validate_shape(*'
_output_shapes
:@

conv3/weights/readIdentityconv3/weights*'
_output_shapes
:@*
T0* 
_class
loc:@conv3/weights

conv3/biases/Initializer/zerosConst*
_class
loc:@conv3/biases*
valueB*    *
dtype0*
_output_shapes	
:

conv3/biases
VariableV2*
_class
loc:@conv3/biases*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name 
Л
conv3/biases/AssignAssignconv3/biasesconv3/biases/Initializer/zeros*
T0*
_class
loc:@conv3/biases*
validate_shape(*
_output_shapes	
:*
use_locking(
r
conv3/biases/readIdentityconv3/biases*
T0*
_class
loc:@conv3/biases*
_output_shapes	
:
j
model/conv3/dilation_rateConst*
_output_shapes
:*
valueB"      *
dtype0
њ
model/conv3/Conv2DConv2Dmodel/conv2/MaxPool2D/MaxPoolconv3/weights/read*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*2
_output_shapes 
:џџџџџџџџџ*
	dilations
*
T0

model/conv3/BiasAddBiasAddmodel/conv3/Conv2Dconv3/biases/read*
T0*
data_formatNHWC*2
_output_shapes 
:џџџџџџџџџ
p
model/conv3/conv3/ReluRelumodel/conv3/BiasAdd*
T0*2
_output_shapes 
:џџџџџџџџџ
Ю
model/conv3/MaxPool2D/MaxPoolMaxPoolmodel/conv3/conv3/Relu*
data_formatNHWC*
strides
*
ksize
*
paddingSAME*1
_output_shapes
:џџџџџџџџџD*
T0
Љ
.conv4/weights/Initializer/random_uniform/shapeConst* 
_class
loc:@conv4/weights*%
valueB"            *
dtype0*
_output_shapes
:

,conv4/weights/Initializer/random_uniform/minConst*
_output_shapes
: * 
_class
loc:@conv4/weights*
valueB
 *   О*
dtype0

,conv4/weights/Initializer/random_uniform/maxConst* 
_class
loc:@conv4/weights*
valueB
 *   >*
dtype0*
_output_shapes
: 
ђ
6conv4/weights/Initializer/random_uniform/RandomUniformRandomUniform.conv4/weights/Initializer/random_uniform/shape*
dtype0*(
_output_shapes
:*

seed *
T0* 
_class
loc:@conv4/weights*
seed2 
в
,conv4/weights/Initializer/random_uniform/subSub,conv4/weights/Initializer/random_uniform/max,conv4/weights/Initializer/random_uniform/min*
T0* 
_class
loc:@conv4/weights*
_output_shapes
: 
ю
,conv4/weights/Initializer/random_uniform/mulMul6conv4/weights/Initializer/random_uniform/RandomUniform,conv4/weights/Initializer/random_uniform/sub*
T0* 
_class
loc:@conv4/weights*(
_output_shapes
:
р
(conv4/weights/Initializer/random_uniformAdd,conv4/weights/Initializer/random_uniform/mul,conv4/weights/Initializer/random_uniform/min*(
_output_shapes
:*
T0* 
_class
loc:@conv4/weights
З
conv4/weights
VariableV2*
shared_name * 
_class
loc:@conv4/weights*
	container *
shape:*
dtype0*(
_output_shapes
:
е
conv4/weights/AssignAssignconv4/weights(conv4/weights/Initializer/random_uniform*
use_locking(*
T0* 
_class
loc:@conv4/weights*
validate_shape(*(
_output_shapes
:

conv4/weights/readIdentityconv4/weights*(
_output_shapes
:*
T0* 
_class
loc:@conv4/weights

conv4/biases/Initializer/zerosConst*
_class
loc:@conv4/biases*
valueB*    *
dtype0*
_output_shapes	
:

conv4/biases
VariableV2*
_output_shapes	
:*
shared_name *
_class
loc:@conv4/biases*
	container *
shape:*
dtype0
Л
conv4/biases/AssignAssignconv4/biasesconv4/biases/Initializer/zeros*
T0*
_class
loc:@conv4/biases*
validate_shape(*
_output_shapes	
:*
use_locking(
r
conv4/biases/readIdentityconv4/biases*
T0*
_class
loc:@conv4/biases*
_output_shapes	
:
j
model/conv4/dilation_rateConst*
_output_shapes
:*
valueB"      *
dtype0
љ
model/conv4/Conv2DConv2Dmodel/conv3/MaxPool2D/MaxPoolconv4/weights/read*1
_output_shapes
:џџџџџџџџџD*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME

model/conv4/BiasAddBiasAddmodel/conv4/Conv2Dconv4/biases/read*
T0*
data_formatNHWC*1
_output_shapes
:џџџџџџџџџD
o
model/conv4/conv4/ReluRelumodel/conv4/BiasAdd*1
_output_shapes
:џџџџџџџџџD*
T0
Э
model/conv4/MaxPool2D/MaxPoolMaxPoolmodel/conv4/conv4/Relu*
paddingSAME*0
_output_shapes
:џџџџџџџџџ"G*
T0*
data_formatNHWC*
strides
*
ksize

Љ
.conv5/weights/Initializer/random_uniform/shapeConst* 
_class
loc:@conv5/weights*%
valueB"            *
dtype0*
_output_shapes
:

,conv5/weights/Initializer/random_uniform/minConst* 
_class
loc:@conv5/weights*
valueB
 *СжО*
dtype0*
_output_shapes
: 

,conv5/weights/Initializer/random_uniform/maxConst* 
_class
loc:@conv5/weights*
valueB
 *Сж>*
dtype0*
_output_shapes
: 
ё
6conv5/weights/Initializer/random_uniform/RandomUniformRandomUniform.conv5/weights/Initializer/random_uniform/shape*'
_output_shapes
:*

seed *
T0* 
_class
loc:@conv5/weights*
seed2 *
dtype0
в
,conv5/weights/Initializer/random_uniform/subSub,conv5/weights/Initializer/random_uniform/max,conv5/weights/Initializer/random_uniform/min* 
_class
loc:@conv5/weights*
_output_shapes
: *
T0
э
,conv5/weights/Initializer/random_uniform/mulMul6conv5/weights/Initializer/random_uniform/RandomUniform,conv5/weights/Initializer/random_uniform/sub*
T0* 
_class
loc:@conv5/weights*'
_output_shapes
:
п
(conv5/weights/Initializer/random_uniformAdd,conv5/weights/Initializer/random_uniform/mul,conv5/weights/Initializer/random_uniform/min*
T0* 
_class
loc:@conv5/weights*'
_output_shapes
:
Е
conv5/weights
VariableV2*
dtype0*'
_output_shapes
:*
shared_name * 
_class
loc:@conv5/weights*
	container *
shape:
д
conv5/weights/AssignAssignconv5/weights(conv5/weights/Initializer/random_uniform*
T0* 
_class
loc:@conv5/weights*
validate_shape(*'
_output_shapes
:*
use_locking(

conv5/weights/readIdentityconv5/weights*
T0* 
_class
loc:@conv5/weights*'
_output_shapes
:

conv5/biases/Initializer/zerosConst*
dtype0*
_output_shapes
:*
_class
loc:@conv5/biases*
valueB*    

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
К
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
ї
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
:џџџџџџџџџ"G

model/conv5/BiasAddBiasAddmodel/conv5/Conv2Dconv5/biases/read*
T0*
data_formatNHWC*/
_output_shapes
:џџџџџџџџџ"G
Щ
model/conv5/MaxPool2D/MaxPoolMaxPoolmodel/conv5/BiasAdd*
T0*
data_formatNHWC*
strides
*
ksize
*
paddingSAME*/
_output_shapes
:џџџџџџџџџ$
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
ч
#model/Flatten/flatten/strided_sliceStridedSlicemodel/Flatten/flatten/Shape)model/Flatten/flatten/strided_slice/stack+model/Flatten/flatten/strided_slice/stack_1+model/Flatten/flatten/strided_slice/stack_2*
_output_shapes
: *
T0*
Index0*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask 
p
%model/Flatten/flatten/Reshape/shape/1Const*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 
Б
#model/Flatten/flatten/Reshape/shapePack#model/Flatten/flatten/strided_slice%model/Flatten/flatten/Reshape/shape/1*
T0*

axis *
N*
_output_shapes
:
Ў
model/Flatten/flatten/ReshapeReshapemodel/conv5/MaxPool2D/MaxPool#model/Flatten/flatten/Reshape/shape*
T0*
Tshape0*)
_output_shapes
:џџџџџџџџџ№
l
model_1/conv1/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
ь
model_1/conv1/Conv2DConv2Dpositive_inputconv1/weights/read*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*1
_output_shapes
:џџџџџџџџџ ч *
	dilations
*
T0

model_1/conv1/BiasAddBiasAddmodel_1/conv1/Conv2Dconv1/biases/read*
T0*
data_formatNHWC*1
_output_shapes
:џџџџџџџџџ ч 
s
model_1/conv1/conv1/ReluRelumodel_1/conv1/BiasAdd*
T0*1
_output_shapes
:џџџџџџџџџ ч 
в
model_1/conv1/MaxPool2D/MaxPoolMaxPoolmodel_1/conv1/conv1/Relu*
T0*
data_formatNHWC*
strides
*
ksize
*
paddingSAME*1
_output_shapes
:џџџџџџџџџД 
l
model_1/conv2/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
§
model_1/conv2/Conv2DConv2Dmodel_1/conv1/MaxPool2D/MaxPoolconv2/weights/read*
paddingSAME*1
_output_shapes
:џџџџџџџџџД@*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(

model_1/conv2/BiasAddBiasAddmodel_1/conv2/Conv2Dconv2/biases/read*
data_formatNHWC*1
_output_shapes
:џџџџџџџџџД@*
T0
s
model_1/conv2/conv2/ReluRelumodel_1/conv2/BiasAdd*1
_output_shapes
:џџџџџџџџџД@*
T0
в
model_1/conv2/MaxPool2D/MaxPoolMaxPoolmodel_1/conv2/conv2/Relu*
ksize
*
paddingSAME*1
_output_shapes
:џџџџџџџџџ@*
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
ў
model_1/conv3/Conv2DConv2Dmodel_1/conv2/MaxPool2D/MaxPoolconv3/weights/read*2
_output_shapes 
:џџџџџџџџџ*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME

model_1/conv3/BiasAddBiasAddmodel_1/conv3/Conv2Dconv3/biases/read*
T0*
data_formatNHWC*2
_output_shapes 
:џџџџџџџџџ
t
model_1/conv3/conv3/ReluRelumodel_1/conv3/BiasAdd*
T0*2
_output_shapes 
:џџџџџџџџџ
в
model_1/conv3/MaxPool2D/MaxPoolMaxPoolmodel_1/conv3/conv3/Relu*
ksize
*
paddingSAME*1
_output_shapes
:џџџџџџџџџD*
T0*
data_formatNHWC*
strides

l
model_1/conv4/dilation_rateConst*
dtype0*
_output_shapes
:*
valueB"      
§
model_1/conv4/Conv2DConv2Dmodel_1/conv3/MaxPool2D/MaxPoolconv4/weights/read*
paddingSAME*1
_output_shapes
:џџџџџџџџџD*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(

model_1/conv4/BiasAddBiasAddmodel_1/conv4/Conv2Dconv4/biases/read*
T0*
data_formatNHWC*1
_output_shapes
:џџџџџџџџџD
s
model_1/conv4/conv4/ReluRelumodel_1/conv4/BiasAdd*
T0*1
_output_shapes
:џџџџџџџџџD
б
model_1/conv4/MaxPool2D/MaxPoolMaxPoolmodel_1/conv4/conv4/Relu*
ksize
*
paddingSAME*0
_output_shapes
:џџџџџџџџџ"G*
T0*
strides
*
data_formatNHWC
l
model_1/conv5/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
ћ
model_1/conv5/Conv2DConv2Dmodel_1/conv4/MaxPool2D/MaxPoolconv5/weights/read*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*/
_output_shapes
:џџџџџџџџџ"G*
	dilations
*
T0

model_1/conv5/BiasAddBiasAddmodel_1/conv5/Conv2Dconv5/biases/read*/
_output_shapes
:џџџџџџџџџ"G*
T0*
data_formatNHWC
Э
model_1/conv5/MaxPool2D/MaxPoolMaxPoolmodel_1/conv5/BiasAdd*
T0*
data_formatNHWC*
strides
*
ksize
*
paddingSAME*/
_output_shapes
:џџџџџџџџџ$
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
-model_1/Flatten/flatten/strided_slice/stack_1Const*
_output_shapes
:*
valueB:*
dtype0
w
-model_1/Flatten/flatten/strided_slice/stack_2Const*
_output_shapes
:*
valueB:*
dtype0
ё
%model_1/Flatten/flatten/strided_sliceStridedSlicemodel_1/Flatten/flatten/Shape+model_1/Flatten/flatten/strided_slice/stack-model_1/Flatten/flatten/strided_slice/stack_1-model_1/Flatten/flatten/strided_slice/stack_2*
_output_shapes
: *
T0*
Index0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask 
r
'model_1/Flatten/flatten/Reshape/shape/1Const*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 
З
%model_1/Flatten/flatten/Reshape/shapePack%model_1/Flatten/flatten/strided_slice'model_1/Flatten/flatten/Reshape/shape/1*
_output_shapes
:*
T0*

axis *
N
Д
model_1/Flatten/flatten/ReshapeReshapemodel_1/conv5/MaxPool2D/MaxPool%model_1/Flatten/flatten/Reshape/shape*
T0*
Tshape0*)
_output_shapes
:џџџџџџџџџ№
l
model_2/conv1/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
ь
model_2/conv1/Conv2DConv2Dnegative_inputconv1/weights/read*
paddingSAME*1
_output_shapes
:џџџџџџџџџ ч *
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(

model_2/conv1/BiasAddBiasAddmodel_2/conv1/Conv2Dconv1/biases/read*
data_formatNHWC*1
_output_shapes
:џџџџџџџџџ ч *
T0
s
model_2/conv1/conv1/ReluRelumodel_2/conv1/BiasAdd*
T0*1
_output_shapes
:џџџџџџџџџ ч 
в
model_2/conv1/MaxPool2D/MaxPoolMaxPoolmodel_2/conv1/conv1/Relu*
ksize
*
paddingSAME*1
_output_shapes
:џџџџџџџџџД *
T0*
data_formatNHWC*
strides

l
model_2/conv2/dilation_rateConst*
dtype0*
_output_shapes
:*
valueB"      
§
model_2/conv2/Conv2DConv2Dmodel_2/conv1/MaxPool2D/MaxPoolconv2/weights/read*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*1
_output_shapes
:џџџџџџџџџД@

model_2/conv2/BiasAddBiasAddmodel_2/conv2/Conv2Dconv2/biases/read*
data_formatNHWC*1
_output_shapes
:џџџџџџџџџД@*
T0
s
model_2/conv2/conv2/ReluRelumodel_2/conv2/BiasAdd*
T0*1
_output_shapes
:џџџџџџџџџД@
в
model_2/conv2/MaxPool2D/MaxPoolMaxPoolmodel_2/conv2/conv2/Relu*
T0*
data_formatNHWC*
strides
*
ksize
*
paddingSAME*1
_output_shapes
:џџџџџџџџџ@
l
model_2/conv3/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
ў
model_2/conv3/Conv2DConv2Dmodel_2/conv2/MaxPool2D/MaxPoolconv3/weights/read*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME*2
_output_shapes 
:џџџџџџџџџ

model_2/conv3/BiasAddBiasAddmodel_2/conv3/Conv2Dconv3/biases/read*
T0*
data_formatNHWC*2
_output_shapes 
:џџџџџџџџџ
t
model_2/conv3/conv3/ReluRelumodel_2/conv3/BiasAdd*2
_output_shapes 
:џџџџџџџџџ*
T0
в
model_2/conv3/MaxPool2D/MaxPoolMaxPoolmodel_2/conv3/conv3/Relu*
T0*
data_formatNHWC*
strides
*
ksize
*
paddingSAME*1
_output_shapes
:џџџџџџџџџD
l
model_2/conv4/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
§
model_2/conv4/Conv2DConv2Dmodel_2/conv3/MaxPool2D/MaxPoolconv4/weights/read*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME*1
_output_shapes
:џџџџџџџџџD

model_2/conv4/BiasAddBiasAddmodel_2/conv4/Conv2Dconv4/biases/read*
T0*
data_formatNHWC*1
_output_shapes
:џџџџџџџџџD
s
model_2/conv4/conv4/ReluRelumodel_2/conv4/BiasAdd*
T0*1
_output_shapes
:џџџџџџџџџD
б
model_2/conv4/MaxPool2D/MaxPoolMaxPoolmodel_2/conv4/conv4/Relu*
ksize
*
paddingSAME*0
_output_shapes
:џџџџџџџџџ"G*
T0*
strides
*
data_formatNHWC
l
model_2/conv5/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
ћ
model_2/conv5/Conv2DConv2Dmodel_2/conv4/MaxPool2D/MaxPoolconv5/weights/read*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME*/
_output_shapes
:џџџџџџџџџ"G*
	dilations
*
T0

model_2/conv5/BiasAddBiasAddmodel_2/conv5/Conv2Dconv5/biases/read*
T0*
data_formatNHWC*/
_output_shapes
:џџџџџџџџџ"G
Э
model_2/conv5/MaxPool2D/MaxPoolMaxPoolmodel_2/conv5/BiasAdd*
T0*
data_formatNHWC*
strides
*
ksize
*
paddingSAME*/
_output_shapes
:џџџџџџџџџ$
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
ё
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
џџџџџџџџџ*
dtype0*
_output_shapes
: 
З
%model_2/Flatten/flatten/Reshape/shapePack%model_2/Flatten/flatten/strided_slice'model_2/Flatten/flatten/Reshape/shape/1*
_output_shapes
:*
T0*

axis *
N
Д
model_2/Flatten/flatten/ReshapeReshapemodel_2/conv5/MaxPool2D/MaxPool%model_2/Flatten/flatten/Reshape/shape*
T0*
Tshape0*)
_output_shapes
:џџџџџџџџџ№
~
subSubmodel/Flatten/flatten/Reshapemodel_1/Flatten/flatten/Reshape*)
_output_shapes
:џџџџџџџџџ№*
T0
J
Pow/yConst*
dtype0*
_output_shapes
: *
valueB
 *   @
J
PowPowsubPow/y*
T0*)
_output_shapes
:џџџџџџџџџ№
W
Sum/reduction_indicesConst*
value	B :*
dtype0*
_output_shapes
: 
u
SumSumPowSum/reduction_indices*

Tidx0*
	keep_dims(*
T0*'
_output_shapes
:џџџџџџџџџ
C
SqrtSqrtSum*
T0*'
_output_shapes
:џџџџџџџџџ

sub_1Submodel/Flatten/flatten/Reshapemodel_2/Flatten/flatten/Reshape*
T0*)
_output_shapes
:џџџџџџџџџ№
L
Pow_1/yConst*
valueB
 *   @*
dtype0*
_output_shapes
: 
P
Pow_1Powsub_1Pow_1/y*)
_output_shapes
:џџџџџџџџџ№*
T0
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
:џџџџџџџџџ*

Tidx0*
	keep_dims(
G
Sqrt_1SqrtSum_1*
T0*'
_output_shapes
:џџџџџџџџџ
L
sub_2SubSqrtSqrt_1*'
_output_shapes
:џџџџџџџџџ*
T0
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
:џџџџџџџџџ
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
:џџџџџџџџџ
V
ConstConst*
dtype0*
_output_shapes
:*
valueB"       
Z
MeanMeanMaximumConst*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
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
Ђ
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
gradients/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
X
gradients/grad_ys_0Const*
valueB
 *  ?*
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

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

gradients/Mean_grad/TileTilegradients/Mean_grad/Reshapegradients/Mean_grad/Shape*

Tmultiples0*
T0*'
_output_shapes
:џџџџџџџџџ
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

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

gradients/Mean_grad/Prod_1Prodgradients/Mean_grad/Shape_2gradients/Mean_grad/Const_1*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
_
gradients/Mean_grad/Maximum/yConst*
value	B :*
dtype0*
_output_shapes
: 

gradients/Mean_grad/MaximumMaximumgradients/Mean_grad/Prod_1gradients/Mean_grad/Maximum/y*
_output_shapes
: *
T0

gradients/Mean_grad/floordivFloorDivgradients/Mean_grad/Prodgradients/Mean_grad/Maximum*
_output_shapes
: *
T0
~
gradients/Mean_grad/CastCastgradients/Mean_grad/floordiv*

SrcT0*
Truncate( *

DstT0*
_output_shapes
: 

gradients/Mean_grad/truedivRealDivgradients/Mean_grad/Tilegradients/Mean_grad/Cast*'
_output_shapes
:џџџџџџџџџ*
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
"gradients/Maximum_grad/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Ќ
gradients/Maximum_grad/zerosFillgradients/Maximum_grad/Shape_2"gradients/Maximum_grad/zeros/Const*
T0*

index_type0*'
_output_shapes
:џџџџџџџџџ
u
#gradients/Maximum_grad/GreaterEqualGreaterEqualadd	Maximum/y*
T0*'
_output_shapes
:џџџџџџџџџ
Р
,gradients/Maximum_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/Maximum_grad/Shapegradients/Maximum_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Й
gradients/Maximum_grad/SelectSelect#gradients/Maximum_grad/GreaterEqualgradients/Mean_grad/truedivgradients/Maximum_grad/zeros*
T0*'
_output_shapes
:џџџџџџџџџ
Л
gradients/Maximum_grad/Select_1Select#gradients/Maximum_grad/GreaterEqualgradients/Maximum_grad/zerosgradients/Mean_grad/truediv*
T0*'
_output_shapes
:џџџџџџџџџ
Ў
gradients/Maximum_grad/SumSumgradients/Maximum_grad/Select,gradients/Maximum_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
Ѓ
gradients/Maximum_grad/ReshapeReshapegradients/Maximum_grad/Sumgradients/Maximum_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Д
gradients/Maximum_grad/Sum_1Sumgradients/Maximum_grad/Select_1.gradients/Maximum_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:

 gradients/Maximum_grad/Reshape_1Reshapegradients/Maximum_grad/Sum_1gradients/Maximum_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
s
'gradients/Maximum_grad/tuple/group_depsNoOp^gradients/Maximum_grad/Reshape!^gradients/Maximum_grad/Reshape_1
ъ
/gradients/Maximum_grad/tuple/control_dependencyIdentitygradients/Maximum_grad/Reshape(^gradients/Maximum_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/Maximum_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
п
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
Д
(gradients/add_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_grad/Shapegradients/add_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0
И
gradients/add_grad/SumSum/gradients/Maximum_grad/tuple/control_dependency(gradients/add_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 

gradients/add_grad/ReshapeReshapegradients/add_grad/Sumgradients/add_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
М
gradients/add_grad/Sum_1Sum/gradients/Maximum_grad/tuple/control_dependency*gradients/add_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:

gradients/add_grad/Reshape_1Reshapegradients/add_grad/Sum_1gradients/add_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
g
#gradients/add_grad/tuple/group_depsNoOp^gradients/add_grad/Reshape^gradients/add_grad/Reshape_1
к
+gradients/add_grad/tuple/control_dependencyIdentitygradients/add_grad/Reshape$^gradients/add_grad/tuple/group_deps*
T0*-
_class#
!loc:@gradients/add_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
Я
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
К
*gradients/sub_2_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/sub_2_grad/Shapegradients/sub_2_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
И
gradients/sub_2_grad/SumSum+gradients/add_grad/tuple/control_dependency*gradients/sub_2_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 

gradients/sub_2_grad/ReshapeReshapegradients/sub_2_grad/Sumgradients/sub_2_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
М
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
Ё
gradients/sub_2_grad/Reshape_1Reshapegradients/sub_2_grad/Neggradients/sub_2_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
m
%gradients/sub_2_grad/tuple/group_depsNoOp^gradients/sub_2_grad/Reshape^gradients/sub_2_grad/Reshape_1
т
-gradients/sub_2_grad/tuple/control_dependencyIdentitygradients/sub_2_grad/Reshape&^gradients/sub_2_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/sub_2_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
ш
/gradients/sub_2_grad/tuple/control_dependency_1Identitygradients/sub_2_grad/Reshape_1&^gradients/sub_2_grad/tuple/group_deps*'
_output_shapes
:џџџџџџџџџ*
T0*1
_class'
%#loc:@gradients/sub_2_grad/Reshape_1

gradients/Sqrt_grad/SqrtGradSqrtGradSqrt-gradients/sub_2_grad/tuple/control_dependency*
T0*'
_output_shapes
:џџџџџџџџџ

gradients/Sqrt_1_grad/SqrtGradSqrtGradSqrt_1/gradients/sub_2_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:џџџџџџџџџ
[
gradients/Sum_grad/ShapeShapePow*
T0*
out_type0*
_output_shapes
:

gradients/Sum_grad/SizeConst*+
_class!
loc:@gradients/Sum_grad/Shape*
value	B :*
dtype0*
_output_shapes
: 

gradients/Sum_grad/addAddSum/reduction_indicesgradients/Sum_grad/Size*
_output_shapes
: *
T0*+
_class!
loc:@gradients/Sum_grad/Shape
Ё
gradients/Sum_grad/modFloorModgradients/Sum_grad/addgradients/Sum_grad/Size*
T0*+
_class!
loc:@gradients/Sum_grad/Shape*
_output_shapes
: 

gradients/Sum_grad/Shape_1Const*
dtype0*
_output_shapes
: *+
_class!
loc:@gradients/Sum_grad/Shape*
valueB 

gradients/Sum_grad/range/startConst*+
_class!
loc:@gradients/Sum_grad/Shape*
value	B : *
dtype0*
_output_shapes
: 

gradients/Sum_grad/range/deltaConst*+
_class!
loc:@gradients/Sum_grad/Shape*
value	B :*
dtype0*
_output_shapes
: 
Я
gradients/Sum_grad/rangeRangegradients/Sum_grad/range/startgradients/Sum_grad/Sizegradients/Sum_grad/range/delta*

Tidx0*+
_class!
loc:@gradients/Sum_grad/Shape*
_output_shapes
:

gradients/Sum_grad/Fill/valueConst*+
_class!
loc:@gradients/Sum_grad/Shape*
value	B :*
dtype0*
_output_shapes
: 
К
gradients/Sum_grad/FillFillgradients/Sum_grad/Shape_1gradients/Sum_grad/Fill/value*
_output_shapes
: *
T0*+
_class!
loc:@gradients/Sum_grad/Shape*

index_type0
ё
 gradients/Sum_grad/DynamicStitchDynamicStitchgradients/Sum_grad/rangegradients/Sum_grad/modgradients/Sum_grad/Shapegradients/Sum_grad/Fill*
T0*+
_class!
loc:@gradients/Sum_grad/Shape*
N*
_output_shapes
:

gradients/Sum_grad/Maximum/yConst*+
_class!
loc:@gradients/Sum_grad/Shape*
value	B :*
dtype0*
_output_shapes
: 
З
gradients/Sum_grad/MaximumMaximum gradients/Sum_grad/DynamicStitchgradients/Sum_grad/Maximum/y*
_output_shapes
:*
T0*+
_class!
loc:@gradients/Sum_grad/Shape
Џ
gradients/Sum_grad/floordivFloorDivgradients/Sum_grad/Shapegradients/Sum_grad/Maximum*
T0*+
_class!
loc:@gradients/Sum_grad/Shape*
_output_shapes
:
Ў
gradients/Sum_grad/ReshapeReshapegradients/Sqrt_grad/SqrtGrad gradients/Sum_grad/DynamicStitch*
T0*
Tshape0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ

gradients/Sum_grad/TileTilegradients/Sum_grad/Reshapegradients/Sum_grad/floordiv*
T0*)
_output_shapes
:џџџџџџџџџ№*

Tmultiples0
_
gradients/Sum_1_grad/ShapeShapePow_1*
_output_shapes
:*
T0*
out_type0

gradients/Sum_1_grad/SizeConst*
dtype0*
_output_shapes
: *-
_class#
!loc:@gradients/Sum_1_grad/Shape*
value	B :
Ѓ
gradients/Sum_1_grad/addAddSum_1/reduction_indicesgradients/Sum_1_grad/Size*
T0*-
_class#
!loc:@gradients/Sum_1_grad/Shape*
_output_shapes
: 
Љ
gradients/Sum_1_grad/modFloorModgradients/Sum_1_grad/addgradients/Sum_1_grad/Size*
T0*-
_class#
!loc:@gradients/Sum_1_grad/Shape*
_output_shapes
: 

gradients/Sum_1_grad/Shape_1Const*-
_class#
!loc:@gradients/Sum_1_grad/Shape*
valueB *
dtype0*
_output_shapes
: 

 gradients/Sum_1_grad/range/startConst*-
_class#
!loc:@gradients/Sum_1_grad/Shape*
value	B : *
dtype0*
_output_shapes
: 

 gradients/Sum_1_grad/range/deltaConst*-
_class#
!loc:@gradients/Sum_1_grad/Shape*
value	B :*
dtype0*
_output_shapes
: 
й
gradients/Sum_1_grad/rangeRange gradients/Sum_1_grad/range/startgradients/Sum_1_grad/Size gradients/Sum_1_grad/range/delta*

Tidx0*-
_class#
!loc:@gradients/Sum_1_grad/Shape*
_output_shapes
:

gradients/Sum_1_grad/Fill/valueConst*-
_class#
!loc:@gradients/Sum_1_grad/Shape*
value	B :*
dtype0*
_output_shapes
: 
Т
gradients/Sum_1_grad/FillFillgradients/Sum_1_grad/Shape_1gradients/Sum_1_grad/Fill/value*
_output_shapes
: *
T0*-
_class#
!loc:@gradients/Sum_1_grad/Shape*

index_type0
§
"gradients/Sum_1_grad/DynamicStitchDynamicStitchgradients/Sum_1_grad/rangegradients/Sum_1_grad/modgradients/Sum_1_grad/Shapegradients/Sum_1_grad/Fill*
T0*-
_class#
!loc:@gradients/Sum_1_grad/Shape*
N*
_output_shapes
:

gradients/Sum_1_grad/Maximum/yConst*-
_class#
!loc:@gradients/Sum_1_grad/Shape*
value	B :*
dtype0*
_output_shapes
: 
П
gradients/Sum_1_grad/MaximumMaximum"gradients/Sum_1_grad/DynamicStitchgradients/Sum_1_grad/Maximum/y*
T0*-
_class#
!loc:@gradients/Sum_1_grad/Shape*
_output_shapes
:
З
gradients/Sum_1_grad/floordivFloorDivgradients/Sum_1_grad/Shapegradients/Sum_1_grad/Maximum*
T0*-
_class#
!loc:@gradients/Sum_1_grad/Shape*
_output_shapes
:
Д
gradients/Sum_1_grad/ReshapeReshapegradients/Sqrt_1_grad/SqrtGrad"gradients/Sum_1_grad/DynamicStitch*
T0*
Tshape0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
Є
gradients/Sum_1_grad/TileTilegradients/Sum_1_grad/Reshapegradients/Sum_1_grad/floordiv*

Tmultiples0*
T0*)
_output_shapes
:џџџџџџџџџ№
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
Д
(gradients/Pow_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/Pow_grad/Shapegradients/Pow_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
q
gradients/Pow_grad/mulMulgradients/Sum_grad/TilePow/y*)
_output_shapes
:џџџџџџџџџ№*
T0
]
gradients/Pow_grad/sub/yConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
_
gradients/Pow_grad/subSubPow/ygradients/Pow_grad/sub/y*
T0*
_output_shapes
: 
n
gradients/Pow_grad/PowPowsubgradients/Pow_grad/sub*
T0*)
_output_shapes
:џџџџџџџџџ№

gradients/Pow_grad/mul_1Mulgradients/Pow_grad/mulgradients/Pow_grad/Pow*
T0*)
_output_shapes
:џџџџџџџџџ№
Ё
gradients/Pow_grad/SumSumgradients/Pow_grad/mul_1(gradients/Pow_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 

gradients/Pow_grad/ReshapeReshapegradients/Pow_grad/Sumgradients/Pow_grad/Shape*
T0*
Tshape0*)
_output_shapes
:џџџџџџџџџ№
a
gradients/Pow_grad/Greater/yConst*
valueB
 *    *
dtype0*
_output_shapes
: 
|
gradients/Pow_grad/GreaterGreatersubgradients/Pow_grad/Greater/y*)
_output_shapes
:џџџџџџџџџ№*
T0
e
"gradients/Pow_grad/ones_like/ShapeShapesub*
T0*
out_type0*
_output_shapes
:
g
"gradients/Pow_grad/ones_like/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *  ?
В
gradients/Pow_grad/ones_likeFill"gradients/Pow_grad/ones_like/Shape"gradients/Pow_grad/ones_like/Const*
T0*

index_type0*)
_output_shapes
:џџџџџџџџџ№

gradients/Pow_grad/SelectSelectgradients/Pow_grad/Greatersubgradients/Pow_grad/ones_like*
T0*)
_output_shapes
:џџџџџџџџџ№
l
gradients/Pow_grad/LogLoggradients/Pow_grad/Select*
T0*)
_output_shapes
:џџџџџџџџџ№
c
gradients/Pow_grad/zeros_like	ZerosLikesub*)
_output_shapes
:џџџџџџџџџ№*
T0
Ќ
gradients/Pow_grad/Select_1Selectgradients/Pow_grad/Greatergradients/Pow_grad/Loggradients/Pow_grad/zeros_like*
T0*)
_output_shapes
:џџџџџџџџџ№
q
gradients/Pow_grad/mul_2Mulgradients/Sum_grad/TilePow*
T0*)
_output_shapes
:џџџџџџџџџ№

gradients/Pow_grad/mul_3Mulgradients/Pow_grad/mul_2gradients/Pow_grad/Select_1*
T0*)
_output_shapes
:џџџџџџџџџ№
Ѕ
gradients/Pow_grad/Sum_1Sumgradients/Pow_grad/mul_3*gradients/Pow_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 

gradients/Pow_grad/Reshape_1Reshapegradients/Pow_grad/Sum_1gradients/Pow_grad/Shape_1*
_output_shapes
: *
T0*
Tshape0
g
#gradients/Pow_grad/tuple/group_depsNoOp^gradients/Pow_grad/Reshape^gradients/Pow_grad/Reshape_1
м
+gradients/Pow_grad/tuple/control_dependencyIdentitygradients/Pow_grad/Reshape$^gradients/Pow_grad/tuple/group_deps*
T0*-
_class#
!loc:@gradients/Pow_grad/Reshape*)
_output_shapes
:џџџџџџџџџ№
Я
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
К
*gradients/Pow_1_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/Pow_1_grad/Shapegradients/Pow_1_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
w
gradients/Pow_1_grad/mulMulgradients/Sum_1_grad/TilePow_1/y*
T0*)
_output_shapes
:џџџџџџџџџ№
_
gradients/Pow_1_grad/sub/yConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
e
gradients/Pow_1_grad/subSubPow_1/ygradients/Pow_1_grad/sub/y*
T0*
_output_shapes
: 
t
gradients/Pow_1_grad/PowPowsub_1gradients/Pow_1_grad/sub*)
_output_shapes
:џџџџџџџџџ№*
T0

gradients/Pow_1_grad/mul_1Mulgradients/Pow_1_grad/mulgradients/Pow_1_grad/Pow*
T0*)
_output_shapes
:џџџџџџџџџ№
Ї
gradients/Pow_1_grad/SumSumgradients/Pow_1_grad/mul_1*gradients/Pow_1_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 

gradients/Pow_1_grad/ReshapeReshapegradients/Pow_1_grad/Sumgradients/Pow_1_grad/Shape*
T0*
Tshape0*)
_output_shapes
:џџџџџџџџџ№
c
gradients/Pow_1_grad/Greater/yConst*
valueB
 *    *
dtype0*
_output_shapes
: 

gradients/Pow_1_grad/GreaterGreatersub_1gradients/Pow_1_grad/Greater/y*
T0*)
_output_shapes
:џџџџџџџџџ№
i
$gradients/Pow_1_grad/ones_like/ShapeShapesub_1*
T0*
out_type0*
_output_shapes
:
i
$gradients/Pow_1_grad/ones_like/ConstConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
И
gradients/Pow_1_grad/ones_likeFill$gradients/Pow_1_grad/ones_like/Shape$gradients/Pow_1_grad/ones_like/Const*
T0*

index_type0*)
_output_shapes
:џџџџџџџџџ№

gradients/Pow_1_grad/SelectSelectgradients/Pow_1_grad/Greatersub_1gradients/Pow_1_grad/ones_like*
T0*)
_output_shapes
:џџџџџџџџџ№
p
gradients/Pow_1_grad/LogLoggradients/Pow_1_grad/Select*
T0*)
_output_shapes
:џџџџџџџџџ№
g
gradients/Pow_1_grad/zeros_like	ZerosLikesub_1*
T0*)
_output_shapes
:џџџџџџџџџ№
Д
gradients/Pow_1_grad/Select_1Selectgradients/Pow_1_grad/Greatergradients/Pow_1_grad/Loggradients/Pow_1_grad/zeros_like*)
_output_shapes
:џџџџџџџџџ№*
T0
w
gradients/Pow_1_grad/mul_2Mulgradients/Sum_1_grad/TilePow_1*
T0*)
_output_shapes
:џџџџџџџџџ№

gradients/Pow_1_grad/mul_3Mulgradients/Pow_1_grad/mul_2gradients/Pow_1_grad/Select_1*
T0*)
_output_shapes
:џџџџџџџџџ№
Ћ
gradients/Pow_1_grad/Sum_1Sumgradients/Pow_1_grad/mul_3,gradients/Pow_1_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 

gradients/Pow_1_grad/Reshape_1Reshapegradients/Pow_1_grad/Sum_1gradients/Pow_1_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
m
%gradients/Pow_1_grad/tuple/group_depsNoOp^gradients/Pow_1_grad/Reshape^gradients/Pow_1_grad/Reshape_1
ф
-gradients/Pow_1_grad/tuple/control_dependencyIdentitygradients/Pow_1_grad/Reshape&^gradients/Pow_1_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/Pow_1_grad/Reshape*)
_output_shapes
:џџџџџџџџџ№
з
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
gradients/sub_grad/Shape_1Shapemodel_1/Flatten/flatten/Reshape*
_output_shapes
:*
T0*
out_type0
Д
(gradients/sub_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/sub_grad/Shapegradients/sub_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Д
gradients/sub_grad/SumSum+gradients/Pow_grad/tuple/control_dependency(gradients/sub_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 

gradients/sub_grad/ReshapeReshapegradients/sub_grad/Sumgradients/sub_grad/Shape*
T0*
Tshape0*)
_output_shapes
:џџџџџџџџџ№
И
gradients/sub_grad/Sum_1Sum+gradients/Pow_grad/tuple/control_dependency*gradients/sub_grad/BroadcastGradientArgs:1*
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

gradients/sub_grad/Reshape_1Reshapegradients/sub_grad/Neggradients/sub_grad/Shape_1*)
_output_shapes
:џџџџџџџџџ№*
T0*
Tshape0
g
#gradients/sub_grad/tuple/group_depsNoOp^gradients/sub_grad/Reshape^gradients/sub_grad/Reshape_1
м
+gradients/sub_grad/tuple/control_dependencyIdentitygradients/sub_grad/Reshape$^gradients/sub_grad/tuple/group_deps*
T0*-
_class#
!loc:@gradients/sub_grad/Reshape*)
_output_shapes
:џџџџџџџџџ№
т
-gradients/sub_grad/tuple/control_dependency_1Identitygradients/sub_grad/Reshape_1$^gradients/sub_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/sub_grad/Reshape_1*)
_output_shapes
:џџџџџџџџџ№
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
К
*gradients/sub_1_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/sub_1_grad/Shapegradients/sub_1_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
К
gradients/sub_1_grad/SumSum-gradients/Pow_1_grad/tuple/control_dependency*gradients/sub_1_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 

gradients/sub_1_grad/ReshapeReshapegradients/sub_1_grad/Sumgradients/sub_1_grad/Shape*
T0*
Tshape0*)
_output_shapes
:џџџџџџџџџ№
О
gradients/sub_1_grad/Sum_1Sum-gradients/Pow_1_grad/tuple/control_dependency,gradients/sub_1_grad/BroadcastGradientArgs:1*
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
Ѓ
gradients/sub_1_grad/Reshape_1Reshapegradients/sub_1_grad/Neggradients/sub_1_grad/Shape_1*)
_output_shapes
:џџџџџџџџџ№*
T0*
Tshape0
m
%gradients/sub_1_grad/tuple/group_depsNoOp^gradients/sub_1_grad/Reshape^gradients/sub_1_grad/Reshape_1
ф
-gradients/sub_1_grad/tuple/control_dependencyIdentitygradients/sub_1_grad/Reshape&^gradients/sub_1_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/sub_1_grad/Reshape*)
_output_shapes
:џџџџџџџџџ№
ъ
/gradients/sub_1_grad/tuple/control_dependency_1Identitygradients/sub_1_grad/Reshape_1&^gradients/sub_1_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/sub_1_grad/Reshape_1*)
_output_shapes
:џџџџџџџџџ№

4gradients/model_1/Flatten/flatten/Reshape_grad/ShapeShapemodel_1/conv5/MaxPool2D/MaxPool*
T0*
out_type0*
_output_shapes
:
ю
6gradients/model_1/Flatten/flatten/Reshape_grad/ReshapeReshape-gradients/sub_grad/tuple/control_dependency_14gradients/model_1/Flatten/flatten/Reshape_grad/Shape*
T0*
Tshape0*/
_output_shapes
:џџџџџџџџџ$
о
gradients/AddNAddN+gradients/sub_grad/tuple/control_dependency-gradients/sub_1_grad/tuple/control_dependency*
T0*-
_class#
!loc:@gradients/sub_grad/Reshape*
N*)
_output_shapes
:џџџџџџџџџ№

2gradients/model/Flatten/flatten/Reshape_grad/ShapeShapemodel/conv5/MaxPool2D/MaxPool*
_output_shapes
:*
T0*
out_type0
Ы
4gradients/model/Flatten/flatten/Reshape_grad/ReshapeReshapegradients/AddN2gradients/model/Flatten/flatten/Reshape_grad/Shape*
T0*
Tshape0*/
_output_shapes
:џџџџџџџџџ$

4gradients/model_2/Flatten/flatten/Reshape_grad/ShapeShapemodel_2/conv5/MaxPool2D/MaxPool*
T0*
out_type0*
_output_shapes
:
№
6gradients/model_2/Flatten/flatten/Reshape_grad/ReshapeReshape/gradients/sub_1_grad/tuple/control_dependency_14gradients/model_2/Flatten/flatten/Reshape_grad/Shape*
T0*
Tshape0*/
_output_shapes
:џџџџџџџџџ$
Х
:gradients/model_1/conv5/MaxPool2D/MaxPool_grad/MaxPoolGradMaxPoolGradmodel_1/conv5/BiasAddmodel_1/conv5/MaxPool2D/MaxPool6gradients/model_1/Flatten/flatten/Reshape_grad/Reshape*/
_output_shapes
:џџџџџџџџџ"G*
T0*
data_formatNHWC*
strides
*
ksize
*
paddingSAME
Н
8gradients/model/conv5/MaxPool2D/MaxPool_grad/MaxPoolGradMaxPoolGradmodel/conv5/BiasAddmodel/conv5/MaxPool2D/MaxPool4gradients/model/Flatten/flatten/Reshape_grad/Reshape*
T0*
data_formatNHWC*
strides
*
ksize
*
paddingSAME*/
_output_shapes
:џџџџџџџџџ"G
Х
:gradients/model_2/conv5/MaxPool2D/MaxPool_grad/MaxPoolGradMaxPoolGradmodel_2/conv5/BiasAddmodel_2/conv5/MaxPool2D/MaxPool6gradients/model_2/Flatten/flatten/Reshape_grad/Reshape*/
_output_shapes
:џџџџџџџџџ"G*
T0*
data_formatNHWC*
strides
*
ksize
*
paddingSAME
З
0gradients/model_1/conv5/BiasAdd_grad/BiasAddGradBiasAddGrad:gradients/model_1/conv5/MaxPool2D/MaxPool_grad/MaxPoolGrad*
T0*
data_formatNHWC*
_output_shapes
:
­
5gradients/model_1/conv5/BiasAdd_grad/tuple/group_depsNoOp1^gradients/model_1/conv5/BiasAdd_grad/BiasAddGrad;^gradients/model_1/conv5/MaxPool2D/MaxPool_grad/MaxPoolGrad
Ц
=gradients/model_1/conv5/BiasAdd_grad/tuple/control_dependencyIdentity:gradients/model_1/conv5/MaxPool2D/MaxPool_grad/MaxPoolGrad6^gradients/model_1/conv5/BiasAdd_grad/tuple/group_deps*
T0*M
_classC
A?loc:@gradients/model_1/conv5/MaxPool2D/MaxPool_grad/MaxPoolGrad*/
_output_shapes
:џџџџџџџџџ"G

?gradients/model_1/conv5/BiasAdd_grad/tuple/control_dependency_1Identity0gradients/model_1/conv5/BiasAdd_grad/BiasAddGrad6^gradients/model_1/conv5/BiasAdd_grad/tuple/group_deps*
_output_shapes
:*
T0*C
_class9
75loc:@gradients/model_1/conv5/BiasAdd_grad/BiasAddGrad
Г
.gradients/model/conv5/BiasAdd_grad/BiasAddGradBiasAddGrad8gradients/model/conv5/MaxPool2D/MaxPool_grad/MaxPoolGrad*
data_formatNHWC*
_output_shapes
:*
T0
Ї
3gradients/model/conv5/BiasAdd_grad/tuple/group_depsNoOp/^gradients/model/conv5/BiasAdd_grad/BiasAddGrad9^gradients/model/conv5/MaxPool2D/MaxPool_grad/MaxPoolGrad
О
;gradients/model/conv5/BiasAdd_grad/tuple/control_dependencyIdentity8gradients/model/conv5/MaxPool2D/MaxPool_grad/MaxPoolGrad4^gradients/model/conv5/BiasAdd_grad/tuple/group_deps*
T0*K
_classA
?=loc:@gradients/model/conv5/MaxPool2D/MaxPool_grad/MaxPoolGrad*/
_output_shapes
:џџџџџџџџџ"G

=gradients/model/conv5/BiasAdd_grad/tuple/control_dependency_1Identity.gradients/model/conv5/BiasAdd_grad/BiasAddGrad4^gradients/model/conv5/BiasAdd_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients/model/conv5/BiasAdd_grad/BiasAddGrad*
_output_shapes
:
З
0gradients/model_2/conv5/BiasAdd_grad/BiasAddGradBiasAddGrad:gradients/model_2/conv5/MaxPool2D/MaxPool_grad/MaxPoolGrad*
T0*
data_formatNHWC*
_output_shapes
:
­
5gradients/model_2/conv5/BiasAdd_grad/tuple/group_depsNoOp1^gradients/model_2/conv5/BiasAdd_grad/BiasAddGrad;^gradients/model_2/conv5/MaxPool2D/MaxPool_grad/MaxPoolGrad
Ц
=gradients/model_2/conv5/BiasAdd_grad/tuple/control_dependencyIdentity:gradients/model_2/conv5/MaxPool2D/MaxPool_grad/MaxPoolGrad6^gradients/model_2/conv5/BiasAdd_grad/tuple/group_deps*
T0*M
_classC
A?loc:@gradients/model_2/conv5/MaxPool2D/MaxPool_grad/MaxPoolGrad*/
_output_shapes
:џџџџџџџџџ"G

?gradients/model_2/conv5/BiasAdd_grad/tuple/control_dependency_1Identity0gradients/model_2/conv5/BiasAdd_grad/BiasAddGrad6^gradients/model_2/conv5/BiasAdd_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/model_2/conv5/BiasAdd_grad/BiasAddGrad*
_output_shapes
:
­
*gradients/model_1/conv5/Conv2D_grad/ShapeNShapeNmodel_1/conv4/MaxPool2D/MaxPoolconv5/weights/read*
T0*
out_type0*
N* 
_output_shapes
::
і
7gradients/model_1/conv5/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput*gradients/model_1/conv5/Conv2D_grad/ShapeNconv5/weights/read=gradients/model_1/conv5/BiasAdd_grad/tuple/control_dependency*
paddingSAME*0
_output_shapes
:џџџџџџџџџ"G*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(
ў
8gradients/model_1/conv5/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFiltermodel_1/conv4/MaxPool2D/MaxPool,gradients/model_1/conv5/Conv2D_grad/ShapeN:1=gradients/model_1/conv5/BiasAdd_grad/tuple/control_dependency*'
_output_shapes
:*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME
Б
4gradients/model_1/conv5/Conv2D_grad/tuple/group_depsNoOp9^gradients/model_1/conv5/Conv2D_grad/Conv2DBackpropFilter8^gradients/model_1/conv5/Conv2D_grad/Conv2DBackpropInput
П
<gradients/model_1/conv5/Conv2D_grad/tuple/control_dependencyIdentity7gradients/model_1/conv5/Conv2D_grad/Conv2DBackpropInput5^gradients/model_1/conv5/Conv2D_grad/tuple/group_deps*
T0*J
_class@
><loc:@gradients/model_1/conv5/Conv2D_grad/Conv2DBackpropInput*0
_output_shapes
:џџџџџџџџџ"G
К
>gradients/model_1/conv5/Conv2D_grad/tuple/control_dependency_1Identity8gradients/model_1/conv5/Conv2D_grad/Conv2DBackpropFilter5^gradients/model_1/conv5/Conv2D_grad/tuple/group_deps*'
_output_shapes
:*
T0*K
_classA
?=loc:@gradients/model_1/conv5/Conv2D_grad/Conv2DBackpropFilter
Љ
(gradients/model/conv5/Conv2D_grad/ShapeNShapeNmodel/conv4/MaxPool2D/MaxPoolconv5/weights/read*
T0*
out_type0*
N* 
_output_shapes
::
№
5gradients/model/conv5/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput(gradients/model/conv5/Conv2D_grad/ShapeNconv5/weights/read;gradients/model/conv5/BiasAdd_grad/tuple/control_dependency*0
_output_shapes
:џџџџџџџџџ"G*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME
і
6gradients/model/conv5/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFiltermodel/conv4/MaxPool2D/MaxPool*gradients/model/conv5/Conv2D_grad/ShapeN:1;gradients/model/conv5/BiasAdd_grad/tuple/control_dependency*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*'
_output_shapes
:*
	dilations
*
T0
Ћ
2gradients/model/conv5/Conv2D_grad/tuple/group_depsNoOp7^gradients/model/conv5/Conv2D_grad/Conv2DBackpropFilter6^gradients/model/conv5/Conv2D_grad/Conv2DBackpropInput
З
:gradients/model/conv5/Conv2D_grad/tuple/control_dependencyIdentity5gradients/model/conv5/Conv2D_grad/Conv2DBackpropInput3^gradients/model/conv5/Conv2D_grad/tuple/group_deps*
T0*H
_class>
<:loc:@gradients/model/conv5/Conv2D_grad/Conv2DBackpropInput*0
_output_shapes
:џџџџџџџџџ"G
В
<gradients/model/conv5/Conv2D_grad/tuple/control_dependency_1Identity6gradients/model/conv5/Conv2D_grad/Conv2DBackpropFilter3^gradients/model/conv5/Conv2D_grad/tuple/group_deps*
T0*I
_class?
=;loc:@gradients/model/conv5/Conv2D_grad/Conv2DBackpropFilter*'
_output_shapes
:
­
*gradients/model_2/conv5/Conv2D_grad/ShapeNShapeNmodel_2/conv4/MaxPool2D/MaxPoolconv5/weights/read*
T0*
out_type0*
N* 
_output_shapes
::
і
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
:џџџџџџџџџ"G
ў
8gradients/model_2/conv5/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFiltermodel_2/conv4/MaxPool2D/MaxPool,gradients/model_2/conv5/Conv2D_grad/ShapeN:1=gradients/model_2/conv5/BiasAdd_grad/tuple/control_dependency*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*'
_output_shapes
:*
	dilations
*
T0
Б
4gradients/model_2/conv5/Conv2D_grad/tuple/group_depsNoOp9^gradients/model_2/conv5/Conv2D_grad/Conv2DBackpropFilter8^gradients/model_2/conv5/Conv2D_grad/Conv2DBackpropInput
П
<gradients/model_2/conv5/Conv2D_grad/tuple/control_dependencyIdentity7gradients/model_2/conv5/Conv2D_grad/Conv2DBackpropInput5^gradients/model_2/conv5/Conv2D_grad/tuple/group_deps*
T0*J
_class@
><loc:@gradients/model_2/conv5/Conv2D_grad/Conv2DBackpropInput*0
_output_shapes
:џџџџџџџџџ"G
К
>gradients/model_2/conv5/Conv2D_grad/tuple/control_dependency_1Identity8gradients/model_2/conv5/Conv2D_grad/Conv2DBackpropFilter5^gradients/model_2/conv5/Conv2D_grad/tuple/group_deps*
T0*K
_classA
?=loc:@gradients/model_2/conv5/Conv2D_grad/Conv2DBackpropFilter*'
_output_shapes
:
Ь
gradients/AddN_1AddN?gradients/model_1/conv5/BiasAdd_grad/tuple/control_dependency_1=gradients/model/conv5/BiasAdd_grad/tuple/control_dependency_1?gradients/model_2/conv5/BiasAdd_grad/tuple/control_dependency_1*
N*
_output_shapes
:*
T0*C
_class9
75loc:@gradients/model_1/conv5/BiasAdd_grad/BiasAddGrad
а
:gradients/model_1/conv4/MaxPool2D/MaxPool_grad/MaxPoolGradMaxPoolGradmodel_1/conv4/conv4/Relumodel_1/conv4/MaxPool2D/MaxPool<gradients/model_1/conv5/Conv2D_grad/tuple/control_dependency*
T0*
data_formatNHWC*
strides
*
ksize
*
paddingSAME*1
_output_shapes
:џџџџџџџџџD
Ш
8gradients/model/conv4/MaxPool2D/MaxPool_grad/MaxPoolGradMaxPoolGradmodel/conv4/conv4/Relumodel/conv4/MaxPool2D/MaxPool:gradients/model/conv5/Conv2D_grad/tuple/control_dependency*
T0*
data_formatNHWC*
strides
*
ksize
*
paddingSAME*1
_output_shapes
:џџџџџџџџџD
а
:gradients/model_2/conv4/MaxPool2D/MaxPool_grad/MaxPoolGradMaxPoolGradmodel_2/conv4/conv4/Relumodel_2/conv4/MaxPool2D/MaxPool<gradients/model_2/conv5/Conv2D_grad/tuple/control_dependency*
ksize
*
paddingSAME*1
_output_shapes
:џџџџџџџџџD*
T0*
data_formatNHWC*
strides

о
gradients/AddN_2AddN>gradients/model_1/conv5/Conv2D_grad/tuple/control_dependency_1<gradients/model/conv5/Conv2D_grad/tuple/control_dependency_1>gradients/model_2/conv5/Conv2D_grad/tuple/control_dependency_1*
N*'
_output_shapes
:*
T0*K
_classA
?=loc:@gradients/model_1/conv5/Conv2D_grad/Conv2DBackpropFilter
Ю
0gradients/model_1/conv4/conv4/Relu_grad/ReluGradReluGrad:gradients/model_1/conv4/MaxPool2D/MaxPool_grad/MaxPoolGradmodel_1/conv4/conv4/Relu*
T0*1
_output_shapes
:џџџџџџџџџD
Ш
.gradients/model/conv4/conv4/Relu_grad/ReluGradReluGrad8gradients/model/conv4/MaxPool2D/MaxPool_grad/MaxPoolGradmodel/conv4/conv4/Relu*
T0*1
_output_shapes
:џџџџџџџџџD
Ю
0gradients/model_2/conv4/conv4/Relu_grad/ReluGradReluGrad:gradients/model_2/conv4/MaxPool2D/MaxPool_grad/MaxPoolGradmodel_2/conv4/conv4/Relu*
T0*1
_output_shapes
:џџџџџџџџџD
Ў
0gradients/model_1/conv4/BiasAdd_grad/BiasAddGradBiasAddGrad0gradients/model_1/conv4/conv4/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes	
:
Ѓ
5gradients/model_1/conv4/BiasAdd_grad/tuple/group_depsNoOp1^gradients/model_1/conv4/BiasAdd_grad/BiasAddGrad1^gradients/model_1/conv4/conv4/Relu_grad/ReluGrad
Д
=gradients/model_1/conv4/BiasAdd_grad/tuple/control_dependencyIdentity0gradients/model_1/conv4/conv4/Relu_grad/ReluGrad6^gradients/model_1/conv4/BiasAdd_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/model_1/conv4/conv4/Relu_grad/ReluGrad*1
_output_shapes
:џџџџџџџџџD
 
?gradients/model_1/conv4/BiasAdd_grad/tuple/control_dependency_1Identity0gradients/model_1/conv4/BiasAdd_grad/BiasAddGrad6^gradients/model_1/conv4/BiasAdd_grad/tuple/group_deps*
_output_shapes	
:*
T0*C
_class9
75loc:@gradients/model_1/conv4/BiasAdd_grad/BiasAddGrad
Њ
.gradients/model/conv4/BiasAdd_grad/BiasAddGradBiasAddGrad.gradients/model/conv4/conv4/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes	
:

3gradients/model/conv4/BiasAdd_grad/tuple/group_depsNoOp/^gradients/model/conv4/BiasAdd_grad/BiasAddGrad/^gradients/model/conv4/conv4/Relu_grad/ReluGrad
Ќ
;gradients/model/conv4/BiasAdd_grad/tuple/control_dependencyIdentity.gradients/model/conv4/conv4/Relu_grad/ReluGrad4^gradients/model/conv4/BiasAdd_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients/model/conv4/conv4/Relu_grad/ReluGrad*1
_output_shapes
:џџџџџџџџџD

=gradients/model/conv4/BiasAdd_grad/tuple/control_dependency_1Identity.gradients/model/conv4/BiasAdd_grad/BiasAddGrad4^gradients/model/conv4/BiasAdd_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients/model/conv4/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:
Ў
0gradients/model_2/conv4/BiasAdd_grad/BiasAddGradBiasAddGrad0gradients/model_2/conv4/conv4/Relu_grad/ReluGrad*
data_formatNHWC*
_output_shapes	
:*
T0
Ѓ
5gradients/model_2/conv4/BiasAdd_grad/tuple/group_depsNoOp1^gradients/model_2/conv4/BiasAdd_grad/BiasAddGrad1^gradients/model_2/conv4/conv4/Relu_grad/ReluGrad
Д
=gradients/model_2/conv4/BiasAdd_grad/tuple/control_dependencyIdentity0gradients/model_2/conv4/conv4/Relu_grad/ReluGrad6^gradients/model_2/conv4/BiasAdd_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/model_2/conv4/conv4/Relu_grad/ReluGrad*1
_output_shapes
:џџџџџџџџџD
 
?gradients/model_2/conv4/BiasAdd_grad/tuple/control_dependency_1Identity0gradients/model_2/conv4/BiasAdd_grad/BiasAddGrad6^gradients/model_2/conv4/BiasAdd_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/model_2/conv4/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:
­
*gradients/model_1/conv4/Conv2D_grad/ShapeNShapeNmodel_1/conv3/MaxPool2D/MaxPoolconv4/weights/read*
N* 
_output_shapes
::*
T0*
out_type0
ї
7gradients/model_1/conv4/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput*gradients/model_1/conv4/Conv2D_grad/ShapeNconv4/weights/read=gradients/model_1/conv4/BiasAdd_grad/tuple/control_dependency*1
_output_shapes
:џџџџџџџџџD*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME
џ
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
:
Б
4gradients/model_1/conv4/Conv2D_grad/tuple/group_depsNoOp9^gradients/model_1/conv4/Conv2D_grad/Conv2DBackpropFilter8^gradients/model_1/conv4/Conv2D_grad/Conv2DBackpropInput
Р
<gradients/model_1/conv4/Conv2D_grad/tuple/control_dependencyIdentity7gradients/model_1/conv4/Conv2D_grad/Conv2DBackpropInput5^gradients/model_1/conv4/Conv2D_grad/tuple/group_deps*1
_output_shapes
:џџџџџџџџџD*
T0*J
_class@
><loc:@gradients/model_1/conv4/Conv2D_grad/Conv2DBackpropInput
Л
>gradients/model_1/conv4/Conv2D_grad/tuple/control_dependency_1Identity8gradients/model_1/conv4/Conv2D_grad/Conv2DBackpropFilter5^gradients/model_1/conv4/Conv2D_grad/tuple/group_deps*(
_output_shapes
:*
T0*K
_classA
?=loc:@gradients/model_1/conv4/Conv2D_grad/Conv2DBackpropFilter
Љ
(gradients/model/conv4/Conv2D_grad/ShapeNShapeNmodel/conv3/MaxPool2D/MaxPoolconv4/weights/read*
T0*
out_type0*
N* 
_output_shapes
::
ё
5gradients/model/conv4/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput(gradients/model/conv4/Conv2D_grad/ShapeNconv4/weights/read;gradients/model/conv4/BiasAdd_grad/tuple/control_dependency*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*1
_output_shapes
:џџџџџџџџџD*
	dilations
*
T0
ї
6gradients/model/conv4/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFiltermodel/conv3/MaxPool2D/MaxPool*gradients/model/conv4/Conv2D_grad/ShapeN:1;gradients/model/conv4/BiasAdd_grad/tuple/control_dependency*(
_output_shapes
:*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME
Ћ
2gradients/model/conv4/Conv2D_grad/tuple/group_depsNoOp7^gradients/model/conv4/Conv2D_grad/Conv2DBackpropFilter6^gradients/model/conv4/Conv2D_grad/Conv2DBackpropInput
И
:gradients/model/conv4/Conv2D_grad/tuple/control_dependencyIdentity5gradients/model/conv4/Conv2D_grad/Conv2DBackpropInput3^gradients/model/conv4/Conv2D_grad/tuple/group_deps*
T0*H
_class>
<:loc:@gradients/model/conv4/Conv2D_grad/Conv2DBackpropInput*1
_output_shapes
:џџџџџџџџџD
Г
<gradients/model/conv4/Conv2D_grad/tuple/control_dependency_1Identity6gradients/model/conv4/Conv2D_grad/Conv2DBackpropFilter3^gradients/model/conv4/Conv2D_grad/tuple/group_deps*
T0*I
_class?
=;loc:@gradients/model/conv4/Conv2D_grad/Conv2DBackpropFilter*(
_output_shapes
:
­
*gradients/model_2/conv4/Conv2D_grad/ShapeNShapeNmodel_2/conv3/MaxPool2D/MaxPoolconv4/weights/read*
T0*
out_type0*
N* 
_output_shapes
::
ї
7gradients/model_2/conv4/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput*gradients/model_2/conv4/Conv2D_grad/ShapeNconv4/weights/read=gradients/model_2/conv4/BiasAdd_grad/tuple/control_dependency*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*1
_output_shapes
:џџџџџџџџџD*
	dilations

џ
8gradients/model_2/conv4/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFiltermodel_2/conv3/MaxPool2D/MaxPool,gradients/model_2/conv4/Conv2D_grad/ShapeN:1=gradients/model_2/conv4/BiasAdd_grad/tuple/control_dependency*
paddingSAME*(
_output_shapes
:*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(
Б
4gradients/model_2/conv4/Conv2D_grad/tuple/group_depsNoOp9^gradients/model_2/conv4/Conv2D_grad/Conv2DBackpropFilter8^gradients/model_2/conv4/Conv2D_grad/Conv2DBackpropInput
Р
<gradients/model_2/conv4/Conv2D_grad/tuple/control_dependencyIdentity7gradients/model_2/conv4/Conv2D_grad/Conv2DBackpropInput5^gradients/model_2/conv4/Conv2D_grad/tuple/group_deps*
T0*J
_class@
><loc:@gradients/model_2/conv4/Conv2D_grad/Conv2DBackpropInput*1
_output_shapes
:џџџџџџџџџD
Л
>gradients/model_2/conv4/Conv2D_grad/tuple/control_dependency_1Identity8gradients/model_2/conv4/Conv2D_grad/Conv2DBackpropFilter5^gradients/model_2/conv4/Conv2D_grad/tuple/group_deps*
T0*K
_classA
?=loc:@gradients/model_2/conv4/Conv2D_grad/Conv2DBackpropFilter*(
_output_shapes
:
Э
gradients/AddN_3AddN?gradients/model_1/conv4/BiasAdd_grad/tuple/control_dependency_1=gradients/model/conv4/BiasAdd_grad/tuple/control_dependency_1?gradients/model_2/conv4/BiasAdd_grad/tuple/control_dependency_1*
T0*C
_class9
75loc:@gradients/model_1/conv4/BiasAdd_grad/BiasAddGrad*
N*
_output_shapes	
:
б
:gradients/model_1/conv3/MaxPool2D/MaxPool_grad/MaxPoolGradMaxPoolGradmodel_1/conv3/conv3/Relumodel_1/conv3/MaxPool2D/MaxPool<gradients/model_1/conv4/Conv2D_grad/tuple/control_dependency*
T0*
data_formatNHWC*
strides
*
ksize
*
paddingSAME*2
_output_shapes 
:џџџџџџџџџ
Щ
8gradients/model/conv3/MaxPool2D/MaxPool_grad/MaxPoolGradMaxPoolGradmodel/conv3/conv3/Relumodel/conv3/MaxPool2D/MaxPool:gradients/model/conv4/Conv2D_grad/tuple/control_dependency*
T0*
data_formatNHWC*
strides
*
ksize
*
paddingSAME*2
_output_shapes 
:џџџџџџџџџ
б
:gradients/model_2/conv3/MaxPool2D/MaxPool_grad/MaxPoolGradMaxPoolGradmodel_2/conv3/conv3/Relumodel_2/conv3/MaxPool2D/MaxPool<gradients/model_2/conv4/Conv2D_grad/tuple/control_dependency*2
_output_shapes 
:џџџџџџџџџ*
T0*
strides
*
data_formatNHWC*
ksize
*
paddingSAME
п
gradients/AddN_4AddN>gradients/model_1/conv4/Conv2D_grad/tuple/control_dependency_1<gradients/model/conv4/Conv2D_grad/tuple/control_dependency_1>gradients/model_2/conv4/Conv2D_grad/tuple/control_dependency_1*
N*(
_output_shapes
:*
T0*K
_classA
?=loc:@gradients/model_1/conv4/Conv2D_grad/Conv2DBackpropFilter
Я
0gradients/model_1/conv3/conv3/Relu_grad/ReluGradReluGrad:gradients/model_1/conv3/MaxPool2D/MaxPool_grad/MaxPoolGradmodel_1/conv3/conv3/Relu*
T0*2
_output_shapes 
:џџџџџџџџџ
Щ
.gradients/model/conv3/conv3/Relu_grad/ReluGradReluGrad8gradients/model/conv3/MaxPool2D/MaxPool_grad/MaxPoolGradmodel/conv3/conv3/Relu*
T0*2
_output_shapes 
:џџџџџџџџџ
Я
0gradients/model_2/conv3/conv3/Relu_grad/ReluGradReluGrad:gradients/model_2/conv3/MaxPool2D/MaxPool_grad/MaxPoolGradmodel_2/conv3/conv3/Relu*
T0*2
_output_shapes 
:џџџџџџџџџ
Ў
0gradients/model_1/conv3/BiasAdd_grad/BiasAddGradBiasAddGrad0gradients/model_1/conv3/conv3/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes	
:
Ѓ
5gradients/model_1/conv3/BiasAdd_grad/tuple/group_depsNoOp1^gradients/model_1/conv3/BiasAdd_grad/BiasAddGrad1^gradients/model_1/conv3/conv3/Relu_grad/ReluGrad
Е
=gradients/model_1/conv3/BiasAdd_grad/tuple/control_dependencyIdentity0gradients/model_1/conv3/conv3/Relu_grad/ReluGrad6^gradients/model_1/conv3/BiasAdd_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/model_1/conv3/conv3/Relu_grad/ReluGrad*2
_output_shapes 
:џџџџџџџџџ
 
?gradients/model_1/conv3/BiasAdd_grad/tuple/control_dependency_1Identity0gradients/model_1/conv3/BiasAdd_grad/BiasAddGrad6^gradients/model_1/conv3/BiasAdd_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/model_1/conv3/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:
Њ
.gradients/model/conv3/BiasAdd_grad/BiasAddGradBiasAddGrad.gradients/model/conv3/conv3/Relu_grad/ReluGrad*
data_formatNHWC*
_output_shapes	
:*
T0

3gradients/model/conv3/BiasAdd_grad/tuple/group_depsNoOp/^gradients/model/conv3/BiasAdd_grad/BiasAddGrad/^gradients/model/conv3/conv3/Relu_grad/ReluGrad
­
;gradients/model/conv3/BiasAdd_grad/tuple/control_dependencyIdentity.gradients/model/conv3/conv3/Relu_grad/ReluGrad4^gradients/model/conv3/BiasAdd_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients/model/conv3/conv3/Relu_grad/ReluGrad*2
_output_shapes 
:џџџџџџџџџ

=gradients/model/conv3/BiasAdd_grad/tuple/control_dependency_1Identity.gradients/model/conv3/BiasAdd_grad/BiasAddGrad4^gradients/model/conv3/BiasAdd_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients/model/conv3/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:
Ў
0gradients/model_2/conv3/BiasAdd_grad/BiasAddGradBiasAddGrad0gradients/model_2/conv3/conv3/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes	
:
Ѓ
5gradients/model_2/conv3/BiasAdd_grad/tuple/group_depsNoOp1^gradients/model_2/conv3/BiasAdd_grad/BiasAddGrad1^gradients/model_2/conv3/conv3/Relu_grad/ReluGrad
Е
=gradients/model_2/conv3/BiasAdd_grad/tuple/control_dependencyIdentity0gradients/model_2/conv3/conv3/Relu_grad/ReluGrad6^gradients/model_2/conv3/BiasAdd_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/model_2/conv3/conv3/Relu_grad/ReluGrad*2
_output_shapes 
:џџџџџџџџџ
 
?gradients/model_2/conv3/BiasAdd_grad/tuple/control_dependency_1Identity0gradients/model_2/conv3/BiasAdd_grad/BiasAddGrad6^gradients/model_2/conv3/BiasAdd_grad/tuple/group_deps*
_output_shapes	
:*
T0*C
_class9
75loc:@gradients/model_2/conv3/BiasAdd_grad/BiasAddGrad
­
*gradients/model_1/conv3/Conv2D_grad/ShapeNShapeNmodel_1/conv2/MaxPool2D/MaxPoolconv3/weights/read*
N* 
_output_shapes
::*
T0*
out_type0
ї
7gradients/model_1/conv3/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput*gradients/model_1/conv3/Conv2D_grad/ShapeNconv3/weights/read=gradients/model_1/conv3/BiasAdd_grad/tuple/control_dependency*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME*1
_output_shapes
:џџџџџџџџџ@
ў
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
:@
Б
4gradients/model_1/conv3/Conv2D_grad/tuple/group_depsNoOp9^gradients/model_1/conv3/Conv2D_grad/Conv2DBackpropFilter8^gradients/model_1/conv3/Conv2D_grad/Conv2DBackpropInput
Р
<gradients/model_1/conv3/Conv2D_grad/tuple/control_dependencyIdentity7gradients/model_1/conv3/Conv2D_grad/Conv2DBackpropInput5^gradients/model_1/conv3/Conv2D_grad/tuple/group_deps*
T0*J
_class@
><loc:@gradients/model_1/conv3/Conv2D_grad/Conv2DBackpropInput*1
_output_shapes
:џџџџџџџџџ@
К
>gradients/model_1/conv3/Conv2D_grad/tuple/control_dependency_1Identity8gradients/model_1/conv3/Conv2D_grad/Conv2DBackpropFilter5^gradients/model_1/conv3/Conv2D_grad/tuple/group_deps*
T0*K
_classA
?=loc:@gradients/model_1/conv3/Conv2D_grad/Conv2DBackpropFilter*'
_output_shapes
:@
Љ
(gradients/model/conv3/Conv2D_grad/ShapeNShapeNmodel/conv2/MaxPool2D/MaxPoolconv3/weights/read*
T0*
out_type0*
N* 
_output_shapes
::
ё
5gradients/model/conv3/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput(gradients/model/conv3/Conv2D_grad/ShapeNconv3/weights/read;gradients/model/conv3/BiasAdd_grad/tuple/control_dependency*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*1
_output_shapes
:џџџџџџџџџ@
і
6gradients/model/conv3/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFiltermodel/conv2/MaxPool2D/MaxPool*gradients/model/conv3/Conv2D_grad/ShapeN:1;gradients/model/conv3/BiasAdd_grad/tuple/control_dependency*
paddingSAME*'
_output_shapes
:@*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
Ћ
2gradients/model/conv3/Conv2D_grad/tuple/group_depsNoOp7^gradients/model/conv3/Conv2D_grad/Conv2DBackpropFilter6^gradients/model/conv3/Conv2D_grad/Conv2DBackpropInput
И
:gradients/model/conv3/Conv2D_grad/tuple/control_dependencyIdentity5gradients/model/conv3/Conv2D_grad/Conv2DBackpropInput3^gradients/model/conv3/Conv2D_grad/tuple/group_deps*
T0*H
_class>
<:loc:@gradients/model/conv3/Conv2D_grad/Conv2DBackpropInput*1
_output_shapes
:џџџџџџџџџ@
В
<gradients/model/conv3/Conv2D_grad/tuple/control_dependency_1Identity6gradients/model/conv3/Conv2D_grad/Conv2DBackpropFilter3^gradients/model/conv3/Conv2D_grad/tuple/group_deps*
T0*I
_class?
=;loc:@gradients/model/conv3/Conv2D_grad/Conv2DBackpropFilter*'
_output_shapes
:@
­
*gradients/model_2/conv3/Conv2D_grad/ShapeNShapeNmodel_2/conv2/MaxPool2D/MaxPoolconv3/weights/read*
N* 
_output_shapes
::*
T0*
out_type0
ї
7gradients/model_2/conv3/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput*gradients/model_2/conv3/Conv2D_grad/ShapeNconv3/weights/read=gradients/model_2/conv3/BiasAdd_grad/tuple/control_dependency*1
_output_shapes
:џџџџџџџџџ@*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME
ў
8gradients/model_2/conv3/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFiltermodel_2/conv2/MaxPool2D/MaxPool,gradients/model_2/conv3/Conv2D_grad/ShapeN:1=gradients/model_2/conv3/BiasAdd_grad/tuple/control_dependency*'
_output_shapes
:@*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME
Б
4gradients/model_2/conv3/Conv2D_grad/tuple/group_depsNoOp9^gradients/model_2/conv3/Conv2D_grad/Conv2DBackpropFilter8^gradients/model_2/conv3/Conv2D_grad/Conv2DBackpropInput
Р
<gradients/model_2/conv3/Conv2D_grad/tuple/control_dependencyIdentity7gradients/model_2/conv3/Conv2D_grad/Conv2DBackpropInput5^gradients/model_2/conv3/Conv2D_grad/tuple/group_deps*
T0*J
_class@
><loc:@gradients/model_2/conv3/Conv2D_grad/Conv2DBackpropInput*1
_output_shapes
:џџџџџџџџџ@
К
>gradients/model_2/conv3/Conv2D_grad/tuple/control_dependency_1Identity8gradients/model_2/conv3/Conv2D_grad/Conv2DBackpropFilter5^gradients/model_2/conv3/Conv2D_grad/tuple/group_deps*'
_output_shapes
:@*
T0*K
_classA
?=loc:@gradients/model_2/conv3/Conv2D_grad/Conv2DBackpropFilter
Э
gradients/AddN_5AddN?gradients/model_1/conv3/BiasAdd_grad/tuple/control_dependency_1=gradients/model/conv3/BiasAdd_grad/tuple/control_dependency_1?gradients/model_2/conv3/BiasAdd_grad/tuple/control_dependency_1*
T0*C
_class9
75loc:@gradients/model_1/conv3/BiasAdd_grad/BiasAddGrad*
N*
_output_shapes	
:
а
:gradients/model_1/conv2/MaxPool2D/MaxPool_grad/MaxPoolGradMaxPoolGradmodel_1/conv2/conv2/Relumodel_1/conv2/MaxPool2D/MaxPool<gradients/model_1/conv3/Conv2D_grad/tuple/control_dependency*1
_output_shapes
:џџџџџџџџџД@*
T0*
strides
*
data_formatNHWC*
ksize
*
paddingSAME
Ш
8gradients/model/conv2/MaxPool2D/MaxPool_grad/MaxPoolGradMaxPoolGradmodel/conv2/conv2/Relumodel/conv2/MaxPool2D/MaxPool:gradients/model/conv3/Conv2D_grad/tuple/control_dependency*
ksize
*
paddingSAME*1
_output_shapes
:џџџџџџџџџД@*
T0*
data_formatNHWC*
strides

а
:gradients/model_2/conv2/MaxPool2D/MaxPool_grad/MaxPoolGradMaxPoolGradmodel_2/conv2/conv2/Relumodel_2/conv2/MaxPool2D/MaxPool<gradients/model_2/conv3/Conv2D_grad/tuple/control_dependency*
data_formatNHWC*
strides
*
ksize
*
paddingSAME*1
_output_shapes
:џџџџџџџџџД@*
T0
о
gradients/AddN_6AddN>gradients/model_1/conv3/Conv2D_grad/tuple/control_dependency_1<gradients/model/conv3/Conv2D_grad/tuple/control_dependency_1>gradients/model_2/conv3/Conv2D_grad/tuple/control_dependency_1*
T0*K
_classA
?=loc:@gradients/model_1/conv3/Conv2D_grad/Conv2DBackpropFilter*
N*'
_output_shapes
:@
Ю
0gradients/model_1/conv2/conv2/Relu_grad/ReluGradReluGrad:gradients/model_1/conv2/MaxPool2D/MaxPool_grad/MaxPoolGradmodel_1/conv2/conv2/Relu*
T0*1
_output_shapes
:џџџџџџџџџД@
Ш
.gradients/model/conv2/conv2/Relu_grad/ReluGradReluGrad8gradients/model/conv2/MaxPool2D/MaxPool_grad/MaxPoolGradmodel/conv2/conv2/Relu*
T0*1
_output_shapes
:џџџџџџџџџД@
Ю
0gradients/model_2/conv2/conv2/Relu_grad/ReluGradReluGrad:gradients/model_2/conv2/MaxPool2D/MaxPool_grad/MaxPoolGradmodel_2/conv2/conv2/Relu*
T0*1
_output_shapes
:џџџџџџџџџД@
­
0gradients/model_1/conv2/BiasAdd_grad/BiasAddGradBiasAddGrad0gradients/model_1/conv2/conv2/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes
:@
Ѓ
5gradients/model_1/conv2/BiasAdd_grad/tuple/group_depsNoOp1^gradients/model_1/conv2/BiasAdd_grad/BiasAddGrad1^gradients/model_1/conv2/conv2/Relu_grad/ReluGrad
Д
=gradients/model_1/conv2/BiasAdd_grad/tuple/control_dependencyIdentity0gradients/model_1/conv2/conv2/Relu_grad/ReluGrad6^gradients/model_1/conv2/BiasAdd_grad/tuple/group_deps*1
_output_shapes
:џџџџџџџџџД@*
T0*C
_class9
75loc:@gradients/model_1/conv2/conv2/Relu_grad/ReluGrad

?gradients/model_1/conv2/BiasAdd_grad/tuple/control_dependency_1Identity0gradients/model_1/conv2/BiasAdd_grad/BiasAddGrad6^gradients/model_1/conv2/BiasAdd_grad/tuple/group_deps*
_output_shapes
:@*
T0*C
_class9
75loc:@gradients/model_1/conv2/BiasAdd_grad/BiasAddGrad
Љ
.gradients/model/conv2/BiasAdd_grad/BiasAddGradBiasAddGrad.gradients/model/conv2/conv2/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes
:@

3gradients/model/conv2/BiasAdd_grad/tuple/group_depsNoOp/^gradients/model/conv2/BiasAdd_grad/BiasAddGrad/^gradients/model/conv2/conv2/Relu_grad/ReluGrad
Ќ
;gradients/model/conv2/BiasAdd_grad/tuple/control_dependencyIdentity.gradients/model/conv2/conv2/Relu_grad/ReluGrad4^gradients/model/conv2/BiasAdd_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients/model/conv2/conv2/Relu_grad/ReluGrad*1
_output_shapes
:џџџџџџџџџД@

=gradients/model/conv2/BiasAdd_grad/tuple/control_dependency_1Identity.gradients/model/conv2/BiasAdd_grad/BiasAddGrad4^gradients/model/conv2/BiasAdd_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients/model/conv2/BiasAdd_grad/BiasAddGrad*
_output_shapes
:@
­
0gradients/model_2/conv2/BiasAdd_grad/BiasAddGradBiasAddGrad0gradients/model_2/conv2/conv2/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes
:@
Ѓ
5gradients/model_2/conv2/BiasAdd_grad/tuple/group_depsNoOp1^gradients/model_2/conv2/BiasAdd_grad/BiasAddGrad1^gradients/model_2/conv2/conv2/Relu_grad/ReluGrad
Д
=gradients/model_2/conv2/BiasAdd_grad/tuple/control_dependencyIdentity0gradients/model_2/conv2/conv2/Relu_grad/ReluGrad6^gradients/model_2/conv2/BiasAdd_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/model_2/conv2/conv2/Relu_grad/ReluGrad*1
_output_shapes
:џџџџџџџџџД@

?gradients/model_2/conv2/BiasAdd_grad/tuple/control_dependency_1Identity0gradients/model_2/conv2/BiasAdd_grad/BiasAddGrad6^gradients/model_2/conv2/BiasAdd_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/model_2/conv2/BiasAdd_grad/BiasAddGrad*
_output_shapes
:@
­
*gradients/model_1/conv2/Conv2D_grad/ShapeNShapeNmodel_1/conv1/MaxPool2D/MaxPoolconv2/weights/read*
T0*
out_type0*
N* 
_output_shapes
::
ї
7gradients/model_1/conv2/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput*gradients/model_1/conv2/Conv2D_grad/ShapeNconv2/weights/read=gradients/model_1/conv2/BiasAdd_grad/tuple/control_dependency*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*1
_output_shapes
:џџџџџџџџџД 
§
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
Б
4gradients/model_1/conv2/Conv2D_grad/tuple/group_depsNoOp9^gradients/model_1/conv2/Conv2D_grad/Conv2DBackpropFilter8^gradients/model_1/conv2/Conv2D_grad/Conv2DBackpropInput
Р
<gradients/model_1/conv2/Conv2D_grad/tuple/control_dependencyIdentity7gradients/model_1/conv2/Conv2D_grad/Conv2DBackpropInput5^gradients/model_1/conv2/Conv2D_grad/tuple/group_deps*1
_output_shapes
:џџџџџџџџџД *
T0*J
_class@
><loc:@gradients/model_1/conv2/Conv2D_grad/Conv2DBackpropInput
Й
>gradients/model_1/conv2/Conv2D_grad/tuple/control_dependency_1Identity8gradients/model_1/conv2/Conv2D_grad/Conv2DBackpropFilter5^gradients/model_1/conv2/Conv2D_grad/tuple/group_deps*&
_output_shapes
: @*
T0*K
_classA
?=loc:@gradients/model_1/conv2/Conv2D_grad/Conv2DBackpropFilter
Љ
(gradients/model/conv2/Conv2D_grad/ShapeNShapeNmodel/conv1/MaxPool2D/MaxPoolconv2/weights/read*
T0*
out_type0*
N* 
_output_shapes
::
ё
5gradients/model/conv2/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput(gradients/model/conv2/Conv2D_grad/ShapeNconv2/weights/read;gradients/model/conv2/BiasAdd_grad/tuple/control_dependency*1
_output_shapes
:џџџџџџџџџД *
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME
ѕ
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
Ћ
2gradients/model/conv2/Conv2D_grad/tuple/group_depsNoOp7^gradients/model/conv2/Conv2D_grad/Conv2DBackpropFilter6^gradients/model/conv2/Conv2D_grad/Conv2DBackpropInput
И
:gradients/model/conv2/Conv2D_grad/tuple/control_dependencyIdentity5gradients/model/conv2/Conv2D_grad/Conv2DBackpropInput3^gradients/model/conv2/Conv2D_grad/tuple/group_deps*1
_output_shapes
:џџџџџџџџџД *
T0*H
_class>
<:loc:@gradients/model/conv2/Conv2D_grad/Conv2DBackpropInput
Б
<gradients/model/conv2/Conv2D_grad/tuple/control_dependency_1Identity6gradients/model/conv2/Conv2D_grad/Conv2DBackpropFilter3^gradients/model/conv2/Conv2D_grad/tuple/group_deps*
T0*I
_class?
=;loc:@gradients/model/conv2/Conv2D_grad/Conv2DBackpropFilter*&
_output_shapes
: @
­
*gradients/model_2/conv2/Conv2D_grad/ShapeNShapeNmodel_2/conv1/MaxPool2D/MaxPoolconv2/weights/read*
T0*
out_type0*
N* 
_output_shapes
::
ї
7gradients/model_2/conv2/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput*gradients/model_2/conv2/Conv2D_grad/ShapeNconv2/weights/read=gradients/model_2/conv2/BiasAdd_grad/tuple/control_dependency*
paddingSAME*1
_output_shapes
:џџџџџџџџџД *
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
§
8gradients/model_2/conv2/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFiltermodel_2/conv1/MaxPool2D/MaxPool,gradients/model_2/conv2/Conv2D_grad/ShapeN:1=gradients/model_2/conv2/BiasAdd_grad/tuple/control_dependency*&
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
Б
4gradients/model_2/conv2/Conv2D_grad/tuple/group_depsNoOp9^gradients/model_2/conv2/Conv2D_grad/Conv2DBackpropFilter8^gradients/model_2/conv2/Conv2D_grad/Conv2DBackpropInput
Р
<gradients/model_2/conv2/Conv2D_grad/tuple/control_dependencyIdentity7gradients/model_2/conv2/Conv2D_grad/Conv2DBackpropInput5^gradients/model_2/conv2/Conv2D_grad/tuple/group_deps*
T0*J
_class@
><loc:@gradients/model_2/conv2/Conv2D_grad/Conv2DBackpropInput*1
_output_shapes
:џџџџџџџџџД 
Й
>gradients/model_2/conv2/Conv2D_grad/tuple/control_dependency_1Identity8gradients/model_2/conv2/Conv2D_grad/Conv2DBackpropFilter5^gradients/model_2/conv2/Conv2D_grad/tuple/group_deps*&
_output_shapes
: @*
T0*K
_classA
?=loc:@gradients/model_2/conv2/Conv2D_grad/Conv2DBackpropFilter
Ь
gradients/AddN_7AddN?gradients/model_1/conv2/BiasAdd_grad/tuple/control_dependency_1=gradients/model/conv2/BiasAdd_grad/tuple/control_dependency_1?gradients/model_2/conv2/BiasAdd_grad/tuple/control_dependency_1*
T0*C
_class9
75loc:@gradients/model_1/conv2/BiasAdd_grad/BiasAddGrad*
N*
_output_shapes
:@
а
:gradients/model_1/conv1/MaxPool2D/MaxPool_grad/MaxPoolGradMaxPoolGradmodel_1/conv1/conv1/Relumodel_1/conv1/MaxPool2D/MaxPool<gradients/model_1/conv2/Conv2D_grad/tuple/control_dependency*1
_output_shapes
:џџџџџџџџџ ч *
T0*
data_formatNHWC*
strides
*
ksize
*
paddingSAME
Ш
8gradients/model/conv1/MaxPool2D/MaxPool_grad/MaxPoolGradMaxPoolGradmodel/conv1/conv1/Relumodel/conv1/MaxPool2D/MaxPool:gradients/model/conv2/Conv2D_grad/tuple/control_dependency*
T0*
data_formatNHWC*
strides
*
ksize
*
paddingSAME*1
_output_shapes
:џџџџџџџџџ ч 
а
:gradients/model_2/conv1/MaxPool2D/MaxPool_grad/MaxPoolGradMaxPoolGradmodel_2/conv1/conv1/Relumodel_2/conv1/MaxPool2D/MaxPool<gradients/model_2/conv2/Conv2D_grad/tuple/control_dependency*
T0*
data_formatNHWC*
strides
*
ksize
*
paddingSAME*1
_output_shapes
:џџџџџџџџџ ч 
н
gradients/AddN_8AddN>gradients/model_1/conv2/Conv2D_grad/tuple/control_dependency_1<gradients/model/conv2/Conv2D_grad/tuple/control_dependency_1>gradients/model_2/conv2/Conv2D_grad/tuple/control_dependency_1*
T0*K
_classA
?=loc:@gradients/model_1/conv2/Conv2D_grad/Conv2DBackpropFilter*
N*&
_output_shapes
: @
Ю
0gradients/model_1/conv1/conv1/Relu_grad/ReluGradReluGrad:gradients/model_1/conv1/MaxPool2D/MaxPool_grad/MaxPoolGradmodel_1/conv1/conv1/Relu*1
_output_shapes
:џџџџџџџџџ ч *
T0
Ш
.gradients/model/conv1/conv1/Relu_grad/ReluGradReluGrad8gradients/model/conv1/MaxPool2D/MaxPool_grad/MaxPoolGradmodel/conv1/conv1/Relu*
T0*1
_output_shapes
:џџџџџџџџџ ч 
Ю
0gradients/model_2/conv1/conv1/Relu_grad/ReluGradReluGrad:gradients/model_2/conv1/MaxPool2D/MaxPool_grad/MaxPoolGradmodel_2/conv1/conv1/Relu*1
_output_shapes
:џџџџџџџџџ ч *
T0
­
0gradients/model_1/conv1/BiasAdd_grad/BiasAddGradBiasAddGrad0gradients/model_1/conv1/conv1/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes
: 
Ѓ
5gradients/model_1/conv1/BiasAdd_grad/tuple/group_depsNoOp1^gradients/model_1/conv1/BiasAdd_grad/BiasAddGrad1^gradients/model_1/conv1/conv1/Relu_grad/ReluGrad
Д
=gradients/model_1/conv1/BiasAdd_grad/tuple/control_dependencyIdentity0gradients/model_1/conv1/conv1/Relu_grad/ReluGrad6^gradients/model_1/conv1/BiasAdd_grad/tuple/group_deps*1
_output_shapes
:џџџџџџџџџ ч *
T0*C
_class9
75loc:@gradients/model_1/conv1/conv1/Relu_grad/ReluGrad

?gradients/model_1/conv1/BiasAdd_grad/tuple/control_dependency_1Identity0gradients/model_1/conv1/BiasAdd_grad/BiasAddGrad6^gradients/model_1/conv1/BiasAdd_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/model_1/conv1/BiasAdd_grad/BiasAddGrad*
_output_shapes
: 
Љ
.gradients/model/conv1/BiasAdd_grad/BiasAddGradBiasAddGrad.gradients/model/conv1/conv1/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes
: 

3gradients/model/conv1/BiasAdd_grad/tuple/group_depsNoOp/^gradients/model/conv1/BiasAdd_grad/BiasAddGrad/^gradients/model/conv1/conv1/Relu_grad/ReluGrad
Ќ
;gradients/model/conv1/BiasAdd_grad/tuple/control_dependencyIdentity.gradients/model/conv1/conv1/Relu_grad/ReluGrad4^gradients/model/conv1/BiasAdd_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients/model/conv1/conv1/Relu_grad/ReluGrad*1
_output_shapes
:џџџџџџџџџ ч 

=gradients/model/conv1/BiasAdd_grad/tuple/control_dependency_1Identity.gradients/model/conv1/BiasAdd_grad/BiasAddGrad4^gradients/model/conv1/BiasAdd_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients/model/conv1/BiasAdd_grad/BiasAddGrad*
_output_shapes
: 
­
0gradients/model_2/conv1/BiasAdd_grad/BiasAddGradBiasAddGrad0gradients/model_2/conv1/conv1/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes
: 
Ѓ
5gradients/model_2/conv1/BiasAdd_grad/tuple/group_depsNoOp1^gradients/model_2/conv1/BiasAdd_grad/BiasAddGrad1^gradients/model_2/conv1/conv1/Relu_grad/ReluGrad
Д
=gradients/model_2/conv1/BiasAdd_grad/tuple/control_dependencyIdentity0gradients/model_2/conv1/conv1/Relu_grad/ReluGrad6^gradients/model_2/conv1/BiasAdd_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/model_2/conv1/conv1/Relu_grad/ReluGrad*1
_output_shapes
:џџџџџџџџџ ч 

?gradients/model_2/conv1/BiasAdd_grad/tuple/control_dependency_1Identity0gradients/model_2/conv1/BiasAdd_grad/BiasAddGrad6^gradients/model_2/conv1/BiasAdd_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/model_2/conv1/BiasAdd_grad/BiasAddGrad*
_output_shapes
: 

*gradients/model_1/conv1/Conv2D_grad/ShapeNShapeNpositive_inputconv1/weights/read*
T0*
out_type0*
N* 
_output_shapes
::
ї
7gradients/model_1/conv1/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput*gradients/model_1/conv1/Conv2D_grad/ShapeNconv1/weights/read=gradients/model_1/conv1/BiasAdd_grad/tuple/control_dependency*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME*1
_output_shapes
:џџџџџџџџџ ч
ь
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
Б
4gradients/model_1/conv1/Conv2D_grad/tuple/group_depsNoOp9^gradients/model_1/conv1/Conv2D_grad/Conv2DBackpropFilter8^gradients/model_1/conv1/Conv2D_grad/Conv2DBackpropInput
Р
<gradients/model_1/conv1/Conv2D_grad/tuple/control_dependencyIdentity7gradients/model_1/conv1/Conv2D_grad/Conv2DBackpropInput5^gradients/model_1/conv1/Conv2D_grad/tuple/group_deps*
T0*J
_class@
><loc:@gradients/model_1/conv1/Conv2D_grad/Conv2DBackpropInput*1
_output_shapes
:џџџџџџџџџ ч
Й
>gradients/model_1/conv1/Conv2D_grad/tuple/control_dependency_1Identity8gradients/model_1/conv1/Conv2D_grad/Conv2DBackpropFilter5^gradients/model_1/conv1/Conv2D_grad/tuple/group_deps*
T0*K
_classA
?=loc:@gradients/model_1/conv1/Conv2D_grad/Conv2DBackpropFilter*&
_output_shapes
: 

(gradients/model/conv1/Conv2D_grad/ShapeNShapeNanchor_inputconv1/weights/read*
T0*
out_type0*
N* 
_output_shapes
::
ё
5gradients/model/conv1/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput(gradients/model/conv1/Conv2D_grad/ShapeNconv1/weights/read;gradients/model/conv1/BiasAdd_grad/tuple/control_dependency*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*1
_output_shapes
:џџџџџџџџџ ч
ф
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
Ћ
2gradients/model/conv1/Conv2D_grad/tuple/group_depsNoOp7^gradients/model/conv1/Conv2D_grad/Conv2DBackpropFilter6^gradients/model/conv1/Conv2D_grad/Conv2DBackpropInput
И
:gradients/model/conv1/Conv2D_grad/tuple/control_dependencyIdentity5gradients/model/conv1/Conv2D_grad/Conv2DBackpropInput3^gradients/model/conv1/Conv2D_grad/tuple/group_deps*1
_output_shapes
:џџџџџџџџџ ч*
T0*H
_class>
<:loc:@gradients/model/conv1/Conv2D_grad/Conv2DBackpropInput
Б
<gradients/model/conv1/Conv2D_grad/tuple/control_dependency_1Identity6gradients/model/conv1/Conv2D_grad/Conv2DBackpropFilter3^gradients/model/conv1/Conv2D_grad/tuple/group_deps*&
_output_shapes
: *
T0*I
_class?
=;loc:@gradients/model/conv1/Conv2D_grad/Conv2DBackpropFilter

*gradients/model_2/conv1/Conv2D_grad/ShapeNShapeNnegative_inputconv1/weights/read*
T0*
out_type0*
N* 
_output_shapes
::
ї
7gradients/model_2/conv1/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput*gradients/model_2/conv1/Conv2D_grad/ShapeNconv1/weights/read=gradients/model_2/conv1/BiasAdd_grad/tuple/control_dependency*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*1
_output_shapes
:џџџџџџџџџ ч*
	dilations

ь
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
Б
4gradients/model_2/conv1/Conv2D_grad/tuple/group_depsNoOp9^gradients/model_2/conv1/Conv2D_grad/Conv2DBackpropFilter8^gradients/model_2/conv1/Conv2D_grad/Conv2DBackpropInput
Р
<gradients/model_2/conv1/Conv2D_grad/tuple/control_dependencyIdentity7gradients/model_2/conv1/Conv2D_grad/Conv2DBackpropInput5^gradients/model_2/conv1/Conv2D_grad/tuple/group_deps*1
_output_shapes
:џџџџџџџџџ ч*
T0*J
_class@
><loc:@gradients/model_2/conv1/Conv2D_grad/Conv2DBackpropInput
Й
>gradients/model_2/conv1/Conv2D_grad/tuple/control_dependency_1Identity8gradients/model_2/conv1/Conv2D_grad/Conv2DBackpropFilter5^gradients/model_2/conv1/Conv2D_grad/tuple/group_deps*
T0*K
_classA
?=loc:@gradients/model_2/conv1/Conv2D_grad/Conv2DBackpropFilter*&
_output_shapes
: 
Ь
gradients/AddN_9AddN?gradients/model_1/conv1/BiasAdd_grad/tuple/control_dependency_1=gradients/model/conv1/BiasAdd_grad/tuple/control_dependency_1?gradients/model_2/conv1/BiasAdd_grad/tuple/control_dependency_1*
T0*C
_class9
75loc:@gradients/model_1/conv1/BiasAdd_grad/BiasAddGrad*
N*
_output_shapes
: 
о
gradients/AddN_10AddN>gradients/model_1/conv1/Conv2D_grad/tuple/control_dependency_1<gradients/model/conv1/Conv2D_grad/tuple/control_dependency_1>gradients/model_2/conv1/Conv2D_grad/tuple/control_dependency_1*
N*&
_output_shapes
: *
T0*K
_classA
?=loc:@gradients/model_1/conv1/Conv2D_grad/Conv2DBackpropFilter
Г
8conv1/weights/Momentum/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*%
valueB"             * 
_class
loc:@conv1/weights

.conv1/weights/Momentum/Initializer/zeros/ConstConst*
valueB
 *    * 
_class
loc:@conv1/weights*
dtype0*
_output_shapes
: 
џ
(conv1/weights/Momentum/Initializer/zerosFill8conv1/weights/Momentum/Initializer/zeros/shape_as_tensor.conv1/weights/Momentum/Initializer/zeros/Const*
T0*

index_type0* 
_class
loc:@conv1/weights*&
_output_shapes
: 
М
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
х
conv1/weights/Momentum/AssignAssignconv1/weights/Momentum(conv1/weights/Momentum/Initializer/zeros*
T0* 
_class
loc:@conv1/weights*
validate_shape(*&
_output_shapes
: *
use_locking(

conv1/weights/Momentum/readIdentityconv1/weights/Momentum*&
_output_shapes
: *
T0* 
_class
loc:@conv1/weights

'conv1/biases/Momentum/Initializer/zerosConst*
dtype0*
_output_shapes
: *
valueB *    *
_class
loc:@conv1/biases
Ђ
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
е
conv1/biases/Momentum/AssignAssignconv1/biases/Momentum'conv1/biases/Momentum/Initializer/zeros*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*
_class
loc:@conv1/biases

conv1/biases/Momentum/readIdentityconv1/biases/Momentum*
T0*
_class
loc:@conv1/biases*
_output_shapes
: 
Г
8conv2/weights/Momentum/Initializer/zeros/shape_as_tensorConst*%
valueB"          @   * 
_class
loc:@conv2/weights*
dtype0*
_output_shapes
:

.conv2/weights/Momentum/Initializer/zeros/ConstConst*
valueB
 *    * 
_class
loc:@conv2/weights*
dtype0*
_output_shapes
: 
џ
(conv2/weights/Momentum/Initializer/zerosFill8conv2/weights/Momentum/Initializer/zeros/shape_as_tensor.conv2/weights/Momentum/Initializer/zeros/Const*
T0*

index_type0* 
_class
loc:@conv2/weights*&
_output_shapes
: @
М
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
х
conv2/weights/Momentum/AssignAssignconv2/weights/Momentum(conv2/weights/Momentum/Initializer/zeros*
use_locking(*
T0* 
_class
loc:@conv2/weights*
validate_shape(*&
_output_shapes
: @

conv2/weights/Momentum/readIdentityconv2/weights/Momentum*
T0* 
_class
loc:@conv2/weights*&
_output_shapes
: @

'conv2/biases/Momentum/Initializer/zerosConst*
dtype0*
_output_shapes
:@*
valueB@*    *
_class
loc:@conv2/biases
Ђ
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
е
conv2/biases/Momentum/AssignAssignconv2/biases/Momentum'conv2/biases/Momentum/Initializer/zeros*
T0*
_class
loc:@conv2/biases*
validate_shape(*
_output_shapes
:@*
use_locking(

conv2/biases/Momentum/readIdentityconv2/biases/Momentum*
T0*
_class
loc:@conv2/biases*
_output_shapes
:@
Г
8conv3/weights/Momentum/Initializer/zeros/shape_as_tensorConst*%
valueB"      @      * 
_class
loc:@conv3/weights*
dtype0*
_output_shapes
:

.conv3/weights/Momentum/Initializer/zeros/ConstConst*
valueB
 *    * 
_class
loc:@conv3/weights*
dtype0*
_output_shapes
: 

(conv3/weights/Momentum/Initializer/zerosFill8conv3/weights/Momentum/Initializer/zeros/shape_as_tensor.conv3/weights/Momentum/Initializer/zeros/Const*'
_output_shapes
:@*
T0*

index_type0* 
_class
loc:@conv3/weights
О
conv3/weights/Momentum
VariableV2*
shared_name * 
_class
loc:@conv3/weights*
	container *
shape:@*
dtype0*'
_output_shapes
:@
ц
conv3/weights/Momentum/AssignAssignconv3/weights/Momentum(conv3/weights/Momentum/Initializer/zeros*
T0* 
_class
loc:@conv3/weights*
validate_shape(*'
_output_shapes
:@*
use_locking(

conv3/weights/Momentum/readIdentityconv3/weights/Momentum*
T0* 
_class
loc:@conv3/weights*'
_output_shapes
:@

'conv3/biases/Momentum/Initializer/zerosConst*
valueB*    *
_class
loc:@conv3/biases*
dtype0*
_output_shapes	
:
Є
conv3/biases/Momentum
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *
_class
loc:@conv3/biases*
	container *
shape:
ж
conv3/biases/Momentum/AssignAssignconv3/biases/Momentum'conv3/biases/Momentum/Initializer/zeros*
use_locking(*
T0*
_class
loc:@conv3/biases*
validate_shape(*
_output_shapes	
:

conv3/biases/Momentum/readIdentityconv3/biases/Momentum*
T0*
_class
loc:@conv3/biases*
_output_shapes	
:
Г
8conv4/weights/Momentum/Initializer/zeros/shape_as_tensorConst*%
valueB"            * 
_class
loc:@conv4/weights*
dtype0*
_output_shapes
:

.conv4/weights/Momentum/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    * 
_class
loc:@conv4/weights

(conv4/weights/Momentum/Initializer/zerosFill8conv4/weights/Momentum/Initializer/zeros/shape_as_tensor.conv4/weights/Momentum/Initializer/zeros/Const*(
_output_shapes
:*
T0*

index_type0* 
_class
loc:@conv4/weights
Р
conv4/weights/Momentum
VariableV2*
dtype0*(
_output_shapes
:*
shared_name * 
_class
loc:@conv4/weights*
	container *
shape:
ч
conv4/weights/Momentum/AssignAssignconv4/weights/Momentum(conv4/weights/Momentum/Initializer/zeros*
validate_shape(*(
_output_shapes
:*
use_locking(*
T0* 
_class
loc:@conv4/weights

conv4/weights/Momentum/readIdentityconv4/weights/Momentum*
T0* 
_class
loc:@conv4/weights*(
_output_shapes
:

'conv4/biases/Momentum/Initializer/zerosConst*
valueB*    *
_class
loc:@conv4/biases*
dtype0*
_output_shapes	
:
Є
conv4/biases/Momentum
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *
_class
loc:@conv4/biases*
	container *
shape:
ж
conv4/biases/Momentum/AssignAssignconv4/biases/Momentum'conv4/biases/Momentum/Initializer/zeros*
use_locking(*
T0*
_class
loc:@conv4/biases*
validate_shape(*
_output_shapes	
:

conv4/biases/Momentum/readIdentityconv4/biases/Momentum*
T0*
_class
loc:@conv4/biases*
_output_shapes	
:
Г
8conv5/weights/Momentum/Initializer/zeros/shape_as_tensorConst*%
valueB"            * 
_class
loc:@conv5/weights*
dtype0*
_output_shapes
:

.conv5/weights/Momentum/Initializer/zeros/ConstConst*
valueB
 *    * 
_class
loc:@conv5/weights*
dtype0*
_output_shapes
: 

(conv5/weights/Momentum/Initializer/zerosFill8conv5/weights/Momentum/Initializer/zeros/shape_as_tensor.conv5/weights/Momentum/Initializer/zeros/Const*
T0*

index_type0* 
_class
loc:@conv5/weights*'
_output_shapes
:
О
conv5/weights/Momentum
VariableV2*
shape:*
dtype0*'
_output_shapes
:*
shared_name * 
_class
loc:@conv5/weights*
	container 
ц
conv5/weights/Momentum/AssignAssignconv5/weights/Momentum(conv5/weights/Momentum/Initializer/zeros*
validate_shape(*'
_output_shapes
:*
use_locking(*
T0* 
_class
loc:@conv5/weights

conv5/weights/Momentum/readIdentityconv5/weights/Momentum*
T0* 
_class
loc:@conv5/weights*'
_output_shapes
:

'conv5/biases/Momentum/Initializer/zerosConst*
valueB*    *
_class
loc:@conv5/biases*
dtype0*
_output_shapes
:
Ђ
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
е
conv5/biases/Momentum/AssignAssignconv5/biases/Momentum'conv5/biases/Momentum/Initializer/zeros*
use_locking(*
T0*
_class
loc:@conv5/biases*
validate_shape(*
_output_shapes
:

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
з#<*
dtype0*
_output_shapes
: 
V
Momentum/momentumConst*
valueB
 *Єp}?*
dtype0*
_output_shapes
: 

+Momentum/update_conv1/weights/ApplyMomentumApplyMomentumconv1/weightsconv1/weights/MomentumMomentum/learning_rategradients/AddN_10Momentum/momentum*
use_locking( *
T0* 
_class
loc:@conv1/weights*
use_nesterov(*&
_output_shapes
: 

*Momentum/update_conv1/biases/ApplyMomentumApplyMomentumconv1/biasesconv1/biases/MomentumMomentum/learning_rategradients/AddN_9Momentum/momentum*
use_locking( *
T0*
_class
loc:@conv1/biases*
use_nesterov(*
_output_shapes
: 

+Momentum/update_conv2/weights/ApplyMomentumApplyMomentumconv2/weightsconv2/weights/MomentumMomentum/learning_rategradients/AddN_8Momentum/momentum*
use_nesterov(*&
_output_shapes
: @*
use_locking( *
T0* 
_class
loc:@conv2/weights

*Momentum/update_conv2/biases/ApplyMomentumApplyMomentumconv2/biasesconv2/biases/MomentumMomentum/learning_rategradients/AddN_7Momentum/momentum*
use_locking( *
T0*
_class
loc:@conv2/biases*
use_nesterov(*
_output_shapes
:@

+Momentum/update_conv3/weights/ApplyMomentumApplyMomentumconv3/weightsconv3/weights/MomentumMomentum/learning_rategradients/AddN_6Momentum/momentum*
use_locking( *
T0* 
_class
loc:@conv3/weights*
use_nesterov(*'
_output_shapes
:@

*Momentum/update_conv3/biases/ApplyMomentumApplyMomentumconv3/biasesconv3/biases/MomentumMomentum/learning_rategradients/AddN_5Momentum/momentum*
T0*
_class
loc:@conv3/biases*
use_nesterov(*
_output_shapes	
:*
use_locking( 

+Momentum/update_conv4/weights/ApplyMomentumApplyMomentumconv4/weightsconv4/weights/MomentumMomentum/learning_rategradients/AddN_4Momentum/momentum*
use_locking( *
T0* 
_class
loc:@conv4/weights*
use_nesterov(*(
_output_shapes
:

*Momentum/update_conv4/biases/ApplyMomentumApplyMomentumconv4/biasesconv4/biases/MomentumMomentum/learning_rategradients/AddN_3Momentum/momentum*
use_locking( *
T0*
_class
loc:@conv4/biases*
use_nesterov(*
_output_shapes	
:

+Momentum/update_conv5/weights/ApplyMomentumApplyMomentumconv5/weightsconv5/weights/MomentumMomentum/learning_rategradients/AddN_2Momentum/momentum*
use_locking( *
T0* 
_class
loc:@conv5/weights*
use_nesterov(*'
_output_shapes
:

*Momentum/update_conv5/biases/ApplyMomentumApplyMomentumconv5/biasesconv5/biases/MomentumMomentum/learning_rategradients/AddN_1Momentum/momentum*
T0*
_class
loc:@conv5/biases*
use_nesterov(*
_output_shapes
:*
use_locking( 
о
Momentum/updateNoOp+^Momentum/update_conv1/biases/ApplyMomentum,^Momentum/update_conv1/weights/ApplyMomentum+^Momentum/update_conv2/biases/ApplyMomentum,^Momentum/update_conv2/weights/ApplyMomentum+^Momentum/update_conv3/biases/ApplyMomentum,^Momentum/update_conv3/weights/ApplyMomentum+^Momentum/update_conv4/biases/ApplyMomentum,^Momentum/update_conv4/weights/ApplyMomentum+^Momentum/update_conv5/biases/ApplyMomentum,^Momentum/update_conv5/weights/ApplyMomentum

Momentum/valueConst^Momentum/update*
_class
loc:@Variable*
value	B :*
dtype0*
_output_shapes
: 

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
ш
save/SaveV2/tensor_namesConst*
valueBBVariableBconv1/biasesBconv1/biases/MomentumBconv1/weightsBconv1/weights/MomentumBconv2/biasesBconv2/biases/MomentumBconv2/weightsBconv2/weights/MomentumBconv3/biasesBconv3/biases/MomentumBconv3/weightsBconv3/weights/MomentumBconv4/biasesBconv4/biases/MomentumBconv4/weightsBconv4/weights/MomentumBconv5/biasesBconv5/biases/MomentumBconv5/weightsBconv5/weights/Momentum*
dtype0*
_output_shapes
:

save/SaveV2/shape_and_slicesConst*
dtype0*
_output_shapes
:*=
value4B2B B B B B B B B B B B B B B B B B B B B B 

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
њ
save/RestoreV2/tensor_namesConst"/device:CPU:0*
valueBBVariableBconv1/biasesBconv1/biases/MomentumBconv1/weightsBconv1/weights/MomentumBconv2/biasesBconv2/biases/MomentumBconv2/weightsBconv2/weights/MomentumBconv3/biasesBconv3/biases/MomentumBconv3/weightsBconv3/weights/MomentumBconv4/biasesBconv4/biases/MomentumBconv4/weightsBconv4/weights/MomentumBconv5/biasesBconv5/biases/MomentumBconv5/weightsBconv5/weights/Momentum*
dtype0*
_output_shapes
:

save/RestoreV2/shape_and_slicesConst"/device:CPU:0*=
value4B2B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:

save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices"/device:CPU:0*h
_output_shapesV
T:::::::::::::::::::::*#
dtypes
2

save/AssignAssignVariablesave/RestoreV2*
T0*
_class
loc:@Variable*
validate_shape(*
_output_shapes
: *
use_locking(
І
save/Assign_1Assignconv1/biasessave/RestoreV2:1*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*
_class
loc:@conv1/biases
Џ
save/Assign_2Assignconv1/biases/Momentumsave/RestoreV2:2*
use_locking(*
T0*
_class
loc:@conv1/biases*
validate_shape(*
_output_shapes
: 
Д
save/Assign_3Assignconv1/weightssave/RestoreV2:3*
T0* 
_class
loc:@conv1/weights*
validate_shape(*&
_output_shapes
: *
use_locking(
Н
save/Assign_4Assignconv1/weights/Momentumsave/RestoreV2:4*
T0* 
_class
loc:@conv1/weights*
validate_shape(*&
_output_shapes
: *
use_locking(
І
save/Assign_5Assignconv2/biasessave/RestoreV2:5*
T0*
_class
loc:@conv2/biases*
validate_shape(*
_output_shapes
:@*
use_locking(
Џ
save/Assign_6Assignconv2/biases/Momentumsave/RestoreV2:6*
use_locking(*
T0*
_class
loc:@conv2/biases*
validate_shape(*
_output_shapes
:@
Д
save/Assign_7Assignconv2/weightssave/RestoreV2:7*
use_locking(*
T0* 
_class
loc:@conv2/weights*
validate_shape(*&
_output_shapes
: @
Н
save/Assign_8Assignconv2/weights/Momentumsave/RestoreV2:8*
use_locking(*
T0* 
_class
loc:@conv2/weights*
validate_shape(*&
_output_shapes
: @
Ї
save/Assign_9Assignconv3/biasessave/RestoreV2:9*
use_locking(*
T0*
_class
loc:@conv3/biases*
validate_shape(*
_output_shapes	
:
В
save/Assign_10Assignconv3/biases/Momentumsave/RestoreV2:10*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*
_class
loc:@conv3/biases
З
save/Assign_11Assignconv3/weightssave/RestoreV2:11*
use_locking(*
T0* 
_class
loc:@conv3/weights*
validate_shape(*'
_output_shapes
:@
Р
save/Assign_12Assignconv3/weights/Momentumsave/RestoreV2:12*
T0* 
_class
loc:@conv3/weights*
validate_shape(*'
_output_shapes
:@*
use_locking(
Љ
save/Assign_13Assignconv4/biasessave/RestoreV2:13*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*
_class
loc:@conv4/biases
В
save/Assign_14Assignconv4/biases/Momentumsave/RestoreV2:14*
use_locking(*
T0*
_class
loc:@conv4/biases*
validate_shape(*
_output_shapes	
:
И
save/Assign_15Assignconv4/weightssave/RestoreV2:15*
use_locking(*
T0* 
_class
loc:@conv4/weights*
validate_shape(*(
_output_shapes
:
С
save/Assign_16Assignconv4/weights/Momentumsave/RestoreV2:16*
use_locking(*
T0* 
_class
loc:@conv4/weights*
validate_shape(*(
_output_shapes
:
Ј
save/Assign_17Assignconv5/biasessave/RestoreV2:17*
use_locking(*
T0*
_class
loc:@conv5/biases*
validate_shape(*
_output_shapes
:
Б
save/Assign_18Assignconv5/biases/Momentumsave/RestoreV2:18*
use_locking(*
T0*
_class
loc:@conv5/biases*
validate_shape(*
_output_shapes
:
З
save/Assign_19Assignconv5/weightssave/RestoreV2:19*
use_locking(*
T0* 
_class
loc:@conv5/weights*
validate_shape(*'
_output_shapes
:
Р
save/Assign_20Assignconv5/weights/Momentumsave/RestoreV2:20*
use_locking(*
T0* 
_class
loc:@conv5/weights*
validate_shape(*'
_output_shapes
:
ё
save/restore_allNoOp^save/Assign^save/Assign_1^save/Assign_10^save/Assign_11^save/Assign_12^save/Assign_13^save/Assign_14^save/Assign_15^save/Assign_16^save/Assign_17^save/Assign_18^save/Assign_19^save/Assign_2^save/Assign_20^save/Assign_3^save/Assign_4^save/Assign_5^save/Assign_6^save/Assign_7^save/Assign_8^save/Assign_9
К
initNoOp^Variable/Assign^conv1/biases/Assign^conv1/biases/Momentum/Assign^conv1/weights/Assign^conv1/weights/Momentum/Assign^conv2/biases/Assign^conv2/biases/Momentum/Assign^conv2/weights/Assign^conv2/weights/Momentum/Assign^conv3/biases/Assign^conv3/biases/Momentum/Assign^conv3/weights/Assign^conv3/weights/Momentum/Assign^conv4/biases/Assign^conv4/biases/Momentum/Assign^conv4/weights/Assign^conv4/weights/Momentum/Assign^conv5/biases/Assign^conv5/biases/Momentum/Assign^conv5/weights/Assign^conv5/weights/Momentum/Assign
N
	step/tagsConst*
dtype0*
_output_shapes
: *
valueB
 Bstep
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
dtype0*
_output_shapes
: *
valueB Bconv5/biases_1
j
conv5/biases_1HistogramSummaryconv5/biases_1/tagconv5/biases/read*
T0*
_output_shapes
: 
є
Merge/MergeSummaryMergeSummarysteplossconv1/weights_1conv1/biases_1conv2/weights_1conv2/biases_1conv3/weights_1conv3/biases_1conv4/weights_1conv4/biases_1conv5/weights_1conv5/biases_1*
N*
_output_shapes
: ""
	variables§
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

conv1/weights/Momentum:0conv1/weights/Momentum/Assignconv1/weights/Momentum/read:02*conv1/weights/Momentum/Initializer/zeros:0

conv1/biases/Momentum:0conv1/biases/Momentum/Assignconv1/biases/Momentum/read:02)conv1/biases/Momentum/Initializer/zeros:0

conv2/weights/Momentum:0conv2/weights/Momentum/Assignconv2/weights/Momentum/read:02*conv2/weights/Momentum/Initializer/zeros:0

conv2/biases/Momentum:0conv2/biases/Momentum/Assignconv2/biases/Momentum/read:02)conv2/biases/Momentum/Initializer/zeros:0

conv3/weights/Momentum:0conv3/weights/Momentum/Assignconv3/weights/Momentum/read:02*conv3/weights/Momentum/Initializer/zeros:0

conv3/biases/Momentum:0conv3/biases/Momentum/Assignconv3/biases/Momentum/read:02)conv3/biases/Momentum/Initializer/zeros:0

conv4/weights/Momentum:0conv4/weights/Momentum/Assignconv4/weights/Momentum/read:02*conv4/weights/Momentum/Initializer/zeros:0

conv4/biases/Momentum:0conv4/biases/Momentum/Assignconv4/biases/Momentum/read:02)conv4/biases/Momentum/Initializer/zeros:0

conv5/weights/Momentum:0conv5/weights/Momentum/Assignconv5/weights/Momentum/read:02*con