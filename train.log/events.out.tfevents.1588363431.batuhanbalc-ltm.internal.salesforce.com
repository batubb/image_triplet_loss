       гK"	  └) л╫Abrain.Event:2г∙жмА     ^[╙	Y┐╠) л╫A"єМ
Б
anchor_inputPlaceholder*
dtype0*0
_output_shapes
:         2Ц*%
shape:         2Ц
Г
positive_inputPlaceholder*
dtype0*0
_output_shapes
:         2Ц*%
shape:         2Ц
Г
negative_inputPlaceholder*%
shape:         2Ц*
dtype0*0
_output_shapes
:         2Ц
й
.conv1/weights/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*%
valueB"             * 
_class
loc:@conv1/weights
У
,conv1/weights/Initializer/random_uniform/minConst*
valueB
 *мEr╜* 
_class
loc:@conv1/weights*
dtype0*
_output_shapes
: 
У
,conv1/weights/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *
valueB
 *мEr=* 
_class
loc:@conv1/weights
Ё
6conv1/weights/Initializer/random_uniform/RandomUniformRandomUniform.conv1/weights/Initializer/random_uniform/shape*
dtype0*&
_output_shapes
: *

seed *
T0* 
_class
loc:@conv1/weights*
seed2 
╥
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
▐
(conv1/weights/Initializer/random_uniformAdd,conv1/weights/Initializer/random_uniform/mul,conv1/weights/Initializer/random_uniform/min*
T0* 
_class
loc:@conv1/weights*&
_output_shapes
: 
│
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
╙
conv1/weights/AssignAssignconv1/weights(conv1/weights/Initializer/random_uniform*
use_locking(*
T0* 
_class
loc:@conv1/weights*
validate_shape(*&
_output_shapes
: 
А
conv1/weights/readIdentityconv1/weights*
T0* 
_class
loc:@conv1/weights*&
_output_shapes
: 
М
conv1/biases/Initializer/zerosConst*
valueB *    *
_class
loc:@conv1/biases*
dtype0*
_output_shapes
: 
Щ
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
║
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
ч
model/conv1/Conv2DConv2Danchor_inputconv1/weights/read*
paddingSAME*0
_output_shapes
:         2Ц *
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
Ч
model/conv1/BiasAddBiasAddmodel/conv1/Conv2Dconv1/biases/read*
T0*
data_formatNHWC*0
_output_shapes
:         2Ц 
n
model/conv1/conv1/ReluRelumodel/conv1/BiasAdd*
T0*0
_output_shapes
:         2Ц 
╠
model/conv1/MaxPool2D/MaxPoolMaxPoolmodel/conv1/conv1/Relu*
T0*
data_formatNHWC*
strides
*
ksize
*
paddingSAME*/
_output_shapes
:         K 
й
.conv2/weights/Initializer/random_uniform/shapeConst*%
valueB"          @   * 
_class
loc:@conv2/weights*
dtype0*
_output_shapes
:
У
,conv2/weights/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *
valueB
 *═╠L╜* 
_class
loc:@conv2/weights
У
,conv2/weights/Initializer/random_uniform/maxConst*
valueB
 *═╠L=* 
_class
loc:@conv2/weights*
dtype0*
_output_shapes
: 
Ё
6conv2/weights/Initializer/random_uniform/RandomUniformRandomUniform.conv2/weights/Initializer/random_uniform/shape*
dtype0*&
_output_shapes
: @*

seed *
T0* 
_class
loc:@conv2/weights*
seed2 
╥
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
▐
(conv2/weights/Initializer/random_uniformAdd,conv2/weights/Initializer/random_uniform/mul,conv2/weights/Initializer/random_uniform/min*&
_output_shapes
: @*
T0* 
_class
loc:@conv2/weights
│
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
╙
conv2/weights/AssignAssignconv2/weights(conv2/weights/Initializer/random_uniform*
T0* 
_class
loc:@conv2/weights*
validate_shape(*&
_output_shapes
: @*
use_locking(
А
conv2/weights/readIdentityconv2/weights*
T0* 
_class
loc:@conv2/weights*&
_output_shapes
: @
М
conv2/biases/Initializer/zerosConst*
valueB@*    *
_class
loc:@conv2/biases*
dtype0*
_output_shapes
:@
Щ
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
║
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
ў
model/conv2/Conv2DConv2Dmodel/conv1/MaxPool2D/MaxPoolconv2/weights/read*/
_output_shapes
:         K@*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME
Ц
model/conv2/BiasAddBiasAddmodel/conv2/Conv2Dconv2/biases/read*
T0*
data_formatNHWC*/
_output_shapes
:         K@
m
model/conv2/conv2/ReluRelumodel/conv2/BiasAdd*
T0*/
_output_shapes
:         K@
╠
model/conv2/MaxPool2D/MaxPoolMaxPoolmodel/conv2/conv2/Relu*/
_output_shapes
:         &@*
T0*
data_formatNHWC*
strides
*
ksize
*
paddingSAME
й
.conv3/weights/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*%
valueB"      @   А   * 
_class
loc:@conv3/weights
У
,conv3/weights/Initializer/random_uniform/minConst*
valueB
 *я[q╜* 
_class
loc:@conv3/weights*
dtype0*
_output_shapes
: 
У
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
:@А*

seed 
╥
,conv3/weights/Initializer/random_uniform/subSub,conv3/weights/Initializer/random_uniform/max,conv3/weights/Initializer/random_uniform/min*
_output_shapes
: *
T0* 
_class
loc:@conv3/weights
э
,conv3/weights/Initializer/random_uniform/mulMul6conv3/weights/Initializer/random_uniform/RandomUniform,conv3/weights/Initializer/random_uniform/sub*'
_output_shapes
:@А*
T0* 
_class
loc:@conv3/weights
▀
(conv3/weights/Initializer/random_uniformAdd,conv3/weights/Initializer/random_uniform/mul,conv3/weights/Initializer/random_uniform/min*
T0* 
_class
loc:@conv3/weights*'
_output_shapes
:@А
╡
conv3/weights
VariableV2*
dtype0*'
_output_shapes
:@А*
shared_name * 
_class
loc:@conv3/weights*
	container *
shape:@А
╘
conv3/weights/AssignAssignconv3/weights(conv3/weights/Initializer/random_uniform*
T0* 
_class
loc:@conv3/weights*
validate_shape(*'
_output_shapes
:@А*
use_locking(
Б
conv3/weights/readIdentityconv3/weights*
T0* 
_class
loc:@conv3/weights*'
_output_shapes
:@А
О
conv3/biases/Initializer/zerosConst*
valueBА*    *
_class
loc:@conv3/biases*
dtype0*
_output_shapes	
:А
Ы
conv3/biases
VariableV2*
dtype0*
_output_shapes	
:А*
shared_name *
_class
loc:@conv3/biases*
	container *
shape:А
╗
conv3/biases/AssignAssignconv3/biasesconv3/biases/Initializer/zeros*
T0*
_class
loc:@conv3/biases*
validate_shape(*
_output_shapes	
:А*
use_locking(
r
conv3/biases/readIdentityconv3/biases*
_output_shapes	
:А*
T0*
_class
loc:@conv3/biases
j
model/conv3/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
°
model/conv3/Conv2DConv2Dmodel/conv2/MaxPool2D/MaxPoolconv3/weights/read*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*0
_output_shapes
:         &А*
	dilations

Ч
model/conv3/BiasAddBiasAddmodel/conv3/Conv2Dconv3/biases/read*
T0*
data_formatNHWC*0
_output_shapes
:         &А
n
model/conv3/conv3/ReluRelumodel/conv3/BiasAdd*
T0*0
_output_shapes
:         &А
═
model/conv3/MaxPool2D/MaxPoolMaxPoolmodel/conv3/conv3/Relu*
T0*
data_formatNHWC*
strides
*
ksize
*
paddingSAME*0
_output_shapes
:         А
й
.conv4/weights/Initializer/random_uniform/shapeConst*%
valueB"      А      * 
_class
loc:@conv4/weights*
dtype0*
_output_shapes
:
У
,conv4/weights/Initializer/random_uniform/minConst*
valueB
 *   ╛* 
_class
loc:@conv4/weights*
dtype0*
_output_shapes
: 
У
,conv4/weights/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *
valueB
 *   >* 
_class
loc:@conv4/weights
Є
6conv4/weights/Initializer/random_uniform/RandomUniformRandomUniform.conv4/weights/Initializer/random_uniform/shape*
dtype0*(
_output_shapes
:АА*

seed *
T0* 
_class
loc:@conv4/weights*
seed2 
╥
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
:АА
р
(conv4/weights/Initializer/random_uniformAdd,conv4/weights/Initializer/random_uniform/mul,conv4/weights/Initializer/random_uniform/min*
T0* 
_class
loc:@conv4/weights*(
_output_shapes
:АА
╖
conv4/weights
VariableV2*
dtype0*(
_output_shapes
:АА*
shared_name * 
_class
loc:@conv4/weights*
	container *
shape:АА
╒
conv4/weights/AssignAssignconv4/weights(conv4/weights/Initializer/random_uniform*
use_locking(*
T0* 
_class
loc:@conv4/weights*
validate_shape(*(
_output_shapes
:АА
В
conv4/weights/readIdentityconv4/weights*
T0* 
_class
loc:@conv4/weights*(
_output_shapes
:АА
О
conv4/biases/Initializer/zerosConst*
valueBА*    *
_class
loc:@conv4/biases*
dtype0*
_output_shapes	
:А
Ы
conv4/biases
VariableV2*
shared_name *
_class
loc:@conv4/biases*
	container *
shape:А*
dtype0*
_output_shapes	
:А
╗
conv4/biases/AssignAssignconv4/biasesconv4/biases/Initializer/zeros*
use_locking(*
T0*
_class
loc:@conv4/biases*
validate_shape(*
_output_shapes	
:А
r
conv4/biases/readIdentityconv4/biases*
T0*
_class
loc:@conv4/biases*
_output_shapes	
:А
j
model/conv4/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
°
model/conv4/Conv2DConv2Dmodel/conv3/MaxPool2D/MaxPoolconv4/weights/read*
paddingSAME*0
_output_shapes
:         А*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
Ч
model/conv4/BiasAddBiasAddmodel/conv4/Conv2Dconv4/biases/read*
T0*
data_formatNHWC*0
_output_shapes
:         А
n
model/conv4/conv4/ReluRelumodel/conv4/BiasAdd*0
_output_shapes
:         А*
T0
═
model/conv4/MaxPool2D/MaxPoolMaxPoolmodel/conv4/conv4/Relu*
T0*
data_formatNHWC*
strides
*
ksize
*
paddingSAME*0
_output_shapes
:         
А
й
.conv5/weights/Initializer/random_uniform/shapeConst*%
valueB"            * 
_class
loc:@conv5/weights*
dtype0*
_output_shapes
:
У
,conv5/weights/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *
valueB
 *┴╓╛* 
_class
loc:@conv5/weights
У
,conv5/weights/Initializer/random_uniform/maxConst*
valueB
 *┴╓>* 
_class
loc:@conv5/weights*
dtype0*
_output_shapes
: 
ё
6conv5/weights/Initializer/random_uniform/RandomUniformRandomUniform.conv5/weights/Initializer/random_uniform/shape*
dtype0*'
_output_shapes
:А*

seed *
T0* 
_class
loc:@conv5/weights*
seed2 
╥
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
:А
▀
(conv5/weights/Initializer/random_uniformAdd,conv5/weights/Initializer/random_uniform/mul,conv5/weights/Initializer/random_uniform/min*
T0* 
_class
loc:@conv5/weights*'
_output_shapes
:А
╡
conv5/weights
VariableV2*
shared_name * 
_class
loc:@conv5/weights*
	container *
shape:А*
dtype0*'
_output_shapes
:А
╘
conv5/weights/AssignAssignconv5/weights(conv5/weights/Initializer/random_uniform*
use_locking(*
T0* 
_class
loc:@conv5/weights*
validate_shape(*'
_output_shapes
:А
Б
conv5/weights/readIdentityconv5/weights*'
_output_shapes
:А*
T0* 
_class
loc:@conv5/weights
М
conv5/biases/Initializer/zerosConst*
valueB*    *
_class
loc:@conv5/biases*
dtype0*
_output_shapes
:
Щ
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
║
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
ў
model/conv5/Conv2DConv2Dmodel/conv4/MaxPool2D/MaxPoolconv5/weights/read*/
_output_shapes
:         
*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME
Ц
model/conv5/BiasAddBiasAddmodel/conv5/Conv2Dconv5/biases/read*
data_formatNHWC*/
_output_shapes
:         
*
T0
╔
model/conv5/MaxPool2D/MaxPoolMaxPoolmodel/conv5/BiasAdd*
T0*
data_formatNHWC*
strides
*
ksize
*
paddingSAME*/
_output_shapes
:         
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
%model/Flatten/flatten/Reshape/shape/1Const*
dtype0*
_output_shapes
: *
valueB :
         
▒
#model/Flatten/flatten/Reshape/shapePack#model/Flatten/flatten/strided_slice%model/Flatten/flatten/Reshape/shape/1*
T0*

axis *
N*
_output_shapes
:
н
model/Flatten/flatten/ReshapeReshapemodel/conv5/MaxPool2D/MaxPool#model/Flatten/flatten/Reshape/shape*
T0*
Tshape0*(
_output_shapes
:         Ш
l
model_1/conv1/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
ы
model_1/conv1/Conv2DConv2Dpositive_inputconv1/weights/read*0
_output_shapes
:         2Ц *
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME
Ы
model_1/conv1/BiasAddBiasAddmodel_1/conv1/Conv2Dconv1/biases/read*
data_formatNHWC*0
_output_shapes
:         2Ц *
T0
r
model_1/conv1/conv1/ReluRelumodel_1/conv1/BiasAdd*
T0*0
_output_shapes
:         2Ц 
╨
model_1/conv1/MaxPool2D/MaxPoolMaxPoolmodel_1/conv1/conv1/Relu*/
_output_shapes
:         K *
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
√
model_1/conv2/Conv2DConv2Dmodel_1/conv1/MaxPool2D/MaxPoolconv2/weights/read*/
_output_shapes
:         K@*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME
Ъ
model_1/conv2/BiasAddBiasAddmodel_1/conv2/Conv2Dconv2/biases/read*/
_output_shapes
:         K@*
T0*
data_formatNHWC
q
model_1/conv2/conv2/ReluRelumodel_1/conv2/BiasAdd*/
_output_shapes
:         K@*
T0
╨
model_1/conv2/MaxPool2D/MaxPoolMaxPoolmodel_1/conv2/conv2/Relu*
T0*
data_formatNHWC*
strides
*
ksize
*
paddingSAME*/
_output_shapes
:         &@
l
model_1/conv3/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
№
model_1/conv3/Conv2DConv2Dmodel_1/conv2/MaxPool2D/MaxPoolconv3/weights/read*0
_output_shapes
:         &А*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME
Ы
model_1/conv3/BiasAddBiasAddmodel_1/conv3/Conv2Dconv3/biases/read*
T0*
data_formatNHWC*0
_output_shapes
:         &А
r
model_1/conv3/conv3/ReluRelumodel_1/conv3/BiasAdd*
T0*0
_output_shapes
:         &А
╤
model_1/conv3/MaxPool2D/MaxPoolMaxPoolmodel_1/conv3/conv3/Relu*
T0*
data_formatNHWC*
strides
*
ksize
*
paddingSAME*0
_output_shapes
:         А
l
model_1/conv4/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
№
model_1/conv4/Conv2DConv2Dmodel_1/conv3/MaxPool2D/MaxPoolconv4/weights/read*0
_output_shapes
:         А*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME
Ы
model_1/conv4/BiasAddBiasAddmodel_1/conv4/Conv2Dconv4/biases/read*
T0*
data_formatNHWC*0
_output_shapes
:         А
r
model_1/conv4/conv4/ReluRelumodel_1/conv4/BiasAdd*0
_output_shapes
:         А*
T0
╤
model_1/conv4/MaxPool2D/MaxPoolMaxPoolmodel_1/conv4/conv4/Relu*
ksize
*
paddingSAME*0
_output_shapes
:         
А*
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
√
model_1/conv5/Conv2DConv2Dmodel_1/conv4/MaxPool2D/MaxPoolconv5/weights/read*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*/
_output_shapes
:         
*
	dilations
*
T0
Ъ
model_1/conv5/BiasAddBiasAddmodel_1/conv5/Conv2Dconv5/biases/read*
T0*
data_formatNHWC*/
_output_shapes
:         

═
model_1/conv5/MaxPool2D/MaxPoolMaxPoolmodel_1/conv5/BiasAdd*
data_formatNHWC*
strides
*
ksize
*
paddingSAME*/
_output_shapes
:         *
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
-model_1/Flatten/flatten/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
ё
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
         *
dtype0*
_output_shapes
: 
╖
%model_1/Flatten/flatten/Reshape/shapePack%model_1/Flatten/flatten/strided_slice'model_1/Flatten/flatten/Reshape/shape/1*
T0*

axis *
N*
_output_shapes
:
│
model_1/Flatten/flatten/ReshapeReshapemodel_1/conv5/MaxPool2D/MaxPool%model_1/Flatten/flatten/Reshape/shape*
T0*
Tshape0*(
_output_shapes
:         Ш
l
model_2/conv1/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
ы
model_2/conv1/Conv2DConv2Dnegative_inputconv1/weights/read*
paddingSAME*0
_output_shapes
:         2Ц *
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
Ы
model_2/conv1/BiasAddBiasAddmodel_2/conv1/Conv2Dconv1/biases/read*
data_formatNHWC*0
_output_shapes
:         2Ц *
T0
r
model_2/conv1/conv1/ReluRelumodel_2/conv1/BiasAdd*
T0*0
_output_shapes
:         2Ц 
╨
model_2/conv1/MaxPool2D/MaxPoolMaxPoolmodel_2/conv1/conv1/Relu*
T0*
data_formatNHWC*
strides
*
ksize
*
paddingSAME*/
_output_shapes
:         K 
l
model_2/conv2/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
√
model_2/conv2/Conv2DConv2Dmodel_2/conv1/MaxPool2D/MaxPoolconv2/weights/read*
paddingSAME*/
_output_shapes
:         K@*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
Ъ
model_2/conv2/BiasAddBiasAddmodel_2/conv2/Conv2Dconv2/biases/read*
data_formatNHWC*/
_output_shapes
:         K@*
T0
q
model_2/conv2/conv2/ReluRelumodel_2/conv2/BiasAdd*
T0*/
_output_shapes
:         K@
╨
model_2/conv2/MaxPool2D/MaxPoolMaxPoolmodel_2/conv2/conv2/Relu*/
_output_shapes
:         &@*
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
№
model_2/conv3/Conv2DConv2Dmodel_2/conv2/MaxPool2D/MaxPoolconv3/weights/read*
paddingSAME*0
_output_shapes
:         &А*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
Ы
model_2/conv3/BiasAddBiasAddmodel_2/conv3/Conv2Dconv3/biases/read*
T0*
data_formatNHWC*0
_output_shapes
:         &А
r
model_2/conv3/conv3/ReluRelumodel_2/conv3/BiasAdd*0
_output_shapes
:         &А*
T0
╤
model_2/conv3/MaxPool2D/MaxPoolMaxPoolmodel_2/conv3/conv3/Relu*
ksize
*
paddingSAME*0
_output_shapes
:         А*
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
№
model_2/conv4/Conv2DConv2Dmodel_2/conv3/MaxPool2D/MaxPoolconv4/weights/read*0
_output_shapes
:         А*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME
Ы
model_2/conv4/BiasAddBiasAddmodel_2/conv4/Conv2Dconv4/biases/read*
T0*
data_formatNHWC*0
_output_shapes
:         А
r
model_2/conv4/conv4/ReluRelumodel_2/conv4/BiasAdd*
T0*0
_output_shapes
:         А
╤
model_2/conv4/MaxPool2D/MaxPoolMaxPoolmodel_2/conv4/conv4/Relu*
T0*
data_formatNHWC*
strides
*
ksize
*
paddingSAME*0
_output_shapes
:         
А
l
model_2/conv5/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
√
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
:         

Ъ
model_2/conv5/BiasAddBiasAddmodel_2/conv5/Conv2Dconv5/biases/read*
data_formatNHWC*/
_output_shapes
:         
*
T0
═
model_2/conv5/MaxPool2D/MaxPoolMaxPoolmodel_2/conv5/BiasAdd*
data_formatNHWC*
strides
*
ksize
*
paddingSAME*/
_output_shapes
:         *
T0
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
'model_2/Flatten/flatten/Reshape/shape/1Const*
valueB :
         *
dtype0*
_output_shapes
: 
╖
%model_2/Flatten/flatten/Reshape/shapePack%model_2/Flatten/flatten/strided_slice'model_2/Flatten/flatten/Reshape/shape/1*
T0*

axis *
N*
_output_shapes
:
│
model_2/Flatten/flatten/ReshapeReshapemodel_2/conv5/MaxPool2D/MaxPool%model_2/Flatten/flatten/Reshape/shape*(
_output_shapes
:         Ш*
T0*
Tshape0

mulMulmodel_1/Flatten/flatten/Reshapemodel_2/Flatten/flatten/Reshape*(
_output_shapes
:         Ш*
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
:         
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
:         Ш
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
:         *
	keep_dims( *

Tidx0
A
SqrtSqrtSum_1*
T0*#
_output_shapes
:         
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
:         Ш*
T0
Y
Sum_2/reduction_indicesConst*
value	B :*
dtype0*
_output_shapes
: 
w
Sum_2SumPow_1Sum_2/reduction_indices*
T0*#
_output_shapes
:         *
	keep_dims( *

Tidx0
C
Sqrt_1SqrtSum_2*
T0*#
_output_shapes
:         
H
mul_1MulSqrt_1Sqrt*
T0*#
_output_shapes
:         
L
truedivRealDivSummul_1*
T0*#
_output_shapes
:         
}
subSubmodel/Flatten/flatten/Reshapemodel_1/Flatten/flatten/Reshape*(
_output_shapes
:         Ш*
T0
L
Pow_2/yConst*
dtype0*
_output_shapes
: *
valueB
 *   @
M
Pow_2PowsubPow_2/y*
T0*(
_output_shapes
:         Ш
Y
Sum_3/reduction_indicesConst*
dtype0*
_output_shapes
: *
value	B :
{
Sum_3SumPow_2Sum_3/reduction_indices*
	keep_dims(*

Tidx0*
T0*'
_output_shapes
:         
G
Sqrt_2SqrtSum_3*
T0*'
_output_shapes
:         

sub_1Submodel/Flatten/flatten/Reshapemodel_2/Flatten/flatten/Reshape*(
_output_shapes
:         Ш*
T0
L
Pow_3/yConst*
valueB
 *   @*
dtype0*
_output_shapes
: 
O
Pow_3Powsub_1Pow_3/y*
T0*(
_output_shapes
:         Ш
Y
Sum_4/reduction_indicesConst*
value	B :*
dtype0*
_output_shapes
: 
{
Sum_4SumPow_3Sum_4/reduction_indices*
	keep_dims(*

Tidx0*
T0*'
_output_shapes
:         
G
Sqrt_3SqrtSum_4*'
_output_shapes
:         *
T0
N
sub_2SubSqrt_2Sqrt_3*'
_output_shapes
:         *
T0
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
:         *
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
:         
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
в
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
gradients/grad_ys_0Const*
dtype0*
_output_shapes
: *
valueB
 *  А?
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
Р
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
Ь
gradients/Mean_grad/TileTilegradients/Mean_grad/Reshapegradients/Mean_grad/Shape*

Tmultiples0*
T0*'
_output_shapes
:         
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
Ц
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
Ъ
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
В
gradients/Mean_grad/MaximumMaximumgradients/Mean_grad/Prod_1gradients/Mean_grad/Maximum/y*
T0*
_output_shapes
: 
А
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
М
gradients/Mean_grad/truedivRealDivgradients/Mean_grad/Tilegradients/Mean_grad/Cast*
T0*'
_output_shapes
:         
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
м
gradients/Maximum_grad/zerosFillgradients/Maximum_grad/Shape_2"gradients/Maximum_grad/zeros/Const*
T0*

index_type0*'
_output_shapes
:         
u
#gradients/Maximum_grad/GreaterEqualGreaterEqualadd	Maximum/y*
T0*'
_output_shapes
:         
└
,gradients/Maximum_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/Maximum_grad/Shapegradients/Maximum_grad/Shape_1*
T0*2
_output_shapes 
:         :         
╣
gradients/Maximum_grad/SelectSelect#gradients/Maximum_grad/GreaterEqualgradients/Mean_grad/truedivgradients/Maximum_grad/zeros*
T0*'
_output_shapes
:         
╗
gradients/Maximum_grad/Select_1Select#gradients/Maximum_grad/GreaterEqualgradients/Maximum_grad/zerosgradients/Mean_grad/truediv*
T0*'
_output_shapes
:         
о
gradients/Maximum_grad/SumSumgradients/Maximum_grad/Select,gradients/Maximum_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
г
gradients/Maximum_grad/ReshapeReshapegradients/Maximum_grad/Sumgradients/Maximum_grad/Shape*
T0*
Tshape0*'
_output_shapes
:         
┤
gradients/Maximum_grad/Sum_1Sumgradients/Maximum_grad/Select_1.gradients/Maximum_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
Ш
 gradients/Maximum_grad/Reshape_1Reshapegradients/Maximum_grad/Sum_1gradients/Maximum_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
s
'gradients/Maximum_grad/tuple/group_depsNoOp^gradients/Maximum_grad/Reshape!^gradients/Maximum_grad/Reshape_1
ъ
/gradients/Maximum_grad/tuple/control_dependencyIdentitygradients/Maximum_grad/Reshape(^gradients/Maximum_grad/tuple/group_deps*'
_output_shapes
:         *
T0*1
_class'
%#loc:@gradients/Maximum_grad/Reshape
▀
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
gradients/add_grad/Shape_1Const*
dtype0*
_output_shapes
: *
valueB 
┤
(gradients/add_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_grad/Shapegradients/add_grad/Shape_1*2
_output_shapes 
:         :         *
T0
╕
gradients/add_grad/SumSum/gradients/Maximum_grad/tuple/control_dependency(gradients/add_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Ч
gradients/add_grad/ReshapeReshapegradients/add_grad/Sumgradients/add_grad/Shape*
T0*
Tshape0*'
_output_shapes
:         
╝
gradients/add_grad/Sum_1Sum/gradients/Maximum_grad/tuple/control_dependency*gradients/add_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
М
gradients/add_grad/Reshape_1Reshapegradients/add_grad/Sum_1gradients/add_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
g
#gradients/add_grad/tuple/group_depsNoOp^gradients/add_grad/Reshape^gradients/add_grad/Reshape_1
┌
+gradients/add_grad/tuple/control_dependencyIdentitygradients/add_grad/Reshape$^gradients/add_grad/tuple/group_deps*
T0*-
_class#
!loc:@gradients/add_grad/Reshape*'
_output_shapes
:         
╧
-gradients/add_grad/tuple/control_dependency_1Identitygradients/add_grad/Reshape_1$^gradients/add_grad/tuple/group_deps*
_output_shapes
: *
T0*/
_class%
#!loc:@gradients/add_grad/Reshape_1
`
gradients/sub_2_grad/ShapeShapeSqrt_2*
T0*
out_type0*
_output_shapes
:
b
gradients/sub_2_grad/Shape_1ShapeSqrt_3*
T0*
out_type0*
_output_shapes
:
║
*gradients/sub_2_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/sub_2_grad/Shapegradients/sub_2_grad/Shape_1*2
_output_shapes 
:         :         *
T0
╕
gradients/sub_2_grad/SumSum+gradients/add_grad/tuple/control_dependency*gradients/sub_2_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
Э
gradients/sub_2_grad/ReshapeReshapegradients/sub_2_grad/Sumgradients/sub_2_grad/Shape*'
_output_shapes
:         *
T0*
Tshape0
╝
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
б
gradients/sub_2_grad/Reshape_1Reshapegradients/sub_2_grad/Neggradients/sub_2_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:         
m
%gradients/sub_2_grad/tuple/group_depsNoOp^gradients/sub_2_grad/Reshape^gradients/sub_2_grad/Reshape_1
т
-gradients/sub_2_grad/tuple/control_dependencyIdentitygradients/sub_2_grad/Reshape&^gradients/sub_2_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/sub_2_grad/Reshape*'
_output_shapes
:         
ш
/gradients/sub_2_grad/tuple/control_dependency_1Identitygradients/sub_2_grad/Reshape_1&^gradients/sub_2_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/sub_2_grad/Reshape_1*'
_output_shapes
:         
У
gradients/Sqrt_2_grad/SqrtGradSqrtGradSqrt_2-gradients/sub_2_grad/tuple/control_dependency*
T0*'
_output_shapes
:         
Х
gradients/Sqrt_3_grad/SqrtGradSqrtGradSqrt_3/gradients/sub_2_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:         
_
gradients/Sum_3_grad/ShapeShapePow_2*
T0*
out_type0*
_output_shapes
:
К
gradients/Sum_3_grad/SizeConst*
value	B :*-
_class#
!loc:@gradients/Sum_3_grad/Shape*
dtype0*
_output_shapes
: 
г
gradients/Sum_3_grad/addAddSum_3/reduction_indicesgradients/Sum_3_grad/Size*
T0*-
_class#
!loc:@gradients/Sum_3_grad/Shape*
_output_shapes
: 
й
gradients/Sum_3_grad/modFloorModgradients/Sum_3_grad/addgradients/Sum_3_grad/Size*
T0*-
_class#
!loc:@gradients/Sum_3_grad/Shape*
_output_shapes
: 
О
gradients/Sum_3_grad/Shape_1Const*
valueB *-
_class#
!loc:@gradients/Sum_3_grad/Shape*
dtype0*
_output_shapes
: 
С
 gradients/Sum_3_grad/range/startConst*
value	B : *-
_class#
!loc:@gradients/Sum_3_grad/Shape*
dtype0*
_output_shapes
: 
С
 gradients/Sum_3_grad/range/deltaConst*
value	B :*-
_class#
!loc:@gradients/Sum_3_grad/Shape*
dtype0*
_output_shapes
: 
┘
gradients/Sum_3_grad/rangeRange gradients/Sum_3_grad/range/startgradients/Sum_3_grad/Size gradients/Sum_3_grad/range/delta*
_output_shapes
:*

Tidx0*-
_class#
!loc:@gradients/Sum_3_grad/Shape
Р
gradients/Sum_3_grad/Fill/valueConst*
value	B :*-
_class#
!loc:@gradients/Sum_3_grad/Shape*
dtype0*
_output_shapes
: 
┬
gradients/Sum_3_grad/FillFillgradients/Sum_3_grad/Shape_1gradients/Sum_3_grad/Fill/value*
T0*

index_type0*-
_class#
!loc:@gradients/Sum_3_grad/Shape*
_output_shapes
: 
¤
"gradients/Sum_3_grad/DynamicStitchDynamicStitchgradients/Sum_3_grad/rangegradients/Sum_3_grad/modgradients/Sum_3_grad/Shapegradients/Sum_3_grad/Fill*
T0*-
_class#
!loc:@gradients/Sum_3_grad/Shape*
N*
_output_shapes
:
П
gradients/Sum_3_grad/Maximum/yConst*
value	B :*-
_class#
!loc:@gradients/Sum_3_grad/Shape*
dtype0*
_output_shapes
: 
┐
gradients/Sum_3_grad/MaximumMaximum"gradients/Sum_3_grad/DynamicStitchgradients/Sum_3_grad/Maximum/y*
T0*-
_class#
!loc:@gradients/Sum_3_grad/Shape*
_output_shapes
:
╖
gradients/Sum_3_grad/floordivFloorDivgradients/Sum_3_grad/Shapegradients/Sum_3_grad/Maximum*
T0*-
_class#
!loc:@gradients/Sum_3_grad/Shape*
_output_shapes
:
┤
gradients/Sum_3_grad/ReshapeReshapegradients/Sqrt_2_grad/SqrtGrad"gradients/Sum_3_grad/DynamicStitch*0
_output_shapes
:                  *
T0*
Tshape0
г
gradients/Sum_3_grad/TileTilegradients/Sum_3_grad/Reshapegradients/Sum_3_grad/floordiv*
T0*(
_output_shapes
:         Ш*

Tmultiples0
_
gradients/Sum_4_grad/ShapeShapePow_3*
T0*
out_type0*
_output_shapes
:
К
gradients/Sum_4_grad/SizeConst*
value	B :*-
_class#
!loc:@gradients/Sum_4_grad/Shape*
dtype0*
_output_shapes
: 
г
gradients/Sum_4_grad/addAddSum_4/reduction_indicesgradients/Sum_4_grad/Size*
T0*-
_class#
!loc:@gradients/Sum_4_grad/Shape*
_output_shapes
: 
й
gradients/Sum_4_grad/modFloorModgradients/Sum_4_grad/addgradients/Sum_4_grad/Size*
_output_shapes
: *
T0*-
_class#
!loc:@gradients/Sum_4_grad/Shape
О
gradients/Sum_4_grad/Shape_1Const*
valueB *-
_class#
!loc:@gradients/Sum_4_grad/Shape*
dtype0*
_output_shapes
: 
С
 gradients/Sum_4_grad/range/startConst*
dtype0*
_output_shapes
: *
value	B : *-
_class#
!loc:@gradients/Sum_4_grad/Shape
С
 gradients/Sum_4_grad/range/deltaConst*
value	B :*-
_class#
!loc:@gradients/Sum_4_grad/Shape*
dtype0*
_output_shapes
: 
┘
gradients/Sum_4_grad/rangeRange gradients/Sum_4_grad/range/startgradients/Sum_4_grad/Size gradients/Sum_4_grad/range/delta*-
_class#
!loc:@gradients/Sum_4_grad/Shape*
_output_shapes
:*

Tidx0
Р
gradients/Sum_4_grad/Fill/valueConst*
value	B :*-
_class#
!loc:@gradients/Sum_4_grad/Shape*
dtype0*
_output_shapes
: 
┬
gradients/Sum_4_grad/FillFillgradients/Sum_4_grad/Shape_1gradients/Sum_4_grad/Fill/value*
T0*

index_type0*-
_class#
!loc:@gradients/Sum_4_grad/Shape*
_output_shapes
: 
¤
"gradients/Sum_4_grad/DynamicStitchDynamicStitchgradients/Sum_4_grad/rangegradients/Sum_4_grad/modgradients/Sum_4_grad/Shapegradients/Sum_4_grad/Fill*
T0*-
_class#
!loc:@gradients/Sum_4_grad/Shape*
N*
_output_shapes
:
П
gradients/Sum_4_grad/Maximum/yConst*
value	B :*-
_class#
!loc:@gradients/Sum_4_grad/Shape*
dtype0*
_output_shapes
: 
┐
gradients/Sum_4_grad/MaximumMaximum"gradients/Sum_4_grad/DynamicStitchgradients/Sum_4_grad/Maximum/y*
_output_shapes
:*
T0*-
_class#
!loc:@gradients/Sum_4_grad/Shape
╖
gradients/Sum_4_grad/floordivFloorDivgradients/Sum_4_grad/Shapegradients/Sum_4_grad/Maximum*
T0*-
_class#
!loc:@gradients/Sum_4_grad/Shape*
_output_shapes
:
┤
gradients/Sum_4_grad/ReshapeReshapegradients/Sqrt_3_grad/SqrtGrad"gradients/Sum_4_grad/DynamicStitch*
T0*
Tshape0*0
_output_shapes
:                  
г
gradients/Sum_4_grad/TileTilegradients/Sum_4_grad/Reshapegradients/Sum_4_grad/floordiv*

Tmultiples0*
T0*(
_output_shapes
:         Ш
]
gradients/Pow_2_grad/ShapeShapesub*
_output_shapes
:*
T0*
out_type0
_
gradients/Pow_2_grad/Shape_1Const*
dtype0*
_output_shapes
: *
valueB 
║
*gradients/Pow_2_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/Pow_2_grad/Shapegradients/Pow_2_grad/Shape_1*
T0*2
_output_shapes 
:         :         
v
gradients/Pow_2_grad/mulMulgradients/Sum_3_grad/TilePow_2/y*
T0*(
_output_shapes
:         Ш
_
gradients/Pow_2_grad/sub/yConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
e
gradients/Pow_2_grad/subSubPow_2/ygradients/Pow_2_grad/sub/y*
T0*
_output_shapes
: 
q
gradients/Pow_2_grad/PowPowsubgradients/Pow_2_grad/sub*(
_output_shapes
:         Ш*
T0
И
gradients/Pow_2_grad/mul_1Mulgradients/Pow_2_grad/mulgradients/Pow_2_grad/Pow*
T0*(
_output_shapes
:         Ш
з
gradients/Pow_2_grad/SumSumgradients/Pow_2_grad/mul_1*gradients/Pow_2_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Ю
gradients/Pow_2_grad/ReshapeReshapegradients/Pow_2_grad/Sumgradients/Pow_2_grad/Shape*
T0*
Tshape0*(
_output_shapes
:         Ш
c
gradients/Pow_2_grad/Greater/yConst*
valueB
 *    *
dtype0*
_output_shapes
: 

gradients/Pow_2_grad/GreaterGreatersubgradients/Pow_2_grad/Greater/y*(
_output_shapes
:         Ш*
T0
g
$gradients/Pow_2_grad/ones_like/ShapeShapesub*
T0*
out_type0*
_output_shapes
:
i
$gradients/Pow_2_grad/ones_like/ConstConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
╖
gradients/Pow_2_grad/ones_likeFill$gradients/Pow_2_grad/ones_like/Shape$gradients/Pow_2_grad/ones_like/Const*
T0*

index_type0*(
_output_shapes
:         Ш
Ы
gradients/Pow_2_grad/SelectSelectgradients/Pow_2_grad/Greatersubgradients/Pow_2_grad/ones_like*
T0*(
_output_shapes
:         Ш
o
gradients/Pow_2_grad/LogLoggradients/Pow_2_grad/Select*
T0*(
_output_shapes
:         Ш
d
gradients/Pow_2_grad/zeros_like	ZerosLikesub*
T0*(
_output_shapes
:         Ш
│
gradients/Pow_2_grad/Select_1Selectgradients/Pow_2_grad/Greatergradients/Pow_2_grad/Loggradients/Pow_2_grad/zeros_like*
T0*(
_output_shapes
:         Ш
v
gradients/Pow_2_grad/mul_2Mulgradients/Sum_3_grad/TilePow_2*
T0*(
_output_shapes
:         Ш
П
gradients/Pow_2_grad/mul_3Mulgradients/Pow_2_grad/mul_2gradients/Pow_2_grad/Select_1*(
_output_shapes
:         Ш*
T0
л
gradients/Pow_2_grad/Sum_1Sumgradients/Pow_2_grad/mul_3,gradients/Pow_2_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
Т
gradients/Pow_2_grad/Reshape_1Reshapegradients/Pow_2_grad/Sum_1gradients/Pow_2_grad/Shape_1*
_output_shapes
: *
T0*
Tshape0
m
%gradients/Pow_2_grad/tuple/group_depsNoOp^gradients/Pow_2_grad/Reshape^gradients/Pow_2_grad/Reshape_1
у
-gradients/Pow_2_grad/tuple/control_dependencyIdentitygradients/Pow_2_grad/Reshape&^gradients/Pow_2_grad/tuple/group_deps*(
_output_shapes
:         Ш*
T0*/
_class%
#!loc:@gradients/Pow_2_grad/Reshape
╫
/gradients/Pow_2_grad/tuple/control_dependency_1Identitygradients/Pow_2_grad/Reshape_1&^gradients/Pow_2_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/Pow_2_grad/Reshape_1*
_output_shapes
: 
_
gradients/Pow_3_grad/ShapeShapesub_1*
T0*
out_type0*
_output_shapes
:
_
gradients/Pow_3_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
║
*gradients/Pow_3_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/Pow_3_grad/Shapegradients/Pow_3_grad/Shape_1*
T0*2
_output_shapes 
:         :         
v
gradients/Pow_3_grad/mulMulgradients/Sum_4_grad/TilePow_3/y*
T0*(
_output_shapes
:         Ш
_
gradients/Pow_3_grad/sub/yConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
e
gradients/Pow_3_grad/subSubPow_3/ygradients/Pow_3_grad/sub/y*
T0*
_output_shapes
: 
s
gradients/Pow_3_grad/PowPowsub_1gradients/Pow_3_grad/sub*(
_output_shapes
:         Ш*
T0
И
gradients/Pow_3_grad/mul_1Mulgradients/Pow_3_grad/mulgradients/Pow_3_grad/Pow*
T0*(
_output_shapes
:         Ш
з
gradients/Pow_3_grad/SumSumgradients/Pow_3_grad/mul_1*gradients/Pow_3_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Ю
gradients/Pow_3_grad/ReshapeReshapegradients/Pow_3_grad/Sumgradients/Pow_3_grad/Shape*
T0*
Tshape0*(
_output_shapes
:         Ш
c
gradients/Pow_3_grad/Greater/yConst*
dtype0*
_output_shapes
: *
valueB
 *    
Б
gradients/Pow_3_grad/GreaterGreatersub_1gradients/Pow_3_grad/Greater/y*(
_output_shapes
:         Ш*
T0
i
$gradients/Pow_3_grad/ones_like/ShapeShapesub_1*
T0*
out_type0*
_output_shapes
:
i
$gradients/Pow_3_grad/ones_like/ConstConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
╖
gradients/Pow_3_grad/ones_likeFill$gradients/Pow_3_grad/ones_like/Shape$gradients/Pow_3_grad/ones_like/Const*(
_output_shapes
:         Ш*
T0*

index_type0
Э
gradients/Pow_3_grad/SelectSelectgradients/Pow_3_grad/Greatersub_1gradients/Pow_3_grad/ones_like*(
_output_shapes
:         Ш*
T0
o
gradients/Pow_3_grad/LogLoggradients/Pow_3_grad/Select*
T0*(
_output_shapes
:         Ш
f
gradients/Pow_3_grad/zeros_like	ZerosLikesub_1*
T0*(
_output_shapes
:         Ш
│
gradients/Pow_3_grad/Select_1Selectgradients/Pow_3_grad/Greatergradients/Pow_3_grad/Loggradients/Pow_3_grad/zeros_like*
T0*(
_output_shapes
:         Ш
v
gradients/Pow_3_grad/mul_2Mulgradients/Sum_4_grad/TilePow_3*
T0*(
_output_shapes
:         Ш
П
gradients/Pow_3_grad/mul_3Mulgradients/Pow_3_grad/mul_2gradients/Pow_3_grad/Select_1*
T0*(
_output_shapes
:         Ш
л
gradients/Pow_3_grad/Sum_1Sumgradients/Pow_3_grad/mul_3,gradients/Pow_3_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Т
gradients/Pow_3_grad/Reshape_1Reshapegradients/Pow_3_grad/Sum_1gradients/Pow_3_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
m
%gradients/Pow_3_grad/tuple/group_depsNoOp^gradients/Pow_3_grad/Reshape^gradients/Pow_3_grad/Reshape_1
у
-gradients/Pow_3_grad/tuple/control_dependencyIdentitygradients/Pow_3_grad/Reshape&^gradients/Pow_3_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/Pow_3_grad/Reshape*(
_output_shapes
:         Ш
╫
/gradients/Pow_3_grad/tuple/control_dependency_1Identitygradients/Pow_3_grad/Reshape_1&^gradients/Pow_3_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/Pow_3_grad/Reshape_1*
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
┤
(gradients/sub_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/sub_grad/Shapegradients/sub_grad/Shape_1*2
_output_shapes 
:         :         *
T0
╢
gradients/sub_grad/SumSum-gradients/Pow_2_grad/tuple/control_dependency(gradients/sub_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
Ш
gradients/sub_grad/ReshapeReshapegradients/sub_grad/Sumgradients/sub_grad/Shape*
T0*
Tshape0*(
_output_shapes
:         Ш
║
gradients/sub_grad/Sum_1Sum-gradients/Pow_2_grad/tuple/control_dependency*gradients/sub_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Z
gradients/sub_grad/NegNeggradients/sub_grad/Sum_1*
_output_shapes
:*
T0
Ь
gradients/sub_grad/Reshape_1Reshapegradients/sub_grad/Neggradients/sub_grad/Shape_1*(
_output_shapes
:         Ш*
T0*
Tshape0
g
#gradients/sub_grad/tuple/group_depsNoOp^gradients/sub_grad/Reshape^gradients/sub_grad/Reshape_1
█
+gradients/sub_grad/tuple/control_dependencyIdentitygradients/sub_grad/Reshape$^gradients/sub_grad/tuple/group_deps*(
_output_shapes
:         Ш*
T0*-
_class#
!loc:@gradients/sub_grad/Reshape
с
-gradients/sub_grad/tuple/control_dependency_1Identitygradients/sub_grad/Reshape_1$^gradients/sub_grad/tuple/group_deps*(
_output_shapes
:         Ш*
T0*/
_class%
#!loc:@gradients/sub_grad/Reshape_1
w
gradients/sub_1_grad/ShapeShapemodel/Flatten/flatten/Reshape*
_output_shapes
:*
T0*
out_type0
{
gradients/sub_1_grad/Shape_1Shapemodel_2/Flatten/flatten/Reshape*
_output_shapes
:*
T0*
out_type0
║
*gradients/sub_1_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/sub_1_grad/Shapegradients/sub_1_grad/Shape_1*2
_output_shapes 
:         :         *
T0
║
gradients/sub_1_grad/SumSum-gradients/Pow_3_grad/tuple/control_dependency*gradients/sub_1_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
Ю
gradients/sub_1_grad/ReshapeReshapegradients/sub_1_grad/Sumgradients/sub_1_grad/Shape*
T0*
Tshape0*(
_output_shapes
:         Ш
╛
gradients/sub_1_grad/Sum_1Sum-gradients/Pow_3_grad/tuple/control_dependency,gradients/sub_1_grad/BroadcastGradientArgs:1*
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
в
gradients/sub_1_grad/Reshape_1Reshapegradients/sub_1_grad/Neggradients/sub_1_grad/Shape_1*(
_output_shapes
:         Ш*
T0*
Tshape0
m
%gradients/sub_1_grad/tuple/group_depsNoOp^gradients/sub_1_grad/Reshape^gradients/sub_1_grad/Reshape_1
у
-gradients/sub_1_grad/tuple/control_dependencyIdentitygradients/sub_1_grad/Reshape&^gradients/sub_1_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/sub_1_grad/Reshape*(
_output_shapes
:         Ш
щ
/gradients/sub_1_grad/tuple/control_dependency_1Identitygradients/sub_1_grad/Reshape_1&^gradients/sub_1_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/sub_1_grad/Reshape_1*(
_output_shapes
:         Ш
У
4gradients/model_1/Flatten/flatten/Reshape_grad/ShapeShapemodel_1/conv5/MaxPool2D/MaxPool*
T0*
out_type0*
_output_shapes
:
ю
6gradients/model_1/Flatten/flatten/Reshape_grad/ReshapeReshape-gradients/sub_grad/tuple/control_dependency_14gradients/model_1/Flatten/flatten/Reshape_grad/Shape*/
_output_shapes
:         *
T0*
Tshape0
▌
gradients/AddNAddN+gradients/sub_grad/tuple/control_dependency-gradients/sub_1_grad/tuple/control_dependency*
T0*-
_class#
!loc:@gradients/sub_grad/Reshape*
N*(
_output_shapes
:         Ш
П
2gradients/model/Flatten/flatten/Reshape_grad/ShapeShapemodel/conv5/MaxPool2D/MaxPool*
T0*
out_type0*
_output_shapes
:
╦
4gradients/model/Flatten/flatten/Reshape_grad/ReshapeReshapegradients/AddN2gradients/model/Flatten/flatten/Reshape_grad/Shape*
T0*
Tshape0*/
_output_shapes
:         
У
4gradients/model_2/Flatten/flatten/Reshape_grad/ShapeShapemodel_2/conv5/MaxPool2D/MaxPool*
T0*
out_type0*
_output_shapes
:
Ё
6gradients/model_2/Flatten/flatten/Reshape_grad/ReshapeReshape/gradients/sub_1_grad/tuple/control_dependency_14gradients/model_2/Flatten/flatten/Reshape_grad/Shape*
T0*
Tshape0*/
_output_shapes
:         
┼
:gradients/model_1/conv5/MaxPool2D/MaxPool_grad/MaxPoolGradMaxPoolGradmodel_1/conv5/BiasAddmodel_1/conv5/MaxPool2D/MaxPool6gradients/model_1/Flatten/flatten/Reshape_grad/Reshape*
ksize
*
paddingSAME*/
_output_shapes
:         
*
T0*
data_formatNHWC*
strides

╜
8gradients/model/conv5/MaxPool2D/MaxPool_grad/MaxPoolGradMaxPoolGradmodel/conv5/BiasAddmodel/conv5/MaxPool2D/MaxPool4gradients/model/Flatten/flatten/Reshape_grad/Reshape*
ksize
*
paddingSAME*/
_output_shapes
:         
*
T0*
data_formatNHWC*
strides

┼
:gradients/model_2/conv5/MaxPool2D/MaxPool_grad/MaxPoolGradMaxPoolGradmodel_2/conv5/BiasAddmodel_2/conv5/MaxPool2D/MaxPool6gradients/model_2/Flatten/flatten/Reshape_grad/Reshape*
ksize
*
paddingSAME*/
_output_shapes
:         
*
T0*
data_formatNHWC*
strides

╖
0gradients/model_1/conv5/BiasAdd_grad/BiasAddGradBiasAddGrad:gradients/model_1/conv5/MaxPool2D/MaxPool_grad/MaxPoolGrad*
T0*
data_formatNHWC*
_output_shapes
:
н
5gradients/model_1/conv5/BiasAdd_grad/tuple/group_depsNoOp1^gradients/model_1/conv5/BiasAdd_grad/BiasAddGrad;^gradients/model_1/conv5/MaxPool2D/MaxPool_grad/MaxPoolGrad
╞
=gradients/model_1/conv5/BiasAdd_grad/tuple/control_dependencyIdentity:gradients/model_1/conv5/MaxPool2D/MaxPool_grad/MaxPoolGrad6^gradients/model_1/conv5/BiasAdd_grad/tuple/group_deps*
T0*M
_classC
A?loc:@gradients/model_1/conv5/MaxPool2D/MaxPool_grad/MaxPoolGrad*/
_output_shapes
:         

Я
?gradients/model_1/conv5/BiasAdd_grad/tuple/control_dependency_1Identity0gradients/model_1/conv5/BiasAdd_grad/BiasAddGrad6^gradients/model_1/conv5/BiasAdd_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/model_1/conv5/BiasAdd_grad/BiasAddGrad*
_output_shapes
:
│
.gradients/model/conv5/BiasAdd_grad/BiasAddGradBiasAddGrad8gradients/model/conv5/MaxPool2D/MaxPool_grad/MaxPoolGrad*
T0*
data_formatNHWC*
_output_shapes
:
з
3gradients/model/conv5/BiasAdd_grad/tuple/group_depsNoOp/^gradients/model/conv5/BiasAdd_grad/BiasAddGrad9^gradients/model/conv5/MaxPool2D/MaxPool_grad/MaxPoolGrad
╛
;gradients/model/conv5/BiasAdd_grad/tuple/control_dependencyIdentity8gradients/model/conv5/MaxPool2D/MaxPool_grad/MaxPoolGrad4^gradients/model/conv5/BiasAdd_grad/tuple/group_deps*
T0*K
_classA
?=loc:@gradients/model/conv5/MaxPool2D/MaxPool_grad/MaxPoolGrad*/
_output_shapes
:         

Ч
=gradients/model/conv5/BiasAdd_grad/tuple/control_dependency_1Identity.gradients/model/conv5/BiasAdd_grad/BiasAddGrad4^gradients/model/conv5/BiasAdd_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients/model/conv5/BiasAdd_grad/BiasAddGrad*
_output_shapes
:
╖
0gradients/model_2/conv5/BiasAdd_grad/BiasAddGradBiasAddGrad:gradients/model_2/conv5/MaxPool2D/MaxPool_grad/MaxPoolGrad*
T0*
data_formatNHWC*
_output_shapes
:
н
5gradients/model_2/conv5/BiasAdd_grad/tuple/group_depsNoOp1^gradients/model_2/conv5/BiasAdd_grad/BiasAddGrad;^gradients/model_2/conv5/MaxPool2D/MaxPool_grad/MaxPoolGrad
╞
=gradients/model_2/conv5/BiasAdd_grad/tuple/control_dependencyIdentity:gradients/model_2/conv5/MaxPool2D/MaxPool_grad/MaxPoolGrad6^gradients/model_2/conv5/BiasAdd_grad/tuple/group_deps*
T0*M
_classC
A?loc:@gradients/model_2/conv5/MaxPool2D/MaxPool_grad/MaxPoolGrad*/
_output_shapes
:         

Я
?gradients/model_2/conv5/BiasAdd_grad/tuple/control_dependency_1Identity0gradients/model_2/conv5/BiasAdd_grad/BiasAddGrad6^gradients/model_2/conv5/BiasAdd_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/model_2/conv5/BiasAdd_grad/BiasAddGrad*
_output_shapes
:
н
*gradients/model_1/conv5/Conv2D_grad/ShapeNShapeNmodel_1/conv4/MaxPool2D/MaxPoolconv5/weights/read*
T0*
out_type0*
N* 
_output_shapes
::
Ў
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
:         
А
■
8gradients/model_1/conv5/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFiltermodel_1/conv4/MaxPool2D/MaxPool,gradients/model_1/conv5/Conv2D_grad/ShapeN:1=gradients/model_1/conv5/BiasAdd_grad/tuple/control_dependency*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*'
_output_shapes
:А*
	dilations

▒
4gradients/model_1/conv5/Conv2D_grad/tuple/group_depsNoOp9^gradients/model_1/conv5/Conv2D_grad/Conv2DBackpropFilter8^gradients/model_1/conv5/Conv2D_grad/Conv2DBackpropInput
┐
<gradients/model_1/conv5/Conv2D_grad/tuple/control_dependencyIdentity7gradients/model_1/conv5/Conv2D_grad/Conv2DBackpropInput5^gradients/model_1/conv5/Conv2D_grad/tuple/group_deps*
T0*J
_class@
><loc:@gradients/model_1/conv5/Conv2D_grad/Conv2DBackpropInput*0
_output_shapes
:         
А
║
>gradients/model_1/conv5/Conv2D_grad/tuple/control_dependency_1Identity8gradients/model_1/conv5/Conv2D_grad/Conv2DBackpropFilter5^gradients/model_1/conv5/Conv2D_grad/tuple/group_deps*'
_output_shapes
:А*
T0*K
_classA
?=loc:@gradients/model_1/conv5/Conv2D_grad/Conv2DBackpropFilter
й
(gradients/model/conv5/Conv2D_grad/ShapeNShapeNmodel/conv4/MaxPool2D/MaxPoolconv5/weights/read*
T0*
out_type0*
N* 
_output_shapes
::
Ё
5gradients/model/conv5/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput(gradients/model/conv5/Conv2D_grad/ShapeNconv5/weights/read;gradients/model/conv5/BiasAdd_grad/tuple/control_dependency*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*0
_output_shapes
:         
А*
	dilations
*
T0
Ў
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
:А
л
2gradients/model/conv5/Conv2D_grad/tuple/group_depsNoOp7^gradients/model/conv5/Conv2D_grad/Conv2DBackpropFilter6^gradients/model/conv5/Conv2D_grad/Conv2DBackpropInput
╖
:gradients/model/conv5/Conv2D_grad/tuple/control_dependencyIdentity5gradients/model/conv5/Conv2D_grad/Conv2DBackpropInput3^gradients/model/conv5/Conv2D_grad/tuple/group_deps*0
_output_shapes
:         
А*
T0*H
_class>
<:loc:@gradients/model/conv5/Conv2D_grad/Conv2DBackpropInput
▓
<gradients/model/conv5/Conv2D_grad/tuple/control_dependency_1Identity6gradients/model/conv5/Conv2D_grad/Conv2DBackpropFilter3^gradients/model/conv5/Conv2D_grad/tuple/group_deps*
T0*I
_class?
=;loc:@gradients/model/conv5/Conv2D_grad/Conv2DBackpropFilter*'
_output_shapes
:А
н
*gradients/model_2/conv5/Conv2D_grad/ShapeNShapeNmodel_2/conv4/MaxPool2D/MaxPoolconv5/weights/read*
N* 
_output_shapes
::*
T0*
out_type0
Ў
7gradients/model_2/conv5/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput*gradients/model_2/conv5/Conv2D_grad/ShapeNconv5/weights/read=gradients/model_2/conv5/BiasAdd_grad/tuple/control_dependency*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*0
_output_shapes
:         
А*
	dilations

■
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
:А
▒
4gradients/model_2/conv5/Conv2D_grad/tuple/group_depsNoOp9^gradients/model_2/conv5/Conv2D_grad/Conv2DBackpropFilter8^gradients/model_2/conv5/Conv2D_grad/Conv2DBackpropInput
┐
<gradients/model_2/conv5/Conv2D_grad/tuple/control_dependencyIdentity7gradients/model_2/conv5/Conv2D_grad/Conv2DBackpropInput5^gradients/model_2/conv5/Conv2D_grad/tuple/group_deps*
T0*J
_class@
><loc:@gradients/model_2/conv5/Conv2D_grad/Conv2DBackpropInput*0
_output_shapes
:         
А
║
>gradients/model_2/conv5/Conv2D_grad/tuple/control_dependency_1Identity8gradients/model_2/conv5/Conv2D_grad/Conv2DBackpropFilter5^gradients/model_2/conv5/Conv2D_grad/tuple/group_deps*
T0*K
_classA
?=loc:@gradients/model_2/conv5/Conv2D_grad/Conv2DBackpropFilter*'
_output_shapes
:А
╠
gradients/AddN_1AddN?gradients/model_1/conv5/BiasAdd_grad/tuple/control_dependency_1=gradients/model/conv5/BiasAdd_grad/tuple/control_dependency_1?gradients/model_2/conv5/BiasAdd_grad/tuple/control_dependency_1*
T0*C
_class9
75loc:@gradients/model_1/conv5/BiasAdd_grad/BiasAddGrad*
N*
_output_shapes
:
╧
:gradients/model_1/conv4/MaxPool2D/MaxPool_grad/MaxPoolGradMaxPoolGradmodel_1/conv4/conv4/Relumodel_1/conv4/MaxPool2D/MaxPool<gradients/model_1/conv5/Conv2D_grad/tuple/control_dependency*
T0*
data_formatNHWC*
strides
*
ksize
*
paddingSAME*0
_output_shapes
:         А
╟
8gradients/model/conv4/MaxPool2D/MaxPool_grad/MaxPoolGradMaxPoolGradmodel/conv4/conv4/Relumodel/conv4/MaxPool2D/MaxPool:gradients/model/conv5/Conv2D_grad/tuple/control_dependency*0
_output_shapes
:         А*
T0*
data_formatNHWC*
strides
*
ksize
*
paddingSAME
╧
:gradients/model_2/conv4/MaxPool2D/MaxPool_grad/MaxPoolGradMaxPoolGradmodel_2/conv4/conv4/Relumodel_2/conv4/MaxPool2D/MaxPool<gradients/model_2/conv5/Conv2D_grad/tuple/control_dependency*
T0*
data_formatNHWC*
strides
*
ksize
*
paddingSAME*0
_output_shapes
:         А
▐
gradients/AddN_2AddN>gradients/model_1/conv5/Conv2D_grad/tuple/control_dependency_1<gradients/model/conv5/Conv2D_grad/tuple/control_dependency_1>gradients/model_2/conv5/Conv2D_grad/tuple/control_dependency_1*
T0*K
_classA
?=loc:@gradients/model_1/conv5/Conv2D_grad/Conv2DBackpropFilter*
N*'
_output_shapes
:А
═
0gradients/model_1/conv4/conv4/Relu_grad/ReluGradReluGrad:gradients/model_1/conv4/MaxPool2D/MaxPool_grad/MaxPoolGradmodel_1/conv4/conv4/Relu*
T0*0
_output_shapes
:         А
╟
.gradients/model/conv4/conv4/Relu_grad/ReluGradReluGrad8gradients/model/conv4/MaxPool2D/MaxPool_grad/MaxPoolGradmodel/conv4/conv4/Relu*
T0*0
_output_shapes
:         А
═
0gradients/model_2/conv4/conv4/Relu_grad/ReluGradReluGrad:gradients/model_2/conv4/MaxPool2D/MaxPool_grad/MaxPoolGradmodel_2/conv4/conv4/Relu*0
_output_shapes
:         А*
T0
о
0gradients/model_1/conv4/BiasAdd_grad/BiasAddGradBiasAddGrad0gradients/model_1/conv4/conv4/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes	
:А
г
5gradients/model_1/conv4/BiasAdd_grad/tuple/group_depsNoOp1^gradients/model_1/conv4/BiasAdd_grad/BiasAddGrad1^gradients/model_1/conv4/conv4/Relu_grad/ReluGrad
│
=gradients/model_1/conv4/BiasAdd_grad/tuple/control_dependencyIdentity0gradients/model_1/conv4/conv4/Relu_grad/ReluGrad6^gradients/model_1/conv4/BiasAdd_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/model_1/conv4/conv4/Relu_grad/ReluGrad*0
_output_shapes
:         А
а
?gradients/model_1/conv4/BiasAdd_grad/tuple/control_dependency_1Identity0gradients/model_1/conv4/BiasAdd_grad/BiasAddGrad6^gradients/model_1/conv4/BiasAdd_grad/tuple/group_deps*
_output_shapes	
:А*
T0*C
_class9
75loc:@gradients/model_1/conv4/BiasAdd_grad/BiasAddGrad
к
.gradients/model/conv4/BiasAdd_grad/BiasAddGradBiasAddGrad.gradients/model/conv4/conv4/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes	
:А
Э
3gradients/model/conv4/BiasAdd_grad/tuple/group_depsNoOp/^gradients/model/conv4/BiasAdd_grad/BiasAddGrad/^gradients/model/conv4/conv4/Relu_grad/ReluGrad
л
;gradients/model/conv4/BiasAdd_grad/tuple/control_dependencyIdentity.gradients/model/conv4/conv4/Relu_grad/ReluGrad4^gradients/model/conv4/BiasAdd_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients/model/conv4/conv4/Relu_grad/ReluGrad*0
_output_shapes
:         А
Ш
=gradients/model/conv4/BiasAdd_grad/tuple/control_dependency_1Identity.gradients/model/conv4/BiasAdd_grad/BiasAddGrad4^gradients/model/conv4/BiasAdd_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients/model/conv4/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:А
о
0gradients/model_2/conv4/BiasAdd_grad/BiasAddGradBiasAddGrad0gradients/model_2/conv4/conv4/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes	
:А
г
5gradients/model_2/conv4/BiasAdd_grad/tuple/group_depsNoOp1^gradients/model_2/conv4/BiasAdd_grad/BiasAddGrad1^gradients/model_2/conv4/conv4/Relu_grad/ReluGrad
│
=gradients/model_2/conv4/BiasAdd_grad/tuple/control_dependencyIdentity0gradients/model_2/conv4/conv4/Relu_grad/ReluGrad6^gradients/model_2/conv4/BiasAdd_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/model_2/conv4/conv4/Relu_grad/ReluGrad*0
_output_shapes
:         А
а
?gradients/model_2/conv4/BiasAdd_grad/tuple/control_dependency_1Identity0gradients/model_2/conv4/BiasAdd_grad/BiasAddGrad6^gradients/model_2/conv4/BiasAdd_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/model_2/conv4/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:А
н
*gradients/model_1/conv4/Conv2D_grad/ShapeNShapeNmodel_1/conv3/MaxPool2D/MaxPoolconv4/weights/read*
T0*
out_type0*
N* 
_output_shapes
::
Ў
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
:         А
 
8gradients/model_1/conv4/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFiltermodel_1/conv3/MaxPool2D/MaxPool,gradients/model_1/conv4/Conv2D_grad/ShapeN:1=gradients/model_1/conv4/BiasAdd_grad/tuple/control_dependency*(
_output_shapes
:АА*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME
▒
4gradients/model_1/conv4/Conv2D_grad/tuple/group_depsNoOp9^gradients/model_1/conv4/Conv2D_grad/Conv2DBackpropFilter8^gradients/model_1/conv4/Conv2D_grad/Conv2DBackpropInput
┐
<gradients/model_1/conv4/Conv2D_grad/tuple/control_dependencyIdentity7gradients/model_1/conv4/Conv2D_grad/Conv2DBackpropInput5^gradients/model_1/conv4/Conv2D_grad/tuple/group_deps*
T0*J
_class@
><loc:@gradients/model_1/conv4/Conv2D_grad/Conv2DBackpropInput*0
_output_shapes
:         А
╗
>gradients/model_1/conv4/Conv2D_grad/tuple/control_dependency_1Identity8gradients/model_1/conv4/Conv2D_grad/Conv2DBackpropFilter5^gradients/model_1/conv4/Conv2D_grad/tuple/group_deps*
T0*K
_classA
?=loc:@gradients/model_1/conv4/Conv2D_grad/Conv2DBackpropFilter*(
_output_shapes
:АА
й
(gradients/model/conv4/Conv2D_grad/ShapeNShapeNmodel/conv3/MaxPool2D/MaxPoolconv4/weights/read*
T0*
out_type0*
N* 
_output_shapes
::
Ё
5gradients/model/conv4/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput(gradients/model/conv4/Conv2D_grad/ShapeNconv4/weights/read;gradients/model/conv4/BiasAdd_grad/tuple/control_dependency*
paddingSAME*0
_output_shapes
:         А*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
ў
6gradients/model/conv4/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFiltermodel/conv3/MaxPool2D/MaxPool*gradients/model/conv4/Conv2D_grad/ShapeN:1;gradients/model/conv4/BiasAdd_grad/tuple/control_dependency*(
_output_shapes
:АА*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME
л
2gradients/model/conv4/Conv2D_grad/tuple/group_depsNoOp7^gradients/model/conv4/Conv2D_grad/Conv2DBackpropFilter6^gradients/model/conv4/Conv2D_grad/Conv2DBackpropInput
╖
:gradients/model/conv4/Conv2D_grad/tuple/control_dependencyIdentity5gradients/model/conv4/Conv2D_grad/Conv2DBackpropInput3^gradients/model/conv4/Conv2D_grad/tuple/group_deps*
T0*H
_class>
<:loc:@gradients/model/conv4/Conv2D_grad/Conv2DBackpropInput*0
_output_shapes
:         А
│
<gradients/model/conv4/Conv2D_grad/tuple/control_dependency_1Identity6gradients/model/conv4/Conv2D_grad/Conv2DBackpropFilter3^gradients/model/conv4/Conv2D_grad/tuple/group_deps*(
_output_shapes
:АА*
T0*I
_class?
=;loc:@gradients/model/conv4/Conv2D_grad/Conv2DBackpropFilter
н
*gradients/model_2/conv4/Conv2D_grad/ShapeNShapeNmodel_2/conv3/MaxPool2D/MaxPoolconv4/weights/read*
T0*
out_type0*
N* 
_output_shapes
::
Ў
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
:         А
 
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
:АА
▒
4gradients/model_2/conv4/Conv2D_grad/tuple/group_depsNoOp9^gradients/model_2/conv4/Conv2D_grad/Conv2DBackpropFilter8^gradients/model_2/conv4/Conv2D_grad/Conv2DBackpropInput
┐
<gradients/model_2/conv4/Conv2D_grad/tuple/control_dependencyIdentity7gradients/model_2/conv4/Conv2D_grad/Conv2DBackpropInput5^gradients/model_2/conv4/Conv2D_grad/tuple/group_deps*
T0*J
_class@
><loc:@gradients/model_2/conv4/Conv2D_grad/Conv2DBackpropInput*0
_output_shapes
:         А
╗
>gradients/model_2/conv4/Conv2D_grad/tuple/control_dependency_1Identity8gradients/model_2/conv4/Conv2D_grad/Conv2DBackpropFilter5^gradients/model_2/conv4/Conv2D_grad/tuple/group_deps*
T0*K
_classA
?=loc:@gradients/model_2/conv4/Conv2D_grad/Conv2DBackpropFilter*(
_output_shapes
:АА
═
gradients/AddN_3AddN?gradients/model_1/conv4/BiasAdd_grad/tuple/control_dependency_1=gradients/model/conv4/BiasAdd_grad/tuple/control_dependency_1?gradients/model_2/conv4/BiasAdd_grad/tuple/control_dependency_1*
T0*C
_class9
75loc:@gradients/model_1/conv4/BiasAdd_grad/BiasAddGrad*
N*
_output_shapes	
:А
╧
:gradients/model_1/conv3/MaxPool2D/MaxPool_grad/MaxPoolGradMaxPoolGradmodel_1/conv3/conv3/Relumodel_1/conv3/MaxPool2D/MaxPool<gradients/model_1/conv4/Conv2D_grad/tuple/control_dependency*0
_output_shapes
:         &А*
T0*
data_formatNHWC*
strides
*
ksize
*
paddingSAME
╟
8gradients/model/conv3/MaxPool2D/MaxPool_grad/MaxPoolGradMaxPoolGradmodel/conv3/conv3/Relumodel/conv3/MaxPool2D/MaxPool:gradients/model/conv4/Conv2D_grad/tuple/control_dependency*0
_output_shapes
:         &А*
T0*
data_formatNHWC*
strides
*
ksize
*
paddingSAME
╧
:gradients/model_2/conv3/MaxPool2D/MaxPool_grad/MaxPoolGradMaxPoolGradmodel_2/conv3/conv3/Relumodel_2/conv3/MaxPool2D/MaxPool<gradients/model_2/conv4/Conv2D_grad/tuple/control_dependency*
ksize
*
paddingSAME*0
_output_shapes
:         &А*
T0*
data_formatNHWC*
strides

▀
gradients/AddN_4AddN>gradients/model_1/conv4/Conv2D_grad/tuple/control_dependency_1<gradients/model/conv4/Conv2D_grad/tuple/control_dependency_1>gradients/model_2/conv4/Conv2D_grad/tuple/control_dependency_1*
T0*K
_classA
?=loc:@gradients/model_1/conv4/Conv2D_grad/Conv2DBackpropFilter*
N*(
_output_shapes
:АА
═
0gradients/model_1/conv3/conv3/Relu_grad/ReluGradReluGrad:gradients/model_1/conv3/MaxPool2D/MaxPool_grad/MaxPoolGradmodel_1/conv3/conv3/Relu*
T0*0
_output_shapes
:         &А
╟
.gradients/model/conv3/conv3/Relu_grad/ReluGradReluGrad8gradients/model/conv3/MaxPool2D/MaxPool_grad/MaxPoolGradmodel/conv3/conv3/Relu*0
_output_shapes
:         &А*
T0
═
0gradients/model_2/conv3/conv3/Relu_grad/ReluGradReluGrad:gradients/model_2/conv3/MaxPool2D/MaxPool_grad/MaxPoolGradmodel_2/conv3/conv3/Relu*
T0*0
_output_shapes
:         &А
о
0gradients/model_1/conv3/BiasAdd_grad/BiasAddGradBiasAddGrad0gradients/model_1/conv3/conv3/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes	
:А
г
5gradients/model_1/conv3/BiasAdd_grad/tuple/group_depsNoOp1^gradients/model_1/conv3/BiasAdd_grad/BiasAddGrad1^gradients/model_1/conv3/conv3/Relu_grad/ReluGrad
│
=gradients/model_1/conv3/BiasAdd_grad/tuple/control_dependencyIdentity0gradients/model_1/conv3/conv3/Relu_grad/ReluGrad6^gradients/model_1/conv3/BiasAdd_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/model_1/conv3/conv3/Relu_grad/ReluGrad*0
_output_shapes
:         &А
а
?gradients/model_1/conv3/BiasAdd_grad/tuple/control_dependency_1Identity0gradients/model_1/conv3/BiasAdd_grad/BiasAddGrad6^gradients/model_1/conv3/BiasAdd_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/model_1/conv3/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:А
к
.gradients/model/conv3/BiasAdd_grad/BiasAddGradBiasAddGrad.gradients/model/conv3/conv3/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes	
:А
Э
3gradients/model/conv3/BiasAdd_grad/tuple/group_depsNoOp/^gradients/model/conv3/BiasAdd_grad/BiasAddGrad/^gradients/model/conv3/conv3/Relu_grad/ReluGrad
л
;gradients/model/conv3/BiasAdd_grad/tuple/control_dependencyIdentity.gradients/model/conv3/conv3/Relu_grad/ReluGrad4^gradients/model/conv3/BiasAdd_grad/tuple/group_deps*0
_output_shapes
:         &А*
T0*A
_class7
53loc:@gradients/model/conv3/conv3/Relu_grad/ReluGrad
Ш
=gradients/model/conv3/BiasAdd_grad/tuple/control_dependency_1Identity.gradients/model/conv3/BiasAdd_grad/BiasAddGrad4^gradients/model/conv3/BiasAdd_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients/model/conv3/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:А
о
0gradients/model_2/conv3/BiasAdd_grad/BiasAddGradBiasAddGrad0gradients/model_2/conv3/conv3/Relu_grad/ReluGrad*
data_formatNHWC*
_output_shapes	
:А*
T0
г
5gradients/model_2/conv3/BiasAdd_grad/tuple/group_depsNoOp1^gradients/model_2/conv3/BiasAdd_grad/BiasAddGrad1^gradients/model_2/conv3/conv3/Relu_grad/ReluGrad
│
=gradients/model_2/conv3/BiasAdd_grad/tuple/control_dependencyIdentity0gradients/model_2/conv3/conv3/Relu_grad/ReluGrad6^gradients/model_2/conv3/BiasAdd_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/model_2/conv3/conv3/Relu_grad/ReluGrad*0
_output_shapes
:         &А
а
?gradients/model_2/conv3/BiasAdd_grad/tuple/control_dependency_1Identity0gradients/model_2/conv3/BiasAdd_grad/BiasAddGrad6^gradients/model_2/conv3/BiasAdd_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/model_2/conv3/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:А
н
*gradients/model_1/conv3/Conv2D_grad/ShapeNShapeNmodel_1/conv2/MaxPool2D/MaxPoolconv3/weights/read*
T0*
out_type0*
N* 
_output_shapes
::
ї
7gradients/model_1/conv3/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput*gradients/model_1/conv3/Conv2D_grad/ShapeNconv3/weights/read=gradients/model_1/conv3/BiasAdd_grad/tuple/control_dependency*
paddingSAME*/
_output_shapes
:         &@*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
■
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
:@А
▒
4gradients/model_1/conv3/Conv2D_grad/tuple/group_depsNoOp9^gradients/model_1/conv3/Conv2D_grad/Conv2DBackpropFilter8^gradients/model_1/conv3/Conv2D_grad/Conv2DBackpropInput
╛
<gradients/model_1/conv3/Conv2D_grad/tuple/control_dependencyIdentity7gradients/model_1/conv3/Conv2D_grad/Conv2DBackpropInput5^gradients/model_1/conv3/Conv2D_grad/tuple/group_deps*
T0*J
_class@
><loc:@gradients/model_1/conv3/Conv2D_grad/Conv2DBackpropInput*/
_output_shapes
:         &@
║
>gradients/model_1/conv3/Conv2D_grad/tuple/control_dependency_1Identity8gradients/model_1/conv3/Conv2D_grad/Conv2DBackpropFilter5^gradients/model_1/conv3/Conv2D_grad/tuple/group_deps*
T0*K
_classA
?=loc:@gradients/model_1/conv3/Conv2D_grad/Conv2DBackpropFilter*'
_output_shapes
:@А
й
(gradients/model/conv3/Conv2D_grad/ShapeNShapeNmodel/conv2/MaxPool2D/MaxPoolconv3/weights/read*
N* 
_output_shapes
::*
T0*
out_type0
я
5gradients/model/conv3/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput(gradients/model/conv3/Conv2D_grad/ShapeNconv3/weights/read;gradients/model/conv3/BiasAdd_grad/tuple/control_dependency*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*/
_output_shapes
:         &@
Ў
6gradients/model/conv3/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFiltermodel/conv2/MaxPool2D/MaxPool*gradients/model/conv3/Conv2D_grad/ShapeN:1;gradients/model/conv3/BiasAdd_grad/tuple/control_dependency*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*'
_output_shapes
:@А*
	dilations

л
2gradients/model/conv3/Conv2D_grad/tuple/group_depsNoOp7^gradients/model/conv3/Conv2D_grad/Conv2DBackpropFilter6^gradients/model/conv3/Conv2D_grad/Conv2DBackpropInput
╢
:gradients/model/conv3/Conv2D_grad/tuple/control_dependencyIdentity5gradients/model/conv3/Conv2D_grad/Conv2DBackpropInput3^gradients/model/conv3/Conv2D_grad/tuple/group_deps*
T0*H
_class>
<:loc:@gradients/model/conv3/Conv2D_grad/Conv2DBackpropInput*/
_output_shapes
:         &@
▓
<gradients/model/conv3/Conv2D_grad/tuple/control_dependency_1Identity6gradients/model/conv3/Conv2D_grad/Conv2DBackpropFilter3^gradients/model/conv3/Conv2D_grad/tuple/group_deps*
T0*I
_class?
=;loc:@gradients/model/conv3/Conv2D_grad/Conv2DBackpropFilter*'
_output_shapes
:@А
н
*gradients/model_2/conv3/Conv2D_grad/ShapeNShapeNmodel_2/conv2/MaxPool2D/MaxPoolconv3/weights/read*
N* 
_output_shapes
::*
T0*
out_type0
ї
7gradients/model_2/conv3/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput*gradients/model_2/conv3/Conv2D_grad/ShapeNconv3/weights/read=gradients/model_2/conv3/BiasAdd_grad/tuple/control_dependency*/
_output_shapes
:         &@*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME
■
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
:@А
▒
4gradients/model_2/conv3/Conv2D_grad/tuple/group_depsNoOp9^gradients/model_2/conv3/Conv2D_grad/Conv2DBackpropFilter8^gradients/model_2/conv3/Conv2D_grad/Conv2DBackpropInput
╛
<gradients/model_2/conv3/Conv2D_grad/tuple/control_dependencyIdentity7gradients/model_2/conv3/Conv2D_grad/Conv2DBackpropInput5^gradients/model_2/conv3/Conv2D_grad/tuple/group_deps*/
_output_shapes
:         &@*
T0*J
_class@
><loc:@gradients/model_2/conv3/Conv2D_grad/Conv2DBackpropInput
║
>gradients/model_2/conv3/Conv2D_grad/tuple/control_dependency_1Identity8gradients/model_2/conv3/Conv2D_grad/Conv2DBackpropFilter5^gradients/model_2/conv3/Conv2D_grad/tuple/group_deps*
T0*K
_classA
?=loc:@gradients/model_2/conv3/Conv2D_grad/Conv2DBackpropFilter*'
_output_shapes
:@А
═
gradients/AddN_5AddN?gradients/model_1/conv3/BiasAdd_grad/tuple/control_dependency_1=gradients/model/conv3/BiasAdd_grad/tuple/control_dependency_1?gradients/model_2/conv3/BiasAdd_grad/tuple/control_dependency_1*
T0*C
_class9
75loc:@gradients/model_1/conv3/BiasAdd_grad/BiasAddGrad*
N*
_output_shapes	
:А
╬
:gradients/model_1/conv2/MaxPool2D/MaxPool_grad/MaxPoolGradMaxPoolGradmodel_1/conv2/conv2/Relumodel_1/conv2/MaxPool2D/MaxPool<gradients/model_1/conv3/Conv2D_grad/tuple/control_dependency*
ksize
*
paddingSAME*/
_output_shapes
:         K@*
T0*
data_formatNHWC*
strides

╞
8gradients/model/conv2/MaxPool2D/MaxPool_grad/MaxPoolGradMaxPoolGradmodel/conv2/conv2/Relumodel/conv2/MaxPool2D/MaxPool:gradients/model/conv3/Conv2D_grad/tuple/control_dependency*
ksize
*
paddingSAME*/
_output_shapes
:         K@*
T0*
data_formatNHWC*
strides

╬
:gradients/model_2/conv2/MaxPool2D/MaxPool_grad/MaxPoolGradMaxPoolGradmodel_2/conv2/conv2/Relumodel_2/conv2/MaxPool2D/MaxPool<gradients/model_2/conv3/Conv2D_grad/tuple/control_dependency*
T0*
data_formatNHWC*
strides
*
ksize
*
paddingSAME*/
_output_shapes
:         K@
▐
gradients/AddN_6AddN>gradients/model_1/conv3/Conv2D_grad/tuple/control_dependency_1<gradients/model/conv3/Conv2D_grad/tuple/control_dependency_1>gradients/model_2/conv3/Conv2D_grad/tuple/control_dependency_1*
T0*K
_classA
?=loc:@gradients/model_1/conv3/Conv2D_grad/Conv2DBackpropFilter*
N*'
_output_shapes
:@А
╠
0gradients/model_1/conv2/conv2/Relu_grad/ReluGradReluGrad:gradients/model_1/conv2/MaxPool2D/MaxPool_grad/MaxPoolGradmodel_1/conv2/conv2/Relu*/
_output_shapes
:         K@*
T0
╞
.gradients/model/conv2/conv2/Relu_grad/ReluGradReluGrad8gradients/model/conv2/MaxPool2D/MaxPool_grad/MaxPoolGradmodel/conv2/conv2/Relu*
T0*/
_output_shapes
:         K@
╠
0gradients/model_2/conv2/conv2/Relu_grad/ReluGradReluGrad:gradients/model_2/conv2/MaxPool2D/MaxPool_grad/MaxPoolGradmodel_2/conv2/conv2/Relu*
T0*/
_output_shapes
:         K@
н
0gradients/model_1/conv2/BiasAdd_grad/BiasAddGradBiasAddGrad0gradients/model_1/conv2/conv2/Relu_grad/ReluGrad*
data_formatNHWC*
_output_shapes
:@*
T0
г
5gradients/model_1/conv2/BiasAdd_grad/tuple/group_depsNoOp1^gradients/model_1/conv2/BiasAdd_grad/BiasAddGrad1^gradients/model_1/conv2/conv2/Relu_grad/ReluGrad
▓
=gradients/model_1/conv2/BiasAdd_grad/tuple/control_dependencyIdentity0gradients/model_1/conv2/conv2/Relu_grad/ReluGrad6^gradients/model_1/conv2/BiasAdd_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/model_1/conv2/conv2/Relu_grad/ReluGrad*/
_output_shapes
:         K@
Я
?gradients/model_1/conv2/BiasAdd_grad/tuple/control_dependency_1Identity0gradients/model_1/conv2/BiasAdd_grad/BiasAddGrad6^gradients/model_1/conv2/BiasAdd_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/model_1/conv2/BiasAdd_grad/BiasAddGrad*
_output_shapes
:@
й
.gradients/model/conv2/BiasAdd_grad/BiasAddGradBiasAddGrad.gradients/model/conv2/conv2/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes
:@
Э
3gradients/model/conv2/BiasAdd_grad/tuple/group_depsNoOp/^gradients/model/conv2/BiasAdd_grad/BiasAddGrad/^gradients/model/conv2/conv2/Relu_grad/ReluGrad
к
;gradients/model/conv2/BiasAdd_grad/tuple/control_dependencyIdentity.gradients/model/conv2/conv2/Relu_grad/ReluGrad4^gradients/model/conv2/BiasAdd_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients/model/conv2/conv2/Relu_grad/ReluGrad*/
_output_shapes
:         K@
Ч
=gradients/model/conv2/BiasAdd_grad/tuple/control_dependency_1Identity.gradients/model/conv2/BiasAdd_grad/BiasAddGrad4^gradients/model/conv2/BiasAdd_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients/model/conv2/BiasAdd_grad/BiasAddGrad*
_output_shapes
:@
н
0gradients/model_2/conv2/BiasAdd_grad/BiasAddGradBiasAddGrad0gradients/model_2/conv2/conv2/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes
:@
г
5gradients/model_2/conv2/BiasAdd_grad/tuple/group_depsNoOp1^gradients/model_2/conv2/BiasAdd_grad/BiasAddGrad1^gradients/model_2/conv2/conv2/Relu_grad/ReluGrad
▓
=gradients/model_2/conv2/BiasAdd_grad/tuple/control_dependencyIdentity0gradients/model_2/conv2/conv2/Relu_grad/ReluGrad6^gradients/model_2/conv2/BiasAdd_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/model_2/conv2/conv2/Relu_grad/ReluGrad*/
_output_shapes
:         K@
Я
?gradients/model_2/conv2/BiasAdd_grad/tuple/control_dependency_1Identity0gradients/model_2/conv2/BiasAdd_grad/BiasAddGrad6^gradients/model_2/conv2/BiasAdd_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/model_2/conv2/BiasAdd_grad/BiasAddGrad*
_output_shapes
:@
н
*gradients/model_1/conv2/Conv2D_grad/ShapeNShapeNmodel_1/conv1/MaxPool2D/MaxPoolconv2/weights/read*
N* 
_output_shapes
::*
T0*
out_type0
ї
7gradients/model_1/conv2/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput*gradients/model_1/conv2/Conv2D_grad/ShapeNconv2/weights/read=gradients/model_1/conv2/BiasAdd_grad/tuple/control_dependency*/
_output_shapes
:         K *
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME
¤
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
▒
4gradients/model_1/conv2/Conv2D_grad/tuple/group_depsNoOp9^gradients/model_1/conv2/Conv2D_grad/Conv2DBackpropFilter8^gradients/model_1/conv2/Conv2D_grad/Conv2DBackpropInput
╛
<gradients/model_1/conv2/Conv2D_grad/tuple/control_dependencyIdentity7gradients/model_1/conv2/Conv2D_grad/Conv2DBackpropInput5^gradients/model_1/conv2/Conv2D_grad/tuple/group_deps*
T0*J
_class@
><loc:@gradients/model_1/conv2/Conv2D_grad/Conv2DBackpropInput*/
_output_shapes
:         K 
╣
>gradients/model_1/conv2/Conv2D_grad/tuple/control_dependency_1Identity8gradients/model_1/conv2/Conv2D_grad/Conv2DBackpropFilter5^gradients/model_1/conv2/Conv2D_grad/tuple/group_deps*
T0*K
_classA
?=loc:@gradients/model_1/conv2/Conv2D_grad/Conv2DBackpropFilter*&
_output_shapes
: @
й
(gradients/model/conv2/Conv2D_grad/ShapeNShapeNmodel/conv1/MaxPool2D/MaxPoolconv2/weights/read*
T0*
out_type0*
N* 
_output_shapes
::
я
5gradients/model/conv2/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput(gradients/model/conv2/Conv2D_grad/ShapeNconv2/weights/read;gradients/model/conv2/BiasAdd_grad/tuple/control_dependency*
paddingSAME*/
_output_shapes
:         K *
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
ї
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
л
2gradients/model/conv2/Conv2D_grad/tuple/group_depsNoOp7^gradients/model/conv2/Conv2D_grad/Conv2DBackpropFilter6^gradients/model/conv2/Conv2D_grad/Conv2DBackpropInput
╢
:gradients/model/conv2/Conv2D_grad/tuple/control_dependencyIdentity5gradients/model/conv2/Conv2D_grad/Conv2DBackpropInput3^gradients/model/conv2/Conv2D_grad/tuple/group_deps*/
_output_shapes
:         K *
T0*H
_class>
<:loc:@gradients/model/conv2/Conv2D_grad/Conv2DBackpropInput
▒
<gradients/model/conv2/Conv2D_grad/tuple/control_dependency_1Identity6gradients/model/conv2/Conv2D_grad/Conv2DBackpropFilter3^gradients/model/conv2/Conv2D_grad/tuple/group_deps*&
_output_shapes
: @*
T0*I
_class?
=;loc:@gradients/model/conv2/Conv2D_grad/Conv2DBackpropFilter
н
*gradients/model_2/conv2/Conv2D_grad/ShapeNShapeNmodel_2/conv1/MaxPool2D/MaxPoolconv2/weights/read*
T0*
out_type0*
N* 
_output_shapes
::
ї
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
:         K 
¤
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
▒
4gradients/model_2/conv2/Conv2D_grad/tuple/group_depsNoOp9^gradients/model_2/conv2/Conv2D_grad/Conv2DBackpropFilter8^gradients/model_2/conv2/Conv2D_grad/Conv2DBackpropInput
╛
<gradients/model_2/conv2/Conv2D_grad/tuple/control_dependencyIdentity7gradients/model_2/conv2/Conv2D_grad/Conv2DBackpropInput5^gradients/model_2/conv2/Conv2D_grad/tuple/group_deps*
T0*J
_class@
><loc:@gradients/model_2/conv2/Conv2D_grad/Conv2DBackpropInput*/
_output_shapes
:         K 
╣
>gradients/model_2/conv2/Conv2D_grad/tuple/control_dependency_1Identity8gradients/model_2/conv2/Conv2D_grad/Conv2DBackpropFilter5^gradients/model_2/conv2/Conv2D_grad/tuple/group_deps*&
_output_shapes
: @*
T0*K
_classA
?=loc:@gradients/model_2/conv2/Conv2D_grad/Conv2DBackpropFilter
╠
gradients/AddN_7AddN?gradients/model_1/conv2/BiasAdd_grad/tuple/control_dependency_1=gradients/model/conv2/BiasAdd_grad/tuple/control_dependency_1?gradients/model_2/conv2/BiasAdd_grad/tuple/control_dependency_1*
N*
_output_shapes
:@*
T0*C
_class9
75loc:@gradients/model_1/conv2/BiasAdd_grad/BiasAddGrad
╧
:gradients/model_1/conv1/MaxPool2D/MaxPool_grad/MaxPoolGradMaxPoolGradmodel_1/conv1/conv1/Relumodel_1/conv1/MaxPool2D/MaxPool<gradients/model_1/conv2/Conv2D_grad/tuple/control_dependency*
ksize
*
paddingSAME*0
_output_shapes
:         2Ц *
T0*
data_formatNHWC*
strides

╟
8gradients/model/conv1/MaxPool2D/MaxPool_grad/MaxPoolGradMaxPoolGradmodel/conv1/conv1/Relumodel/conv1/MaxPool2D/MaxPool:gradients/model/conv2/Conv2D_grad/tuple/control_dependency*0
_output_shapes
:         2Ц *
T0*
data_formatNHWC*
strides
*
ksize
*
paddingSAME
╧
:gradients/model_2/conv1/MaxPool2D/MaxPool_grad/MaxPoolGradMaxPoolGradmodel_2/conv1/conv1/Relumodel_2/conv1/MaxPool2D/MaxPool<gradients/model_2/conv2/Conv2D_grad/tuple/control_dependency*
T0*
data_formatNHWC*
strides
*
ksize
*
paddingSAME*0
_output_shapes
:         2Ц 
▌
gradients/AddN_8AddN>gradients/model_1/conv2/Conv2D_grad/tuple/control_dependency_1<gradients/model/conv2/Conv2D_grad/tuple/control_dependency_1>gradients/model_2/conv2/Conv2D_grad/tuple/control_dependency_1*
T0*K
_classA
?=loc:@gradients/model_1/conv2/Conv2D_grad/Conv2DBackpropFilter*
N*&
_output_shapes
: @
═
0gradients/model_1/conv1/conv1/Relu_grad/ReluGradReluGrad:gradients/model_1/conv1/MaxPool2D/MaxPool_grad/MaxPoolGradmodel_1/conv1/conv1/Relu*
T0*0
_output_shapes
:         2Ц 
╟
.gradients/model/conv1/conv1/Relu_grad/ReluGradReluGrad8gradients/model/conv1/MaxPool2D/MaxPool_grad/MaxPoolGradmodel/conv1/conv1/Relu*
T0*0
_output_shapes
:         2Ц 
═
0gradients/model_2/conv1/conv1/Relu_grad/ReluGradReluGrad:gradients/model_2/conv1/MaxPool2D/MaxPool_grad/MaxPoolGradmodel_2/conv1/conv1/Relu*0
_output_shapes
:         2Ц *
T0
н
0gradients/model_1/conv1/BiasAdd_grad/BiasAddGradBiasAddGrad0gradients/model_1/conv1/conv1/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes
: 
г
5gradients/model_1/conv1/BiasAdd_grad/tuple/group_depsNoOp1^gradients/model_1/conv1/BiasAdd_grad/BiasAddGrad1^gradients/model_1/conv1/conv1/Relu_grad/ReluGrad
│
=gradients/model_1/conv1/BiasAdd_grad/tuple/control_dependencyIdentity0gradients/model_1/conv1/conv1/Relu_grad/ReluGrad6^gradients/model_1/conv1/BiasAdd_grad/tuple/group_deps*0
_output_shapes
:         2Ц *
T0*C
_class9
75loc:@gradients/model_1/conv1/conv1/Relu_grad/ReluGrad
Я
?gradients/model_1/conv1/BiasAdd_grad/tuple/control_dependency_1Identity0gradients/model_1/conv1/BiasAdd_grad/BiasAddGrad6^gradients/model_1/conv1/BiasAdd_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/model_1/conv1/BiasAdd_grad/BiasAddGrad*
_output_shapes
: 
й
.gradients/model/conv1/BiasAdd_grad/BiasAddGradBiasAddGrad.gradients/model/conv1/conv1/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes
: 
Э
3gradients/model/conv1/BiasAdd_grad/tuple/group_depsNoOp/^gradients/model/conv1/BiasAdd_grad/BiasAddGrad/^gradients/model/conv1/conv1/Relu_grad/ReluGrad
л
;gradients/model/conv1/BiasAdd_grad/tuple/control_dependencyIdentity.gradients/model/conv1/conv1/Relu_grad/ReluGrad4^gradients/model/conv1/BiasAdd_grad/tuple/group_deps*0
_output_shapes
:         2Ц *
T0*A
_class7
53loc:@gradients/model/conv1/conv1/Relu_grad/ReluGrad
Ч
=gradients/model/conv1/BiasAdd_grad/tuple/control_dependency_1Identity.gradients/model/conv1/BiasAdd_grad/BiasAddGrad4^gradients/model/conv1/BiasAdd_grad/tuple/group_deps*
_output_shapes
: *
T0*A
_class7
53loc:@gradients/model/conv1/BiasAdd_grad/BiasAddGrad
н
0gradients/model_2/conv1/BiasAdd_grad/BiasAddGradBiasAddGrad0gradients/model_2/conv1/conv1/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes
: 
г
5gradients/model_2/conv1/BiasAdd_grad/tuple/group_depsNoOp1^gradients/model_2/conv1/BiasAdd_grad/BiasAddGrad1^gradients/model_2/conv1/conv1/Relu_grad/ReluGrad
│
=gradients/model_2/conv1/BiasAdd_grad/tuple/control_dependencyIdentity0gradients/model_2/conv1/conv1/Relu_grad/ReluGrad6^gradients/model_2/conv1/BiasAdd_grad/tuple/group_deps*0
_output_shapes
:         2Ц *
T0*C
_class9
75loc:@gradients/model_2/conv1/conv1/Relu_grad/ReluGrad
Я
?gradients/model_2/conv1/BiasAdd_grad/tuple/control_dependency_1Identity0gradients/model_2/conv1/BiasAdd_grad/BiasAddGrad6^gradients/model_2/conv1/BiasAdd_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/model_2/conv1/BiasAdd_grad/BiasAddGrad*
_output_shapes
: 
Ь
*gradients/model_1/conv1/Conv2D_grad/ShapeNShapeNpositive_inputconv1/weights/read*
T0*
out_type0*
N* 
_output_shapes
::
Ў
7gradients/model_1/conv1/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput*gradients/model_1/conv1/Conv2D_grad/ShapeNconv1/weights/read=gradients/model_1/conv1/BiasAdd_grad/tuple/control_dependency*
paddingSAME*0
_output_shapes
:         2Ц*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
ь
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
▒
4gradients/model_1/conv1/Conv2D_grad/tuple/group_depsNoOp9^gradients/model_1/conv1/Conv2D_grad/Conv2DBackpropFilter8^gradients/model_1/conv1/Conv2D_grad/Conv2DBackpropInput
┐
<gradients/model_1/conv1/Conv2D_grad/tuple/control_dependencyIdentity7gradients/model_1/conv1/Conv2D_grad/Conv2DBackpropInput5^gradients/model_1/conv1/Conv2D_grad/tuple/group_deps*
T0*J
_class@
><loc:@gradients/model_1/conv1/Conv2D_grad/Conv2DBackpropInput*0
_output_shapes
:         2Ц
╣
>gradients/model_1/conv1/Conv2D_grad/tuple/control_dependency_1Identity8gradients/model_1/conv1/Conv2D_grad/Conv2DBackpropFilter5^gradients/model_1/conv1/Conv2D_grad/tuple/group_deps*&
_output_shapes
: *
T0*K
_classA
?=loc:@gradients/model_1/conv1/Conv2D_grad/Conv2DBackpropFilter
Ш
(gradients/model/conv1/Conv2D_grad/ShapeNShapeNanchor_inputconv1/weights/read*
T0*
out_type0*
N* 
_output_shapes
::
Ё
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
:         2Ц
ф
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
л
2gradients/model/conv1/Conv2D_grad/tuple/group_depsNoOp7^gradients/model/conv1/Conv2D_grad/Conv2DBackpropFilter6^gradients/model/conv1/Conv2D_grad/Conv2DBackpropInput
╖
:gradients/model/conv1/Conv2D_grad/tuple/control_dependencyIdentity5gradients/model/conv1/Conv2D_grad/Conv2DBackpropInput3^gradients/model/conv1/Conv2D_grad/tuple/group_deps*0
_output_shapes
:         2Ц*
T0*H
_class>
<:loc:@gradients/model/conv1/Conv2D_grad/Conv2DBackpropInput
▒
<gradients/model/conv1/Conv2D_grad/tuple/control_dependency_1Identity6gradients/model/conv1/Conv2D_grad/Conv2DBackpropFilter3^gradients/model/conv1/Conv2D_grad/tuple/group_deps*&
_output_shapes
: *
T0*I
_class?
=;loc:@gradients/model/conv1/Conv2D_grad/Conv2DBackpropFilter
Ь
*gradients/model_2/conv1/Conv2D_grad/ShapeNShapeNnegative_inputconv1/weights/read*
T0*
out_type0*
N* 
_output_shapes
::
Ў
7gradients/model_2/conv1/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput*gradients/model_2/conv1/Conv2D_grad/ShapeNconv1/weights/read=gradients/model_2/conv1/BiasAdd_grad/tuple/control_dependency*0
_output_shapes
:         2Ц*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME
ь
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
▒
4gradients/model_2/conv1/Conv2D_grad/tuple/group_depsNoOp9^gradients/model_2/conv1/Conv2D_grad/Conv2DBackpropFilter8^gradients/model_2/conv1/Conv2D_grad/Conv2DBackpropInput
┐
<gradients/model_2/conv1/Conv2D_grad/tuple/control_dependencyIdentity7gradients/model_2/conv1/Conv2D_grad/Conv2DBackpropInput5^gradients/model_2/conv1/Conv2D_grad/tuple/group_deps*0
_output_shapes
:         2Ц*
T0*J
_class@
><loc:@gradients/model_2/conv1/Conv2D_grad/Conv2DBackpropInput
╣
>gradients/model_2/conv1/Conv2D_grad/tuple/control_dependency_1Identity8gradients/model_2/conv1/Conv2D_grad/Conv2DBackpropFilter5^gradients/model_2/conv1/Conv2D_grad/tuple/group_deps*&
_output_shapes
: *
T0*K
_classA
?=loc:@gradients/model_2/conv1/Conv2D_grad/Conv2DBackpropFilter
╠
gradients/AddN_9AddN?gradients/model_1/conv1/BiasAdd_grad/tuple/control_dependency_1=gradients/model/conv1/BiasAdd_grad/tuple/control_dependency_1?gradients/model_2/conv1/BiasAdd_grad/tuple/control_dependency_1*
T0*C
_class9
75loc:@gradients/model_1/conv1/BiasAdd_grad/BiasAddGrad*
N*
_output_shapes
: 
▐
gradients/AddN_10AddN>gradients/model_1/conv1/Conv2D_grad/tuple/control_dependency_1<gradients/model/conv1/Conv2D_grad/tuple/control_dependency_1>gradients/model_2/conv1/Conv2D_grad/tuple/control_dependency_1*
T0*K
_classA
?=loc:@gradients/model_1/conv1/Conv2D_grad/Conv2DBackpropFilter*
N*&
_output_shapes
: 
│
8conv1/weights/Momentum/Initializer/zeros/shape_as_tensorConst* 
_class
loc:@conv1/weights*%
valueB"             *
dtype0*
_output_shapes
:
Х
.conv1/weights/Momentum/Initializer/zeros/ConstConst* 
_class
loc:@conv1/weights*
valueB
 *    *
dtype0*
_output_shapes
: 
 
(conv1/weights/Momentum/Initializer/zerosFill8conv1/weights/Momentum/Initializer/zeros/shape_as_tensor.conv1/weights/Momentum/Initializer/zeros/Const*
T0* 
_class
loc:@conv1/weights*

index_type0*&
_output_shapes
: 
╝
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
conv1/weights/Momentum/AssignAssignconv1/weights/Momentum(conv1/weights/Momentum/Initializer/zeros*
T0* 
_class
loc:@conv1/weights*
validate_shape(*&
_output_shapes
: *
use_locking(
Т
conv1/weights/Momentum/readIdentityconv1/weights/Momentum*
T0* 
_class
loc:@conv1/weights*&
_output_shapes
: 
Х
'conv1/biases/Momentum/Initializer/zerosConst*
_class
loc:@conv1/biases*
valueB *    *
dtype0*
_output_shapes
: 
в
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
╒
conv1/biases/Momentum/AssignAssignconv1/biases/Momentum'conv1/biases/Momentum/Initializer/zeros*
T0*
_class
loc:@conv1/biases*
validate_shape(*
_output_shapes
: *
use_locking(
Г
conv1/biases/Momentum/readIdentityconv1/biases/Momentum*
T0*
_class
loc:@conv1/biases*
_output_shapes
: 
│
8conv2/weights/Momentum/Initializer/zeros/shape_as_tensorConst* 
_class
loc:@conv2/weights*%
valueB"          @   *
dtype0*
_output_shapes
:
Х
.conv2/weights/Momentum/Initializer/zeros/ConstConst* 
_class
loc:@conv2/weights*
valueB
 *    *
dtype0*
_output_shapes
: 
 
(conv2/weights/Momentum/Initializer/zerosFill8conv2/weights/Momentum/Initializer/zeros/shape_as_tensor.conv2/weights/Momentum/Initializer/zeros/Const*
T0* 
_class
loc:@conv2/weights*

index_type0*&
_output_shapes
: @
╝
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
Т
conv2/weights/Momentum/readIdentityconv2/weights/Momentum*
T0* 
_class
loc:@conv2/weights*&
_output_shapes
: @
Х
'conv2/biases/Momentum/Initializer/zerosConst*
_class
loc:@conv2/biases*
valueB@*    *
dtype0*
_output_shapes
:@
в
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
╒
conv2/biases/Momentum/AssignAssignconv2/biases/Momentum'conv2/biases/Momentum/Initializer/zeros*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*
_class
loc:@conv2/biases
Г
conv2/biases/Momentum/readIdentityconv2/biases/Momentum*
T0*
_class
loc:@conv2/biases*
_output_shapes
:@
│
8conv3/weights/Momentum/Initializer/zeros/shape_as_tensorConst* 
_class
loc:@conv3/weights*%
valueB"      @   А   *
dtype0*
_output_shapes
:
Х
.conv3/weights/Momentum/Initializer/zeros/ConstConst* 
_class
loc:@conv3/weights*
valueB
 *    *
dtype0*
_output_shapes
: 
А
(conv3/weights/Momentum/Initializer/zerosFill8conv3/weights/Momentum/Initializer/zeros/shape_as_tensor.conv3/weights/Momentum/Initializer/zeros/Const*'
_output_shapes
:@А*
T0* 
_class
loc:@conv3/weights*

index_type0
╛
conv3/weights/Momentum
VariableV2*
dtype0*'
_output_shapes
:@А*
shared_name * 
_class
loc:@conv3/weights*
	container *
shape:@А
ц
conv3/weights/Momentum/AssignAssignconv3/weights/Momentum(conv3/weights/Momentum/Initializer/zeros*
use_locking(*
T0* 
_class
loc:@conv3/weights*
validate_shape(*'
_output_shapes
:@А
У
conv3/weights/Momentum/readIdentityconv3/weights/Momentum*
T0* 
_class
loc:@conv3/weights*'
_output_shapes
:@А
Ч
'conv3/biases/Momentum/Initializer/zerosConst*
_class
loc:@conv3/biases*
valueBА*    *
dtype0*
_output_shapes	
:А
д
conv3/biases/Momentum
VariableV2*
	container *
shape:А*
dtype0*
_output_shapes	
:А*
shared_name *
_class
loc:@conv3/biases
╓
conv3/biases/Momentum/AssignAssignconv3/biases/Momentum'conv3/biases/Momentum/Initializer/zeros*
validate_shape(*
_output_shapes	
:А*
use_locking(*
T0*
_class
loc:@conv3/biases
Д
conv3/biases/Momentum/readIdentityconv3/biases/Momentum*
T0*
_class
loc:@conv3/biases*
_output_shapes	
:А
│
8conv4/weights/Momentum/Initializer/zeros/shape_as_tensorConst* 
_class
loc:@conv4/weights*%
valueB"      А      *
dtype0*
_output_shapes
:
Х
.conv4/weights/Momentum/Initializer/zeros/ConstConst* 
_class
loc:@conv4/weights*
valueB
 *    *
dtype0*
_output_shapes
: 
Б
(conv4/weights/Momentum/Initializer/zerosFill8conv4/weights/Momentum/Initializer/zeros/shape_as_tensor.conv4/weights/Momentum/Initializer/zeros/Const*
T0* 
_class
loc:@conv4/weights*

index_type0*(
_output_shapes
:АА
└
conv4/weights/Momentum
VariableV2* 
_class
loc:@conv4/weights*
	container *
shape:АА*
dtype0*(
_output_shapes
:АА*
shared_name 
ч
conv4/weights/Momentum/AssignAssignconv4/weights/Momentum(conv4/weights/Momentum/Initializer/zeros*
validate_shape(*(
_output_shapes
:АА*
use_locking(*
T0* 
_class
loc:@conv4/weights
Ф
conv4/weights/Momentum/readIdentityconv4/weights/Momentum*
T0* 
_class
loc:@conv4/weights*(
_output_shapes
:АА
Ч
'conv4/biases/Momentum/Initializer/zerosConst*
_class
loc:@conv4/biases*
valueBА*    *
dtype0*
_output_shapes	
:А
д
conv4/biases/Momentum
VariableV2*
	container *
shape:А*
dtype0*
_output_shapes	
:А*
shared_name *
_class
loc:@conv4/biases
╓
conv4/biases/Momentum/AssignAssignconv4/biases/Momentum'conv4/biases/Momentum/Initializer/zeros*
T0*
_class
loc:@conv4/biases*
validate_shape(*
_output_shapes	
:А*
use_locking(
Д
conv4/biases/Momentum/readIdentityconv4/biases/Momentum*
T0*
_class
loc:@conv4/biases*
_output_shapes	
:А
│
8conv5/weights/Momentum/Initializer/zeros/shape_as_tensorConst* 
_class
loc:@conv5/weights*%
valueB"            *
dtype0*
_output_shapes
:
Х
.conv5/weights/Momentum/Initializer/zeros/ConstConst* 
_class
loc:@conv5/weights*
valueB
 *    *
dtype0*
_output_shapes
: 
А
(conv5/weights/Momentum/Initializer/zerosFill8conv5/weights/Momentum/Initializer/zeros/shape_as_tensor.conv5/weights/Momentum/Initializer/zeros/Const*'
_output_shapes
:А*
T0* 
_class
loc:@conv5/weights*

index_type0
╛
conv5/weights/Momentum
VariableV2*
dtype0*'
_output_shapes
:А*
shared_name * 
_class
loc:@conv5/weights*
	container *
shape:А
ц
conv5/weights/Momentum/AssignAssignconv5/weights/Momentum(conv5/weights/Momentum/Initializer/zeros*
validate_shape(*'
_output_shapes
:А*
use_locking(*
T0* 
_class
loc:@conv5/weights
У
conv5/weights/Momentum/readIdentityconv5/weights/Momentum*
T0* 
_class
loc:@conv5/weights*'
_output_shapes
:А
Х
'conv5/biases/Momentum/Initializer/zerosConst*
_class
loc:@conv5/biases*
valueB*    *
dtype0*
_output_shapes
:
в
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
╒
conv5/biases/Momentum/AssignAssignconv5/biases/Momentum'conv5/biases/Momentum/Initializer/zeros*
use_locking(*
T0*
_class
loc:@conv5/biases*
validate_shape(*
_output_shapes
:
Г
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
╫#<
V
Momentum/momentumConst*
valueB
 *дp}?*
dtype0*
_output_shapes
: 
Ы
+Momentum/update_conv1/weights/ApplyMomentumApplyMomentumconv1/weightsconv1/weights/MomentumMomentum/learning_rategradients/AddN_10Momentum/momentum*
use_locking( *
T0* 
_class
loc:@conv1/weights*
use_nesterov(*&
_output_shapes
: 
К
*Momentum/update_conv1/biases/ApplyMomentumApplyMomentumconv1/biasesconv1/biases/MomentumMomentum/learning_rategradients/AddN_9Momentum/momentum*
T0*
_class
loc:@conv1/biases*
use_nesterov(*
_output_shapes
: *
use_locking( 
Ъ
+Momentum/update_conv2/weights/ApplyMomentumApplyMomentumconv2/weightsconv2/weights/MomentumMomentum/learning_rategradients/AddN_8Momentum/momentum*
use_locking( *
T0* 
_class
loc:@conv2/weights*
use_nesterov(*&
_output_shapes
: @
К
*Momentum/update_conv2/biases/ApplyMomentumApplyMomentumconv2/biasesconv2/biases/MomentumMomentum/learning_rategradients/AddN_7Momentum/momentum*
use_locking( *
T0*
_class
loc:@conv2/biases*
use_nesterov(*
_output_shapes
:@
Ы
+Momentum/update_conv3/weights/ApplyMomentumApplyMomentumconv3/weightsconv3/weights/MomentumMomentum/learning_rategradients/AddN_6Momentum/momentum*
T0* 
_class
loc:@conv3/weights*
use_nesterov(*'
_output_shapes
:@А*
use_locking( 
Л
*Momentum/update_conv3/biases/ApplyMomentumApplyMomentumconv3/biasesconv3/biases/MomentumMomentum/learning_rategradients/AddN_5Momentum/momentum*
T0*
_class
loc:@conv3/biases*
use_nesterov(*
_output_shapes	
:А*
use_locking( 
Ь
+Momentum/update_conv4/weights/ApplyMomentumApplyMomentumconv4/weightsconv4/weights/MomentumMomentum/learning_rategradients/AddN_4Momentum/momentum*
use_locking( *
T0* 
_class
loc:@conv4/weights*
use_nesterov(*(
_output_shapes
:АА
Л
*Momentum/update_conv4/biases/ApplyMomentumApplyMomentumconv4/biasesconv4/biases/MomentumMomentum/learning_rategradients/AddN_3Momentum/momentum*
T0*
_class
loc:@conv4/biases*
use_nesterov(*
_output_shapes	
:А*
use_locking( 
Ы
+Momentum/update_conv5/weights/ApplyMomentumApplyMomentumconv5/weightsconv5/weights/MomentumMomentum/learning_rategradients/AddN_2Momentum/momentum*
use_locking( *
T0* 
_class
loc:@conv5/weights*
use_nesterov(*'
_output_shapes
:А
К
*Momentum/update_conv5/biases/ApplyMomentumApplyMomentumconv5/biasesconv5/biases/MomentumMomentum/learning_rategradients/AddN_1Momentum/momentum*
use_locking( *
T0*
_class
loc:@conv5/biases*
use_nesterov(*
_output_shapes
:
▐
Momentum/updateNoOp+^Momentum/update_conv1/biases/ApplyMomentum,^Momentum/update_conv1/weights/ApplyMomentum+^Momentum/update_conv2/biases/ApplyMomentum,^Momentum/update_conv2/weights/ApplyMomentum+^Momentum/update_conv3/biases/ApplyMomentum,^Momentum/update_conv3/weights/ApplyMomentum+^Momentum/update_conv4/biases/ApplyMomentum,^Momentum/update_conv4/weights/ApplyMomentum+^Momentum/update_conv5/biases/ApplyMomentum,^Momentum/update_conv5/weights/ApplyMomentum

Momentum/valueConst^Momentum/update*
value	B :*
_class
loc:@Variable*
dtype0*
_output_shapes
: 
А
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
ш
save/SaveV2/tensor_namesConst*Ы
valueСBОBVariableBconv1/biasesBconv1/biases/MomentumBconv1/weightsBconv1/weights/MomentumBconv2/biasesBconv2/biases/MomentumBconv2/weightsBconv2/weights/MomentumBconv3/biasesBconv3/biases/MomentumBconv3/weightsBconv3/weights/MomentumBconv4/biasesBconv4/biases/MomentumBconv4/weightsBconv4/weights/MomentumBconv5/biasesBconv5/biases/MomentumBconv5/weightsBconv5/weights/Momentum*
dtype0*
_output_shapes
:
Н
save/SaveV2/shape_and_slicesConst*=
value4B2B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:
Д
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
·
save/RestoreV2/tensor_namesConst"/device:CPU:0*
dtype0*
_output_shapes
:*Ы
valueСBОBVariableBconv1/biasesBconv1/biases/MomentumBconv1/weightsBconv1/weights/MomentumBconv2/biasesBconv2/biases/MomentumBconv2/weightsBconv2/weights/MomentumBconv3/biasesBconv3/biases/MomentumBconv3/weightsBconv3/weights/MomentumBconv4/biasesBconv4/biases/MomentumBconv4/weightsBconv4/weights/MomentumBconv5/biasesBconv5/biases/MomentumBconv5/weightsBconv5/weights/Momentum
Я
save/RestoreV2/shape_and_slicesConst"/device:CPU:0*=
value4B2B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:
Г
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices"/device:CPU:0*h
_output_shapesV
T:::::::::::::::::::::*#
dtypes
2
Ц
save/AssignAssignVariablesave/RestoreV2*
use_locking(*
T0*
_class
loc:@Variable*
validate_shape(*
_output_shapes
: 
ж
save/Assign_1Assignconv1/biasessave/RestoreV2:1*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*
_class
loc:@conv1/biases
п
save/Assign_2Assignconv1/biases/Momentumsave/RestoreV2:2*
T0*
_class
loc:@conv1/biases*
validate_shape(*
_output_shapes
: *
use_locking(
┤
save/Assign_3Assignconv1/weightssave/RestoreV2:3*
use_locking(*
T0* 
_class
loc:@conv1/weights*
validate_shape(*&
_output_shapes
: 
╜
save/Assign_4Assignconv1/weights/Momentumsave/RestoreV2:4*
validate_shape(*&
_output_shapes
: *
use_locking(*
T0* 
_class
loc:@conv1/weights
ж
save/Assign_5Assignconv2/biasessave/RestoreV2:5*
use_locking(*
T0*
_class
loc:@conv2/biases*
validate_shape(*
_output_shapes
:@
п
save/Assign_6Assignconv2/biases/Momentumsave/RestoreV2:6*
T0*
_class
loc:@conv2/biases*
validate_shape(*
_output_shapes
:@*
use_locking(
┤
save/Assign_7Assignconv2/weightssave/RestoreV2:7*
validate_shape(*&
_output_shapes
: @*
use_locking(*
T0* 
_class
loc:@conv2/weights
╜
save/Assign_8Assignconv2/weights/Momentumsave/RestoreV2:8*
use_locking(*
T0* 
_class
loc:@conv2/weights*
validate_shape(*&
_output_shapes
: @
з
save/Assign_9Assignconv3/biasessave/RestoreV2:9*
T0*
_class
loc:@conv3/biases*
validate_shape(*
_output_shapes	
:А*
use_locking(
▓
save/Assign_10Assignconv3/biases/Momentumsave/RestoreV2:10*
use_locking(*
T0*
_class
loc:@conv3/biases*
validate_shape(*
_output_shapes	
:А
╖
save/Assign_11Assignconv3/weightssave/RestoreV2:11*
T0* 
_class
loc:@conv3/weights*
validate_shape(*'
_output_shapes
:@А*
use_locking(
└
save/Assign_12Assignconv3/weights/Momentumsave/RestoreV2:12*
use_locking(*
T0* 
_class
loc:@conv3/weights*
validate_shape(*'
_output_shapes
:@А
й
save/Assign_13Assignconv4/biasessave/RestoreV2:13*
validate_shape(*
_output_shapes	
:А*
use_locking(*
T0*
_class
loc:@conv4/biases
▓
save/Assign_14Assignconv4/biases/Momentumsave/RestoreV2:14*
validate_shape(*
_output_shapes	
:А*
use_locking(*
T0*
_class
loc:@conv4/biases
╕
save/Assign_15Assignconv4/weightssave/RestoreV2:15*
use_locking(*
T0* 
_class
loc:@conv4/weights*
validate_shape(*(
_output_shapes
:АА
┴
save/Assign_16Assignconv4/weights/Momentumsave/RestoreV2:16*
T0* 
_class
loc:@conv4/weights*
validate_shape(*(
_output_shapes
:АА*
use_locking(
и
save/Assign_17Assignconv5/biasessave/RestoreV2:17*
use_locking(*
T0*
_class
loc:@conv5/biases*
validate_shape(*
_output_shapes
:
▒
save/Assign_18Assignconv5/biases/Momentumsave/RestoreV2:18*
use_locking(*
T0*
_class
loc:@conv5/biases*
validate_shape(*
_output_shapes
:
╖
save/Assign_19Assignconv5/weightssave/RestoreV2:19*
use_locking(*
T0* 
_class
loc:@conv5/weights*
validate_shape(*'
_output_shapes
:А
└
save/Assign_20Assignconv5/weights/Momentumsave/RestoreV2:20*
validate_shape(*'
_output_shapes
:А*
use_locking(*
T0* 
_class
loc:@conv5/weights
ё
save/restore_allNoOp^save/Assign^save/Assign_1^save/Assign_10^save/Assign_11^save/Assign_12^save/Assign_13^save/Assign_14^save/Assign_15^save/Assign_16^save/Assign_17^save/Assign_18^save/Assign_19^save/Assign_2^save/Assign_20^save/Assign_3^save/Assign_4^save/Assign_5^save/Assign_6^save/Assign_7^save/Assign_8^save/Assign_9
║
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
Ї
Merge/MergeSummaryMergeSummarysteplossconv1/weights_1conv1/biases_1conv2/weights_1conv2/biases_1conv3/weights_1conv3/biases_1conv4/weights_1conv4/biases_1conv5/weights_1conv5/biases_1*
N*
_output_shapes
: "куу▒А/     D7	Я╧) л╫AJє▐
╖,О,
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
2	АР
░
ApplyMomentum
var"TА
accum"TА
lr"T	
grad"T
momentum"T
out"TА" 
Ttype:
2	"
use_lockingbool( "
use_nesterovbool( 
x
Assign
ref"TА

value"T

output_ref"TА"	
Ttype"
validate_shapebool("
use_lockingbool(Ш
s
	AssignAdd
ref"TА

value"T

output_ref"TА" 
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
Т
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
С
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
╘
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

2	Р
Н
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
2	Р
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
Н
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
2	И
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
list(type)(0И
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
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
Ў
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
М
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
ref"dtypeА"
shapeshape"
dtypetype"
	containerstring "
shared_namestring И
&
	ZerosLike
x"T
y"T"	
Ttype*1.13.12b'v1.13.0-rc2-5-g6612da8951'єМ
Б
anchor_inputPlaceholder*
dtype0*0
_output_shapes
:         2Ц*%
shape:         2Ц
Г
positive_inputPlaceholder*
dtype0*0
_output_shapes
:         2Ц*%
shape:         2Ц
Г
negative_inputPlaceholder*%
shape:         2Ц*
dtype0*0
_output_shapes
:         2Ц
й
.conv1/weights/Initializer/random_uniform/shapeConst* 
_class
loc:@conv1/weights*%
valueB"             *
dtype0*
_output_shapes
:
У
,conv1/weights/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: * 
_class
loc:@conv1/weights*
valueB
 *мEr╜
У
,conv1/weights/Initializer/random_uniform/maxConst* 
_class
loc:@conv1/weights*
valueB
 *мEr=*
dtype0*
_output_shapes
: 
Ё
6conv1/weights/Initializer/random_uniform/RandomUniformRandomUniform.conv1/weights/Initializer/random_uniform/shape*
dtype0*&
_output_shapes
: *

seed *
T0* 
_class
loc:@conv1/weights*
seed2 
╥
,conv1/weights/Initializer/random_uniform/subSub,conv1/weights/Initializer/random_uniform/max,conv1/weights/Initializer/random_uniform/min*
_output_shapes
: *
T0* 
_class
loc:@conv1/weights
ь
,conv1/weights/Initializer/random_uniform/mulMul6conv1/weights/Initializer/random_uniform/RandomUniform,conv1/weights/Initializer/random_uniform/sub*
T0* 
_class
loc:@conv1/weights*&
_output_shapes
: 
▐
(conv1/weights/Initializer/random_uniformAdd,conv1/weights/Initializer/random_uniform/mul,conv1/weights/Initializer/random_uniform/min*
T0* 
_class
loc:@conv1/weights*&
_output_shapes
: 
│
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
╙
conv1/weights/AssignAssignconv1/weights(conv1/weights/Initializer/random_uniform*
use_locking(*
T0* 
_class
loc:@conv1/weights*
validate_shape(*&
_output_shapes
: 
А
conv1/weights/readIdentityconv1/weights*
T0* 
_class
loc:@conv1/weights*&
_output_shapes
: 
М
conv1/biases/Initializer/zerosConst*
dtype0*
_output_shapes
: *
_class
loc:@conv1/biases*
valueB *    
Щ
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
║
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
ч
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
:         2Ц 
Ч
model/conv1/BiasAddBiasAddmodel/conv1/Conv2Dconv1/biases/read*
T0*
data_formatNHWC*0
_output_shapes
:         2Ц 
n
model/conv1/conv1/ReluRelumodel/conv1/BiasAdd*
T0*0
_output_shapes
:         2Ц 
╠
model/conv1/MaxPool2D/MaxPoolMaxPoolmodel/conv1/conv1/Relu*
T0*
data_formatNHWC*
strides
*
ksize
*
paddingSAME*/
_output_shapes
:         K 
й
.conv2/weights/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:* 
_class
loc:@conv2/weights*%
valueB"          @   
У
,conv2/weights/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: * 
_class
loc:@conv2/weights*
valueB
 *═╠L╜
У
,conv2/weights/Initializer/random_uniform/maxConst* 
_class
loc:@conv2/weights*
valueB
 *═╠L=*
dtype0*
_output_shapes
: 
Ё
6conv2/weights/Initializer/random_uniform/RandomUniformRandomUniform.conv2/weights/Initializer/random_uniform/shape*
dtype0*&
_output_shapes
: @*

seed *
T0* 
_class
loc:@conv2/weights*
seed2 
╥
,conv2/weights/Initializer/random_uniform/subSub,conv2/weights/Initializer/random_uniform/max,conv2/weights/Initializer/random_uniform/min*
T0* 
_class
loc:@conv2/weights*
_output_shapes
: 
ь
,conv2/weights/Initializer/random_uniform/mulMul6conv2/weights/Initializer/random_uniform/RandomUniform,conv2/weights/Initializer/random_uniform/sub*&
_output_shapes
: @*
T0* 
_class
loc:@conv2/weights
▐
(conv2/weights/Initializer/random_uniformAdd,conv2/weights/Initializer/random_uniform/mul,conv2/weights/Initializer/random_uniform/min*&
_output_shapes
: @*
T0* 
_class
loc:@conv2/weights
│
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
╙
conv2/weights/AssignAssignconv2/weights(conv2/weights/Initializer/random_uniform*
T0* 
_class
loc:@conv2/weights*
validate_shape(*&
_output_shapes
: @*
use_locking(
А
conv2/weights/readIdentityconv2/weights*
T0* 
_class
loc:@conv2/weights*&
_output_shapes
: @
М
conv2/biases/Initializer/zerosConst*
dtype0*
_output_shapes
:@*
_class
loc:@conv2/biases*
valueB@*    
Щ
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
║
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
ў
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
:         K@
Ц
model/conv2/BiasAddBiasAddmodel/conv2/Conv2Dconv2/biases/read*
data_formatNHWC*/
_output_shapes
:         K@*
T0
m
model/conv2/conv2/ReluRelumodel/conv2/BiasAdd*
T0*/
_output_shapes
:         K@
╠
model/conv2/MaxPool2D/MaxPoolMaxPoolmodel/conv2/conv2/Relu*
data_formatNHWC*
strides
*
ksize
*
paddingSAME*/
_output_shapes
:         &@*
T0
й
.conv3/weights/Initializer/random_uniform/shapeConst* 
_class
loc:@conv3/weights*%
valueB"      @   А   *
dtype0*
_output_shapes
:
У
,conv3/weights/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: * 
_class
loc:@conv3/weights*
valueB
 *я[q╜
У
,conv3/weights/Initializer/random_uniform/maxConst* 
_class
loc:@conv3/weights*
valueB
 *я[q=*
dtype0*
_output_shapes
: 
ё
6conv3/weights/Initializer/random_uniform/RandomUniformRandomUniform.conv3/weights/Initializer/random_uniform/shape*
dtype0*'
_output_shapes
:@А*

seed *
T0* 
_class
loc:@conv3/weights*
seed2 
╥
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
:@А
▀
(conv3/weights/Initializer/random_uniformAdd,conv3/weights/Initializer/random_uniform/mul,conv3/weights/Initializer/random_uniform/min*
T0* 
_class
loc:@conv3/weights*'
_output_shapes
:@А
╡
conv3/weights
VariableV2* 
_class
loc:@conv3/weights*
	container *
shape:@А*
dtype0*'
_output_shapes
:@А*
shared_name 
╘
conv3/weights/AssignAssignconv3/weights(conv3/weights/Initializer/random_uniform*
use_locking(*
T0* 
_class
loc:@conv3/weights*
validate_shape(*'
_output_shapes
:@А
Б
conv3/weights/readIdentityconv3/weights*
T0* 
_class
loc:@conv3/weights*'
_output_shapes
:@А
О
conv3/biases/Initializer/zerosConst*
_class
loc:@conv3/biases*
valueBА*    *
dtype0*
_output_shapes	
:А
Ы
conv3/biases
VariableV2*
	container *
shape:А*
dtype0*
_output_shapes	
:А*
shared_name *
_class
loc:@conv3/biases
╗
conv3/biases/AssignAssignconv3/biasesconv3/biases/Initializer/zeros*
validate_shape(*
_output_shapes	
:А*
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
:А
j
model/conv3/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
°
model/conv3/Conv2DConv2Dmodel/conv2/MaxPool2D/MaxPoolconv3/weights/read*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*0
_output_shapes
:         &А*
	dilations

Ч
model/conv3/BiasAddBiasAddmodel/conv3/Conv2Dconv3/biases/read*
T0*
data_formatNHWC*0
_output_shapes
:         &А
n
model/conv3/conv3/ReluRelumodel/conv3/BiasAdd*
T0*0
_output_shapes
:         &А
═
model/conv3/MaxPool2D/MaxPoolMaxPoolmodel/conv3/conv3/Relu*
ksize
*
paddingSAME*0
_output_shapes
:         А*
T0*
data_formatNHWC*
strides

й
.conv4/weights/Initializer/random_uniform/shapeConst* 
_class
loc:@conv4/weights*%
valueB"      А      *
dtype0*
_output_shapes
:
У
,conv4/weights/Initializer/random_uniform/minConst* 
_class
loc:@conv4/weights*
valueB
 *   ╛*
dtype0*
_output_shapes
: 
У
,conv4/weights/Initializer/random_uniform/maxConst* 
_class
loc:@conv4/weights*
valueB
 *   >*
dtype0*
_output_shapes
: 
Є
6conv4/weights/Initializer/random_uniform/RandomUniformRandomUniform.conv4/weights/Initializer/random_uniform/shape*
T0* 
_class
loc:@conv4/weights*
seed2 *
dtype0*(
_output_shapes
:АА*

seed 
╥
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
:АА
р
(conv4/weights/Initializer/random_uniformAdd,conv4/weights/Initializer/random_uniform/mul,conv4/weights/Initializer/random_uniform/min*
T0* 
_class
loc:@conv4/weights*(
_output_shapes
:АА
╖
conv4/weights
VariableV2*
dtype0*(
_output_shapes
:АА*
shared_name * 
_class
loc:@conv4/weights*
	container *
shape:АА
╒
conv4/weights/AssignAssignconv4/weights(conv4/weights/Initializer/random_uniform*
T0* 
_class
loc:@conv4/weights*
validate_shape(*(
_output_shapes
:АА*
use_locking(
В
conv4/weights/readIdentityconv4/weights*
T0* 
_class
loc:@conv4/weights*(
_output_shapes
:АА
О
conv4/biases/Initializer/zerosConst*
dtype0*
_output_shapes	
:А*
_class
loc:@conv4/biases*
valueBА*    
Ы
conv4/biases
VariableV2*
shared_name *
_class
loc:@conv4/biases*
	container *
shape:А*
dtype0*
_output_shapes	
:А
╗
conv4/biases/AssignAssignconv4/biasesconv4/biases/Initializer/zeros*
use_locking(*
T0*
_class
loc:@conv4/biases*
validate_shape(*
_output_shapes	
:А
r
conv4/biases/readIdentityconv4/biases*
T0*
_class
loc:@conv4/biases*
_output_shapes	
:А
j
model/conv4/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
°
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
:         А
Ч
model/conv4/BiasAddBiasAddmodel/conv4/Conv2Dconv4/biases/read*
T0*
data_formatNHWC*0
_output_shapes
:         А
n
model/conv4/conv4/ReluRelumodel/conv4/BiasAdd*0
_output_shapes
:         А*
T0
═
model/conv4/MaxPool2D/MaxPoolMaxPoolmodel/conv4/conv4/Relu*
T0*
data_formatNHWC*
strides
*
ksize
*
paddingSAME*0
_output_shapes
:         
А
й
.conv5/weights/Initializer/random_uniform/shapeConst* 
_class
loc:@conv5/weights*%
valueB"            *
dtype0*
_output_shapes
:
У
,conv5/weights/Initializer/random_uniform/minConst* 
_class
loc:@conv5/weights*
valueB
 *┴╓╛*
dtype0*
_output_shapes
: 
У
,conv5/weights/Initializer/random_uniform/maxConst* 
_class
loc:@conv5/weights*
valueB
 *┴╓>*
dtype0*
_output_shapes
: 
ё
6conv5/weights/Initializer/random_uniform/RandomUniformRandomUniform.conv5/weights/Initializer/random_uniform/shape*
seed2 *
dtype0*'
_output_shapes
:А*

seed *
T0* 
_class
loc:@conv5/weights
╥
,conv5/weights/Initializer/random_uniform/subSub,conv5/weights/Initializer/random_uniform/max,conv5/weights/Initializer/random_uniform/min*
T0* 
_class
loc:@conv5/weights*
_output_shapes
: 
э
,conv5/weights/Initializer/random_uniform/mulMul6conv5/weights/Initializer/random_uniform/RandomUniform,conv5/weights/Initializer/random_uniform/sub*'
_output_shapes
:А*
T0* 
_class
loc:@conv5/weights
▀
(conv5/weights/Initializer/random_uniformAdd,conv5/weights/Initializer/random_uniform/mul,conv5/weights/Initializer/random_uniform/min*
T0* 
_class
loc:@conv5/weights*'
_output_shapes
:А
╡
conv5/weights
VariableV2* 
_class
loc:@conv5/weights*
	container *
shape:А*
dtype0*'
_output_shapes
:А*
shared_name 
╘
conv5/weights/AssignAssignconv5/weights(conv5/weights/Initializer/random_uniform*
T0* 
_class
loc:@conv5/weights*
validate_shape(*'
_output_shapes
:А*
use_locking(
Б
conv5/weights/readIdentityconv5/weights*
T0* 
_class
loc:@conv5/weights*'
_output_shapes
:А
М
conv5/biases/Initializer/zerosConst*
dtype0*
_output_shapes
:*
_class
loc:@conv5/biases*
valueB*    
Щ
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
║
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
dtype0*
_output_shapes
:*
valueB"      
ў
model/conv5/Conv2DConv2Dmodel/conv4/MaxPool2D/MaxPoolconv5/weights/read*/
_output_shapes
:         
*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME
Ц
model/conv5/BiasAddBiasAddmodel/conv5/Conv2Dconv5/biases/read*
data_formatNHWC*/
_output_shapes
:         
*
T0
╔
model/conv5/MaxPool2D/MaxPoolMaxPoolmodel/conv5/BiasAdd*/
_output_shapes
:         *
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
ч
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
         *
dtype0*
_output_shapes
: 
▒
#model/Flatten/flatten/Reshape/shapePack#model/Flatten/flatten/strided_slice%model/Flatten/flatten/Reshape/shape/1*
T0*

axis *
N*
_output_shapes
:
н
model/Flatten/flatten/ReshapeReshapemodel/conv5/MaxPool2D/MaxPool#model/Flatten/flatten/Reshape/shape*
T0*
Tshape0*(
_output_shapes
:         Ш
l
model_1/conv1/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
ы
model_1/conv1/Conv2DConv2Dpositive_inputconv1/weights/read*0
_output_shapes
:         2Ц *
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME
Ы
model_1/conv1/BiasAddBiasAddmodel_1/conv1/Conv2Dconv1/biases/read*
T0*
data_formatNHWC*0
_output_shapes
:         2Ц 
r
model_1/conv1/conv1/ReluRelumodel_1/conv1/BiasAdd*0
_output_shapes
:         2Ц *
T0
╨
model_1/conv1/MaxPool2D/MaxPoolMaxPoolmodel_1/conv1/conv1/Relu*
strides
*
data_formatNHWC*
ksize
*
paddingSAME*/
_output_shapes
:         K *
T0
l
model_1/conv2/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
√
model_1/conv2/Conv2DConv2Dmodel_1/conv1/MaxPool2D/MaxPoolconv2/weights/read*
paddingSAME*/
_output_shapes
:         K@*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
Ъ
model_1/conv2/BiasAddBiasAddmodel_1/conv2/Conv2Dconv2/biases/read*
data_formatNHWC*/
_output_shapes
:         K@*
T0
q
model_1/conv2/conv2/ReluRelumodel_1/conv2/BiasAdd*
T0*/
_output_shapes
:         K@
╨
model_1/conv2/MaxPool2D/MaxPoolMaxPoolmodel_1/conv2/conv2/Relu*
ksize
*
paddingSAME*/
_output_shapes
:         &@*
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
№
model_1/conv3/Conv2DConv2Dmodel_1/conv2/MaxPool2D/MaxPoolconv3/weights/read*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*0
_output_shapes
:         &А*
	dilations
*
T0
Ы
model_1/conv3/BiasAddBiasAddmodel_1/conv3/Conv2Dconv3/biases/read*
T0*
data_formatNHWC*0
_output_shapes
:         &А
r
model_1/conv3/conv3/ReluRelumodel_1/conv3/BiasAdd*
T0*0
_output_shapes
:         &А
╤
model_1/conv3/MaxPool2D/MaxPoolMaxPoolmodel_1/conv3/conv3/Relu*
T0*
strides
*
data_formatNHWC*
ksize
*
paddingSAME*0
_output_shapes
:         А
l
model_1/conv4/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
№
model_1/conv4/Conv2DConv2Dmodel_1/conv3/MaxPool2D/MaxPoolconv4/weights/read*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*0
_output_shapes
:         А*
	dilations

Ы
model_1/conv4/BiasAddBiasAddmodel_1/conv4/Conv2Dconv4/biases/read*
data_formatNHWC*0
_output_shapes
:         А*
T0
r
model_1/conv4/conv4/ReluRelumodel_1/conv4/BiasAdd*
T0*0
_output_shapes
:         А
╤
model_1/conv4/MaxPool2D/MaxPoolMaxPoolmodel_1/conv4/conv4/Relu*
T0*
strides
*
data_formatNHWC*
ksize
*
paddingSAME*0
_output_shapes
:         
А
l
model_1/conv5/dilation_rateConst*
dtype0*
_output_shapes
:*
valueB"      
√
model_1/conv5/Conv2DConv2Dmodel_1/conv4/MaxPool2D/MaxPoolconv5/weights/read*/
_output_shapes
:         
*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME
Ъ
model_1/conv5/BiasAddBiasAddmodel_1/conv5/Conv2Dconv5/biases/read*
T0*
data_formatNHWC*/
_output_shapes
:         

═
model_1/conv5/MaxPool2D/MaxPoolMaxPoolmodel_1/conv5/BiasAdd*
T0*
strides
*
data_formatNHWC*
ksize
*
paddingSAME*/
_output_shapes
:         
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
ё
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
'model_1/Flatten/flatten/Reshape/shape/1Const*
valueB :
         *
dtype0*
_output_shapes
: 
╖
%model_1/Flatten/flatten/Reshape/shapePack%model_1/Flatten/flatten/strided_slice'model_1/Flatten/flatten/Reshape/shape/1*
T0*

axis *
N*
_output_shapes
:
│
model_1/Flatten/flatten/ReshapeReshapemodel_1/conv5/MaxPool2D/MaxPool%model_1/Flatten/flatten/Reshape/shape*
T0*
Tshape0*(
_output_shapes
:         Ш
l
model_2/conv1/dilation_rateConst*
dtype0*
_output_shapes
:*
valueB"      
ы
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
:         2Ц 
Ы
model_2/conv1/BiasAddBiasAddmodel_2/conv1/Conv2Dconv1/biases/read*
T0*
data_formatNHWC*0
_output_shapes
:         2Ц 
r
model_2/conv1/conv1/ReluRelumodel_2/conv1/BiasAdd*
T0*0
_output_shapes
:         2Ц 
╨
model_2/conv1/MaxPool2D/MaxPoolMaxPoolmodel_2/conv1/conv1/Relu*/
_output_shapes
:         K *
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
√
model_2/conv2/Conv2DConv2Dmodel_2/conv1/MaxPool2D/MaxPoolconv2/weights/read*
paddingSAME*/
_output_shapes
:         K@*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(
Ъ
model_2/conv2/BiasAddBiasAddmodel_2/conv2/Conv2Dconv2/biases/read*
data_formatNHWC*/
_output_shapes
:         K@*
T0
q
model_2/conv2/conv2/ReluRelumodel_2/conv2/BiasAdd*
T0*/
_output_shapes
:         K@
╨
model_2/conv2/MaxPool2D/MaxPoolMaxPoolmodel_2/conv2/conv2/Relu*/
_output_shapes
:         &@*
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
№
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
:         &А
Ы
model_2/conv3/BiasAddBiasAddmodel_2/conv3/Conv2Dconv3/biases/read*
T0*
data_formatNHWC*0
_output_shapes
:         &А
r
model_2/conv3/conv3/ReluRelumodel_2/conv3/BiasAdd*
T0*0
_output_shapes
:         &А
╤
model_2/conv3/MaxPool2D/MaxPoolMaxPoolmodel_2/conv3/conv3/Relu*
ksize
*
paddingSAME*0
_output_shapes
:         А*
T0*
strides
*
data_formatNHWC
l
model_2/conv4/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
№
model_2/conv4/Conv2DConv2Dmodel_2/conv3/MaxPool2D/MaxPoolconv4/weights/read*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*0
_output_shapes
:         А*
	dilations
*
T0
Ы
model_2/conv4/BiasAddBiasAddmodel_2/conv4/Conv2Dconv4/biases/read*
T0*
data_formatNHWC*0
_output_shapes
:         А
r
model_2/conv4/conv4/ReluRelumodel_2/conv4/BiasAdd*
T0*0
_output_shapes
:         А
╤
model_2/conv4/MaxPool2D/MaxPoolMaxPoolmodel_2/conv4/conv4/Relu*
ksize
*
paddingSAME*0
_output_shapes
:         
А*
T0*
data_formatNHWC*
strides

l
model_2/conv5/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
√
model_2/conv5/Conv2DConv2Dmodel_2/conv4/MaxPool2D/MaxPoolconv5/weights/read*/
_output_shapes
:         
*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME
Ъ
model_2/conv5/BiasAddBiasAddmodel_2/conv5/Conv2Dconv5/biases/read*
T0*
data_formatNHWC*/
_output_shapes
:         

═
model_2/conv5/MaxPool2D/MaxPoolMaxPoolmodel_2/conv5/BiasAdd*
T0*
data_formatNHWC*
strides
*
ksize
*
paddingSAME*/
_output_shapes
:         
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
'model_2/Flatten/flatten/Reshape/shape/1Const*
valueB :
         *
dtype0*
_output_shapes
: 
╖
%model_2/Flatten/flatten/Reshape/shapePack%model_2/Flatten/flatten/strided_slice'model_2/Flatten/flatten/Reshape/shape/1*
T0*

axis *
N*
_output_shapes
:
│
model_2/Flatten/flatten/ReshapeReshapemodel_2/conv5/MaxPool2D/MaxPool%model_2/Flatten/flatten/Reshape/shape*
T0*
Tshape0*(
_output_shapes
:         Ш

mulMulmodel_1/Flatten/flatten/Reshapemodel_2/Flatten/flatten/Reshape*
T0*(
_output_shapes
:         Ш
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
:         
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
:         Ш
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
:         *

Tidx0*
	keep_dims( 
A
SqrtSqrtSum_1*#
_output_shapes
:         *
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
:         Ш
Y
Sum_2/reduction_indicesConst*
value	B :*
dtype0*
_output_shapes
: 
w
Sum_2SumPow_1Sum_2/reduction_indices*#
_output_shapes
:         *

Tidx0*
	keep_dims( *
T0
C
Sqrt_1SqrtSum_2*
T0*#
_output_shapes
:         
H
mul_1MulSqrt_1Sqrt*
T0*#
_output_shapes
:         
L
truedivRealDivSummul_1*
T0*#
_output_shapes
:         
}
subSubmodel/Flatten/flatten/Reshapemodel_1/Flatten/flatten/Reshape*(
_output_shapes
:         Ш*
T0
L
Pow_2/yConst*
valueB
 *   @*
dtype0*
_output_shapes
: 
M
Pow_2PowsubPow_2/y*(
_output_shapes
:         Ш*
T0
Y
Sum_3/reduction_indicesConst*
value	B :*
dtype0*
_output_shapes
: 
{
Sum_3SumPow_2Sum_3/reduction_indices*'
_output_shapes
:         *

Tidx0*
	keep_dims(*
T0
G
Sqrt_2SqrtSum_3*'
_output_shapes
:         *
T0

sub_1Submodel/Flatten/flatten/Reshapemodel_2/Flatten/flatten/Reshape*(
_output_shapes
:         Ш*
T0
L
Pow_3/yConst*
valueB
 *   @*
dtype0*
_output_shapes
: 
O
Pow_3Powsub_1Pow_3/y*
T0*(
_output_shapes
:         Ш
Y
Sum_4/reduction_indicesConst*
value	B :*
dtype0*
_output_shapes
: 
{
Sum_4SumPow_3Sum_4/reduction_indices*
T0*'
_output_shapes
:         *

Tidx0*
	keep_dims(
G
Sqrt_3SqrtSum_4*'
_output_shapes
:         *
T0
N
sub_2SubSqrt_2Sqrt_3*
T0*'
_output_shapes
:         
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
:         
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
:         *
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
shape: *
shared_name *
dtype0*
	container *
_output_shapes
: 
в
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
 *  А?*
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
Р
gradients/Mean_grad/ReshapeReshapegradients/Fill!gradients/Mean_grad/Reshape/shape*
_output_shapes

:*
T0*
Tshape0
`
gradients/Mean_grad/ShapeShapeMaximum*
_output_shapes
:*
T0*
out_type0
Ь
gradients/Mean_grad/TileTilegradients/Mean_grad/Reshapegradients/Mean_grad/Shape*

Tmultiples0*
T0*'
_output_shapes
:         
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
Ц
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
Ъ
gradients/Mean_grad/Prod_1Prodgradients/Mean_grad/Shape_2gradients/Mean_grad/Const_1*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
_
gradients/Mean_grad/Maximum/yConst*
dtype0*
_output_shapes
: *
value	B :
В
gradients/Mean_grad/MaximumMaximumgradients/Mean_grad/Prod_1gradients/Mean_grad/Maximum/y*
T0*
_output_shapes
: 
А
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
М
gradients/Mean_grad/truedivRealDivgradients/Mean_grad/Tilegradients/Mean_grad/Cast*
T0*'
_output_shapes
:         
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
м
gradients/Maximum_grad/zerosFillgradients/Maximum_grad/Shape_2"gradients/Maximum_grad/zeros/Const*'
_output_shapes
:         *
T0*

index_type0
u
#gradients/Maximum_grad/GreaterEqualGreaterEqualadd	Maximum/y*
T0*'
_output_shapes
:         
└
,gradients/Maximum_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/Maximum_grad/Shapegradients/Maximum_grad/Shape_1*
T0*2
_output_shapes 
:         :         
╣
gradients/Maximum_grad/SelectSelect#gradients/Maximum_grad/GreaterEqualgradients/Mean_grad/truedivgradients/Maximum_grad/zeros*
T0*'
_output_shapes
:         
╗
gradients/Maximum_grad/Select_1Select#gradients/Maximum_grad/GreaterEqualgradients/Maximum_grad/zerosgradients/Mean_grad/truediv*
T0*'
_output_shapes
:         
о
gradients/Maximum_grad/SumSumgradients/Maximum_grad/Select,gradients/Maximum_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
г
gradients/Maximum_grad/ReshapeReshapegradients/Maximum_grad/Sumgradients/Maximum_grad/Shape*'
_output_shapes
:         *
T0*
Tshape0
┤
gradients/Maximum_grad/Sum_1Sumgradients/Maximum_grad/Select_1.gradients/Maximum_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
Ш
 gradients/Maximum_grad/Reshape_1Reshapegradients/Maximum_grad/Sum_1gradients/Maximum_grad/Shape_1*
_output_shapes
: *
T0*
Tshape0
s
'gradients/Maximum_grad/tuple/group_depsNoOp^gradients/Maximum_grad/Reshape!^gradients/Maximum_grad/Reshape_1
ъ
/gradients/Maximum_grad/tuple/control_dependencyIdentitygradients/Maximum_grad/Reshape(^gradients/Maximum_grad/tuple/group_deps*'
_output_shapes
:         *
T0*1
_class'
%#loc:@gradients/Maximum_grad/Reshape
▀
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
┤
(gradients/add_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_grad/Shapegradients/add_grad/Shape_1*
T0*2
_output_shapes 
:         :         
╕
gradients/add_grad/SumSum/gradients/Maximum_grad/tuple/control_dependency(gradients/add_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
Ч
gradients/add_grad/ReshapeReshapegradients/add_grad/Sumgradients/add_grad/Shape*'
_output_shapes
:         *
T0*
Tshape0
╝
gradients/add_grad/Sum_1Sum/gradients/Maximum_grad/tuple/control_dependency*gradients/add_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
М
gradients/add_grad/Reshape_1Reshapegradients/add_grad/Sum_1gradients/add_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
g
#gradients/add_grad/tuple/group_depsNoOp^gradients/add_grad/Reshape^gradients/add_grad/Reshape_1
┌
+gradients/add_grad/tuple/control_dependencyIdentitygradients/add_grad/Reshape$^gradients/add_grad/tuple/group_deps*'
_output_shapes
:         *
T0*-
_class#
!loc:@gradients/add_grad/Reshape
╧
-gradients/add_grad/tuple/control_dependency_1Identitygradients/add_grad/Reshape_1$^gradients/add_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/add_grad/Reshape_1*
_output_shapes
: 
`
gradients/sub_2_grad/ShapeShapeSqrt_2*
T0*
out_type0*
_output_shapes
:
b
gradients/sub_2_grad/Shape_1ShapeSqrt_3*
T0*
out_type0*
_output_shapes
:
║
*gradients/sub_2_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/sub_2_grad/Shapegradients/sub_2_grad/Shape_1*2
_output_shapes 
:         :         *
T0
╕
gradients/sub_2_grad/SumSum+gradients/add_grad/tuple/control_dependency*gradients/sub_2_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
Э
gradients/sub_2_grad/ReshapeReshapegradients/sub_2_grad/Sumgradients/sub_2_grad/Shape*
T0*
Tshape0*'
_output_shapes
:         
╝
gradients/sub_2_grad/Sum_1Sum+gradients/add_grad/tuple/control_dependency,gradients/sub_2_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
^
gradients/sub_2_grad/NegNeggradients/sub_2_grad/Sum_1*
T0*
_output_shapes
:
б
gradients/sub_2_grad/Reshape_1Reshapegradients/sub_2_grad/Neggradients/sub_2_grad/Shape_1*'
_output_shapes
:         *
T0*
Tshape0
m
%gradients/sub_2_grad/tuple/group_depsNoOp^gradients/sub_2_grad/Reshape^gradients/sub_2_grad/Reshape_1
т
-gradients/sub_2_grad/tuple/control_dependencyIdentitygradients/sub_2_grad/Reshape&^gradients/sub_2_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/sub_2_grad/Reshape*'
_output_shapes
:         
ш
/gradients/sub_2_grad/tuple/control_dependency_1Identitygradients/sub_2_grad/Reshape_1&^gradients/sub_2_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/sub_2_grad/Reshape_1*'
_output_shapes
:         
У
gradients/Sqrt_2_grad/SqrtGradSqrtGradSqrt_2-gradients/sub_2_grad/tuple/control_dependency*'
_output_shapes
:         *
T0
Х
gradients/Sqrt_3_grad/SqrtGradSqrtGradSqrt_3/gradients/sub_2_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:         
_
gradients/Sum_3_grad/ShapeShapePow_2*
T0*
out_type0*
_output_shapes
:
К
gradients/Sum_3_grad/SizeConst*-
_class#
!loc:@gradients/Sum_3_grad/Shape*
value	B :*
dtype0*
_output_shapes
: 
г
gradients/Sum_3_grad/addAddSum_3/reduction_indicesgradients/Sum_3_grad/Size*
T0*-
_class#
!loc:@gradients/Sum_3_grad/Shape*
_output_shapes
: 
й
gradients/Sum_3_grad/modFloorModgradients/Sum_3_grad/addgradients/Sum_3_grad/Size*
_output_shapes
: *
T0*-
_class#
!loc:@gradients/Sum_3_grad/Shape
О
gradients/Sum_3_grad/Shape_1Const*
dtype0*
_output_shapes
: *-
_class#
!loc:@gradients/Sum_3_grad/Shape*
valueB 
С
 gradients/Sum_3_grad/range/startConst*-
_class#
!loc:@gradients/Sum_3_grad/Shape*
value	B : *
dtype0*
_output_shapes
: 
С
 gradients/Sum_3_grad/range/deltaConst*-
_class#
!loc:@gradients/Sum_3_grad/Shape*
value	B :*
dtype0*
_output_shapes
: 
┘
gradients/Sum_3_grad/rangeRange gradients/Sum_3_grad/range/startgradients/Sum_3_grad/Size gradients/Sum_3_grad/range/delta*-
_class#
!loc:@gradients/Sum_3_grad/Shape*
_output_shapes
:*

Tidx0
Р
gradients/Sum_3_grad/Fill/valueConst*-
_class#
!loc:@gradients/Sum_3_grad/Shape*
value	B :*
dtype0*
_output_shapes
: 
┬
gradients/Sum_3_grad/FillFillgradients/Sum_3_grad/Shape_1gradients/Sum_3_grad/Fill/value*
_output_shapes
: *
T0*-
_class#
!loc:@gradients/Sum_3_grad/Shape*

index_type0
¤
"gradients/Sum_3_grad/DynamicStitchDynamicStitchgradients/Sum_3_grad/rangegradients/Sum_3_grad/modgradients/Sum_3_grad/Shapegradients/Sum_3_grad/Fill*
T0*-
_class#
!loc:@gradients/Sum_3_grad/Shape*
N*
_output_shapes
:
П
gradients/Sum_3_grad/Maximum/yConst*-
_class#
!loc:@gradients/Sum_3_grad/Shape*
value	B :*
dtype0*
_output_shapes
: 
┐
gradients/Sum_3_grad/MaximumMaximum"gradients/Sum_3_grad/DynamicStitchgradients/Sum_3_grad/Maximum/y*
T0*-
_class#
!loc:@gradients/Sum_3_grad/Shape*
_output_shapes
:
╖
gradients/Sum_3_grad/floordivFloorDivgradients/Sum_3_grad/Shapegradients/Sum_3_grad/Maximum*
T0*-
_class#
!loc:@gradients/Sum_3_grad/Shape*
_output_shapes
:
┤
gradients/Sum_3_grad/ReshapeReshapegradients/Sqrt_2_grad/SqrtGrad"gradients/Sum_3_grad/DynamicStitch*0
_output_shapes
:                  *
T0*
Tshape0
г
gradients/Sum_3_grad/TileTilegradients/Sum_3_grad/Reshapegradients/Sum_3_grad/floordiv*(
_output_shapes
:         Ш*

Tmultiples0*
T0
_
gradients/Sum_4_grad/ShapeShapePow_3*
T0*
out_type0*
_output_shapes
:
К
gradients/Sum_4_grad/SizeConst*-
_class#
!loc:@gradients/Sum_4_grad/Shape*
value	B :*
dtype0*
_output_shapes
: 
г
gradients/Sum_4_grad/addAddSum_4/reduction_indicesgradients/Sum_4_grad/Size*
_output_shapes
: *
T0*-
_class#
!loc:@gradients/Sum_4_grad/Shape
й
gradients/Sum_4_grad/modFloorModgradients/Sum_4_grad/addgradients/Sum_4_grad/Size*
T0*-
_class#
!loc:@gradients/Sum_4_grad/Shape*
_output_shapes
: 
О
gradients/Sum_4_grad/Shape_1Const*
dtype0*
_output_shapes
: *-
_class#
!loc:@gradients/Sum_4_grad/Shape*
valueB 
С
 gradients/Sum_4_grad/range/startConst*-
_class#
!loc:@gradients/Sum_4_grad/Shape*
value	B : *
dtype0*
_output_shapes
: 
С
 gradients/Sum_4_grad/range/deltaConst*-
_class#
!loc:@gradients/Sum_4_grad/Shape*
value	B :*
dtype0*
_output_shapes
: 
┘
gradients/Sum_4_grad/rangeRange gradients/Sum_4_grad/range/startgradients/Sum_4_grad/Size gradients/Sum_4_grad/range/delta*
_output_shapes
:*

Tidx0*-
_class#
!loc:@gradients/Sum_4_grad/Shape
Р
gradients/Sum_4_grad/Fill/valueConst*-
_class#
!loc:@gradients/Sum_4_grad/Shape*
value	B :*
dtype0*
_output_shapes
: 
┬
gradients/Sum_4_grad/FillFillgradients/Sum_4_grad/Shape_1gradients/Sum_4_grad/Fill/value*
T0*-
_class#
!loc:@gradients/Sum_4_grad/Shape*

index_type0*
_output_shapes
: 
¤
"gradients/Sum_4_grad/DynamicStitchDynamicStitchgradients/Sum_4_grad/rangegradients/Sum_4_grad/modgradients/Sum_4_grad/Shapegradients/Sum_4_grad/Fill*
T0*-
_class#
!loc:@gradients/Sum_4_grad/Shape*
N*
_output_shapes
:
П
gradients/Sum_4_grad/Maximum/yConst*-
_class#
!loc:@gradients/Sum_4_grad/Shape*
value	B :*
dtype0*
_output_shapes
: 
┐
gradients/Sum_4_grad/MaximumMaximum"gradients/Sum_4_grad/DynamicStitchgradients/Sum_4_grad/Maximum/y*
T0*-
_class#
!loc:@gradients/Sum_4_grad/Shape*
_output_shapes
:
╖
gradients/Sum_4_grad/floordivFloorDivgradients/Sum_4_grad/Shapegradients/Sum_4_grad/Maximum*
T0*-
_class#
!loc:@gradients/Sum_4_grad/Shape*
_output_shapes
:
┤
gradients/Sum_4_grad/ReshapeReshapegradients/Sqrt_3_grad/SqrtGrad"gradients/Sum_4_grad/DynamicStitch*
T0*
Tshape0*0
_output_shapes
:                  
г
gradients/Sum_4_grad/TileTilegradients/Sum_4_grad/Reshapegradients/Sum_4_grad/floordiv*

Tmultiples0*
T0*(
_output_shapes
:         Ш
]
gradients/Pow_2_grad/ShapeShapesub*
T0*
out_type0*
_output_shapes
:
_
gradients/Pow_2_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
║
*gradients/Pow_2_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/Pow_2_grad/Shapegradients/Pow_2_grad/Shape_1*
T0*2
_output_shapes 
:         :         
v
gradients/Pow_2_grad/mulMulgradients/Sum_3_grad/TilePow_2/y*
T0*(
_output_shapes
:         Ш
_
gradients/Pow_2_grad/sub/yConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
e
gradients/Pow_2_grad/subSubPow_2/ygradients/Pow_2_grad/sub/y*
T0*
_output_shapes
: 
q
gradients/Pow_2_grad/PowPowsubgradients/Pow_2_grad/sub*(
_output_shapes
:         Ш*
T0
И
gradients/Pow_2_grad/mul_1Mulgradients/Pow_2_grad/mulgradients/Pow_2_grad/Pow*(
_output_shapes
:         Ш*
T0
з
gradients/Pow_2_grad/SumSumgradients/Pow_2_grad/mul_1*gradients/Pow_2_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
Ю
gradients/Pow_2_grad/ReshapeReshapegradients/Pow_2_grad/Sumgradients/Pow_2_grad/Shape*
T0*
Tshape0*(
_output_shapes
:         Ш
c
gradients/Pow_2_grad/Greater/yConst*
valueB
 *    *
dtype0*
_output_shapes
: 

gradients/Pow_2_grad/GreaterGreatersubgradients/Pow_2_grad/Greater/y*
T0*(
_output_shapes
:         Ш
g
$gradients/Pow_2_grad/ones_like/ShapeShapesub*
T0*
out_type0*
_output_shapes
:
i
$gradients/Pow_2_grad/ones_like/ConstConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
╖
gradients/Pow_2_grad/ones_likeFill$gradients/Pow_2_grad/ones_like/Shape$gradients/Pow_2_grad/ones_like/Const*
T0*

index_type0*(
_output_shapes
:         Ш
Ы
gradients/Pow_2_grad/SelectSelectgradients/Pow_2_grad/Greatersubgradients/Pow_2_grad/ones_like*(
_output_shapes
:         Ш*
T0
o
gradients/Pow_2_grad/LogLoggradients/Pow_2_grad/Select*
T0*(
_output_shapes
:         Ш
d
gradients/Pow_2_grad/zeros_like	ZerosLikesub*
T0*(
_output_shapes
:         Ш
│
gradients/Pow_2_grad/Select_1Selectgradients/Pow_2_grad/Greatergradients/Pow_2_grad/Loggradients/Pow_2_grad/zeros_like*
T0*(
_output_shapes
:         Ш
v
gradients/Pow_2_grad/mul_2Mulgradients/Sum_3_grad/TilePow_2*
T0*(
_output_shapes
:         Ш
П
gradients/Pow_2_grad/mul_3Mulgradients/Pow_2_grad/mul_2gradients/Pow_2_grad/Select_1*
T0*(
_output_shapes
:         Ш
л
gradients/Pow_2_grad/Sum_1Sumgradients/Pow_2_grad/mul_3,gradients/Pow_2_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
Т
gradients/Pow_2_grad/Reshape_1Reshapegradients/Pow_2_grad/Sum_1gradients/Pow_2_grad/Shape_1*
_output_shapes
: *
T0*
Tshape0
m
%gradients/Pow_2_grad/tuple/group_depsNoOp^gradients/Pow_2_grad/Reshape^gradients/Pow_2_grad/Reshape_1
у
-gradients/Pow_2_grad/tuple/control_dependencyIdentitygradients/Pow_2_grad/Reshape&^gradients/Pow_2_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/Pow_2_grad/Reshape*(
_output_shapes
:         Ш
╫
/gradients/Pow_2_grad/tuple/control_dependency_1Identitygradients/Pow_2_grad/Reshape_1&^gradients/Pow_2_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/Pow_2_grad/Reshape_1*
_output_shapes
: 
_
gradients/Pow_3_grad/ShapeShapesub_1*
T0*
out_type0*
_output_shapes
:
_
gradients/Pow_3_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
║
*gradients/Pow_3_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/Pow_3_grad/Shapegradients/Pow_3_grad/Shape_1*
T0*2
_output_shapes 
:         :         
v
gradients/Pow_3_grad/mulMulgradients/Sum_4_grad/TilePow_3/y*
T0*(
_output_shapes
:         Ш
_
gradients/Pow_3_grad/sub/yConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
e
gradients/Pow_3_grad/subSubPow_3/ygradients/Pow_3_grad/sub/y*
_output_shapes
: *
T0
s
gradients/Pow_3_grad/PowPowsub_1gradients/Pow_3_grad/sub*
T0*(
_output_shapes
:         Ш
И
gradients/Pow_3_grad/mul_1Mulgradients/Pow_3_grad/mulgradients/Pow_3_grad/Pow*
T0*(
_output_shapes
:         Ш
з
gradients/Pow_3_grad/SumSumgradients/Pow_3_grad/mul_1*gradients/Pow_3_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
Ю
gradients/Pow_3_grad/ReshapeReshapegradients/Pow_3_grad/Sumgradients/Pow_3_grad/Shape*
T0*
Tshape0*(
_output_shapes
:         Ш
c
gradients/Pow_3_grad/Greater/yConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Б
gradients/Pow_3_grad/GreaterGreatersub_1gradients/Pow_3_grad/Greater/y*(
_output_shapes
:         Ш*
T0
i
$gradients/Pow_3_grad/ones_like/ShapeShapesub_1*
T0*
out_type0*
_output_shapes
:
i
$gradients/Pow_3_grad/ones_like/ConstConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
╖
gradients/Pow_3_grad/ones_likeFill$gradients/Pow_3_grad/ones_like/Shape$gradients/Pow_3_grad/ones_like/Const*
T0*

index_type0*(
_output_shapes
:         Ш
Э
gradients/Pow_3_grad/SelectSelectgradients/Pow_3_grad/Greatersub_1gradients/Pow_3_grad/ones_like*
T0*(
_output_shapes
:         Ш
o
gradients/Pow_3_grad/LogLoggradients/Pow_3_grad/Select*
T0*(
_output_shapes
:         Ш
f
gradients/Pow_3_grad/zeros_like	ZerosLikesub_1*
T0*(
_output_shapes
:         Ш
│
gradients/Pow_3_grad/Select_1Selectgradients/Pow_3_grad/Greatergradients/Pow_3_grad/Loggradients/Pow_3_grad/zeros_like*
T0*(
_output_shapes
:         Ш
v
gradients/Pow_3_grad/mul_2Mulgradients/Sum_4_grad/TilePow_3*(
_output_shapes
:         Ш*
T0
П
gradients/Pow_3_grad/mul_3Mulgradients/Pow_3_grad/mul_2gradients/Pow_3_grad/Select_1*
T0*(
_output_shapes
:         Ш
л
gradients/Pow_3_grad/Sum_1Sumgradients/Pow_3_grad/mul_3,gradients/Pow_3_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
Т
gradients/Pow_3_grad/Reshape_1Reshapegradients/Pow_3_grad/Sum_1gradients/Pow_3_grad/Shape_1*
_output_shapes
: *
T0*
Tshape0
m
%gradients/Pow_3_grad/tuple/group_depsNoOp^gradients/Pow_3_grad/Reshape^gradients/Pow_3_grad/Reshape_1
у
-gradients/Pow_3_grad/tuple/control_dependencyIdentitygradients/Pow_3_grad/Reshape&^gradients/Pow_3_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/Pow_3_grad/Reshape*(
_output_shapes
:         Ш
╫
/gradients/Pow_3_grad/tuple/control_dependency_1Identitygradients/Pow_3_grad/Reshape_1&^gradients/Pow_3_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/Pow_3_grad/Reshape_1*
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
┤
(gradients/sub_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/sub_grad/Shapegradients/sub_grad/Shape_1*
T0*2
_output_shapes 
:         :         
╢
gradients/sub_grad/SumSum-gradients/Pow_2_grad/tuple/control_dependency(gradients/sub_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
Ш
gradients/sub_grad/ReshapeReshapegradients/sub_grad/Sumgradients/sub_grad/Shape*
T0*
Tshape0*(
_output_shapes
:         Ш
║
gradients/sub_grad/Sum_1Sum-gradients/Pow_2_grad/tuple/control_dependency*gradients/sub_grad/BroadcastGradientArgs:1*
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
Ь
gradients/sub_grad/Reshape_1Reshapegradients/sub_grad/Neggradients/sub_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:         Ш
g
#gradients/sub_grad/tuple/group_depsNoOp^gradients/sub_grad/Reshape^gradients/sub_grad/Reshape_1
█
+gradients/sub_grad/tuple/control_dependencyIdentitygradients/sub_grad/Reshape$^gradients/sub_grad/tuple/group_deps*
T0*-
_class#
!loc:@gradients/sub_grad/Reshape*(
_output_shapes
:         Ш
с
-gradients/sub_grad/tuple/control_dependency_1Identitygradients/sub_grad/Reshape_1$^gradients/sub_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/sub_grad/Reshape_1*(
_output_shapes
:         Ш
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
║
*gradients/sub_1_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/sub_1_grad/Shapegradients/sub_1_grad/Shape_1*
T0*2
_output_shapes 
:         :         
║
gradients/sub_1_grad/SumSum-gradients/Pow_3_grad/tuple/control_dependency*gradients/sub_1_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
Ю
gradients/sub_1_grad/ReshapeReshapegradients/sub_1_grad/Sumgradients/sub_1_grad/Shape*(
_output_shapes
:         Ш*
T0*
Tshape0
╛
gradients/sub_1_grad/Sum_1Sum-gradients/Pow_3_grad/tuple/control_dependency,gradients/sub_1_grad/BroadcastGradientArgs:1*
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
в
gradients/sub_1_grad/Reshape_1Reshapegradients/sub_1_grad/Neggradients/sub_1_grad/Shape_1*(
_output_shapes
:         Ш*
T0*
Tshape0
m
%gradients/sub_1_grad/tuple/group_depsNoOp^gradients/sub_1_grad/Reshape^gradients/sub_1_grad/Reshape_1
у
-gradients/sub_1_grad/tuple/control_dependencyIdentitygradients/sub_1_grad/Reshape&^gradients/sub_1_grad/tuple/group_deps*(
_output_shapes
:         Ш*
T0*/
_class%
#!loc:@gradients/sub_1_grad/Reshape
щ
/gradients/sub_1_grad/tuple/control_dependency_1Identitygradients/sub_1_grad/Reshape_1&^gradients/sub_1_grad/tuple/group_deps*(
_output_shapes
:         Ш*
T0*1
_class'
%#loc:@gradients/sub_1_grad/Reshape_1
У
4gradients/model_1/Flatten/flatten/Reshape_grad/ShapeShapemodel_1/conv5/MaxPool2D/MaxPool*
_output_shapes
:*
T0*
out_type0
ю
6gradients/model_1/Flatten/flatten/Reshape_grad/ReshapeReshape-gradients/sub_grad/tuple/control_dependency_14gradients/model_1/Flatten/flatten/Reshape_grad/Shape*/
_output_shapes
:         *
T0*
Tshape0
▌
gradients/AddNAddN+gradients/sub_grad/tuple/control_dependency-gradients/sub_1_grad/tuple/control_dependency*
T0*-
_class#
!loc:@gradients/sub_grad/Reshape*
N*(
_output_shapes
:         Ш
П
2gradients/model/Flatten/flatten/Reshape_grad/ShapeShapemodel/conv5/MaxPool2D/MaxPool*
T0*
out_type0*
_output_shapes
:
╦
4gradients/model/Flatten/flatten/Reshape_grad/ReshapeReshapegradients/AddN2gradients/model/Flatten/flatten/Reshape_grad/Shape*
T0*
Tshape0*/
_output_shapes
:         
У
4gradients/model_2/Flatten/flatten/Reshape_grad/ShapeShapemodel_2/conv5/MaxPool2D/MaxPool*
T0*
out_type0*
_output_shapes
:
Ё
6gradients/model_2/Flatten/flatten/Reshape_grad/ReshapeReshape/gradients/sub_1_grad/tuple/control_dependency_14gradients/model_2/Flatten/flatten/Reshape_grad/Shape*
T0*
Tshape0*/
_output_shapes
:         
┼
:gradients/model_1/conv5/MaxPool2D/MaxPool_grad/MaxPoolGradMaxPoolGradmodel_1/conv5/BiasAddmodel_1/conv5/MaxPool2D/MaxPool6gradients/model_1/Flatten/flatten/Reshape_grad/Reshape*/
_output_shapes
:         
*
T0*
strides
*
data_formatNHWC*
ksize
*
paddingSAME
╜
8gradients/model/conv5/MaxPool2D/MaxPool_grad/MaxPoolGradMaxPoolGradmodel/conv5/BiasAddmodel/conv5/MaxPool2D/MaxPool4gradients/model/Flatten/flatten/Reshape_grad/Reshape*/
_output_shapes
:         
*
T0*
data_formatNHWC*
strides
*
ksize
*
paddingSAME
┼
:gradients/model_2/conv5/MaxPool2D/MaxPool_grad/MaxPoolGradMaxPoolGradmodel_2/conv5/BiasAddmodel_2/conv5/MaxPool2D/MaxPool6gradients/model_2/Flatten/flatten/Reshape_grad/Reshape*
data_formatNHWC*
strides
*
ksize
*
paddingSAME*/
_output_shapes
:         
*
T0
╖
0gradients/model_1/conv5/BiasAdd_grad/BiasAddGradBiasAddGrad:gradients/model_1/conv5/MaxPool2D/MaxPool_grad/MaxPoolGrad*
data_formatNHWC*
_output_shapes
:*
T0
н
5gradients/model_1/conv5/BiasAdd_grad/tuple/group_depsNoOp1^gradients/model_1/conv5/BiasAdd_grad/BiasAddGrad;^gradients/model_1/conv5/MaxPool2D/MaxPool_grad/MaxPoolGrad
╞
=gradients/model_1/conv5/BiasAdd_grad/tuple/control_dependencyIdentity:gradients/model_1/conv5/MaxPool2D/MaxPool_grad/MaxPoolGrad6^gradients/model_1/conv5/BiasAdd_grad/tuple/group_deps*/
_output_shapes
:         
*
T0*M
_classC
A?loc:@gradients/model_1/conv5/MaxPool2D/MaxPool_grad/MaxPoolGrad
Я
?gradients/model_1/conv5/BiasAdd_grad/tuple/control_dependency_1Identity0gradients/model_1/conv5/BiasAdd_grad/BiasAddGrad6^gradients/model_1/conv5/BiasAdd_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/model_1/conv5/BiasAdd_grad/BiasAddGrad*
_output_shapes
:
│
.gradients/model/conv5/BiasAdd_grad/BiasAddGradBiasAddGrad8gradients/model/conv5/MaxPool2D/MaxPool_grad/MaxPoolGrad*
T0*
data_formatNHWC*
_output_shapes
:
з
3gradients/model/conv5/BiasAdd_grad/tuple/group_depsNoOp/^gradients/model/conv5/BiasAdd_grad/BiasAddGrad9^gradients/model/conv5/MaxPool2D/MaxPool_grad/MaxPoolGrad
╛
;gradients/model/conv5/BiasAdd_grad/tuple/control_dependencyIdentity8gradients/model/conv5/MaxPool2D/MaxPool_grad/MaxPoolGrad4^gradients/model/conv5/BiasAdd_grad/tuple/group_deps*/
_output_shapes
:         
*
T0*K
_classA
?=loc:@gradients/model/conv5/MaxPool2D/MaxPool_grad/MaxPoolGrad
Ч
=gradients/model/conv5/BiasAdd_grad/tuple/control_dependency_1Identity.gradients/model/conv5/BiasAdd_grad/BiasAddGrad4^gradients/model/conv5/BiasAdd_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients/model/conv5/BiasAdd_grad/BiasAddGrad*
_output_shapes
:
╖
0gradients/model_2/conv5/BiasAdd_grad/BiasAddGradBiasAddGrad:gradients/model_2/conv5/MaxPool2D/MaxPool_grad/MaxPoolGrad*
T0*
data_formatNHWC*
_output_shapes
:
н
5gradients/model_2/conv5/BiasAdd_grad/tuple/group_depsNoOp1^gradients/model_2/conv5/BiasAdd_grad/BiasAddGrad;^gradients/model_2/conv5/MaxPool2D/MaxPool_grad/MaxPoolGrad
╞
=gradients/model_2/conv5/BiasAdd_grad/tuple/control_dependencyIdentity:gradients/model_2/conv5/MaxPool2D/MaxPool_grad/MaxPoolGrad6^gradients/model_2/conv5/BiasAdd_grad/tuple/group_deps*
T0*M
_classC
A?loc:@gradients/model_2/conv5/MaxPool2D/MaxPool_grad/MaxPoolGrad*/
_output_shapes
:         

Я
?gradients/model_2/conv5/BiasAdd_grad/tuple/control_dependency_1Identity0gradients/model_2/conv5/BiasAdd_grad/BiasAddGrad6^gradients/model_2/conv5/BiasAdd_grad/tuple/group_deps*
_output_shapes
:*
T0*C
_class9
75loc:@gradients/model_2/conv5/BiasAdd_grad/BiasAddGrad
н
*gradients/model_1/conv5/Conv2D_grad/ShapeNShapeNmodel_1/conv4/MaxPool2D/MaxPoolconv5/weights/read*
N* 
_output_shapes
::*
T0*
out_type0
Ў
7gradients/model_1/conv5/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput*gradients/model_1/conv5/Conv2D_grad/ShapeNconv5/weights/read=gradients/model_1/conv5/BiasAdd_grad/tuple/control_dependency*
paddingSAME*0
_output_shapes
:         
А*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
■
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
:А
▒
4gradients/model_1/conv5/Conv2D_grad/tuple/group_depsNoOp9^gradients/model_1/conv5/Conv2D_grad/Conv2DBackpropFilter8^gradients/model_1/conv5/Conv2D_grad/Conv2DBackpropInput
┐
<gradients/model_1/conv5/Conv2D_grad/tuple/control_dependencyIdentity7gradients/model_1/conv5/Conv2D_grad/Conv2DBackpropInput5^gradients/model_1/conv5/Conv2D_grad/tuple/group_deps*0
_output_shapes
:         
А*
T0*J
_class@
><loc:@gradients/model_1/conv5/Conv2D_grad/Conv2DBackpropInput
║
>gradients/model_1/conv5/Conv2D_grad/tuple/control_dependency_1Identity8gradients/model_1/conv5/Conv2D_grad/Conv2DBackpropFilter5^gradients/model_1/conv5/Conv2D_grad/tuple/group_deps*
T0*K
_classA
?=loc:@gradients/model_1/conv5/Conv2D_grad/Conv2DBackpropFilter*'
_output_shapes
:А
й
(gradients/model/conv5/Conv2D_grad/ShapeNShapeNmodel/conv4/MaxPool2D/MaxPoolconv5/weights/read*
T0*
out_type0*
N* 
_output_shapes
::
Ё
5gradients/model/conv5/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput(gradients/model/conv5/Conv2D_grad/ShapeNconv5/weights/read;gradients/model/conv5/BiasAdd_grad/tuple/control_dependency*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*0
_output_shapes
:         
А*
	dilations

Ў
6gradients/model/conv5/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFiltermodel/conv4/MaxPool2D/MaxPool*gradients/model/conv5/Conv2D_grad/ShapeN:1;gradients/model/conv5/BiasAdd_grad/tuple/control_dependency*
paddingSAME*'
_output_shapes
:А*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
л
2gradients/model/conv5/Conv2D_grad/tuple/group_depsNoOp7^gradients/model/conv5/Conv2D_grad/Conv2DBackpropFilter6^gradients/model/conv5/Conv2D_grad/Conv2DBackpropInput
╖
:gradients/model/conv5/Conv2D_grad/tuple/control_dependencyIdentity5gradients/model/conv5/Conv2D_grad/Conv2DBackpropInput3^gradients/model/conv5/Conv2D_grad/tuple/group_deps*0
_output_shapes
:         
А*
T0*H
_class>
<:loc:@gradients/model/conv5/Conv2D_grad/Conv2DBackpropInput
▓
<gradients/model/conv5/Conv2D_grad/tuple/control_dependency_1Identity6gradients/model/conv5/Conv2D_grad/Conv2DBackpropFilter3^gradients/model/conv5/Conv2D_grad/tuple/group_deps*
T0*I
_class?
=;loc:@gradients/model/conv5/Conv2D_grad/Conv2DBackpropFilter*'
_output_shapes
:А
н
*gradients/model_2/conv5/Conv2D_grad/ShapeNShapeNmodel_2/conv4/MaxPool2D/MaxPoolconv5/weights/read*
T0*
out_type0*
N* 
_output_shapes
::
Ў
7gradients/model_2/conv5/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput*gradients/model_2/conv5/Conv2D_grad/ShapeNconv5/weights/read=gradients/model_2/conv5/BiasAdd_grad/tuple/control_dependency*
paddingSAME*0
_output_shapes
:         
А*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(
■
8gradients/model_2/conv5/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFiltermodel_2/conv4/MaxPool2D/MaxPool,gradients/model_2/conv5/Conv2D_grad/ShapeN:1=gradients/model_2/conv5/BiasAdd_grad/tuple/control_dependency*
paddingSAME*'
_output_shapes
:А*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
▒
4gradients/model_2/conv5/Conv2D_grad/tuple/group_depsNoOp9^gradients/model_2/conv5/Conv2D_grad/Conv2DBackpropFilter8^gradients/model_2/conv5/Conv2D_grad/Conv2DBackpropInput
┐
<gradients/model_2/conv5/Conv2D_grad/tuple/control_dependencyIdentity7gradients/model_2/conv5/Conv2D_grad/Conv2DBackpropInput5^gradients/model_2/conv5/Conv2D_grad/tuple/group_deps*
T0*J
_class@
><loc:@gradients/model_2/conv5/Conv2D_grad/Conv2DBackpropInput*0
_output_shapes
:         
А
║
>gradients/model_2/conv5/Conv2D_grad/tuple/control_dependency_1Identity8gradients/model_2/conv5/Conv2D_grad/Conv2DBackpropFilter5^gradients/model_2/conv5/Conv2D_grad/tuple/group_deps*
T0*K
_classA
?=loc:@gradients/model_2/conv5/Conv2D_grad/Conv2DBackpropFilter*'
_output_shapes
:А
╠
gradients/AddN_1AddN?gradients/model_1/conv5/BiasAdd_grad/tuple/control_dependency_1=gradients/model/conv5/BiasAdd_grad/tuple/control_dependency_1?gradients/model_2/conv5/BiasAdd_grad/tuple/control_dependency_1*
T0*C
_class9
75loc:@gradients/model_1/conv5/BiasAdd_grad/BiasAddGrad*
N*
_output_shapes
:
╧
:gradients/model_1/conv4/MaxPool2D/MaxPool_grad/MaxPoolGradMaxPoolGradmodel_1/conv4/conv4/Relumodel_1/conv4/MaxPool2D/MaxPool<gradients/model_1/conv5/Conv2D_grad/tuple/control_dependency*
T0*
data_formatNHWC*
strides
*
ksize
*
paddingSAME*0
_output_shapes
:         А
╟
8gradients/model/conv4/MaxPool2D/MaxPool_grad/MaxPoolGradMaxPoolGradmodel/conv4/conv4/Relumodel/conv4/MaxPool2D/MaxPool:gradients/model/conv5/Conv2D_grad/tuple/control_dependency*
T0*
strides
*
data_formatNHWC*
ksize
*
paddingSAME*0
_output_shapes
:         А
╧
:gradients/model_2/conv4/MaxPool2D/MaxPool_grad/MaxPoolGradMaxPoolGradmodel_2/conv4/conv4/Relumodel_2/conv4/MaxPool2D/MaxPool<gradients/model_2/conv5/Conv2D_grad/tuple/control_dependency*
ksize
*
paddingSAME*0
_output_shapes
:         А*
T0*
strides
*
data_formatNHWC
▐
gradients/AddN_2AddN>gradients/model_1/conv5/Conv2D_grad/tuple/control_dependency_1<gradients/model/conv5/Conv2D_grad/tuple/control_dependency_1>gradients/model_2/conv5/Conv2D_grad/tuple/control_dependency_1*
T0*K
_classA
?=loc:@gradients/model_1/conv5/Conv2D_grad/Conv2DBackpropFilter*
N*'
_output_shapes
:А
═
0gradients/model_1/conv4/conv4/Relu_grad/ReluGradReluGrad:gradients/model_1/conv4/MaxPool2D/MaxPool_grad/MaxPoolGradmodel_1/conv4/conv4/Relu*
T0*0
_output_shapes
:         А
╟
.gradients/model/conv4/conv4/Relu_grad/ReluGradReluGrad8gradients/model/conv4/MaxPool2D/MaxPool_grad/MaxPoolGradmodel/conv4/conv4/Relu*
T0*0
_output_shapes
:         А
═
0gradients/model_2/conv4/conv4/Relu_grad/ReluGradReluGrad:gradients/model_2/conv4/MaxPool2D/MaxPool_grad/MaxPoolGradmodel_2/conv4/conv4/Relu*
T0*0
_output_shapes
:         А
о
0gradients/model_1/conv4/BiasAdd_grad/BiasAddGradBiasAddGrad0gradients/model_1/conv4/conv4/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes	
:А
г
5gradients/model_1/conv4/BiasAdd_grad/tuple/group_depsNoOp1^gradients/model_1/conv4/BiasAdd_grad/BiasAddGrad1^gradients/model_1/conv4/conv4/Relu_grad/ReluGrad
│
=gradients/model_1/conv4/BiasAdd_grad/tuple/control_dependencyIdentity0gradients/model_1/conv4/conv4/Relu_grad/ReluGrad6^gradients/model_1/conv4/BiasAdd_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/model_1/conv4/conv4/Relu_grad/ReluGrad*0
_output_shapes
:         А
а
?gradients/model_1/conv4/BiasAdd_grad/tuple/control_dependency_1Identity0gradients/model_1/conv4/BiasAdd_grad/BiasAddGrad6^gradients/model_1/conv4/BiasAdd_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/model_1/conv4/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:А
к
.gradients/model/conv4/BiasAdd_grad/BiasAddGradBiasAddGrad.gradients/model/conv4/conv4/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes	
:А
Э
3gradients/model/conv4/BiasAdd_grad/tuple/group_depsNoOp/^gradients/model/conv4/BiasAdd_grad/BiasAddGrad/^gradients/model/conv4/conv4/Relu_grad/ReluGrad
л
;gradients/model/conv4/BiasAdd_grad/tuple/control_dependencyIdentity.gradients/model/conv4/conv4/Relu_grad/ReluGrad4^gradients/model/conv4/BiasAdd_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients/model/conv4/conv4/Relu_grad/ReluGrad*0
_output_shapes
:         А
Ш
=gradients/model/conv4/BiasAdd_grad/tuple/control_dependency_1Identity.gradients/model/conv4/BiasAdd_grad/BiasAddGrad4^gradients/model/conv4/BiasAdd_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients/model/conv4/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:А
о
0gradients/model_2/conv4/BiasAdd_grad/BiasAddGradBiasAddGrad0gradients/model_2/conv4/conv4/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes	
:А
г
5gradients/model_2/conv4/BiasAdd_grad/tuple/group_depsNoOp1^gradients/model_2/conv4/BiasAdd_grad/BiasAddGrad1^gradients/model_2/conv4/conv4/Relu_grad/ReluGrad
│
=gradients/model_2/conv4/BiasAdd_grad/tuple/control_dependencyIdentity0gradients/model_2/conv4/conv4/Relu_grad/ReluGrad6^gradients/model_2/conv4/BiasAdd_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/model_2/conv4/conv4/Relu_grad/ReluGrad*0
_output_shapes
:         А
а
?gradients/model_2/conv4/BiasAdd_grad/tuple/control_dependency_1Identity0gradients/model_2/conv4/BiasAdd_grad/BiasAddGrad6^gradients/model_2/conv4/BiasAdd_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/model_2/conv4/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:А
н
*gradients/model_1/conv4/Conv2D_grad/ShapeNShapeNmodel_1/conv3/MaxPool2D/MaxPoolconv4/weights/read*
T0*
out_type0*
N* 
_output_shapes
::
Ў
7gradients/model_1/conv4/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput*gradients/model_1/conv4/Conv2D_grad/ShapeNconv4/weights/read=gradients/model_1/conv4/BiasAdd_grad/tuple/control_dependency*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME*0
_output_shapes
:         А*
	dilations
*
T0
 
8gradients/model_1/conv4/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFiltermodel_1/conv3/MaxPool2D/MaxPool,gradients/model_1/conv4/Conv2D_grad/ShapeN:1=gradients/model_1/conv4/BiasAdd_grad/tuple/control_dependency*(
_output_shapes
:АА*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME
▒
4gradients/model_1/conv4/Conv2D_grad/tuple/group_depsNoOp9^gradients/model_1/conv4/Conv2D_grad/Conv2DBackpropFilter8^gradients/model_1/conv4/Conv2D_grad/Conv2DBackpropInput
┐
<gradients/model_1/conv4/Conv2D_grad/tuple/control_dependencyIdentity7gradients/model_1/conv4/Conv2D_grad/Conv2DBackpropInput5^gradients/model_1/conv4/Conv2D_grad/tuple/group_deps*
T0*J
_class@
><loc:@gradients/model_1/conv4/Conv2D_grad/Conv2DBackpropInput*0
_output_shapes
:         А
╗
>gradients/model_1/conv4/Conv2D_grad/tuple/control_dependency_1Identity8gradients/model_1/conv4/Conv2D_grad/Conv2DBackpropFilter5^gradients/model_1/conv4/Conv2D_grad/tuple/group_deps*
T0*K
_classA
?=loc:@gradients/model_1/conv4/Conv2D_grad/Conv2DBackpropFilter*(
_output_shapes
:АА
й
(gradients/model/conv4/Conv2D_grad/ShapeNShapeNmodel/conv3/MaxPool2D/MaxPoolconv4/weights/read*
T0*
out_type0*
N* 
_output_shapes
::
Ё
5gradients/model/conv4/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput(gradients/model/conv4/Conv2D_grad/ShapeNconv4/weights/read;gradients/model/conv4/BiasAdd_grad/tuple/control_dependency*0
_output_shapes
:         А*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME
ў
6gradients/model/conv4/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFiltermodel/conv3/MaxPool2D/MaxPool*gradients/model/conv4/Conv2D_grad/ShapeN:1;gradients/model/conv4/BiasAdd_grad/tuple/control_dependency*(
_output_shapes
:АА*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME
л
2gradients/model/conv4/Conv2D_grad/tuple/group_depsNoOp7^gradients/model/conv4/Conv2D_grad/Conv2DBackpropFilter6^gradients/model/conv4/Conv2D_grad/Conv2DBackpropInput
╖
:gradients/model/conv4/Conv2D_grad/tuple/control_dependencyIdentity5gradients/model/conv4/Conv2D_grad/Conv2DBackpropInput3^gradients/model/conv4/Conv2D_grad/tuple/group_deps*
T0*H
_class>
<:loc:@gradients/model/conv4/Conv2D_grad/Conv2DBackpropInput*0
_output_shapes
:         А
│
<gradients/model/conv4/Conv2D_grad/tuple/control_dependency_1Identity6gradients/model/conv4/Conv2D_grad/Conv2DBackpropFilter3^gradients/model/conv4/Conv2D_grad/tuple/group_deps*(
_output_shapes
:АА*
T0*I
_class?
=;loc:@gradients/model/conv4/Conv2D_grad/Conv2DBackpropFilter
н
*gradients/model_2/conv4/Conv2D_grad/ShapeNShapeNmodel_2/conv3/MaxPool2D/MaxPoolconv4/weights/read*
T0*
out_type0*
N* 
_output_shapes
::
Ў
7gradients/model_2/conv4/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput*gradients/model_2/conv4/Conv2D_grad/ShapeNconv4/weights/read=gradients/model_2/conv4/BiasAdd_grad/tuple/control_dependency*0
_output_shapes
:         А*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME
 
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
:АА
▒
4gradients/model_2/conv4/Conv2D_grad/tuple/group_depsNoOp9^gradients/model_2/conv4/Conv2D_grad/Conv2DBackpropFilter8^gradients/model_2/conv4/Conv2D_grad/Conv2DBackpropInput
┐
<gradients/model_2/conv4/Conv2D_grad/tuple/control_dependencyIdentity7gradients/model_2/conv4/Conv2D_grad/Conv2DBackpropInput5^gradients/model_2/conv4/Conv2D_grad/tuple/group_deps*
T0*J
_class@
><loc:@gradients/model_2/conv4/Conv2D_grad/Conv2DBackpropInput*0
_output_shapes
:         А
╗
>gradients/model_2/conv4/Conv2D_grad/tuple/control_dependency_1Identity8gradients/model_2/conv4/Conv2D_grad/Conv2DBackpropFilter5^gradients/model_2/conv4/Conv2D_grad/tuple/group_deps*
T0*K
_classA
?=loc:@gradients/model_2/conv4/Conv2D_grad/Conv2DBackpropFilter*(
_output_shapes
:АА
═
gradients/AddN_3AddN?gradients/model_1/conv4/BiasAdd_grad/tuple/control_dependency_1=gradients/model/conv4/BiasAdd_grad/tuple/control_dependency_1?gradients/model_2/conv4/BiasAdd_grad/tuple/control_dependency_1*
N*
_output_shapes	
:А*
T0*C
_class9
75loc:@gradients/model_1/conv4/BiasAdd_grad/BiasAddGrad
╧
:gradients/model_1/conv3/MaxPool2D/MaxPool_grad/MaxPoolGradMaxPoolGradmodel_1/conv3/conv3/Relumodel_1/conv3/MaxPool2D/MaxPool<gradients/model_1/conv4/Conv2D_grad/tuple/control_dependency*
data_formatNHWC*
strides
*
ksize
*
paddingSAME*0
_output_shapes
:         &А*
T0
╟
8gradients/model/conv3/MaxPool2D/MaxPool_grad/MaxPoolGradMaxPoolGradmodel/conv3/conv3/Relumodel/conv3/MaxPool2D/MaxPool:gradients/model/conv4/Conv2D_grad/tuple/control_dependency*
T0*
data_formatNHWC*
strides
*
ksize
*
paddingSAME*0
_output_shapes
:         &А
╧
:gradients/model_2/conv3/MaxPool2D/MaxPool_grad/MaxPoolGradMaxPoolGradmodel_2/conv3/conv3/Relumodel_2/conv3/MaxPool2D/MaxPool<gradients/model_2/conv4/Conv2D_grad/tuple/control_dependency*
ksize
*
paddingSAME*0
_output_shapes
:         &А*
T0*
strides
*
data_formatNHWC
▀
gradients/AddN_4AddN>gradients/model_1/conv4/Conv2D_grad/tuple/control_dependency_1<gradients/model/conv4/Conv2D_grad/tuple/control_dependency_1>gradients/model_2/conv4/Conv2D_grad/tuple/control_dependency_1*
T0*K
_classA
?=loc:@gradients/model_1/conv4/Conv2D_grad/Conv2DBackpropFilter*
N*(
_output_shapes
:АА
═
0gradients/model_1/conv3/conv3/Relu_grad/ReluGradReluGrad:gradients/model_1/conv3/MaxPool2D/MaxPool_grad/MaxPoolGradmodel_1/conv3/conv3/Relu*
T0*0
_output_shapes
:         &А
╟
.gradients/model/conv3/conv3/Relu_grad/ReluGradReluGrad8gradients/model/conv3/MaxPool2D/MaxPool_grad/MaxPoolGradmodel/conv3/conv3/Relu*
T0*0
_output_shapes
:         &А
═
0gradients/model_2/conv3/conv3/Relu_grad/ReluGradReluGrad:gradients/model_2/conv3/MaxPool2D/MaxPool_grad/MaxPoolGradmodel_2/conv3/conv3/Relu*
T0*0
_output_shapes
:         &А
о
0gradients/model_1/conv3/BiasAdd_grad/BiasAddGradBiasAddGrad0gradients/model_1/conv3/conv3/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes	
:А
г
5gradients/model_1/conv3/BiasAdd_grad/tuple/group_depsNoOp1^gradients/model_1/conv3/BiasAdd_grad/BiasAddGrad1^gradients/model_1/conv3/conv3/Relu_grad/ReluGrad
│
=gradients/model_1/conv3/BiasAdd_grad/tuple/control_dependencyIdentity0gradients/model_1/conv3/conv3/Relu_grad/ReluGrad6^gradients/model_1/conv3/BiasAdd_grad/tuple/group_deps*0
_output_shapes
:         &А*
T0*C
_class9
75loc:@gradients/model_1/conv3/conv3/Relu_grad/ReluGrad
а
?gradients/model_1/conv3/BiasAdd_grad/tuple/control_dependency_1Identity0gradients/model_1/conv3/BiasAdd_grad/BiasAddGrad6^gradients/model_1/conv3/BiasAdd_grad/tuple/group_deps*
_output_shapes	
:А*
T0*C
_class9
75loc:@gradients/model_1/conv3/BiasAdd_grad/BiasAddGrad
к
.gradients/model/conv3/BiasAdd_grad/BiasAddGradBiasAddGrad.gradients/model/conv3/conv3/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes	
:А
Э
3gradients/model/conv3/BiasAdd_grad/tuple/group_depsNoOp/^gradients/model/conv3/BiasAdd_grad/BiasAddGrad/^gradients/model/conv3/conv3/Relu_grad/ReluGrad
л
;gradients/model/conv3/BiasAdd_grad/tuple/control_dependencyIdentity.gradients/model/conv3/conv3/Relu_grad/ReluGrad4^gradients/model/conv3/BiasAdd_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients/model/conv3/conv3/Relu_grad/ReluGrad*0
_output_shapes
:         &А
Ш
=gradients/model/conv3/BiasAdd_grad/tuple/control_dependency_1Identity.gradients/model/conv3/BiasAdd_grad/BiasAddGrad4^gradients/model/conv3/BiasAdd_grad/tuple/group_deps*
_output_shapes	
:А*
T0*A
_class7
53loc:@gradients/model/conv3/BiasAdd_grad/BiasAddGrad
о
0gradients/model_2/conv3/BiasAdd_grad/BiasAddGradBiasAddGrad0gradients/model_2/conv3/conv3/Relu_grad/ReluGrad*
data_formatNHWC*
_output_shapes	
:А*
T0
г
5gradients/model_2/conv3/BiasAdd_grad/tuple/group_depsNoOp1^gradients/model_2/conv3/BiasAdd_grad/BiasAddGrad1^gradients/model_2/conv3/conv3/Relu_grad/ReluGrad
│
=gradients/model_2/conv3/BiasAdd_grad/tuple/control_dependencyIdentity0gradients/model_2/conv3/conv3/Relu_grad/ReluGrad6^gradients/model_2/conv3/BiasAdd_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/model_2/conv3/conv3/Relu_grad/ReluGrad*0
_output_shapes
:         &А
а
?gradients/model_2/conv3/BiasAdd_grad/tuple/control_dependency_1Identity0gradients/model_2/conv3/BiasAdd_grad/BiasAddGrad6^gradients/model_2/conv3/BiasAdd_grad/tuple/group_deps*
_output_shapes	
:А*
T0*C
_class9
75loc:@gradients/model_2/conv3/BiasAdd_grad/BiasAddGrad
н
*gradients/model_1/conv3/Conv2D_grad/ShapeNShapeNmodel_1/conv2/MaxPool2D/MaxPoolconv3/weights/read*
T0*
out_type0*
N* 
_output_shapes
::
ї
7gradients/model_1/conv3/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput*gradients/model_1/conv3/Conv2D_grad/ShapeNconv3/weights/read=gradients/model_1/conv3/BiasAdd_grad/tuple/control_dependency*
paddingSAME*/
_output_shapes
:         &@*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
■
8gradients/model_1/conv3/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFiltermodel_1/conv2/MaxPool2D/MaxPool,gradients/model_1/conv3/Conv2D_grad/ShapeN:1=gradients/model_1/conv3/BiasAdd_grad/tuple/control_dependency*
paddingSAME*'
_output_shapes
:@А*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
▒
4gradients/model_1/conv3/Conv2D_grad/tuple/group_depsNoOp9^gradients/model_1/conv3/Conv2D_grad/Conv2DBackpropFilter8^gradients/model_1/conv3/Conv2D_grad/Conv2DBackpropInput
╛
<gradients/model_1/conv3/Conv2D_grad/tuple/control_dependencyIdentity7gradients/model_1/conv3/Conv2D_grad/Conv2DBackpropInput5^gradients/model_1/conv3/Conv2D_grad/tuple/group_deps*
T0*J
_class@
><loc:@gradients/model_1/conv3/Conv2D_grad/Conv2DBackpropInput*/
_output_shapes
:         &@
║
>gradients/model_1/conv3/Conv2D_grad/tuple/control_dependency_1Identity8gradients/model_1/conv3/Conv2D_grad/Conv2DBackpropFilter5^gradients/model_1/conv3/Conv2D_grad/tuple/group_deps*
T0*K
_classA
?=loc:@gradients/model_1/conv3/Conv2D_grad/Conv2DBackpropFilter*'
_output_shapes
:@А
й
(gradients/model/conv3/Conv2D_grad/ShapeNShapeNmodel/conv2/MaxPool2D/MaxPoolconv3/weights/read*
N* 
_output_shapes
::*
T0*
out_type0
я
5gradients/model/conv3/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput(gradients/model/conv3/Conv2D_grad/ShapeNconv3/weights/read;gradients/model/conv3/BiasAdd_grad/tuple/control_dependency*/
_output_shapes
:         &@*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME
Ў
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
:@А
л
2gradients/model/conv3/Conv2D_grad/tuple/group_depsNoOp7^gradients/model/conv3/Conv2D_grad/Conv2DBackpropFilter6^gradients/model/conv3/Conv2D_grad/Conv2DBackpropInput
╢
:gradients/model/conv3/Conv2D_grad/tuple/control_dependencyIdentity5gradients/model/conv3/Conv2D_grad/Conv2DBackpropInput3^gradients/model/conv3/Conv2D_grad/tuple/group_deps*
T0*H
_class>
<:loc:@gradients/model/conv3/Conv2D_grad/Conv2DBackpropInput*/
_output_shapes
:         &@
▓
<gradients/model/conv3/Conv2D_grad/tuple/control_dependency_1Identity6gradients/model/conv3/Conv2D_grad/Conv2DBackpropFilter3^gradients/model/conv3/Conv2D_grad/tuple/group_deps*
T0*I
_class?
=;loc:@gradients/model/conv3/Conv2D_grad/Conv2DBackpropFilter*'
_output_shapes
:@А
н
*gradients/model_2/conv3/Conv2D_grad/ShapeNShapeNmodel_2/conv2/MaxPool2D/MaxPoolconv3/weights/read*
T0*
out_type0*
N* 
_output_shapes
::
ї
7gradients/model_2/conv3/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput*gradients/model_2/conv3/Conv2D_grad/ShapeNconv3/weights/read=gradients/model_2/conv3/BiasAdd_grad/tuple/control_dependency*
paddingSAME*/
_output_shapes
:         &@*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
■
8gradients/model_2/conv3/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFiltermodel_2/conv2/MaxPool2D/MaxPool,gradients/model_2/conv3/Conv2D_grad/ShapeN:1=gradients/model_2/conv3/BiasAdd_grad/tuple/control_dependency*
paddingSAME*'
_output_shapes
:@А*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
▒
4gradients/model_2/conv3/Conv2D_grad/tuple/group_depsNoOp9^gradients/model_2/conv3/Conv2D_grad/Conv2DBackpropFilter8^gradients/model_2/conv3/Conv2D_grad/Conv2DBackpropInput
╛
<gradients/model_2/conv3/Conv2D_grad/tuple/control_dependencyIdentity7gradients/model_2/conv3/Conv2D_grad/Conv2DBackpropInput5^gradients/model_2/conv3/Conv2D_grad/tuple/group_deps*
T0*J
_class@
><loc:@gradients/model_2/conv3/Conv2D_grad/Conv2DBackpropInput*/
_output_shapes
:         &@
║
>gradients/model_2/conv3/Conv2D_grad/tuple/control_dependency_1Identity8gradients/model_2/conv3/Conv2D_grad/Conv2DBackpropFilter5^gradients/model_2/conv3/Conv2D_grad/tuple/group_deps*
T0*K
_classA
?=loc:@gradients/model_2/conv3/Conv2D_grad/Conv2DBackpropFilter*'
_output_shapes
:@А
═
gradients/AddN_5AddN?gradients/model_1/conv3/BiasAdd_grad/tuple/control_dependency_1=gradients/model/conv3/BiasAdd_grad/tuple/control_dependency_1?gradients/model_2/conv3/BiasAdd_grad/tuple/control_dependency_1*
T0*C
_class9
75loc:@gradients/model_1/conv3/BiasAdd_grad/BiasAddGrad*
N*
_output_shapes	
:А
╬
:gradients/model_1/conv2/MaxPool2D/MaxPool_grad/MaxPoolGradMaxPoolGradmodel_1/conv2/conv2/Relumodel_1/conv2/MaxPool2D/MaxPool<gradients/model_1/conv3/Conv2D_grad/tuple/control_dependency*
T0*
data_formatNHWC*
strides
*
ksize
*
paddingSAME*/
_output_shapes
:         K@
╞
8gradients/model/conv2/MaxPool2D/MaxPool_grad/MaxPoolGradMaxPoolGradmodel/conv2/conv2/Relumodel/conv2/MaxPool2D/MaxPool:gradients/model/conv3/Conv2D_grad/tuple/control_dependency*
T0*
data_formatNHWC*
strides
*
ksize
*
paddingSAME*/
_output_shapes
:         K@
╬
:gradients/model_2/conv2/MaxPool2D/MaxPool_grad/MaxPoolGradMaxPoolGradmodel_2/conv2/conv2/Relumodel_2/conv2/MaxPool2D/MaxPool<gradients/model_2/conv3/Conv2D_grad/tuple/control_dependency*
ksize
*
paddingSAME*/
_output_shapes
:         K@*
T0*
data_formatNHWC*
strides

▐
gradients/AddN_6AddN>gradients/model_1/conv3/Conv2D_grad/tuple/control_dependency_1<gradients/model/conv3/Conv2D_grad/tuple/control_dependency_1>gradients/model_2/conv3/Conv2D_grad/tuple/control_dependency_1*
T0*K
_classA
?=loc:@gradients/model_1/conv3/Conv2D_grad/Conv2DBackpropFilter*
N*'
_output_shapes
:@А
╠
0gradients/model_1/conv2/conv2/Relu_grad/ReluGradReluGrad:gradients/model_1/conv2/MaxPool2D/MaxPool_grad/MaxPoolGradmodel_1/conv2/conv2/Relu*/
_output_shapes
:         K@*
T0
╞
.gradients/model/conv2/conv2/Relu_grad/ReluGradReluGrad8gradients/model/conv2/MaxPool2D/MaxPool_grad/MaxPoolGradmodel/conv2/conv2/Relu*/
_output_shapes
:         K@*
T0
╠
0gradients/model_2/conv2/conv2/Relu_grad/ReluGradReluGrad:gradients/model_2/conv2/MaxPool2D/MaxPool_grad/MaxPoolGradmodel_2/conv2/conv2/Relu*/
_output_shapes
:         K@*
T0
н
0gradients/model_1/conv2/BiasAdd_grad/BiasAddGradBiasAddGrad0gradients/model_1/conv2/conv2/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes
:@
г
5gradients/model_1/conv2/BiasAdd_grad/tuple/group_depsNoOp1^gradients/model_1/conv2/BiasAdd_grad/BiasAddGrad1^gradients/model_1/conv2/conv2/Relu_grad/ReluGrad
▓
=gradients/model_1/conv2/BiasAdd_grad/tuple/control_dependencyIdentity0gradients/model_1/conv2/conv2/Relu_grad/ReluGrad6^gradients/model_1/conv2/BiasAdd_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/model_1/conv2/conv2/Relu_grad/ReluGrad*/
_output_shapes
:         K@
Я
?gradients/model_1/conv2/BiasAdd_grad/tuple/control_dependency_1Identity0gradients/model_1/conv2/BiasAdd_grad/BiasAddGrad6^gradients/model_1/conv2/BiasAdd_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/model_1/conv2/BiasAdd_grad/BiasAddGrad*
_output_shapes
:@
й
.gradients/model/conv2/BiasAdd_grad/BiasAddGradBiasAddGrad.gradients/model/conv2/conv2/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes
:@
Э
3gradients/model/conv2/BiasAdd_grad/tuple/group_depsNoOp/^gradients/model/conv2/BiasAdd_grad/BiasAddGrad/^gradients/model/conv2/conv2/Relu_grad/ReluGrad
к
;gradients/model/conv2/BiasAdd_grad/tuple/control_dependencyIdentity.gradients/model/conv2/conv2/Relu_grad/ReluGrad4^gradients/model/conv2/BiasAdd_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients/model/conv2/conv2/Relu_grad/ReluGrad*/
_output_shapes
:         K@
Ч
=gradients/model/conv2/BiasAdd_grad/tuple/control_dependency_1Identity.gradients/model/conv2/BiasAdd_grad/BiasAddGrad4^gradients/model/conv2/BiasAdd_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients/model/conv2/BiasAdd_grad/BiasAddGrad*
_output_shapes
:@
н
0gradients/model_2/conv2/BiasAdd_grad/BiasAddGradBiasAddGrad0gradients/model_2/conv2/conv2/Relu_grad/ReluGrad*
data_formatNHWC*
_output_shapes
:@*
T0
г
5gradients/model_2/conv2/BiasAdd_grad/tuple/group_depsNoOp1^gradients/model_2/conv2/BiasAdd_grad/BiasAddGrad1^gradients/model_2/conv2/conv2/Relu_grad/ReluGrad
▓
=gradients/model_2/conv2/BiasAdd_grad/tuple/control_dependencyIdentity0gradients/model_2/conv2/conv2/Relu_grad/ReluGrad6^gradients/model_2/conv2/BiasAdd_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/model_2/conv2/conv2/Relu_grad/ReluGrad*/
_output_shapes
:         K@
Я
?gradients/model_2/conv2/BiasAdd_grad/tuple/control_dependency_1Identity0gradients/model_2/conv2/BiasAdd_grad/BiasAddGrad6^gradients/model_2/conv2/BiasAdd_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/model_2/conv2/BiasAdd_grad/BiasAddGrad*
_output_shapes
:@
н
*gradients/model_1/conv2/Conv2D_grad/ShapeNShapeNmodel_1/conv1/MaxPool2D/MaxPoolconv2/weights/read*
T0*
out_type0*
N* 
_output_shapes
::
ї
7gradients/model_1/conv2/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput*gradients/model_1/conv2/Conv2D_grad/ShapeNconv2/weights/read=gradients/model_1/conv2/BiasAdd_grad/tuple/control_dependency*/
_output_shapes
:         K *
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME
¤
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
▒
4gradients/model_1/conv2/Conv2D_grad/tuple/group_depsNoOp9^gradients/model_1/conv2/Conv2D_grad/Conv2DBackpropFilter8^gradients/model_1/conv2/Conv2D_grad/Conv2DBackpropInput
╛
<gradients/model_1/conv2/Conv2D_grad/tuple/control_dependencyIdentity7gradients/model_1/conv2/Conv2D_grad/Conv2DBackpropInput5^gradients/model_1/conv2/Conv2D_grad/tuple/group_deps*
T0*J
_class@
><loc:@gradients/model_1/conv2/Conv2D_grad/Conv2DBackpropInput*/
_output_shapes
:         K 
╣
>gradients/model_1/conv2/Conv2D_grad/tuple/control_dependency_1Identity8gradients/model_1/conv2/Conv2D_grad/Conv2DBackpropFilter5^gradients/model_1/conv2/Conv2D_grad/tuple/group_deps*
T0*K
_classA
?=loc:@gradients/model_1/conv2/Conv2D_grad/Conv2DBackpropFilter*&
_output_shapes
: @
й
(gradients/model/conv2/Conv2D_grad/ShapeNShapeNmodel/conv1/MaxPool2D/MaxPoolconv2/weights/read*
T0*
out_type0*
N* 
_output_shapes
::
я
5gradients/model/conv2/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput(gradients/model/conv2/Conv2D_grad/ShapeNconv2/weights/read;gradients/model/conv2/BiasAdd_grad/tuple/control_dependency*/
_output_shapes
:         K *
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME
ї
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
л
2gradients/model/conv2/Conv2D_grad/tuple/group_depsNoOp7^gradients/model/conv2/Conv2D_grad/Conv2DBackpropFilter6^gradients/model/conv2/Conv2D_grad/Conv2DBackpropInput
╢
:gradients/model/conv2/Conv2D_grad/tuple/control_dependencyIdentity5gradients/model/conv2/Conv2D_grad/Conv2DBackpropInput3^gradients/model/conv2/Conv2D_grad/tuple/group_deps*
T0*H
_class>
<:loc:@gradients/model/conv2/Conv2D_grad/Conv2DBackpropInput*/
_output_shapes
:         K 
▒
<gradients/model/conv2/Conv2D_grad/tuple/control_dependency_1Identity6gradients/model/conv2/Conv2D_grad/Conv2DBackpropFilter3^gradients/model/conv2/Conv2D_grad/tuple/group_deps*&
_output_shapes
: @*
T0*I
_class?
=;loc:@gradients/model/conv2/Conv2D_grad/Conv2DBackpropFilter
н
*gradients/model_2/conv2/Conv2D_grad/ShapeNShapeNmodel_2/conv1/MaxPool2D/MaxPoolconv2/weights/read*
T0*
out_type0*
N* 
_output_shapes
::
ї
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
:         K 
¤
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
▒
4gradients/model_2/conv2/Conv2D_grad/tuple/group_depsNoOp9^gradients/model_2/conv2/Conv2D_grad/Conv2DBackpropFilter8^gradients/model_2/conv2/Conv2D_grad/Conv2DBackpropInput
╛
<gradients/model_2/conv2/Conv2D_grad/tuple/control_dependencyIdentity7gradients/model_2/conv2/Conv2D_grad/Conv2DBackpropInput5^gradients/model_2/conv2/Conv2D_grad/tuple/group_deps*
T0*J
_class@
><loc:@gradients/model_2/conv2/Conv2D_grad/Conv2DBackpropInput*/
_output_shapes
:         K 
╣
>gradients/model_2/conv2/Conv2D_grad/tuple/control_dependency_1Identity8gradients/model_2/conv2/Conv2D_grad/Conv2DBackpropFilter5^gradients/model_2/conv2/Conv2D_grad/tuple/group_deps*
T0*K
_classA
?=loc:@gradients/model_2/conv2/Conv2D_grad/Conv2DBackpropFilter*&
_output_shapes
: @
╠
gradients/AddN_7AddN?gradients/model_1/conv2/BiasAdd_grad/tuple/control_dependency_1=gradients/model/conv2/BiasAdd_grad/tuple/control_dependency_1?gradients/model_2/conv2/BiasAdd_grad/tuple/control_dependency_1*C
_class9
75loc:@gradients/model_1/conv2/BiasAdd_grad/BiasAddGrad*
N*
_output_shapes
:@*
T0
╧
:gradients/model_1/conv1/MaxPool2D/MaxPool_grad/MaxPoolGradMaxPoolGradmodel_1/conv1/conv1/Relumodel_1/conv1/MaxPool2D/MaxPool<gradients/model_1/conv2/Conv2D_grad/tuple/control_dependency*
paddingSAME*0
_output_shapes
:         2Ц *
T0*
data_formatNHWC*
strides
*
ksize

╟
8gradients/model/conv1/MaxPool2D/MaxPool_grad/MaxPoolGradMaxPoolGradmodel/conv1/conv1/Relumodel/conv1/MaxPool2D/MaxPool:gradients/model/conv2/Conv2D_grad/tuple/control_dependency*
T0*
data_formatNHWC*
strides
*
ksize
*
paddingSAME*0
_output_shapes
:         2Ц 
╧
:gradients/model_2/conv1/MaxPool2D/MaxPool_grad/MaxPoolGradMaxPoolGradmodel_2/conv1/conv1/Relumodel_2/conv1/MaxPool2D/MaxPool<gradients/model_2/conv2/Conv2D_grad/tuple/control_dependency*
T0*
data_formatNHWC*
strides
*
ksize
*
paddingSAME*0
_output_shapes
:         2Ц 
▌
gradients/AddN_8AddN>gradients/model_1/conv2/Conv2D_grad/tuple/control_dependency_1<gradients/model/conv2/Conv2D_grad/tuple/control_dependency_1>gradients/model_2/conv2/Conv2D_grad/tuple/control_dependency_1*K
_classA
?=loc:@gradients/model_1/conv2/Conv2D_grad/Conv2DBackpropFilter*
N*&
_output_shapes
: @*
T0
═
0gradients/model_1/conv1/conv1/Relu_grad/ReluGradReluGrad:gradients/model_1/conv1/MaxPool2D/MaxPool_grad/MaxPoolGradmodel_1/conv1/conv1/Relu*
T0*0
_output_shapes
:         2Ц 
╟
.gradients/model/conv1/conv1/Relu_grad/ReluGradReluGrad8gradients/model/conv1/MaxPool2D/MaxPool_grad/MaxPoolGradmodel/conv1/conv1/Relu*
T0*0
_output_shapes
:         2Ц 
═
0gradients/model_2/conv1/conv1/Relu_grad/ReluGradReluGrad:gradients/model_2/conv1/MaxPool2D/MaxPool_grad/MaxPoolGradmodel_2/conv1/conv1/Relu*0
_output_shapes
:         2Ц *
T0
н
0gradients/model_1/conv1/BiasAdd_grad/BiasAddGradBiasAddGrad0gradients/model_1/conv1/conv1/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes
: 
г
5gradients/model_1/conv1/BiasAdd_grad/tuple/group_depsNoOp1^gradients/model_1/conv1/BiasAdd_grad/BiasAddGrad1^gradients/model_1/conv1/conv1/Relu_grad/ReluGrad
│
=gradients/model_1/conv1/BiasAdd_grad/tuple/control_dependencyIdentity0gradients/model_1/conv1/conv1/Relu_grad/ReluGrad6^gradients/model_1/conv1/BiasAdd_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/model_1/conv1/conv1/Relu_grad/ReluGrad*0
_output_shapes
:         2Ц 
Я
?gradients/model_1/conv1/BiasAdd_grad/tuple/control_dependency_1Identity0gradients/model_1/conv1/BiasAdd_grad/BiasAddGrad6^gradients/model_1/conv1/BiasAdd_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/model_1/conv1/BiasAdd_grad/BiasAddGrad*
_output_shapes
: 
й
.gradients/model/conv1/BiasAdd_grad/BiasAddGradBiasAddGrad.gradients/model/conv1/conv1/Relu_grad/ReluGrad*
data_formatNHWC*
_output_shapes
: *
T0
Э
3gradients/model/conv1/BiasAdd_grad/tuple/group_depsNoOp/^gradients/model/conv1/BiasAdd_grad/BiasAddGrad/^gradients/model/conv1/conv1/Relu_grad/ReluGrad
л
;gradients/model/conv1/BiasAdd_grad/tuple/control_dependencyIdentity.gradients/model/conv1/conv1/Relu_grad/ReluGrad4^gradients/model/conv1/BiasAdd_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients/model/conv1/conv1/Relu_grad/ReluGrad*0
_output_shapes
:         2Ц 
Ч
=gradients/model/conv1/BiasAdd_grad/tuple/control_dependency_1Identity.gradients/model/conv1/BiasAdd_grad/BiasAddGrad4^gradients/model/conv1/BiasAdd_grad/tuple/group_deps*
_output_shapes
: *
T0*A
_class7
53loc:@gradients/model/conv1/BiasAdd_grad/BiasAddGrad
н
0gradients/model_2/conv1/BiasAdd_grad/BiasAddGradBiasAddGrad0gradients/model_2/conv1/conv1/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes
: 
г
5gradients/model_2/conv1/BiasAdd_grad/tuple/group_depsNoOp1^gradients/model_2/conv1/BiasAdd_grad/BiasAddGrad1^gradients/model_2/conv1/conv1/Relu_grad/ReluGrad
│
=gradients/model_2/conv1/BiasAdd_grad/tuple/control_dependencyIdentity0gradients/model_2/conv1/conv1/Relu_grad/ReluGrad6^gradients/model_2/conv1/BiasAdd_grad/tuple/group_deps*0
_output_shapes
:         2Ц *
T0*C
_class9
75loc:@gradients/model_2/conv1/conv1/Relu_grad/ReluGrad
Я
?gradients/model_2/conv1/BiasAdd_grad/tuple/control_dependency_1Identity0gradients/model_2/conv1/BiasAdd_grad/BiasAddGrad6^gradients/model_2/conv1/BiasAdd_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/model_2/conv1/BiasAdd_grad/BiasAddGrad*
_output_shapes
: 
Ь
*gradients/model_1/conv1/Conv2D_grad/ShapeNShapeNpositive_inputconv1/weights/read* 
_output_shapes
::*
T0*
out_type0*
N
Ў
7gradients/model_1/conv1/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput*gradients/model_1/conv1/Conv2D_grad/ShapeNconv1/weights/read=gradients/model_1/conv1/BiasAdd_grad/tuple/control_dependency*0
_output_shapes
:         2Ц*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME
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
▒
4gradients/model_1/conv1/Conv2D_grad/tuple/group_depsNoOp9^gradients/model_1/conv1/Conv2D_grad/Conv2DBackpropFilter8^gradients/model_1/conv1/Conv2D_grad/Conv2DBackpropInput
┐
<gradients/model_1/conv1/Conv2D_grad/tuple/control_dependencyIdentity7gradients/model_1/conv1/Conv2D_grad/Conv2DBackpropInput5^gradients/model_1/conv1/Conv2D_grad/tuple/group_deps*J
_class@
><loc:@gradients/model_1/conv1/Conv2D_grad/Conv2DBackpropInput*0
_output_shapes
:         2Ц*
T0
╣
>gradients/model_1/conv1/Conv2D_grad/tuple/control_dependency_1Identity8gradients/model_1/conv1/Conv2D_grad/Conv2DBackpropFilter5^gradients/model_1/conv1/Conv2D_grad/tuple/group_deps*
T0*K
_classA
?=loc:@gradients/model_1/conv1/Conv2D_grad/Conv2DBackpropFilter*&
_output_shapes
: 
Ш
(gradients/model/conv1/Conv2D_grad/ShapeNShapeNanchor_inputconv1/weights/read*
N* 
_output_shapes
::*
T0*
out_type0
Ё
5gradients/model/conv1/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput(gradients/model/conv1/Conv2D_grad/ShapeNconv1/weights/read;gradients/model/conv1/BiasAdd_grad/tuple/control_dependency*0
_output_shapes
:         2Ц*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME
ф
6gradients/model/conv1/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilteranchor_input*gradients/model/conv1/Conv2D_grad/ShapeN:1;gradients/model/conv1/BiasAdd_grad/tuple/control_dependency*
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
л
2gradients/model/conv1/Conv2D_grad/tuple/group_depsNoOp7^gradients/model/conv1/Conv2D_grad/Conv2DBackpropFilter6^gradients/model/conv1/Conv2D_grad/Conv2DBackpropInput
╖
:gradients/model/conv1/Conv2D_grad/tuple/control_dependencyIdentity5gradients/model/conv1/Conv2D_grad/Conv2DBackpropInput3^gradients/model/conv1/Conv2D_grad/tuple/group_deps*H
_class>
<:loc:@gradients/model/conv1/Conv2D_grad/Conv2DBackpropInput*0
_output_shapes
:         2Ц*
T0
▒
<gradients/model/conv1/Conv2D_grad/tuple/control_dependency_1Identity6gradients/model/conv1/Conv2D_grad/Conv2DBackpropFilter3^gradients/model/conv1/Conv2D_grad/tuple/group_deps*
T0*I
_class?
=;loc:@gradients/model/conv1/Conv2D_grad/Conv2DBackpropFilter*&
_output_shapes
: 
Ь
*gradients/model_2/conv1/Conv2D_grad/ShapeNShapeNnegative_inputconv1/weights/read*
T0*
out_type0*
N* 
_output_shapes
::
Ў
7gradients/model_2/conv1/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput*gradients/model_2/conv1/Conv2D_grad/ShapeNconv1/weights/read=gradients/model_2/conv1/BiasAdd_grad/tuple/control_dependency*
paddingSAME*0
_output_shapes
:         2Ц*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
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
▒
4gradients/model_2/conv1/Conv2D_grad/tuple/group_depsNoOp9^gradients/model_2/conv1/Conv2D_grad/Conv2DBackpropFilter8^gradients/model_2/conv1/Conv2D_grad/Conv2DBackpropInput
┐
<gradients/model_2/conv1/Conv2D_grad/tuple/control_dependencyIdentity7gradients/model_2/conv1/Conv2D_grad/Conv2DBackpropInput5^gradients/model_2/conv1/Conv2D_grad/tuple/group_deps*
T0*J
_class@
><loc:@gradients/model_2/conv1/Conv2D_grad/Conv2DBackpropInput*0
_output_shapes
:         2Ц
╣
>gradients/model_2/conv1/Conv2D_grad/tuple/control_dependency_1Identity8gradients/model_2/conv1/Conv2D_grad/Conv2DBackpropFilter5^gradients/model_2/conv1/Conv2D_grad/tuple/group_deps*
T0*K
_classA
?=loc:@gradients/model_2/conv1/Conv2D_grad/Conv2DBackpropFilter*&
_output_shapes
: 
╠
gradients/AddN_9AddN?gradients/model_1/conv1/BiasAdd_grad/tuple/control_dependency_1=gradients/model/conv1/BiasAdd_grad/tuple/control_dependency_1?gradients/model_2/conv1/BiasAdd_grad/tuple/control_dependency_1*
T0*C
_class9
75loc:@gradients/model_1/conv1/BiasAdd_grad/BiasAddGrad*
N*
_output_shapes
: 
▐
gradients/AddN_10AddN>gradients/model_1/conv1/Conv2D_grad/tuple/control_dependency_1<gradients/model/conv1/Conv2D_grad/tuple/control_dependency_1>gradients/model_2/conv1/Conv2D_grad/tuple/control_dependency_1*&
_output_shapes
: *
T0*K
_classA
?=loc:@gradients/model_1/conv1/Conv2D_grad/Conv2DBackpropFilter*
N
│
8conv1/weights/Momentum/Initializer/zeros/shape_as_tensorConst*
_output_shapes
:*%
valueB"             * 
_class
loc:@conv1/weights*
dtype0
Х
.conv1/weights/Momentum/Initializer/zeros/ConstConst*
valueB
 *    * 
_class
loc:@conv1/weights*
dtype0*
_output_shapes
: 
 
(conv1/weights/Momentum/Initializer/zerosFill8conv1/weights/Momentum/Initializer/zeros/shape_as_tensor.conv1/weights/Momentum/Initializer/zeros/Const*

index_type0* 
_class
loc:@conv1/weights*&
_output_shapes
: *
T0
╝
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
conv1/weights/Momentum/AssignAssignconv1/weights/Momentum(conv1/weights/Momentum/Initializer/zeros*
T0* 
_class
loc:@conv1/weights*
validate_shape(*&
_output_shapes
: *
use_locking(
Т
conv1/weights/Momentum/readIdentityconv1/weights/Momentum*&
_output_shapes
: *
T0* 
_class
loc:@conv1/weights
Х
'conv1/biases/Momentum/Initializer/zerosConst*
valueB *    *
_class
loc:@conv1/biases*
dtype0*
_output_shapes
: 
в
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
╒
conv1/biases/Momentum/AssignAssignconv1/biases/Momentum'conv1/biases/Momentum/Initializer/zeros*
use_locking(*
T0*
_class
loc:@conv1/biases*
validate_shape(*
_output_shapes
: 
Г
conv1/biases/Momentum/readIdentityconv1/biases/Momentum*
T0*
_class
loc:@conv1/biases*
_output_shapes
: 
│
8conv2/weights/Momentum/Initializer/zeros/shape_as_tensorConst*%
valueB"          @   * 
_class
loc:@conv2/weights*
dtype0*
_output_shapes
:
Х
.conv2/weights/Momentum/Initializer/zeros/ConstConst*
valueB
 *    * 
_class
loc:@conv2/weights*
dtype0*
_output_shapes
: 
 
(conv2/weights/Momentum/Initializer/zerosFill8conv2/weights/Momentum/Initializer/zeros/shape_as_tensor.conv2/weights/Momentum/Initializer/zeros/Const*
T0*

index_type0* 
_class
loc:@conv2/weights*&
_output_shapes
: @
╝
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
Т
conv2/weights/Momentum/readIdentityconv2/weights/Momentum*
T0* 
_class
loc:@conv2/weights*&
_output_shapes
: @
Х
'conv2/biases/Momentum/Initializer/zerosConst*
valueB@*    *
_class
loc:@conv2/biases*
dtype0*
_output_shapes
:@
в
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
╒
conv2/biases/Momentum/AssignAssignconv2/biases/Momentum'conv2/biases/Momentum/Initializer/zeros*
T0*
_class
loc:@conv2/biases*
validate_shape(*
_output_shapes
:@*
use_locking(
Г
conv2/biases/Momentum/readIdentityconv2/biases/Momentum*
T0*
_class
loc:@conv2/biases*
_output_shapes
:@
│
8conv3/weights/Momentum/Initializer/zeros/shape_as_tensorConst*%
valueB"      @   А   * 
_class
loc:@conv3/weights*
dtype0*
_output_shapes
:
Х
.conv3/weights/Momentum/Initializer/zeros/ConstConst*
valueB
 *    * 
_class
loc:@conv3/weights*
dtype0*
_output_shapes
: 
А
(conv3/weights/Momentum/Initializer/zerosFill8conv3/weights/Momentum/Initializer/zeros/shape_as_tensor.conv3/weights/Momentum/Initializer/zeros/Const*
T0*

index_type0* 
_class
loc:@conv3/weights*'
_output_shapes
:@А
╛
conv3/weights/Momentum
VariableV2* 
_class
loc:@conv3/weights*
	container *
shape:@А*
dtype0*'
_output_shapes
:@А*
shared_name 
ц
conv3/weights/Momentum/AssignAssignconv3/weights/Momentum(conv3/weights/Momentum/Initializer/zeros*
use_locking(*
T0* 
_class
loc:@conv3/weights*
validate_shape(*'
_output_shapes
:@А
У
conv3/weights/Momentum/readIdentityconv3/weights/Momentum*
T0* 
_class
loc:@conv3/weights*'
_output_shapes
:@А
Ч
'conv3/biases/Momentum/Initializer/zerosConst*
valueBА*    *
_class
loc:@conv3/biases*
dtype0*
_output_shapes	
:А
д
conv3/biases/Momentum
VariableV2*
shared_name *
_class
loc:@conv3/biases*
	container *
shape:А*
dtype0*
_output_shapes	
:А
╓
conv3/biases/Momentum/AssignAssignconv3/biases/Momentum'conv3/biases/Momentum/Initializer/zeros*
use_locking(*
T0*
_class
loc:@conv3/biases*
validate_shape(*
_output_shapes	
:А
Д
conv3/biases/Momentum/readIdentityconv3/biases/Momentum*
T0*
_class
loc:@conv3/biases*
_output_shapes	
:А
│
8conv4/weights/Momentum/Initializer/zeros/shape_as_tensorConst*%
valueB"      А      * 
_class
loc:@conv4/weights*
dtype0*
_output_shapes
:
Х
.conv4/weights/Momentum/Initializer/zeros/ConstConst*
valueB
 *    * 
_class
loc:@conv4/weights*
dtype0*
_output_shapes
: 
Б
(conv4/weights/Momentum/Initializer/zerosFill8conv4/weights/Momentum/Initializer/zeros/shape_as_tensor.conv4/weights/Momentum/Initializer/zeros/Const*
T0*

index_type0* 
_class
loc:@conv4/weights*(
_output_shapes
:АА
└
conv4/weights/Momentum
VariableV2*
shape:АА*
dtype0*(
_output_shapes
:АА*
shared_name * 
_class
loc:@conv4/weights*
	container 
ч
conv4/weights/Momentum/AssignAssignconv4/weights/Momentum(conv4/weights/Momentum/Initializer/zeros*
validate_shape(*(
_output_shapes
:АА*
use_locking(*
T0* 
_class
loc:@conv4/weights
Ф
conv4/weights/Momentum/readIdentityconv4/weights/Momentum*(
_output_shapes
:АА*
T0* 
_class
loc:@conv4/weights
Ч
'conv4/biases/Momentum/Initializer/zerosConst*
dtype0*
_output_shapes	
:А*
valueBА*    *
_class
loc:@conv4/biases
д
conv4/biases/Momentum
VariableV2*
_class
loc:@conv4/biases*
	container *
shape:А*
dtype0*
_output_shapes	
:А*
shared_name 
╓
conv4/biases/Momentum/AssignAssignconv4/biases/Momentum'conv4/biases/Momentum/Initializer/zeros*
use_locking(*
T0*
_class
loc:@conv4/biases*
validate_shape(*
_output_shapes	
:А
Д
conv4/biases/Momentum/readIdentityconv4/biases/Momentum*
_output_shapes	
:А*
T0*
_class
loc:@conv4/biases
│
8conv5/weights/Momentum/Initializer/zeros/shape_as_tensorConst*%
valueB"            * 
_class
loc:@conv5/weights*
dtype0*
_output_shapes
:
Х
.conv5/weights/Momentum/Initializer/zeros/ConstConst*
valueB
 *    * 
_class
loc:@conv5/weights*
dtype0*
_output_shapes
: 
А
(conv5/weights/Momentum/Initializer/zerosFill8conv5/weights/Momentum/Initializer/zeros/shape_as_tensor.conv5/weights/Momentum/Initializer/zeros/Const*'
_output_shapes
:А*
T0*

index_type0* 
_class
loc:@conv5/weights
╛
conv5/weights/Momentum
VariableV2* 
_class
loc:@conv5/weights*
	container *
shape:А*
dtype0*'
_output_shapes
:А*
shared_name 
ц
conv5/weights/Momentum/AssignAssignconv5/weights/Momentum(conv5/weights/Momentum/Initializer/zeros*
use_locking(*
T0* 
_class
loc:@conv5/weights*
validate_shape(*'
_output_shapes
:А
У
conv5/weights/Momentum/readIdentityconv5/weights/Momentum*
T0* 
_class
loc:@conv5/weights*'
_output_shapes
:А
Х
'conv5/biases/Momentum/Initializer/zerosConst*
valueB*    *
_class
loc:@conv5/biases*
dtype0*
_output_shapes
:
в
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
╒
conv5/biases/Momentum/AssignAssignconv5/biases/Momentum'conv5/biases/Momentum/Initializer/zeros*
_output_shapes
:*
use_locking(*
T0*
_class
loc:@conv5/biases*
validate_shape(
Г
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
╫#<*
dtype0*
_output_shapes
: 
V
Momentum/momentumConst*
valueB
 *дp}?*
dtype0*
_output_shapes
: 
Ы
+Momentum/update_conv1/weights/ApplyMomentumApplyMomentumconv1/weightsconv1/weights/MomentumMomentum/learning_rategradients/AddN_10Momentum/momentum*
use_locking( *
T0* 
_class
loc:@conv1/weights*
use_nesterov(*&
_output_shapes
: 
К
*Momentum/update_conv1/biases/ApplyMomentumApplyMomentumconv1/biasesconv1/biases/MomentumMomentum/learning_rategradients/AddN_9Momentum/momentum*
use_locking( *
T0*
_class
loc:@conv1/biases*
use_nesterov(*
_output_shapes
: 
Ъ
+Momentum/update_conv2/weights/ApplyMomentumApplyMomentumconv2/weightsconv2/weights/MomentumMomentum/learning_rategradients/AddN_8Momentum/momentum*
use_nesterov(*&
_output_shapes
: @*
use_locking( *
T0* 
_class
loc:@conv2/weights
К
*Momentum/update_conv2/biases/ApplyMomentumApplyMomentumconv2/biasesconv2/biases/MomentumMomentum/learning_rategradients/AddN_7Momentum/momentum*
use_locking( *
T0*
_class
loc:@conv2/biases*
use_nesterov(*
_output_shapes
:@
Ы
+Momentum/update_conv3/weights/ApplyMomentumApplyMomentumconv3/weightsconv3/weights/MomentumMomentum/learning_rategradients/AddN_6Momentum/momentum*
T0* 
_class
loc:@conv3/weights*
use_nesterov(*'
_output_shapes
:@А*
use_locking( 
Л
*Momentum/update_conv3/biases/ApplyMomentumApplyMomentumconv3/biasesconv3/biases/MomentumMomentum/learning_rategradients/AddN_5Momentum/momentum*
use_locking( *
T0*
_class
loc:@conv3/biases*
use_nesterov(*
_output_shapes	
:А
Ь
+Momentum/update_conv4/weights/ApplyMomentumApplyMomentumconv4/weightsconv4/weights/MomentumMomentum/learning_rategradients/AddN_4Momentum/momentum*
use_nesterov(*(
_output_shapes
:АА*
use_locking( *
T0* 
_class
loc:@conv4/weights
Л
*Momentum/update_conv4/biases/ApplyMomentumApplyMomentumconv4/biasesconv4/biases/MomentumMomentum/learning_rategradients/AddN_3Momentum/momentum*
use_locking( *
T0*
_class
loc:@conv4/biases*
use_nesterov(*
_output_shapes	
:А
Ы
+Momentum/update_conv5/weights/ApplyMomentumApplyMomentumconv5/weightsconv5/weights/MomentumMomentum/learning_rategradients/AddN_2Momentum/momentum*
use_locking( *
T0* 
_class
loc:@conv5/weights*
use_nesterov(*'
_output_shapes
:А
К
*Momentum/update_conv5/biases/ApplyMomentumApplyMomentumconv5/biasesconv5/biases/MomentumMomentum/learning_rategradients/AddN_1Momentum/momentum*
_output_shapes
:*
use_locking( *
T0*
_class
loc:@conv5/biases*
use_nesterov(
▐
Momentum/updateNoOp+^Momentum/update_conv1/biases/ApplyMomentum,^Momentum/update_conv1/weights/ApplyMomentum+^Momentum/update_conv2/biases/ApplyMomentum,^Momentum/update_conv2/weights/ApplyMomentum+^Momentum/update_conv3/biases/ApplyMomentum,^Momentum/update_conv3/weights/ApplyMomentum+^Momentum/update_conv4/biases/ApplyMomentum,^Momentum/update_conv4/weights/ApplyMomentum+^Momentum/update_conv5/biases/ApplyMomentum,^Momentum/update_conv5/weights/ApplyMomentum

Momentum/valueConst^Momentum/update*
_class
loc:@Variable*
value	B :*
dtype0*
_output_shapes
: 
А
Momentum	AssignAddVariableMomentum/value*
_class
loc:@Variable*
_output_shapes
: *
use_locking( *
T0
Y
save/filename/inputConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
n
save/filenamePlaceholderWithDefaultsave/filename/input*
_output_shapes
: *
shape: *
dtype0
e

save/ConstPlaceholderWithDefaultsave/filename*
dtype0*
_output_shapes
: *
shape: 
ш
save/SaveV2/tensor_namesConst*Ы
valueСBОBVariableBconv1/biasesBconv1/biases/MomentumBconv1/weightsBconv1/weights/MomentumBconv2/biasesBconv2/biases/MomentumBconv2/weightsBconv2/weights/MomentumBconv3/biasesBconv3/biases/MomentumBconv3/weightsBconv3/weights/MomentumBconv4/biasesBconv4/biases/MomentumBconv4/weightsBconv4/weights/MomentumBconv5/biasesBconv5/biases/MomentumBconv5/weightsBconv5/weights/Momentum*
dtype0*
_output_shapes
:
Н
save/SaveV2/shape_and_slicesConst*=
value4B2B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:
Д
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
·
save/RestoreV2/tensor_namesConst"/device:CPU:0*
dtype0*
_output_shapes
:*Ы
valueСBОBVariableBconv1/biasesBconv1/biases/MomentumBconv1/weightsBconv1/weights/MomentumBconv2/biasesBconv2/biases/MomentumBconv2/weightsBconv2/weights/MomentumBconv3/biasesBconv3/biases/MomentumBconv3/weightsBconv3/weights/MomentumBconv4/biasesBconv4/biases/MomentumBconv4/weightsBconv4/weights/MomentumBconv5/biasesBconv5/biases/MomentumBconv5/weightsBconv5/weights/Momentum
Я
save/RestoreV2/shape_and_slicesConst"/device:CPU:0*=
value4B2B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:
Г
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices"/device:CPU:0*h
_output_shapesV
T:::::::::::::::::::::*#
dtypes
2
Ц
save/AssignAssignVariablesave/RestoreV2*
T0*
_class
loc:@Variable*
validate_shape(*
_output_shapes
: *
use_locking(
ж
save/Assign_1Assignconv1/biasessave/RestoreV2:1*
T0*
_class
loc:@conv1/biases*
validate_shape(*
_output_shapes
: *
use_locking(
п
save/Assign_2Assignconv1/biases/Momentumsave/RestoreV2:2*
use_locking(*
T0*
_class
loc:@conv1/biases*
validate_shape(*
_output_shapes
: 
┤
save/Assign_3Assignconv1/weightssave/RestoreV2:3*
validate_shape(*&
_output_shapes
: *
use_locking(*
T0* 
_class
loc:@conv1/weights
╜
save/Assign_4Assignconv1/weights/Momentumsave/RestoreV2:4*
use_locking(*
T0* 
_class
loc:@conv1/weights*
validate_shape(*&
_output_shapes
: 
ж
save/Assign_5Assignconv2/biasessave/RestoreV2:5*
_output_shapes
:@*
use_locking(*
T0*
_class
loc:@conv2/biases*
validate_shape(
п
save/Assign_6Assignconv2/biases/Momentumsave/RestoreV2:6*
_output_shapes
:@*
use_locking(*
T0*
_class
loc:@conv2/biases*
validate_shape(
┤
save/Assign_7Assignconv2/weightssave/RestoreV2:7*
validate_shape(*&
_output_shapes
: @*
use_locking(*
T0* 
_class
loc:@conv2/weights
╜
save/Assign_8Assignconv2/weights/Momentumsave/RestoreV2:8*
use_locking(*
T0* 
_class
loc:@conv2/weights*
validate_shape(*&
_output_shapes
: @
з
save/Assign_9Assignconv3/biasessave/RestoreV2:9*
_output_shapes	
:А*
use_locking(*
T0*
_class
loc:@conv3/biases*
validate_shape(
▓
save/Assign_10Assignconv3/biases/Momentumsave/RestoreV2:10*
use_locking(*
T0*
_class
loc:@conv3/biases*
validate_shape(*
_output_shapes	
:А
╖
save/Assign_11Assignconv3/weightssave/RestoreV2:11*
validate_shape(*'
_output_shapes
:@А*
use_locking(*
T0* 
_class
loc:@conv3/weights
└
save/Assign_12Assignconv3/weights/Momentumsave/RestoreV2:12*
T0* 
_class
loc:@conv3/weights*
validate_shape(*'
_output_shapes
:@А*
use_locking(
й
save/Assign_13Assignconv4/biasessave/RestoreV2:13*
_output_shapes	
:А*
use_locking(*
T0*
_class
loc:@conv4/biases*
validate_shape(
▓
save/Assign_14Assignconv4/biases/Momentumsave/RestoreV2:14*
T0*
_class
loc:@conv4/biases*
validate_shape(*
_output_shapes	
:А*
use_locking(
╕
save/Assign_15Assignconv4/weightssave/RestoreV2:15*
use_locking(*
T0* 
_class
loc:@conv4/weights*
validate_shape(*(
_output_shapes
:АА
┴
save/Assign_16Assignconv4/weights/Momentumsave/RestoreV2:16*
use_locking(*
T0* 
_class
loc:@conv4/weights*
validate_shape(*(
_output_shapes
:АА
и
save/Assign_17Assignconv5/biasessave/RestoreV2:17*
T0*
_class
loc:@conv5/biases*
validate_shape(*
_output_shapes
:*
use_locking(
▒
save/Assign_18Assignconv5/biases/Momentumsave/RestoreV2:18*
_output_shapes
:*
use_locking(*
T0*
_class
loc:@conv5/biases*
validate_shape(
╖
save/Assign_19Assignconv5/weightssave/RestoreV2:19*
use_locking(*
T0* 
_class
loc:@conv5/weights*
validate_shape(*'
_output_shapes
:А
└
save/Assign_20Assignconv5/weights/Momentumsave/RestoreV2:20*
use_locking(*
T0* 
_class
loc:@conv5/weights*
validate_shape(*'
_output_shapes
:А
ё
save/restore_allNoOp^save/Assign^save/Assign_1^save/Assign_10^save/Assign_11^save/Assign_12^save/Assign_13^save/Assign_14^save/Assign_15^save/Assign_16^save/Assign_17^save/Assign_18^save/Assign_19^save/Assign_2^save/Assign_20^save/Assign_3^save/Assign_4^save/Assign_5^save/Assign_6^save/Assign_7^save/Assign_8^save/Assign_9
║
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
conv3/biases_1/tagConst*
_output_shapes
: *
valueB Bconv3/biases_1*
dtype0
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
conv5/biases_1HistogramSummaryconv5/biases_1/tagconv5/biases/read*
T0*
_output_shapes
: 
Ї
Merge/MergeSummaryMergeSummarysteplossconv1/weights_1conv1/biases_1conv2/weights_1conv2/biases_1conv3/weights_1conv3/biases_1conv4/weights_1conv4/biases_1conv5/weights_1conv5/biases_1*
N*
_output_shapes
: ""
train_op


Momentum"О
	variablesА¤
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
Д
conv1/weights/Momentum:0conv1/weights/Momentum/Assignconv1/weights/Momentum/read:02*conv1/weights/Momentum/Initializer/zeros:0
А
conv1/biases/Momentum:0conv1/biases/Momentum/Assignconv1/biases/Momentum/read:02)conv1/biases/Momentum/Initializer/zeros:0
Д
conv2/weights/Momentum:0conv2/weights/Momentum/Assignconv2/weights/Momentum/read:02*conv2/weights/Momentum/Initializer/zeros:0
А
conv2/biases/Momentum:0conv2/biases/Momentum/Assignconv2/biases/Momentum/read:02)conv2/biases/Momentum/Initializer/zeros:0
Д
conv3/weights/Momentum:0conv3/weights/Momentum/Assignconv3/weights/Momentum/read:02*conv3/weights/Momentum/Initializer/zeros:0
А
conv3/biases/Momentum:0conv3/biases/Momentum/Assignconv3/biases/Momentum/read:02)conv3/biases/Momentum/Initializer/zeros:0
Д
conv4/weights/Momentum:0conv4/weights/Momentum/Assignconv4/weights/Momentum/read:02*conv4/weights/Momentum/Initializer/zeros:0
А
conv4/biases/Momentum:0conv4/biases/Momentum/Assignconv4/biases/Momentum/read:02)conv4/biases/Momentum/Initializer/zeros:0
Д
conv5/weights/Momentum:0conv5/weights/Momentum/Assignconv5/weights/Momentum/read:02*conv5/weights/Momentum/Initializer/zeros:0
А
conv5/biases/Momentum:0conv5/biases/Momentum/Assignconv5/biases/Momentum/read:02)conv5/biases/Momentum/Initializer/zeros:0"Ш
model_variablesДБ
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
conv5/biases:0conv5/biases/Assignconv5/biases/read:02 conv5/biases/Initializer/zeros:08"┌
	summaries╠
╔
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
conv5/biases_1:0"Ь
trainable_variablesДБ
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
conv5/biases:0conv5/biases/Assignconv5/biases/read:02 conv5/biases/Initializer/zeros:08Б/╘9