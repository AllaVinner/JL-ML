??
??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
?
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

?
Conv2DBackpropInput
input_sizes
filter"T
out_backprop"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
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
@
ReadVariableOp
resource
value"dtype"
dtypetype?
E
Relu
features"T
activations"T"
Ttype:
2	
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
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
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
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
@
StaticRegexFullMatch	
input

output
"
patternstring
?
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
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
P
Unpack

value"T
output"T*num"
numint("	
Ttype"
axisint 
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.6.02v2.6.0-0-g919f693420e8??
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_2
[
total_2/Read/ReadVariableOpReadVariableOptotal_2*
_output_shapes
: *
dtype0
b
count_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_2
[
count_2/Read/ReadVariableOpReadVariableOpcount_2*
_output_shapes
: *
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
?
conv2d_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: * 
shared_nameconv2d_4/kernel
{
#conv2d_4/kernel/Read/ReadVariableOpReadVariableOpconv2d_4/kernel*&
_output_shapes
: *
dtype0
r
conv2d_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_4/bias
k
!conv2d_4/bias/Read/ReadVariableOpReadVariableOpconv2d_4/bias*
_output_shapes
: *
dtype0
?
conv2d_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @* 
shared_nameconv2d_5/kernel
{
#conv2d_5/kernel/Read/ReadVariableOpReadVariableOpconv2d_5/kernel*&
_output_shapes
: @*
dtype0
r
conv2d_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_5/bias
k
!conv2d_5/bias/Read/ReadVariableOpReadVariableOpconv2d_5/bias*
_output_shapes
:@*
dtype0
y
dense_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*
shared_namedense_4/kernel
r
"dense_4/kernel/Read/ReadVariableOpReadVariableOpdense_4/kernel*
_output_shapes
:	?*
dtype0
p
dense_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_4/bias
i
 dense_4/bias/Read/ReadVariableOpReadVariableOpdense_4/bias*
_output_shapes
:*
dtype0
y
dense_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*
shared_namedense_5/kernel
r
"dense_5/kernel/Read/ReadVariableOpReadVariableOpdense_5/kernel*
_output_shapes
:	?*
dtype0
q
dense_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_5/bias
j
 dense_5/bias/Read/ReadVariableOpReadVariableOpdense_5/bias*
_output_shapes	
:?*
dtype0
?
conv2d_transpose_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@ **
shared_nameconv2d_transpose_6/kernel
?
-conv2d_transpose_6/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_6/kernel*&
_output_shapes
:@ *
dtype0
?
conv2d_transpose_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*(
shared_nameconv2d_transpose_6/bias

+conv2d_transpose_6/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose_6/bias*
_output_shapes
:@*
dtype0
?
conv2d_transpose_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @**
shared_nameconv2d_transpose_7/kernel
?
-conv2d_transpose_7/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_7/kernel*&
_output_shapes
: @*
dtype0
?
conv2d_transpose_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameconv2d_transpose_7/bias

+conv2d_transpose_7/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose_7/bias*
_output_shapes
: *
dtype0
?
conv2d_transpose_8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: **
shared_nameconv2d_transpose_8/kernel
?
-conv2d_transpose_8/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_8/kernel*&
_output_shapes
: *
dtype0
?
conv2d_transpose_8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameconv2d_transpose_8/bias

+conv2d_transpose_8/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose_8/bias*
_output_shapes
:*
dtype0
?
Adam/conv2d_4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_nameAdam/conv2d_4/kernel/m
?
*Adam/conv2d_4/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_4/kernel/m*&
_output_shapes
: *
dtype0
?
Adam/conv2d_4/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/conv2d_4/bias/m
y
(Adam/conv2d_4/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_4/bias/m*
_output_shapes
: *
dtype0
?
Adam/conv2d_5/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*'
shared_nameAdam/conv2d_5/kernel/m
?
*Adam/conv2d_5/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_5/kernel/m*&
_output_shapes
: @*
dtype0
?
Adam/conv2d_5/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/conv2d_5/bias/m
y
(Adam/conv2d_5/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_5/bias/m*
_output_shapes
:@*
dtype0
?
Adam/dense_4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*&
shared_nameAdam/dense_4/kernel/m
?
)Adam/dense_4/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_4/kernel/m*
_output_shapes
:	?*
dtype0
~
Adam/dense_4/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_4/bias/m
w
'Adam/dense_4/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_4/bias/m*
_output_shapes
:*
dtype0
?
Adam/dense_5/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*&
shared_nameAdam/dense_5/kernel/m
?
)Adam/dense_5/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_5/kernel/m*
_output_shapes
:	?*
dtype0

Adam/dense_5/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*$
shared_nameAdam/dense_5/bias/m
x
'Adam/dense_5/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_5/bias/m*
_output_shapes	
:?*
dtype0
?
 Adam/conv2d_transpose_6/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@ *1
shared_name" Adam/conv2d_transpose_6/kernel/m
?
4Adam/conv2d_transpose_6/kernel/m/Read/ReadVariableOpReadVariableOp Adam/conv2d_transpose_6/kernel/m*&
_output_shapes
:@ *
dtype0
?
Adam/conv2d_transpose_6/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*/
shared_name Adam/conv2d_transpose_6/bias/m
?
2Adam/conv2d_transpose_6/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_transpose_6/bias/m*
_output_shapes
:@*
dtype0
?
 Adam/conv2d_transpose_7/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*1
shared_name" Adam/conv2d_transpose_7/kernel/m
?
4Adam/conv2d_transpose_7/kernel/m/Read/ReadVariableOpReadVariableOp Adam/conv2d_transpose_7/kernel/m*&
_output_shapes
: @*
dtype0
?
Adam/conv2d_transpose_7/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: */
shared_name Adam/conv2d_transpose_7/bias/m
?
2Adam/conv2d_transpose_7/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_transpose_7/bias/m*
_output_shapes
: *
dtype0
?
 Adam/conv2d_transpose_8/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *1
shared_name" Adam/conv2d_transpose_8/kernel/m
?
4Adam/conv2d_transpose_8/kernel/m/Read/ReadVariableOpReadVariableOp Adam/conv2d_transpose_8/kernel/m*&
_output_shapes
: *
dtype0
?
Adam/conv2d_transpose_8/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name Adam/conv2d_transpose_8/bias/m
?
2Adam/conv2d_transpose_8/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_transpose_8/bias/m*
_output_shapes
:*
dtype0
?
Adam/conv2d_4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_nameAdam/conv2d_4/kernel/v
?
*Adam/conv2d_4/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_4/kernel/v*&
_output_shapes
: *
dtype0
?
Adam/conv2d_4/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/conv2d_4/bias/v
y
(Adam/conv2d_4/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_4/bias/v*
_output_shapes
: *
dtype0
?
Adam/conv2d_5/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*'
shared_nameAdam/conv2d_5/kernel/v
?
*Adam/conv2d_5/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_5/kernel/v*&
_output_shapes
: @*
dtype0
?
Adam/conv2d_5/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/conv2d_5/bias/v
y
(Adam/conv2d_5/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_5/bias/v*
_output_shapes
:@*
dtype0
?
Adam/dense_4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*&
shared_nameAdam/dense_4/kernel/v
?
)Adam/dense_4/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_4/kernel/v*
_output_shapes
:	?*
dtype0
~
Adam/dense_4/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_4/bias/v
w
'Adam/dense_4/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_4/bias/v*
_output_shapes
:*
dtype0
?
Adam/dense_5/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*&
shared_nameAdam/dense_5/kernel/v
?
)Adam/dense_5/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_5/kernel/v*
_output_shapes
:	?*
dtype0

Adam/dense_5/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*$
shared_nameAdam/dense_5/bias/v
x
'Adam/dense_5/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_5/bias/v*
_output_shapes	
:?*
dtype0
?
 Adam/conv2d_transpose_6/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@ *1
shared_name" Adam/conv2d_transpose_6/kernel/v
?
4Adam/conv2d_transpose_6/kernel/v/Read/ReadVariableOpReadVariableOp Adam/conv2d_transpose_6/kernel/v*&
_output_shapes
:@ *
dtype0
?
Adam/conv2d_transpose_6/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*/
shared_name Adam/conv2d_transpose_6/bias/v
?
2Adam/conv2d_transpose_6/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_transpose_6/bias/v*
_output_shapes
:@*
dtype0
?
 Adam/conv2d_transpose_7/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*1
shared_name" Adam/conv2d_transpose_7/kernel/v
?
4Adam/conv2d_transpose_7/kernel/v/Read/ReadVariableOpReadVariableOp Adam/conv2d_transpose_7/kernel/v*&
_output_shapes
: @*
dtype0
?
Adam/conv2d_transpose_7/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: */
shared_name Adam/conv2d_transpose_7/bias/v
?
2Adam/conv2d_transpose_7/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_transpose_7/bias/v*
_output_shapes
: *
dtype0
?
 Adam/conv2d_transpose_8/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *1
shared_name" Adam/conv2d_transpose_8/kernel/v
?
4Adam/conv2d_transpose_8/kernel/v/Read/ReadVariableOpReadVariableOp Adam/conv2d_transpose_8/kernel/v*&
_output_shapes
: *
dtype0
?
Adam/conv2d_transpose_8/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name Adam/conv2d_transpose_8/bias/v
?
2Adam/conv2d_transpose_8/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_transpose_8/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
?`
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?`
value?`B?` B?`
?
encoder
decoder
sampler
loss_tracker_total
loss_tracker_reconstruction
loss_tracker_latent
	optimizer
loss
	trainable_variables

regularization_losses
	variables
	keras_api

signatures
?
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer-5
trainable_variables
regularization_losses
	variables
	keras_api
?
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer-5
trainable_variables
regularization_losses
 	variables
!	keras_api
R
"regularization_losses
#trainable_variables
$	variables
%	keras_api
4
	&total
	'count
(	variables
)	keras_api
4
	*total
	+count
,	variables
-	keras_api
4
	.total
	/count
0	variables
1	keras_api
?
2iter

3beta_1

4beta_2
	5decay
6learning_rate7m?8m?9m?:m?;m?<m?=m?>m??m?@m?Am?Bm?Cm?Dm?7v?8v?9v?:v?;v?<v?=v?>v??v?@v?Av?Bv?Cv?Dv?
 
f
70
81
92
:3
;4
<5
=6
>7
?8
@9
A10
B11
C12
D13
 
?
70
81
92
:3
;4
<5
=6
>7
?8
@9
A10
B11
C12
D13
&14
'15
*16
+17
.18
/19
?
Elayer_metrics

Flayers
Glayer_regularization_losses
	trainable_variables

regularization_losses
	variables
Hnon_trainable_variables
Imetrics
 
R
Jregularization_losses
Ktrainable_variables
L	variables
M	keras_api
h

7kernel
8bias
Nregularization_losses
Otrainable_variables
P	variables
Q	keras_api
h

9kernel
:bias
Rregularization_losses
Strainable_variables
T	variables
U	keras_api
R
Vregularization_losses
Wtrainable_variables
X	variables
Y	keras_api
h

;kernel
<bias
Zregularization_losses
[trainable_variables
\	variables
]	keras_api
R
^regularization_losses
_trainable_variables
`	variables
a	keras_api
*
70
81
92
:3
;4
<5
 
*
70
81
92
:3
;4
<5
?
blayer_metrics

clayers
dlayer_regularization_losses
trainable_variables
regularization_losses
	variables
enon_trainable_variables
fmetrics
h

=kernel
>bias
gregularization_losses
htrainable_variables
i	variables
j	keras_api
R
kregularization_losses
ltrainable_variables
m	variables
n	keras_api
h

?kernel
@bias
oregularization_losses
ptrainable_variables
q	variables
r	keras_api
h

Akernel
Bbias
sregularization_losses
ttrainable_variables
u	variables
v	keras_api
h

Ckernel
Dbias
wregularization_losses
xtrainable_variables
y	variables
z	keras_api
R
{regularization_losses
|trainable_variables
}	variables
~	keras_api
8
=0
>1
?2
@3
A4
B5
C6
D7
 
8
=0
>1
?2
@3
A4
B5
C6
D7
?
layer_metrics
?layers
 ?layer_regularization_losses
trainable_variables
regularization_losses
 	variables
?non_trainable_variables
?metrics
 
 
 
?
?layer_metrics
?non_trainable_variables
 ?layer_regularization_losses
"regularization_losses
#trainable_variables
$	variables
?layers
?metrics
NL
VARIABLE_VALUEtotal3loss_tracker_total/total/.ATTRIBUTES/VARIABLE_VALUE
NL
VARIABLE_VALUEcount3loss_tracker_total/count/.ATTRIBUTES/VARIABLE_VALUE

&0
'1

(	variables
YW
VARIABLE_VALUEtotal_1<loss_tracker_reconstruction/total/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEcount_1<loss_tracker_reconstruction/count/.ATTRIBUTES/VARIABLE_VALUE

*0
+1

,	variables
QO
VARIABLE_VALUEtotal_24loss_tracker_latent/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_24loss_tracker_latent/count/.ATTRIBUTES/VARIABLE_VALUE

.0
/1

0	variables
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEconv2d_4/kernel0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEconv2d_4/bias0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEconv2d_5/kernel0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEconv2d_5/bias0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEdense_4/kernel0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUEdense_4/bias0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEdense_5/kernel0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUEdense_5/bias0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUEconv2d_transpose_6/kernel0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUEconv2d_transpose_6/bias0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUE
`^
VARIABLE_VALUEconv2d_transpose_7/kernel1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUEconv2d_transpose_7/bias1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUE
`^
VARIABLE_VALUEconv2d_transpose_8/kernel1trainable_variables/12/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUEconv2d_transpose_8/bias1trainable_variables/13/.ATTRIBUTES/VARIABLE_VALUE
:

total_loss
reconstruction_loss
latent_loss

0
1
2
 
*
&0
'1
*2
+3
.4
/5

0
1
2
 
 
 
?
?layer_metrics
?non_trainable_variables
 ?layer_regularization_losses
Jregularization_losses
Ktrainable_variables
L	variables
?layers
?metrics
 

70
81

70
81
?
?layer_metrics
?non_trainable_variables
 ?layer_regularization_losses
Nregularization_losses
Otrainable_variables
P	variables
?layers
?metrics
 

90
:1

90
:1
?
?layer_metrics
?non_trainable_variables
 ?layer_regularization_losses
Rregularization_losses
Strainable_variables
T	variables
?layers
?metrics
 
 
 
?
?layer_metrics
?non_trainable_variables
 ?layer_regularization_losses
Vregularization_losses
Wtrainable_variables
X	variables
?layers
?metrics
 

;0
<1

;0
<1
?
?layer_metrics
?non_trainable_variables
 ?layer_regularization_losses
Zregularization_losses
[trainable_variables
\	variables
?layers
?metrics
 
 
 
?
?layer_metrics
?non_trainable_variables
 ?layer_regularization_losses
^regularization_losses
_trainable_variables
`	variables
?layers
?metrics
 
*
0
1
2
3
4
5
 
 
 
 

=0
>1

=0
>1
?
?layer_metrics
?non_trainable_variables
 ?layer_regularization_losses
gregularization_losses
htrainable_variables
i	variables
?layers
?metrics
 
 
 
?
?layer_metrics
?non_trainable_variables
 ?layer_regularization_losses
kregularization_losses
ltrainable_variables
m	variables
?layers
?metrics
 

?0
@1

?0
@1
?
?layer_metrics
?non_trainable_variables
 ?layer_regularization_losses
oregularization_losses
ptrainable_variables
q	variables
?layers
?metrics
 

A0
B1

A0
B1
?
?layer_metrics
?non_trainable_variables
 ?layer_regularization_losses
sregularization_losses
ttrainable_variables
u	variables
?layers
?metrics
 

C0
D1

C0
D1
?
?layer_metrics
?non_trainable_variables
 ?layer_regularization_losses
wregularization_losses
xtrainable_variables
y	variables
?layers
?metrics
 
 
 
?
?layer_metrics
?non_trainable_variables
 ?layer_regularization_losses
{regularization_losses
|trainable_variables
}	variables
?layers
?metrics
 
*
0
1
2
3
4
5
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
xv
VARIABLE_VALUEAdam/conv2d_4/kernel/mLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/conv2d_4/bias/mLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/conv2d_5/kernel/mLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/conv2d_5/bias/mLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense_4/kernel/mLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUEAdam/dense_4/bias/mLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense_5/kernel/mLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUEAdam/dense_5/bias/mLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE Adam/conv2d_transpose_6/kernel/mLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/conv2d_transpose_6/bias/mLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE Adam/conv2d_transpose_7/kernel/mMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUEAdam/conv2d_transpose_7/bias/mMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE Adam/conv2d_transpose_8/kernel/mMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUEAdam/conv2d_transpose_8/bias/mMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/conv2d_4/kernel/vLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/conv2d_4/bias/vLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/conv2d_5/kernel/vLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/conv2d_5/bias/vLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense_4/kernel/vLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUEAdam/dense_4/bias/vLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense_5/kernel/vLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUEAdam/dense_5/bias/vLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE Adam/conv2d_transpose_6/kernel/vLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/conv2d_transpose_6/bias/vLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE Adam/conv2d_transpose_7/kernel/vMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUEAdam/conv2d_transpose_7/bias/vMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE Adam/conv2d_transpose_8/kernel/vMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUEAdam/conv2d_transpose_8/bias/vMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
serving_default_input_1Placeholder*/
_output_shapes
:?????????*
dtype0*$
shape:?????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1conv2d_4/kernelconv2d_4/biasconv2d_5/kernelconv2d_5/biasdense_4/kerneldense_4/biasdense_5/kerneldense_5/biasconv2d_transpose_6/kernelconv2d_transpose_6/biasconv2d_transpose_7/kernelconv2d_transpose_7/biasconv2d_transpose_8/kernelconv2d_transpose_8/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *,
f'R%
#__inference_signature_wrapper_46212
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenametotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal_2/Read/ReadVariableOpcount_2/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp#conv2d_4/kernel/Read/ReadVariableOp!conv2d_4/bias/Read/ReadVariableOp#conv2d_5/kernel/Read/ReadVariableOp!conv2d_5/bias/Read/ReadVariableOp"dense_4/kernel/Read/ReadVariableOp dense_4/bias/Read/ReadVariableOp"dense_5/kernel/Read/ReadVariableOp dense_5/bias/Read/ReadVariableOp-conv2d_transpose_6/kernel/Read/ReadVariableOp+conv2d_transpose_6/bias/Read/ReadVariableOp-conv2d_transpose_7/kernel/Read/ReadVariableOp+conv2d_transpose_7/bias/Read/ReadVariableOp-conv2d_transpose_8/kernel/Read/ReadVariableOp+conv2d_transpose_8/bias/Read/ReadVariableOp*Adam/conv2d_4/kernel/m/Read/ReadVariableOp(Adam/conv2d_4/bias/m/Read/ReadVariableOp*Adam/conv2d_5/kernel/m/Read/ReadVariableOp(Adam/conv2d_5/bias/m/Read/ReadVariableOp)Adam/dense_4/kernel/m/Read/ReadVariableOp'Adam/dense_4/bias/m/Read/ReadVariableOp)Adam/dense_5/kernel/m/Read/ReadVariableOp'Adam/dense_5/bias/m/Read/ReadVariableOp4Adam/conv2d_transpose_6/kernel/m/Read/ReadVariableOp2Adam/conv2d_transpose_6/bias/m/Read/ReadVariableOp4Adam/conv2d_transpose_7/kernel/m/Read/ReadVariableOp2Adam/conv2d_transpose_7/bias/m/Read/ReadVariableOp4Adam/conv2d_transpose_8/kernel/m/Read/ReadVariableOp2Adam/conv2d_transpose_8/bias/m/Read/ReadVariableOp*Adam/conv2d_4/kernel/v/Read/ReadVariableOp(Adam/conv2d_4/bias/v/Read/ReadVariableOp*Adam/conv2d_5/kernel/v/Read/ReadVariableOp(Adam/conv2d_5/bias/v/Read/ReadVariableOp)Adam/dense_4/kernel/v/Read/ReadVariableOp'Adam/dense_4/bias/v/Read/ReadVariableOp)Adam/dense_5/kernel/v/Read/ReadVariableOp'Adam/dense_5/bias/v/Read/ReadVariableOp4Adam/conv2d_transpose_6/kernel/v/Read/ReadVariableOp2Adam/conv2d_transpose_6/bias/v/Read/ReadVariableOp4Adam/conv2d_transpose_7/kernel/v/Read/ReadVariableOp2Adam/conv2d_transpose_7/bias/v/Read/ReadVariableOp4Adam/conv2d_transpose_8/kernel/v/Read/ReadVariableOp2Adam/conv2d_transpose_8/bias/v/Read/ReadVariableOpConst*B
Tin;
927	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *'
f"R 
__inference__traced_save_47533
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenametotalcounttotal_1count_1total_2count_2	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_rateconv2d_4/kernelconv2d_4/biasconv2d_5/kernelconv2d_5/biasdense_4/kerneldense_4/biasdense_5/kerneldense_5/biasconv2d_transpose_6/kernelconv2d_transpose_6/biasconv2d_transpose_7/kernelconv2d_transpose_7/biasconv2d_transpose_8/kernelconv2d_transpose_8/biasAdam/conv2d_4/kernel/mAdam/conv2d_4/bias/mAdam/conv2d_5/kernel/mAdam/conv2d_5/bias/mAdam/dense_4/kernel/mAdam/dense_4/bias/mAdam/dense_5/kernel/mAdam/dense_5/bias/m Adam/conv2d_transpose_6/kernel/mAdam/conv2d_transpose_6/bias/m Adam/conv2d_transpose_7/kernel/mAdam/conv2d_transpose_7/bias/m Adam/conv2d_transpose_8/kernel/mAdam/conv2d_transpose_8/bias/mAdam/conv2d_4/kernel/vAdam/conv2d_4/bias/vAdam/conv2d_5/kernel/vAdam/conv2d_5/bias/vAdam/dense_4/kernel/vAdam/dense_4/bias/vAdam/dense_5/kernel/vAdam/dense_5/bias/v Adam/conv2d_transpose_6/kernel/vAdam/conv2d_transpose_6/bias/v Adam/conv2d_transpose_7/kernel/vAdam/conv2d_transpose_7/bias/v Adam/conv2d_transpose_8/kernel/vAdam/conv2d_transpose_8/bias/v*A
Tin:
826*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? **
f%R#
!__inference__traced_restore_47702??
?
?
M__inference_conv2d_transpose_8_layer_call_and_return_conditional_losses_45620

inputsB
(conv2d_transpose_readvariableop_resource: -
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliceT
stack/1Const*
_output_shapes
: *
dtype0*
value	B :2	
stack/1T
stack/2Const*
_output_shapes
: *
dtype0*
value	B :2	
stack/2T
stack/3Const*
_output_shapes
: *
dtype0*
value	B :2	
stack/3?
stackPackstrided_slice:output:0stack/1:output:0stack/2:output:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSlicestack:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_transpose/ReadVariableOp?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
conv2d_transpose?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2	
BiasAddi
SigmoidSigmoidBiasAdd:output:0*
T0*/
_output_shapes
:?????????2	
Sigmoidn
IdentityIdentitySigmoid:y:0^NoOp*
T0*/
_output_shapes
:?????????2

Identity?
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
`
D__inference_flatten_2_layer_call_and_return_conditional_losses_47023

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"???? 	  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?	
?
,__inference_sequential_4_layer_call_fn_45070
input_9!
unknown: 
	unknown_0: #
	unknown_1: @
	unknown_2:@
	unknown_3:	?
	unknown_4:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_9unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_sequential_4_layer_call_and_return_conditional_losses_450552
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:?????????
!
_user_specified_name	input_9
?
a
E__inference_reshape_10_layer_call_and_return_conditional_losses_47099

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliced
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/1d
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/2d
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B : 2
Reshape/shape/3?
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapew
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:????????? 2	
Reshapel
IdentityIdentityReshape:output:0*
T0*/
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
9__inference_variational_autoencoder_2_layer_call_fn_45929
input_1!
unknown: 
	unknown_0: #
	unknown_1: @
	unknown_2:@
	unknown_3:	?
	unknown_4:
	unknown_5:	?
	unknown_6:	?#
	unknown_7:@ 
	unknown_8:@#
	unknown_9: @

unknown_10: $

unknown_11: 

unknown_12:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *]
fXRV
T__inference_variational_autoencoder_2_layer_call_and_return_conditional_losses_458982
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:?????????: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:?????????
!
_user_specified_name	input_1
?
?
T__inference_variational_autoencoder_2_layer_call_and_return_conditional_losses_46171
input_1,
sequential_4_46139:  
sequential_4_46141: ,
sequential_4_46143: @ 
sequential_4_46145:@%
sequential_4_46147:	? 
sequential_4_46149:%
sequential_5_46153:	?!
sequential_5_46155:	?,
sequential_5_46157:@  
sequential_5_46159:@,
sequential_5_46161: @ 
sequential_5_46163: ,
sequential_5_46165:  
sequential_5_46167:
identity??/normal_sampling_layer_2/StatefulPartitionedCall?$sequential_4/StatefulPartitionedCall?$sequential_5/StatefulPartitionedCall?
$sequential_4/StatefulPartitionedCallStatefulPartitionedCallinput_1sequential_4_46139sequential_4_46141sequential_4_46143sequential_4_46145sequential_4_46147sequential_4_46149*
Tin
	2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_sequential_4_layer_call_and_return_conditional_losses_451592&
$sequential_4/StatefulPartitionedCall?
/normal_sampling_layer_2/StatefulPartitionedCallStatefulPartitionedCall-sequential_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_normal_sampling_layer_2_layer_call_and_return_conditional_losses_4596421
/normal_sampling_layer_2/StatefulPartitionedCall?
$sequential_5/StatefulPartitionedCallStatefulPartitionedCall8normal_sampling_layer_2/StatefulPartitionedCall:output:0sequential_5_46153sequential_5_46155sequential_5_46157sequential_5_46159sequential_5_46161sequential_5_46163sequential_5_46165sequential_5_46167*
Tin
2	*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_sequential_5_layer_call_and_return_conditional_losses_457602&
$sequential_5/StatefulPartitionedCall?
IdentityIdentity-sequential_5/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????2

Identity?
NoOpNoOp0^normal_sampling_layer_2/StatefulPartitionedCall%^sequential_4/StatefulPartitionedCall%^sequential_5/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:?????????: : : : : : : : : : : : : : 2b
/normal_sampling_layer_2/StatefulPartitionedCall/normal_sampling_layer_2/StatefulPartitionedCall2L
$sequential_4/StatefulPartitionedCall$sequential_4/StatefulPartitionedCall2L
$sequential_5/StatefulPartitionedCall$sequential_5/StatefulPartitionedCall:X T
/
_output_shapes
:?????????
!
_user_specified_name	input_1
?&
?
M__inference_conv2d_transpose_8_layer_call_and_return_conditional_losses_45449

inputsB
(conv2d_transpose_readvariableop_resource: -
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulT
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/yb
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1T
stack/3Const*
_output_shapes
: *
dtype0*
value	B :2	
stack/3?
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_transpose/ReadVariableOp?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+???????????????????????????*
paddingSAME*
strides
2
conv2d_transpose?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????2	
BiasAdd{
SigmoidSigmoidBiasAdd:output:0*
T0*A
_output_shapes/
-:+???????????????????????????2	
Sigmoid?
IdentityIdentitySigmoid:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity?
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+??????????????????????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
n
R__inference_normal_sampling_layer_2_layer_call_and_return_conditional_losses_46921

inputs
identity?
unstackUnpackinputs*
T0*:
_output_shapes(
&:?????????:?????????*

axis*	
num2	
unstackd
IdentityIdentityunstack:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
C__inference_conv2d_5_layer_call_and_return_conditional_losses_47008

inputs8
conv2d_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
Reluu
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????@2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
`
D__inference_reshape_8_layer_call_and_return_conditional_losses_46972

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliced
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/1d
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/2d
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/3?
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapew
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:?????????2	
Reshapel
IdentityIdentityReshape:output:0*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?%
?
M__inference_conv2d_transpose_6_layer_call_and_return_conditional_losses_45273

inputsB
(conv2d_transpose_readvariableop_resource:@ -
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulT
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/yb
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1T
stack/3Const*
_output_shapes
: *
dtype0*
value	B :@2	
stack/3?
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:@ *
dtype02!
conv2d_transpose/ReadVariableOp?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+???????????????????????????@*
paddingSAME*
strides
2
conv2d_transpose?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????@2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+???????????????????????????@2
Relu?
IdentityIdentityRelu:activations:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity?
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+??????????????????????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
G__inference_sequential_4_layer_call_and_return_conditional_losses_45055

inputs(
conv2d_4_44993: 
conv2d_4_44995: (
conv2d_5_45010: @
conv2d_5_45012:@ 
dense_4_45034:	?
dense_4_45036:
identity?? conv2d_4/StatefulPartitionedCall? conv2d_5/StatefulPartitionedCall?dense_4/StatefulPartitionedCall?
reshape_8/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_reshape_8_layer_call_and_return_conditional_losses_449792
reshape_8/PartitionedCall?
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall"reshape_8/PartitionedCall:output:0conv2d_4_44993conv2d_4_44995*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_conv2d_4_layer_call_and_return_conditional_losses_449922"
 conv2d_4/StatefulPartitionedCall?
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0conv2d_5_45010conv2d_5_45012*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_conv2d_5_layer_call_and_return_conditional_losses_450092"
 conv2d_5/StatefulPartitionedCall?
flatten_2/PartitionedCallPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_flatten_2_layer_call_and_return_conditional_losses_450212
flatten_2/PartitionedCall?
dense_4/StatefulPartitionedCallStatefulPartitionedCall"flatten_2/PartitionedCall:output:0dense_4_45034dense_4_45036*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dense_4_layer_call_and_return_conditional_losses_450332!
dense_4/StatefulPartitionedCall?
reshape_9/PartitionedCallPartitionedCall(dense_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_reshape_9_layer_call_and_return_conditional_losses_450522
reshape_9/PartitionedCall?
IdentityIdentity"reshape_9/PartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????2

Identity?
NoOpNoOp!^conv2d_4/StatefulPartitionedCall!^conv2d_5/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : 2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
n
R__inference_normal_sampling_layer_2_layer_call_and_return_conditional_losses_45878

inputs
identity?
unstackUnpackinputs*
T0*:
_output_shapes(
&:?????????:?????????*

axis*	
num2	
unstackd
IdentityIdentityunstack:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
C__inference_conv2d_4_layer_call_and_return_conditional_losses_44992

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
Reluu
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:????????? 2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?&
?
M__inference_conv2d_transpose_8_layer_call_and_return_conditional_losses_47290

inputsB
(conv2d_transpose_readvariableop_resource: -
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulT
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/yb
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1T
stack/3Const*
_output_shapes
: *
dtype0*
value	B :2	
stack/3?
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_transpose/ReadVariableOp?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+???????????????????????????*
paddingSAME*
strides
2
conv2d_transpose?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????2	
BiasAdd{
SigmoidSigmoidBiasAdd:output:0*
T0*A
_output_shapes/
-:+???????????????????????????2	
Sigmoid?
IdentityIdentitySigmoid:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity?
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+??????????????????????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
2__inference_conv2d_transpose_8_layer_call_fn_47332

inputs!
unknown: 
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *V
fQRO
M__inference_conv2d_transpose_8_layer_call_and_return_conditional_losses_456202
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:????????? : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?%
?
M__inference_conv2d_transpose_7_layer_call_and_return_conditional_losses_45361

inputsB
(conv2d_transpose_readvariableop_resource: @-
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulT
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/yb
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1T
stack/3Const*
_output_shapes
: *
dtype0*
value	B : 2	
stack/3?
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype02!
conv2d_transpose/ReadVariableOp?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+??????????????????????????? *
paddingSAME*
strides
2
conv2d_transpose?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2
Relu?
IdentityIdentityRelu:activations:0^NoOp*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity?
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?9
?
G__inference_sequential_4_layer_call_and_return_conditional_losses_46612

inputsA
'conv2d_4_conv2d_readvariableop_resource: 6
(conv2d_4_biasadd_readvariableop_resource: A
'conv2d_5_conv2d_readvariableop_resource: @6
(conv2d_5_biasadd_readvariableop_resource:@9
&dense_4_matmul_readvariableop_resource:	?5
'dense_4_biasadd_readvariableop_resource:
identity??conv2d_4/BiasAdd/ReadVariableOp?conv2d_4/Conv2D/ReadVariableOp?conv2d_5/BiasAdd/ReadVariableOp?conv2d_5/Conv2D/ReadVariableOp?dense_4/BiasAdd/ReadVariableOp?dense_4/MatMul/ReadVariableOpX
reshape_8/ShapeShapeinputs*
T0*
_output_shapes
:2
reshape_8/Shape?
reshape_8/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape_8/strided_slice/stack?
reshape_8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_8/strided_slice/stack_1?
reshape_8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_8/strided_slice/stack_2?
reshape_8/strided_sliceStridedSlicereshape_8/Shape:output:0&reshape_8/strided_slice/stack:output:0(reshape_8/strided_slice/stack_1:output:0(reshape_8/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_8/strided_slicex
reshape_8/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_8/Reshape/shape/1x
reshape_8/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_8/Reshape/shape/2x
reshape_8/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_8/Reshape/shape/3?
reshape_8/Reshape/shapePack reshape_8/strided_slice:output:0"reshape_8/Reshape/shape/1:output:0"reshape_8/Reshape/shape/2:output:0"reshape_8/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
reshape_8/Reshape/shape?
reshape_8/ReshapeReshapeinputs reshape_8/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????2
reshape_8/Reshape?
conv2d_4/Conv2D/ReadVariableOpReadVariableOp'conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02 
conv2d_4/Conv2D/ReadVariableOp?
conv2d_4/Conv2DConv2Dreshape_8/Reshape:output:0&conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
2
conv2d_4/Conv2D?
conv2d_4/BiasAdd/ReadVariableOpReadVariableOp(conv2d_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv2d_4/BiasAdd/ReadVariableOp?
conv2d_4/BiasAddBiasAddconv2d_4/Conv2D:output:0'conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2
conv2d_4/BiasAdd{
conv2d_4/ReluReluconv2d_4/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
conv2d_4/Relu?
conv2d_5/Conv2D/ReadVariableOpReadVariableOp'conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02 
conv2d_5/Conv2D/ReadVariableOp?
conv2d_5/Conv2DConv2Dconv2d_4/Relu:activations:0&conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
2
conv2d_5/Conv2D?
conv2d_5/BiasAdd/ReadVariableOpReadVariableOp(conv2d_5_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_5/BiasAdd/ReadVariableOp?
conv2d_5/BiasAddBiasAddconv2d_5/Conv2D:output:0'conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
conv2d_5/BiasAdd{
conv2d_5/ReluReluconv2d_5/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
conv2d_5/Relus
flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"???? 	  2
flatten_2/Const?
flatten_2/ReshapeReshapeconv2d_5/Relu:activations:0flatten_2/Const:output:0*
T0*(
_output_shapes
:??????????2
flatten_2/Reshape?
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
dense_4/MatMul/ReadVariableOp?
dense_4/MatMulMatMulflatten_2/Reshape:output:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_4/MatMul?
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_4/BiasAdd/ReadVariableOp?
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_4/BiasAddj
reshape_9/ShapeShapedense_4/BiasAdd:output:0*
T0*
_output_shapes
:2
reshape_9/Shape?
reshape_9/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape_9/strided_slice/stack?
reshape_9/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_9/strided_slice/stack_1?
reshape_9/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_9/strided_slice/stack_2?
reshape_9/strided_sliceStridedSlicereshape_9/Shape:output:0&reshape_9/strided_slice/stack:output:0(reshape_9/strided_slice/stack_1:output:0(reshape_9/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_9/strided_slicex
reshape_9/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_9/Reshape/shape/1x
reshape_9/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_9/Reshape/shape/2?
reshape_9/Reshape/shapePack reshape_9/strided_slice:output:0"reshape_9/Reshape/shape/1:output:0"reshape_9/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
reshape_9/Reshape/shape?
reshape_9/ReshapeReshapedense_4/BiasAdd:output:0 reshape_9/Reshape/shape:output:0*
T0*+
_output_shapes
:?????????2
reshape_9/Reshapey
IdentityIdentityreshape_9/Reshape:output:0^NoOp*
T0*+
_output_shapes
:?????????2

Identity?
NoOpNoOp ^conv2d_4/BiasAdd/ReadVariableOp^conv2d_4/Conv2D/ReadVariableOp ^conv2d_5/BiasAdd/ReadVariableOp^conv2d_5/Conv2D/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : 2B
conv2d_4/BiasAdd/ReadVariableOpconv2d_4/BiasAdd/ReadVariableOp2@
conv2d_4/Conv2D/ReadVariableOpconv2d_4/Conv2D/ReadVariableOp2B
conv2d_5/BiasAdd/ReadVariableOpconv2d_5/BiasAdd/ReadVariableOp2@
conv2d_5/Conv2D/ReadVariableOpconv2d_5/Conv2D/ReadVariableOp2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
2__inference_conv2d_transpose_6_layer_call_fn_47180

inputs!
unknown:@ 
	unknown_0:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *V
fQRO
M__inference_conv2d_transpose_6_layer_call_and_return_conditional_losses_455622
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:????????? : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
p
7__inference_normal_sampling_layer_2_layer_call_fn_46958

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_normal_sampling_layer_2_layer_call_and_return_conditional_losses_459642
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
a
E__inference_reshape_11_layer_call_and_return_conditional_losses_45640

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliced
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/1d
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/2d
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/3?
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapew
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:?????????2	
Reshapel
IdentityIdentityReshape:output:0*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
(__inference_conv2d_5_layer_call_fn_47017

inputs!
unknown: @
	unknown_0:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_conv2d_5_layer_call_and_return_conditional_losses_450092
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:????????? : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
9__inference_variational_autoencoder_2_layer_call_fn_46534

inputs!
unknown: 
	unknown_0: #
	unknown_1: @
	unknown_2:@
	unknown_3:	?
	unknown_4:
	unknown_5:	?
	unknown_6:	?#
	unknown_7:@ 
	unknown_8:@#
	unknown_9: @

unknown_10: $

unknown_11: 

unknown_12:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *]
fXRV
T__inference_variational_autoencoder_2_layer_call_and_return_conditional_losses_458982
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:?????????: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
2__inference_conv2d_transpose_8_layer_call_fn_47323

inputs!
unknown: 
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *V
fQRO
M__inference_conv2d_transpose_8_layer_call_and_return_conditional_losses_454492
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+??????????????????????????? : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
2__inference_conv2d_transpose_6_layer_call_fn_47171

inputs!
unknown:@ 
	unknown_0:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *V
fQRO
M__inference_conv2d_transpose_6_layer_call_and_return_conditional_losses_452732
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+??????????????????????????? : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?!
?
G__inference_sequential_5_layer_call_and_return_conditional_losses_45760

inputs 
dense_5_45737:	?
dense_5_45739:	?2
conv2d_transpose_6_45743:@ &
conv2d_transpose_6_45745:@2
conv2d_transpose_7_45748: @&
conv2d_transpose_7_45750: 2
conv2d_transpose_8_45753: &
conv2d_transpose_8_45755:
identity??*conv2d_transpose_6/StatefulPartitionedCall?*conv2d_transpose_7/StatefulPartitionedCall?*conv2d_transpose_8/StatefulPartitionedCall?dense_5/StatefulPartitionedCall?
dense_5/StatefulPartitionedCallStatefulPartitionedCallinputsdense_5_45737dense_5_45739*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dense_5_layer_call_and_return_conditional_losses_455172!
dense_5/StatefulPartitionedCall?
reshape_10/PartitionedCallPartitionedCall(dense_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_reshape_10_layer_call_and_return_conditional_losses_455372
reshape_10/PartitionedCall?
*conv2d_transpose_6/StatefulPartitionedCallStatefulPartitionedCall#reshape_10/PartitionedCall:output:0conv2d_transpose_6_45743conv2d_transpose_6_45745*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *V
fQRO
M__inference_conv2d_transpose_6_layer_call_and_return_conditional_losses_455622,
*conv2d_transpose_6/StatefulPartitionedCall?
*conv2d_transpose_7/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_6/StatefulPartitionedCall:output:0conv2d_transpose_7_45748conv2d_transpose_7_45750*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *V
fQRO
M__inference_conv2d_transpose_7_layer_call_and_return_conditional_losses_455912,
*conv2d_transpose_7/StatefulPartitionedCall?
*conv2d_transpose_8/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_7/StatefulPartitionedCall:output:0conv2d_transpose_8_45753conv2d_transpose_8_45755*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *V
fQRO
M__inference_conv2d_transpose_8_layer_call_and_return_conditional_losses_456202,
*conv2d_transpose_8/StatefulPartitionedCall?
reshape_11/PartitionedCallPartitionedCall3conv2d_transpose_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_reshape_11_layer_call_and_return_conditional_losses_456402
reshape_11/PartitionedCall?
IdentityIdentity#reshape_11/PartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????2

Identity?
NoOpNoOp+^conv2d_transpose_6/StatefulPartitionedCall+^conv2d_transpose_7/StatefulPartitionedCall+^conv2d_transpose_8/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : : : : : 2X
*conv2d_transpose_6/StatefulPartitionedCall*conv2d_transpose_6/StatefulPartitionedCall2X
*conv2d_transpose_7/StatefulPartitionedCall*conv2d_transpose_7/StatefulPartitionedCall2X
*conv2d_transpose_8/StatefulPartitionedCall*conv2d_transpose_8/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
B__inference_dense_5_layer_call_and_return_conditional_losses_45517

inputs1
matmul_readvariableop_resource:	?.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relun
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
a
E__inference_reshape_11_layer_call_and_return_conditional_losses_47346

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliced
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/1d
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/2d
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/3?
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapew
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:?????????2	
Reshapel
IdentityIdentityReshape:output:0*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
F
*__inference_reshape_10_layer_call_fn_47104

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_reshape_10_layer_call_and_return_conditional_losses_455372
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
9__inference_variational_autoencoder_2_layer_call_fn_46101
input_1!
unknown: 
	unknown_0: #
	unknown_1: @
	unknown_2:@
	unknown_3:	?
	unknown_4:
	unknown_5:	?
	unknown_6:	?#
	unknown_7:@ 
	unknown_8:@#
	unknown_9: @

unknown_10: $

unknown_11: 

unknown_12:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *]
fXRV
T__inference_variational_autoencoder_2_layer_call_and_return_conditional_losses_460372
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:?????????: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:?????????
!
_user_specified_name	input_1
?

?
B__inference_dense_4_layer_call_and_return_conditional_losses_47038

inputs1
matmul_readvariableop_resource:	?-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddk
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
??
?
G__inference_sequential_5_layer_call_and_return_conditional_losses_46782

inputs9
&dense_5_matmul_readvariableop_resource:	?6
'dense_5_biasadd_readvariableop_resource:	?U
;conv2d_transpose_6_conv2d_transpose_readvariableop_resource:@ @
2conv2d_transpose_6_biasadd_readvariableop_resource:@U
;conv2d_transpose_7_conv2d_transpose_readvariableop_resource: @@
2conv2d_transpose_7_biasadd_readvariableop_resource: U
;conv2d_transpose_8_conv2d_transpose_readvariableop_resource: @
2conv2d_transpose_8_biasadd_readvariableop_resource:
identity??)conv2d_transpose_6/BiasAdd/ReadVariableOp?2conv2d_transpose_6/conv2d_transpose/ReadVariableOp?)conv2d_transpose_7/BiasAdd/ReadVariableOp?2conv2d_transpose_7/conv2d_transpose/ReadVariableOp?)conv2d_transpose_8/BiasAdd/ReadVariableOp?2conv2d_transpose_8/conv2d_transpose/ReadVariableOp?dense_5/BiasAdd/ReadVariableOp?dense_5/MatMul/ReadVariableOp?
dense_5/MatMul/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
dense_5/MatMul/ReadVariableOp?
dense_5/MatMulMatMulinputs%dense_5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_5/MatMul?
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
dense_5/BiasAdd/ReadVariableOp?
dense_5/BiasAddBiasAdddense_5/MatMul:product:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_5/BiasAddq
dense_5/ReluReludense_5/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_5/Relun
reshape_10/ShapeShapedense_5/Relu:activations:0*
T0*
_output_shapes
:2
reshape_10/Shape?
reshape_10/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
reshape_10/strided_slice/stack?
 reshape_10/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_10/strided_slice/stack_1?
 reshape_10/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_10/strided_slice/stack_2?
reshape_10/strided_sliceStridedSlicereshape_10/Shape:output:0'reshape_10/strided_slice/stack:output:0)reshape_10/strided_slice/stack_1:output:0)reshape_10/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_10/strided_slicez
reshape_10/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_10/Reshape/shape/1z
reshape_10/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_10/Reshape/shape/2z
reshape_10/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B : 2
reshape_10/Reshape/shape/3?
reshape_10/Reshape/shapePack!reshape_10/strided_slice:output:0#reshape_10/Reshape/shape/1:output:0#reshape_10/Reshape/shape/2:output:0#reshape_10/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
reshape_10/Reshape/shape?
reshape_10/ReshapeReshapedense_5/Relu:activations:0!reshape_10/Reshape/shape:output:0*
T0*/
_output_shapes
:????????? 2
reshape_10/Reshape
conv2d_transpose_6/ShapeShapereshape_10/Reshape:output:0*
T0*
_output_shapes
:2
conv2d_transpose_6/Shape?
&conv2d_transpose_6/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv2d_transpose_6/strided_slice/stack?
(conv2d_transpose_6/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_6/strided_slice/stack_1?
(conv2d_transpose_6/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_6/strided_slice/stack_2?
 conv2d_transpose_6/strided_sliceStridedSlice!conv2d_transpose_6/Shape:output:0/conv2d_transpose_6/strided_slice/stack:output:01conv2d_transpose_6/strided_slice/stack_1:output:01conv2d_transpose_6/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv2d_transpose_6/strided_slicez
conv2d_transpose_6/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_6/stack/1z
conv2d_transpose_6/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_6/stack/2z
conv2d_transpose_6/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@2
conv2d_transpose_6/stack/3?
conv2d_transpose_6/stackPack)conv2d_transpose_6/strided_slice:output:0#conv2d_transpose_6/stack/1:output:0#conv2d_transpose_6/stack/2:output:0#conv2d_transpose_6/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_6/stack?
(conv2d_transpose_6/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(conv2d_transpose_6/strided_slice_1/stack?
*conv2d_transpose_6/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_6/strided_slice_1/stack_1?
*conv2d_transpose_6/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_6/strided_slice_1/stack_2?
"conv2d_transpose_6/strided_slice_1StridedSlice!conv2d_transpose_6/stack:output:01conv2d_transpose_6/strided_slice_1/stack:output:03conv2d_transpose_6/strided_slice_1/stack_1:output:03conv2d_transpose_6/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv2d_transpose_6/strided_slice_1?
2conv2d_transpose_6/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_6_conv2d_transpose_readvariableop_resource*&
_output_shapes
:@ *
dtype024
2conv2d_transpose_6/conv2d_transpose/ReadVariableOp?
#conv2d_transpose_6/conv2d_transposeConv2DBackpropInput!conv2d_transpose_6/stack:output:0:conv2d_transpose_6/conv2d_transpose/ReadVariableOp:value:0reshape_10/Reshape:output:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2%
#conv2d_transpose_6/conv2d_transpose?
)conv2d_transpose_6/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_6_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02+
)conv2d_transpose_6/BiasAdd/ReadVariableOp?
conv2d_transpose_6/BiasAddBiasAdd,conv2d_transpose_6/conv2d_transpose:output:01conv2d_transpose_6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
conv2d_transpose_6/BiasAdd?
conv2d_transpose_6/ReluRelu#conv2d_transpose_6/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
conv2d_transpose_6/Relu?
conv2d_transpose_7/ShapeShape%conv2d_transpose_6/Relu:activations:0*
T0*
_output_shapes
:2
conv2d_transpose_7/Shape?
&conv2d_transpose_7/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv2d_transpose_7/strided_slice/stack?
(conv2d_transpose_7/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_7/strided_slice/stack_1?
(conv2d_transpose_7/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_7/strided_slice/stack_2?
 conv2d_transpose_7/strided_sliceStridedSlice!conv2d_transpose_7/Shape:output:0/conv2d_transpose_7/strided_slice/stack:output:01conv2d_transpose_7/strided_slice/stack_1:output:01conv2d_transpose_7/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv2d_transpose_7/strided_slicez
conv2d_transpose_7/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_7/stack/1z
conv2d_transpose_7/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_7/stack/2z
conv2d_transpose_7/stack/3Const*
_output_shapes
: *
dtype0*
value	B : 2
conv2d_transpose_7/stack/3?
conv2d_transpose_7/stackPack)conv2d_transpose_7/strided_slice:output:0#conv2d_transpose_7/stack/1:output:0#conv2d_transpose_7/stack/2:output:0#conv2d_transpose_7/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_7/stack?
(conv2d_transpose_7/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(conv2d_transpose_7/strided_slice_1/stack?
*conv2d_transpose_7/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_7/strided_slice_1/stack_1?
*conv2d_transpose_7/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_7/strided_slice_1/stack_2?
"conv2d_transpose_7/strided_slice_1StridedSlice!conv2d_transpose_7/stack:output:01conv2d_transpose_7/strided_slice_1/stack:output:03conv2d_transpose_7/strided_slice_1/stack_1:output:03conv2d_transpose_7/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv2d_transpose_7/strided_slice_1?
2conv2d_transpose_7/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_7_conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype024
2conv2d_transpose_7/conv2d_transpose/ReadVariableOp?
#conv2d_transpose_7/conv2d_transposeConv2DBackpropInput!conv2d_transpose_7/stack:output:0:conv2d_transpose_7/conv2d_transpose/ReadVariableOp:value:0%conv2d_transpose_6/Relu:activations:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
2%
#conv2d_transpose_7/conv2d_transpose?
)conv2d_transpose_7/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_7_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02+
)conv2d_transpose_7/BiasAdd/ReadVariableOp?
conv2d_transpose_7/BiasAddBiasAdd,conv2d_transpose_7/conv2d_transpose:output:01conv2d_transpose_7/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2
conv2d_transpose_7/BiasAdd?
conv2d_transpose_7/ReluRelu#conv2d_transpose_7/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
conv2d_transpose_7/Relu?
conv2d_transpose_8/ShapeShape%conv2d_transpose_7/Relu:activations:0*
T0*
_output_shapes
:2
conv2d_transpose_8/Shape?
&conv2d_transpose_8/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv2d_transpose_8/strided_slice/stack?
(conv2d_transpose_8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_8/strided_slice/stack_1?
(conv2d_transpose_8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_8/strided_slice/stack_2?
 conv2d_transpose_8/strided_sliceStridedSlice!conv2d_transpose_8/Shape:output:0/conv2d_transpose_8/strided_slice/stack:output:01conv2d_transpose_8/strided_slice/stack_1:output:01conv2d_transpose_8/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv2d_transpose_8/strided_slicez
conv2d_transpose_8/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_8/stack/1z
conv2d_transpose_8/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_8/stack/2z
conv2d_transpose_8/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_8/stack/3?
conv2d_transpose_8/stackPack)conv2d_transpose_8/strided_slice:output:0#conv2d_transpose_8/stack/1:output:0#conv2d_transpose_8/stack/2:output:0#conv2d_transpose_8/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_8/stack?
(conv2d_transpose_8/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(conv2d_transpose_8/strided_slice_1/stack?
*conv2d_transpose_8/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_8/strided_slice_1/stack_1?
*conv2d_transpose_8/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_8/strided_slice_1/stack_2?
"conv2d_transpose_8/strided_slice_1StridedSlice!conv2d_transpose_8/stack:output:01conv2d_transpose_8/strided_slice_1/stack:output:03conv2d_transpose_8/strided_slice_1/stack_1:output:03conv2d_transpose_8/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv2d_transpose_8/strided_slice_1?
2conv2d_transpose_8/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_8_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype024
2conv2d_transpose_8/conv2d_transpose/ReadVariableOp?
#conv2d_transpose_8/conv2d_transposeConv2DBackpropInput!conv2d_transpose_8/stack:output:0:conv2d_transpose_8/conv2d_transpose/ReadVariableOp:value:0%conv2d_transpose_7/Relu:activations:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2%
#conv2d_transpose_8/conv2d_transpose?
)conv2d_transpose_8/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)conv2d_transpose_8/BiasAdd/ReadVariableOp?
conv2d_transpose_8/BiasAddBiasAdd,conv2d_transpose_8/conv2d_transpose:output:01conv2d_transpose_8/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
conv2d_transpose_8/BiasAdd?
conv2d_transpose_8/SigmoidSigmoid#conv2d_transpose_8/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
conv2d_transpose_8/Sigmoidr
reshape_11/ShapeShapeconv2d_transpose_8/Sigmoid:y:0*
T0*
_output_shapes
:2
reshape_11/Shape?
reshape_11/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
reshape_11/strided_slice/stack?
 reshape_11/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_11/strided_slice/stack_1?
 reshape_11/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_11/strided_slice/stack_2?
reshape_11/strided_sliceStridedSlicereshape_11/Shape:output:0'reshape_11/strided_slice/stack:output:0)reshape_11/strided_slice/stack_1:output:0)reshape_11/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_11/strided_slicez
reshape_11/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_11/Reshape/shape/1z
reshape_11/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_11/Reshape/shape/2z
reshape_11/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_11/Reshape/shape/3?
reshape_11/Reshape/shapePack!reshape_11/strided_slice:output:0#reshape_11/Reshape/shape/1:output:0#reshape_11/Reshape/shape/2:output:0#reshape_11/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
reshape_11/Reshape/shape?
reshape_11/ReshapeReshapeconv2d_transpose_8/Sigmoid:y:0!reshape_11/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????2
reshape_11/Reshape~
IdentityIdentityreshape_11/Reshape:output:0^NoOp*
T0*/
_output_shapes
:?????????2

Identity?
NoOpNoOp*^conv2d_transpose_6/BiasAdd/ReadVariableOp3^conv2d_transpose_6/conv2d_transpose/ReadVariableOp*^conv2d_transpose_7/BiasAdd/ReadVariableOp3^conv2d_transpose_7/conv2d_transpose/ReadVariableOp*^conv2d_transpose_8/BiasAdd/ReadVariableOp3^conv2d_transpose_8/conv2d_transpose/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : : : : : 2V
)conv2d_transpose_6/BiasAdd/ReadVariableOp)conv2d_transpose_6/BiasAdd/ReadVariableOp2h
2conv2d_transpose_6/conv2d_transpose/ReadVariableOp2conv2d_transpose_6/conv2d_transpose/ReadVariableOp2V
)conv2d_transpose_7/BiasAdd/ReadVariableOp)conv2d_transpose_7/BiasAdd/ReadVariableOp2h
2conv2d_transpose_7/conv2d_transpose/ReadVariableOp2conv2d_transpose_7/conv2d_transpose/ReadVariableOp2V
)conv2d_transpose_8/BiasAdd/ReadVariableOp)conv2d_transpose_8/BiasAdd/ReadVariableOp2h
2conv2d_transpose_8/conv2d_transpose/ReadVariableOp2conv2d_transpose_8/conv2d_transpose/ReadVariableOp2@
dense_5/BiasAdd/ReadVariableOpdense_5/BiasAdd/ReadVariableOp2>
dense_5/MatMul/ReadVariableOpdense_5/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?%
?
M__inference_conv2d_transpose_6_layer_call_and_return_conditional_losses_47138

inputsB
(conv2d_transpose_readvariableop_resource:@ -
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulT
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/yb
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1T
stack/3Const*
_output_shapes
: *
dtype0*
value	B :@2	
stack/3?
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:@ *
dtype02!
conv2d_transpose/ReadVariableOp?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+???????????????????????????@*
paddingSAME*
strides
2
conv2d_transpose?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????@2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+???????????????????????????@2
Relu?
IdentityIdentityRelu:activations:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity?
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+??????????????????????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
??
?!
!__inference__traced_restore_47702
file_prefix 
assignvariableop_total: "
assignvariableop_1_count: $
assignvariableop_2_total_1: $
assignvariableop_3_count_1: $
assignvariableop_4_total_2: $
assignvariableop_5_count_2: &
assignvariableop_6_adam_iter:	 (
assignvariableop_7_adam_beta_1: (
assignvariableop_8_adam_beta_2: '
assignvariableop_9_adam_decay: 0
&assignvariableop_10_adam_learning_rate: =
#assignvariableop_11_conv2d_4_kernel: /
!assignvariableop_12_conv2d_4_bias: =
#assignvariableop_13_conv2d_5_kernel: @/
!assignvariableop_14_conv2d_5_bias:@5
"assignvariableop_15_dense_4_kernel:	?.
 assignvariableop_16_dense_4_bias:5
"assignvariableop_17_dense_5_kernel:	?/
 assignvariableop_18_dense_5_bias:	?G
-assignvariableop_19_conv2d_transpose_6_kernel:@ 9
+assignvariableop_20_conv2d_transpose_6_bias:@G
-assignvariableop_21_conv2d_transpose_7_kernel: @9
+assignvariableop_22_conv2d_transpose_7_bias: G
-assignvariableop_23_conv2d_transpose_8_kernel: 9
+assignvariableop_24_conv2d_transpose_8_bias:D
*assignvariableop_25_adam_conv2d_4_kernel_m: 6
(assignvariableop_26_adam_conv2d_4_bias_m: D
*assignvariableop_27_adam_conv2d_5_kernel_m: @6
(assignvariableop_28_adam_conv2d_5_bias_m:@<
)assignvariableop_29_adam_dense_4_kernel_m:	?5
'assignvariableop_30_adam_dense_4_bias_m:<
)assignvariableop_31_adam_dense_5_kernel_m:	?6
'assignvariableop_32_adam_dense_5_bias_m:	?N
4assignvariableop_33_adam_conv2d_transpose_6_kernel_m:@ @
2assignvariableop_34_adam_conv2d_transpose_6_bias_m:@N
4assignvariableop_35_adam_conv2d_transpose_7_kernel_m: @@
2assignvariableop_36_adam_conv2d_transpose_7_bias_m: N
4assignvariableop_37_adam_conv2d_transpose_8_kernel_m: @
2assignvariableop_38_adam_conv2d_transpose_8_bias_m:D
*assignvariableop_39_adam_conv2d_4_kernel_v: 6
(assignvariableop_40_adam_conv2d_4_bias_v: D
*assignvariableop_41_adam_conv2d_5_kernel_v: @6
(assignvariableop_42_adam_conv2d_5_bias_v:@<
)assignvariableop_43_adam_dense_4_kernel_v:	?5
'assignvariableop_44_adam_dense_4_bias_v:<
)assignvariableop_45_adam_dense_5_kernel_v:	?6
'assignvariableop_46_adam_dense_5_bias_v:	?N
4assignvariableop_47_adam_conv2d_transpose_6_kernel_v:@ @
2assignvariableop_48_adam_conv2d_transpose_6_bias_v:@N
4assignvariableop_49_adam_conv2d_transpose_7_kernel_v: @@
2assignvariableop_50_adam_conv2d_transpose_7_bias_v: N
4assignvariableop_51_adam_conv2d_transpose_8_kernel_v: @
2assignvariableop_52_adam_conv2d_transpose_8_bias_v:
identity_54??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_43?AssignVariableOp_44?AssignVariableOp_45?AssignVariableOp_46?AssignVariableOp_47?AssignVariableOp_48?AssignVariableOp_49?AssignVariableOp_5?AssignVariableOp_50?AssignVariableOp_51?AssignVariableOp_52?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:6*
dtype0*?
value?B?6B3loss_tracker_total/total/.ATTRIBUTES/VARIABLE_VALUEB3loss_tracker_total/count/.ATTRIBUTES/VARIABLE_VALUEB<loss_tracker_reconstruction/total/.ATTRIBUTES/VARIABLE_VALUEB<loss_tracker_reconstruction/count/.ATTRIBUTES/VARIABLE_VALUEB4loss_tracker_latent/total/.ATTRIBUTES/VARIABLE_VALUEB4loss_tracker_latent/count/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/12/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/13/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:6*
dtype0*
valuevBt6B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::::::::::::::::::::*D
dtypes:
826	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOpassignvariableop_totalIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOpassignvariableop_1_countIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOpassignvariableop_2_total_1Identity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOpassignvariableop_3_count_1Identity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOpassignvariableop_4_total_2Identity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOpassignvariableop_5_count_2Identity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOpassignvariableop_6_adam_iterIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOpassignvariableop_7_adam_beta_1Identity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOpassignvariableop_8_adam_beta_2Identity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOpassignvariableop_9_adam_decayIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp&assignvariableop_10_adam_learning_rateIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp#assignvariableop_11_conv2d_4_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp!assignvariableop_12_conv2d_4_biasIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOp#assignvariableop_13_conv2d_5_kernelIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOp!assignvariableop_14_conv2d_5_biasIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOp"assignvariableop_15_dense_4_kernelIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOp assignvariableop_16_dense_4_biasIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOp"assignvariableop_17_dense_5_kernelIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOp assignvariableop_18_dense_5_biasIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOp-assignvariableop_19_conv2d_transpose_6_kernelIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp+assignvariableop_20_conv2d_transpose_6_biasIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp-assignvariableop_21_conv2d_transpose_7_kernelIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp+assignvariableop_22_conv2d_transpose_7_biasIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp-assignvariableop_23_conv2d_transpose_8_kernelIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp+assignvariableop_24_conv2d_transpose_8_biasIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOp*assignvariableop_25_adam_conv2d_4_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOp(assignvariableop_26_adam_conv2d_4_bias_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOp*assignvariableop_27_adam_conv2d_5_kernel_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOp(assignvariableop_28_adam_conv2d_5_bias_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOp)assignvariableop_29_adam_dense_4_kernel_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOp'assignvariableop_30_adam_dense_4_bias_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOp)assignvariableop_31_adam_dense_5_kernel_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOp'assignvariableop_32_adam_dense_5_bias_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33?
AssignVariableOp_33AssignVariableOp4assignvariableop_33_adam_conv2d_transpose_6_kernel_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34?
AssignVariableOp_34AssignVariableOp2assignvariableop_34_adam_conv2d_transpose_6_bias_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35?
AssignVariableOp_35AssignVariableOp4assignvariableop_35_adam_conv2d_transpose_7_kernel_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36?
AssignVariableOp_36AssignVariableOp2assignvariableop_36_adam_conv2d_transpose_7_bias_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37?
AssignVariableOp_37AssignVariableOp4assignvariableop_37_adam_conv2d_transpose_8_kernel_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38?
AssignVariableOp_38AssignVariableOp2assignvariableop_38_adam_conv2d_transpose_8_bias_mIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39?
AssignVariableOp_39AssignVariableOp*assignvariableop_39_adam_conv2d_4_kernel_vIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40?
AssignVariableOp_40AssignVariableOp(assignvariableop_40_adam_conv2d_4_bias_vIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41?
AssignVariableOp_41AssignVariableOp*assignvariableop_41_adam_conv2d_5_kernel_vIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42?
AssignVariableOp_42AssignVariableOp(assignvariableop_42_adam_conv2d_5_bias_vIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43?
AssignVariableOp_43AssignVariableOp)assignvariableop_43_adam_dense_4_kernel_vIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44?
AssignVariableOp_44AssignVariableOp'assignvariableop_44_adam_dense_4_bias_vIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45?
AssignVariableOp_45AssignVariableOp)assignvariableop_45_adam_dense_5_kernel_vIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46?
AssignVariableOp_46AssignVariableOp'assignvariableop_46_adam_dense_5_bias_vIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47?
AssignVariableOp_47AssignVariableOp4assignvariableop_47_adam_conv2d_transpose_6_kernel_vIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48?
AssignVariableOp_48AssignVariableOp2assignvariableop_48_adam_conv2d_transpose_6_bias_vIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49?
AssignVariableOp_49AssignVariableOp4assignvariableop_49_adam_conv2d_transpose_7_kernel_vIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50?
AssignVariableOp_50AssignVariableOp2assignvariableop_50_adam_conv2d_transpose_7_bias_vIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51?
AssignVariableOp_51AssignVariableOp4assignvariableop_51_adam_conv2d_transpose_8_kernel_vIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52?
AssignVariableOp_52AssignVariableOp2assignvariableop_52_adam_conv2d_transpose_8_bias_vIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_529
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?	
Identity_53Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_53f
Identity_54IdentityIdentity_53:output:0^NoOp_1*
T0*
_output_shapes
: 2
Identity_54?	
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 2
NoOp_1"#
identity_54Identity_54:output:0*
_input_shapesn
l: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
?
M__inference_conv2d_transpose_6_layer_call_and_return_conditional_losses_47162

inputsB
(conv2d_transpose_readvariableop_resource:@ -
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliceT
stack/1Const*
_output_shapes
: *
dtype0*
value	B :2	
stack/1T
stack/2Const*
_output_shapes
: *
dtype0*
value	B :2	
stack/2T
stack/3Const*
_output_shapes
: *
dtype0*
value	B :@2	
stack/3?
stackPackstrided_slice:output:0stack/1:output:0stack/2:output:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSlicestack:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:@ *
dtype02!
conv2d_transpose/ReadVariableOp?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
conv2d_transpose?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
Reluu
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????@2

Identity?
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
`
D__inference_flatten_2_layer_call_and_return_conditional_losses_45021

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"???? 	  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
T__inference_variational_autoencoder_2_layer_call_and_return_conditional_losses_46136
input_1,
sequential_4_46104:  
sequential_4_46106: ,
sequential_4_46108: @ 
sequential_4_46110:@%
sequential_4_46112:	? 
sequential_4_46114:%
sequential_5_46118:	?!
sequential_5_46120:	?,
sequential_5_46122:@  
sequential_5_46124:@,
sequential_5_46126: @ 
sequential_5_46128: ,
sequential_5_46130:  
sequential_5_46132:
identity??$sequential_4/StatefulPartitionedCall?$sequential_5/StatefulPartitionedCall?
$sequential_4/StatefulPartitionedCallStatefulPartitionedCallinput_1sequential_4_46104sequential_4_46106sequential_4_46108sequential_4_46110sequential_4_46112sequential_4_46114*
Tin
	2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_sequential_4_layer_call_and_return_conditional_losses_450552&
$sequential_4/StatefulPartitionedCall?
'normal_sampling_layer_2/PartitionedCallPartitionedCall-sequential_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_normal_sampling_layer_2_layer_call_and_return_conditional_losses_458782)
'normal_sampling_layer_2/PartitionedCall?
$sequential_5/StatefulPartitionedCallStatefulPartitionedCall0normal_sampling_layer_2/PartitionedCall:output:0sequential_5_46118sequential_5_46120sequential_5_46122sequential_5_46124sequential_5_46126sequential_5_46128sequential_5_46130sequential_5_46132*
Tin
2	*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_sequential_5_layer_call_and_return_conditional_losses_456432&
$sequential_5/StatefulPartitionedCall?
IdentityIdentity-sequential_5/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????2

Identity?
NoOpNoOp%^sequential_4/StatefulPartitionedCall%^sequential_5/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:?????????: : : : : : : : : : : : : : 2L
$sequential_4/StatefulPartitionedCall$sequential_4/StatefulPartitionedCall2L
$sequential_5/StatefulPartitionedCall$sequential_5/StatefulPartitionedCall:X T
/
_output_shapes
:?????????
!
_user_specified_name	input_1
?!
?
G__inference_sequential_5_layer_call_and_return_conditional_losses_45826
input_10 
dense_5_45803:	?
dense_5_45805:	?2
conv2d_transpose_6_45809:@ &
conv2d_transpose_6_45811:@2
conv2d_transpose_7_45814: @&
conv2d_transpose_7_45816: 2
conv2d_transpose_8_45819: &
conv2d_transpose_8_45821:
identity??*conv2d_transpose_6/StatefulPartitionedCall?*conv2d_transpose_7/StatefulPartitionedCall?*conv2d_transpose_8/StatefulPartitionedCall?dense_5/StatefulPartitionedCall?
dense_5/StatefulPartitionedCallStatefulPartitionedCallinput_10dense_5_45803dense_5_45805*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dense_5_layer_call_and_return_conditional_losses_455172!
dense_5/StatefulPartitionedCall?
reshape_10/PartitionedCallPartitionedCall(dense_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_reshape_10_layer_call_and_return_conditional_losses_455372
reshape_10/PartitionedCall?
*conv2d_transpose_6/StatefulPartitionedCallStatefulPartitionedCall#reshape_10/PartitionedCall:output:0conv2d_transpose_6_45809conv2d_transpose_6_45811*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *V
fQRO
M__inference_conv2d_transpose_6_layer_call_and_return_conditional_losses_455622,
*conv2d_transpose_6/StatefulPartitionedCall?
*conv2d_transpose_7/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_6/StatefulPartitionedCall:output:0conv2d_transpose_7_45814conv2d_transpose_7_45816*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *V
fQRO
M__inference_conv2d_transpose_7_layer_call_and_return_conditional_losses_455912,
*conv2d_transpose_7/StatefulPartitionedCall?
*conv2d_transpose_8/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_7/StatefulPartitionedCall:output:0conv2d_transpose_8_45819conv2d_transpose_8_45821*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *V
fQRO
M__inference_conv2d_transpose_8_layer_call_and_return_conditional_losses_456202,
*conv2d_transpose_8/StatefulPartitionedCall?
reshape_11/PartitionedCallPartitionedCall3conv2d_transpose_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_reshape_11_layer_call_and_return_conditional_losses_456402
reshape_11/PartitionedCall?
IdentityIdentity#reshape_11/PartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????2

Identity?
NoOpNoOp+^conv2d_transpose_6/StatefulPartitionedCall+^conv2d_transpose_7/StatefulPartitionedCall+^conv2d_transpose_8/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : : : : : 2X
*conv2d_transpose_6/StatefulPartitionedCall*conv2d_transpose_6/StatefulPartitionedCall2X
*conv2d_transpose_7/StatefulPartitionedCall*conv2d_transpose_7/StatefulPartitionedCall2X
*conv2d_transpose_8/StatefulPartitionedCall*conv2d_transpose_8/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall:Q M
'
_output_shapes
:?????????
"
_user_specified_name
input_10
?
?
T__inference_variational_autoencoder_2_layer_call_and_return_conditional_losses_46037

inputs,
sequential_4_46005:  
sequential_4_46007: ,
sequential_4_46009: @ 
sequential_4_46011:@%
sequential_4_46013:	? 
sequential_4_46015:%
sequential_5_46019:	?!
sequential_5_46021:	?,
sequential_5_46023:@  
sequential_5_46025:@,
sequential_5_46027: @ 
sequential_5_46029: ,
sequential_5_46031:  
sequential_5_46033:
identity??/normal_sampling_layer_2/StatefulPartitionedCall?$sequential_4/StatefulPartitionedCall?$sequential_5/StatefulPartitionedCall?
$sequential_4/StatefulPartitionedCallStatefulPartitionedCallinputssequential_4_46005sequential_4_46007sequential_4_46009sequential_4_46011sequential_4_46013sequential_4_46015*
Tin
	2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_sequential_4_layer_call_and_return_conditional_losses_451592&
$sequential_4/StatefulPartitionedCall?
/normal_sampling_layer_2/StatefulPartitionedCallStatefulPartitionedCall-sequential_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_normal_sampling_layer_2_layer_call_and_return_conditional_losses_4596421
/normal_sampling_layer_2/StatefulPartitionedCall?
$sequential_5/StatefulPartitionedCallStatefulPartitionedCall8normal_sampling_layer_2/StatefulPartitionedCall:output:0sequential_5_46019sequential_5_46021sequential_5_46023sequential_5_46025sequential_5_46027sequential_5_46029sequential_5_46031sequential_5_46033*
Tin
2	*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_sequential_5_layer_call_and_return_conditional_losses_457602&
$sequential_5/StatefulPartitionedCall?
IdentityIdentity-sequential_5/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????2

Identity?
NoOpNoOp0^normal_sampling_layer_2/StatefulPartitionedCall%^sequential_4/StatefulPartitionedCall%^sequential_5/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:?????????: : : : : : : : : : : : : : 2b
/normal_sampling_layer_2/StatefulPartitionedCall/normal_sampling_layer_2/StatefulPartitionedCall2L
$sequential_4/StatefulPartitionedCall$sequential_4/StatefulPartitionedCall2L
$sequential_5/StatefulPartitionedCall$sequential_5/StatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
C__inference_conv2d_5_layer_call_and_return_conditional_losses_45009

inputs8
conv2d_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
Reluu
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????@2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
M__inference_conv2d_transpose_7_layer_call_and_return_conditional_losses_47238

inputsB
(conv2d_transpose_readvariableop_resource: @-
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliceT
stack/1Const*
_output_shapes
: *
dtype0*
value	B :2	
stack/1T
stack/2Const*
_output_shapes
: *
dtype0*
value	B :2	
stack/2T
stack/3Const*
_output_shapes
: *
dtype0*
value	B : 2	
stack/3?
stackPackstrided_slice:output:0stack/1:output:0stack/2:output:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSlicestack:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype02!
conv2d_transpose/ReadVariableOp?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
2
conv2d_transpose?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
Reluu
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:????????? 2

Identity?
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
'__inference_dense_5_layer_call_fn_47085

inputs
unknown:	?
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dense_5_layer_call_and_return_conditional_losses_455172
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
B__inference_dense_4_layer_call_and_return_conditional_losses_45033

inputs1
matmul_readvariableop_resource:	?-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddk
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
M__inference_conv2d_transpose_7_layer_call_and_return_conditional_losses_45591

inputsB
(conv2d_transpose_readvariableop_resource: @-
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliceT
stack/1Const*
_output_shapes
: *
dtype0*
value	B :2	
stack/1T
stack/2Const*
_output_shapes
: *
dtype0*
value	B :2	
stack/2T
stack/3Const*
_output_shapes
: *
dtype0*
value	B : 2	
stack/3?
stackPackstrided_slice:output:0stack/1:output:0stack/2:output:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSlicestack:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype02!
conv2d_transpose/ReadVariableOp?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
2
conv2d_transpose?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
Reluu
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:????????? 2

Identity?
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
G__inference_sequential_4_layer_call_and_return_conditional_losses_45213
input_9(
conv2d_4_45195: 
conv2d_4_45197: (
conv2d_5_45200: @
conv2d_5_45202:@ 
dense_4_45206:	?
dense_4_45208:
identity?? conv2d_4/StatefulPartitionedCall? conv2d_5/StatefulPartitionedCall?dense_4/StatefulPartitionedCall?
reshape_8/PartitionedCallPartitionedCallinput_9*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_reshape_8_layer_call_and_return_conditional_losses_449792
reshape_8/PartitionedCall?
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall"reshape_8/PartitionedCall:output:0conv2d_4_45195conv2d_4_45197*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_conv2d_4_layer_call_and_return_conditional_losses_449922"
 conv2d_4/StatefulPartitionedCall?
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0conv2d_5_45200conv2d_5_45202*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_conv2d_5_layer_call_and_return_conditional_losses_450092"
 conv2d_5/StatefulPartitionedCall?
flatten_2/PartitionedCallPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_flatten_2_layer_call_and_return_conditional_losses_450212
flatten_2/PartitionedCall?
dense_4/StatefulPartitionedCallStatefulPartitionedCall"flatten_2/PartitionedCall:output:0dense_4_45206dense_4_45208*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dense_4_layer_call_and_return_conditional_losses_450332!
dense_4/StatefulPartitionedCall?
reshape_9/PartitionedCallPartitionedCall(dense_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_reshape_9_layer_call_and_return_conditional_losses_450522
reshape_9/PartitionedCall?
IdentityIdentity"reshape_9/PartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????2

Identity?
NoOpNoOp!^conv2d_4/StatefulPartitionedCall!^conv2d_5/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : 2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall:X T
/
_output_shapes
:?????????
!
_user_specified_name	input_9
?

?
,__inference_sequential_5_layer_call_fn_46894

inputs
unknown:	?
	unknown_0:	?#
	unknown_1:@ 
	unknown_2:@#
	unknown_3: @
	unknown_4: #
	unknown_5: 
	unknown_6:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_sequential_5_layer_call_and_return_conditional_losses_456432
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
E
)__inference_reshape_8_layer_call_fn_46977

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_reshape_8_layer_call_and_return_conditional_losses_449792
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
,__inference_sequential_5_layer_call_fn_45662
input_10
unknown:	?
	unknown_0:	?#
	unknown_1:@ 
	unknown_2:@#
	unknown_3: @
	unknown_4: #
	unknown_5: 
	unknown_6:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_10unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_sequential_5_layer_call_and_return_conditional_losses_456432
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:?????????
"
_user_specified_name
input_10
?
`
D__inference_reshape_9_layer_call_and_return_conditional_losses_47060

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliced
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/1d
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/2?
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapes
ReshapeReshapeinputsReshape/shape:output:0*
T0*+
_output_shapes
:?????????2	
Reshapeh
IdentityIdentityReshape:output:0*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
??
?
T__inference_variational_autoencoder_2_layer_call_and_return_conditional_losses_46346

inputsN
4sequential_4_conv2d_4_conv2d_readvariableop_resource: C
5sequential_4_conv2d_4_biasadd_readvariableop_resource: N
4sequential_4_conv2d_5_conv2d_readvariableop_resource: @C
5sequential_4_conv2d_5_biasadd_readvariableop_resource:@F
3sequential_4_dense_4_matmul_readvariableop_resource:	?B
4sequential_4_dense_4_biasadd_readvariableop_resource:F
3sequential_5_dense_5_matmul_readvariableop_resource:	?C
4sequential_5_dense_5_biasadd_readvariableop_resource:	?b
Hsequential_5_conv2d_transpose_6_conv2d_transpose_readvariableop_resource:@ M
?sequential_5_conv2d_transpose_6_biasadd_readvariableop_resource:@b
Hsequential_5_conv2d_transpose_7_conv2d_transpose_readvariableop_resource: @M
?sequential_5_conv2d_transpose_7_biasadd_readvariableop_resource: b
Hsequential_5_conv2d_transpose_8_conv2d_transpose_readvariableop_resource: M
?sequential_5_conv2d_transpose_8_biasadd_readvariableop_resource:
identity??,sequential_4/conv2d_4/BiasAdd/ReadVariableOp?+sequential_4/conv2d_4/Conv2D/ReadVariableOp?,sequential_4/conv2d_5/BiasAdd/ReadVariableOp?+sequential_4/conv2d_5/Conv2D/ReadVariableOp?+sequential_4/dense_4/BiasAdd/ReadVariableOp?*sequential_4/dense_4/MatMul/ReadVariableOp?6sequential_5/conv2d_transpose_6/BiasAdd/ReadVariableOp??sequential_5/conv2d_transpose_6/conv2d_transpose/ReadVariableOp?6sequential_5/conv2d_transpose_7/BiasAdd/ReadVariableOp??sequential_5/conv2d_transpose_7/conv2d_transpose/ReadVariableOp?6sequential_5/conv2d_transpose_8/BiasAdd/ReadVariableOp??sequential_5/conv2d_transpose_8/conv2d_transpose/ReadVariableOp?+sequential_5/dense_5/BiasAdd/ReadVariableOp?*sequential_5/dense_5/MatMul/ReadVariableOpr
sequential_4/reshape_8/ShapeShapeinputs*
T0*
_output_shapes
:2
sequential_4/reshape_8/Shape?
*sequential_4/reshape_8/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*sequential_4/reshape_8/strided_slice/stack?
,sequential_4/reshape_8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_4/reshape_8/strided_slice/stack_1?
,sequential_4/reshape_8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_4/reshape_8/strided_slice/stack_2?
$sequential_4/reshape_8/strided_sliceStridedSlice%sequential_4/reshape_8/Shape:output:03sequential_4/reshape_8/strided_slice/stack:output:05sequential_4/reshape_8/strided_slice/stack_1:output:05sequential_4/reshape_8/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2&
$sequential_4/reshape_8/strided_slice?
&sequential_4/reshape_8/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2(
&sequential_4/reshape_8/Reshape/shape/1?
&sequential_4/reshape_8/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2(
&sequential_4/reshape_8/Reshape/shape/2?
&sequential_4/reshape_8/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2(
&sequential_4/reshape_8/Reshape/shape/3?
$sequential_4/reshape_8/Reshape/shapePack-sequential_4/reshape_8/strided_slice:output:0/sequential_4/reshape_8/Reshape/shape/1:output:0/sequential_4/reshape_8/Reshape/shape/2:output:0/sequential_4/reshape_8/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2&
$sequential_4/reshape_8/Reshape/shape?
sequential_4/reshape_8/ReshapeReshapeinputs-sequential_4/reshape_8/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????2 
sequential_4/reshape_8/Reshape?
+sequential_4/conv2d_4/Conv2D/ReadVariableOpReadVariableOp4sequential_4_conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02-
+sequential_4/conv2d_4/Conv2D/ReadVariableOp?
sequential_4/conv2d_4/Conv2DConv2D'sequential_4/reshape_8/Reshape:output:03sequential_4/conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
2
sequential_4/conv2d_4/Conv2D?
,sequential_4/conv2d_4/BiasAdd/ReadVariableOpReadVariableOp5sequential_4_conv2d_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02.
,sequential_4/conv2d_4/BiasAdd/ReadVariableOp?
sequential_4/conv2d_4/BiasAddBiasAdd%sequential_4/conv2d_4/Conv2D:output:04sequential_4/conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2
sequential_4/conv2d_4/BiasAdd?
sequential_4/conv2d_4/ReluRelu&sequential_4/conv2d_4/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
sequential_4/conv2d_4/Relu?
+sequential_4/conv2d_5/Conv2D/ReadVariableOpReadVariableOp4sequential_4_conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02-
+sequential_4/conv2d_5/Conv2D/ReadVariableOp?
sequential_4/conv2d_5/Conv2DConv2D(sequential_4/conv2d_4/Relu:activations:03sequential_4/conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
2
sequential_4/conv2d_5/Conv2D?
,sequential_4/conv2d_5/BiasAdd/ReadVariableOpReadVariableOp5sequential_4_conv2d_5_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02.
,sequential_4/conv2d_5/BiasAdd/ReadVariableOp?
sequential_4/conv2d_5/BiasAddBiasAdd%sequential_4/conv2d_5/Conv2D:output:04sequential_4/conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
sequential_4/conv2d_5/BiasAdd?
sequential_4/conv2d_5/ReluRelu&sequential_4/conv2d_5/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
sequential_4/conv2d_5/Relu?
sequential_4/flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"???? 	  2
sequential_4/flatten_2/Const?
sequential_4/flatten_2/ReshapeReshape(sequential_4/conv2d_5/Relu:activations:0%sequential_4/flatten_2/Const:output:0*
T0*(
_output_shapes
:??????????2 
sequential_4/flatten_2/Reshape?
*sequential_4/dense_4/MatMul/ReadVariableOpReadVariableOp3sequential_4_dense_4_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02,
*sequential_4/dense_4/MatMul/ReadVariableOp?
sequential_4/dense_4/MatMulMatMul'sequential_4/flatten_2/Reshape:output:02sequential_4/dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential_4/dense_4/MatMul?
+sequential_4/dense_4/BiasAdd/ReadVariableOpReadVariableOp4sequential_4_dense_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02-
+sequential_4/dense_4/BiasAdd/ReadVariableOp?
sequential_4/dense_4/BiasAddBiasAdd%sequential_4/dense_4/MatMul:product:03sequential_4/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential_4/dense_4/BiasAdd?
sequential_4/reshape_9/ShapeShape%sequential_4/dense_4/BiasAdd:output:0*
T0*
_output_shapes
:2
sequential_4/reshape_9/Shape?
*sequential_4/reshape_9/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*sequential_4/reshape_9/strided_slice/stack?
,sequential_4/reshape_9/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_4/reshape_9/strided_slice/stack_1?
,sequential_4/reshape_9/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_4/reshape_9/strided_slice/stack_2?
$sequential_4/reshape_9/strided_sliceStridedSlice%sequential_4/reshape_9/Shape:output:03sequential_4/reshape_9/strided_slice/stack:output:05sequential_4/reshape_9/strided_slice/stack_1:output:05sequential_4/reshape_9/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2&
$sequential_4/reshape_9/strided_slice?
&sequential_4/reshape_9/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2(
&sequential_4/reshape_9/Reshape/shape/1?
&sequential_4/reshape_9/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2(
&sequential_4/reshape_9/Reshape/shape/2?
$sequential_4/reshape_9/Reshape/shapePack-sequential_4/reshape_9/strided_slice:output:0/sequential_4/reshape_9/Reshape/shape/1:output:0/sequential_4/reshape_9/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2&
$sequential_4/reshape_9/Reshape/shape?
sequential_4/reshape_9/ReshapeReshape%sequential_4/dense_4/BiasAdd:output:0-sequential_4/reshape_9/Reshape/shape:output:0*
T0*+
_output_shapes
:?????????2 
sequential_4/reshape_9/Reshape?
normal_sampling_layer_2/unstackUnpack'sequential_4/reshape_9/Reshape:output:0*
T0*:
_output_shapes(
&:?????????:?????????*

axis*	
num2!
normal_sampling_layer_2/unstack?
*sequential_5/dense_5/MatMul/ReadVariableOpReadVariableOp3sequential_5_dense_5_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02,
*sequential_5/dense_5/MatMul/ReadVariableOp?
sequential_5/dense_5/MatMulMatMul(normal_sampling_layer_2/unstack:output:02sequential_5/dense_5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential_5/dense_5/MatMul?
+sequential_5/dense_5/BiasAdd/ReadVariableOpReadVariableOp4sequential_5_dense_5_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02-
+sequential_5/dense_5/BiasAdd/ReadVariableOp?
sequential_5/dense_5/BiasAddBiasAdd%sequential_5/dense_5/MatMul:product:03sequential_5/dense_5/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential_5/dense_5/BiasAdd?
sequential_5/dense_5/ReluRelu%sequential_5/dense_5/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
sequential_5/dense_5/Relu?
sequential_5/reshape_10/ShapeShape'sequential_5/dense_5/Relu:activations:0*
T0*
_output_shapes
:2
sequential_5/reshape_10/Shape?
+sequential_5/reshape_10/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2-
+sequential_5/reshape_10/strided_slice/stack?
-sequential_5/reshape_10/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_5/reshape_10/strided_slice/stack_1?
-sequential_5/reshape_10/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_5/reshape_10/strided_slice/stack_2?
%sequential_5/reshape_10/strided_sliceStridedSlice&sequential_5/reshape_10/Shape:output:04sequential_5/reshape_10/strided_slice/stack:output:06sequential_5/reshape_10/strided_slice/stack_1:output:06sequential_5/reshape_10/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%sequential_5/reshape_10/strided_slice?
'sequential_5/reshape_10/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2)
'sequential_5/reshape_10/Reshape/shape/1?
'sequential_5/reshape_10/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2)
'sequential_5/reshape_10/Reshape/shape/2?
'sequential_5/reshape_10/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B : 2)
'sequential_5/reshape_10/Reshape/shape/3?
%sequential_5/reshape_10/Reshape/shapePack.sequential_5/reshape_10/strided_slice:output:00sequential_5/reshape_10/Reshape/shape/1:output:00sequential_5/reshape_10/Reshape/shape/2:output:00sequential_5/reshape_10/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2'
%sequential_5/reshape_10/Reshape/shape?
sequential_5/reshape_10/ReshapeReshape'sequential_5/dense_5/Relu:activations:0.sequential_5/reshape_10/Reshape/shape:output:0*
T0*/
_output_shapes
:????????? 2!
sequential_5/reshape_10/Reshape?
%sequential_5/conv2d_transpose_6/ShapeShape(sequential_5/reshape_10/Reshape:output:0*
T0*
_output_shapes
:2'
%sequential_5/conv2d_transpose_6/Shape?
3sequential_5/conv2d_transpose_6/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 25
3sequential_5/conv2d_transpose_6/strided_slice/stack?
5sequential_5/conv2d_transpose_6/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential_5/conv2d_transpose_6/strided_slice/stack_1?
5sequential_5/conv2d_transpose_6/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential_5/conv2d_transpose_6/strided_slice/stack_2?
-sequential_5/conv2d_transpose_6/strided_sliceStridedSlice.sequential_5/conv2d_transpose_6/Shape:output:0<sequential_5/conv2d_transpose_6/strided_slice/stack:output:0>sequential_5/conv2d_transpose_6/strided_slice/stack_1:output:0>sequential_5/conv2d_transpose_6/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2/
-sequential_5/conv2d_transpose_6/strided_slice?
'sequential_5/conv2d_transpose_6/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2)
'sequential_5/conv2d_transpose_6/stack/1?
'sequential_5/conv2d_transpose_6/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2)
'sequential_5/conv2d_transpose_6/stack/2?
'sequential_5/conv2d_transpose_6/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@2)
'sequential_5/conv2d_transpose_6/stack/3?
%sequential_5/conv2d_transpose_6/stackPack6sequential_5/conv2d_transpose_6/strided_slice:output:00sequential_5/conv2d_transpose_6/stack/1:output:00sequential_5/conv2d_transpose_6/stack/2:output:00sequential_5/conv2d_transpose_6/stack/3:output:0*
N*
T0*
_output_shapes
:2'
%sequential_5/conv2d_transpose_6/stack?
5sequential_5/conv2d_transpose_6/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 27
5sequential_5/conv2d_transpose_6/strided_slice_1/stack?
7sequential_5/conv2d_transpose_6/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:29
7sequential_5/conv2d_transpose_6/strided_slice_1/stack_1?
7sequential_5/conv2d_transpose_6/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:29
7sequential_5/conv2d_transpose_6/strided_slice_1/stack_2?
/sequential_5/conv2d_transpose_6/strided_slice_1StridedSlice.sequential_5/conv2d_transpose_6/stack:output:0>sequential_5/conv2d_transpose_6/strided_slice_1/stack:output:0@sequential_5/conv2d_transpose_6/strided_slice_1/stack_1:output:0@sequential_5/conv2d_transpose_6/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask21
/sequential_5/conv2d_transpose_6/strided_slice_1?
?sequential_5/conv2d_transpose_6/conv2d_transpose/ReadVariableOpReadVariableOpHsequential_5_conv2d_transpose_6_conv2d_transpose_readvariableop_resource*&
_output_shapes
:@ *
dtype02A
?sequential_5/conv2d_transpose_6/conv2d_transpose/ReadVariableOp?
0sequential_5/conv2d_transpose_6/conv2d_transposeConv2DBackpropInput.sequential_5/conv2d_transpose_6/stack:output:0Gsequential_5/conv2d_transpose_6/conv2d_transpose/ReadVariableOp:value:0(sequential_5/reshape_10/Reshape:output:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
22
0sequential_5/conv2d_transpose_6/conv2d_transpose?
6sequential_5/conv2d_transpose_6/BiasAdd/ReadVariableOpReadVariableOp?sequential_5_conv2d_transpose_6_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype028
6sequential_5/conv2d_transpose_6/BiasAdd/ReadVariableOp?
'sequential_5/conv2d_transpose_6/BiasAddBiasAdd9sequential_5/conv2d_transpose_6/conv2d_transpose:output:0>sequential_5/conv2d_transpose_6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2)
'sequential_5/conv2d_transpose_6/BiasAdd?
$sequential_5/conv2d_transpose_6/ReluRelu0sequential_5/conv2d_transpose_6/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2&
$sequential_5/conv2d_transpose_6/Relu?
%sequential_5/conv2d_transpose_7/ShapeShape2sequential_5/conv2d_transpose_6/Relu:activations:0*
T0*
_output_shapes
:2'
%sequential_5/conv2d_transpose_7/Shape?
3sequential_5/conv2d_transpose_7/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 25
3sequential_5/conv2d_transpose_7/strided_slice/stack?
5sequential_5/conv2d_transpose_7/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential_5/conv2d_transpose_7/strided_slice/stack_1?
5sequential_5/conv2d_transpose_7/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential_5/conv2d_transpose_7/strided_slice/stack_2?
-sequential_5/conv2d_transpose_7/strided_sliceStridedSlice.sequential_5/conv2d_transpose_7/Shape:output:0<sequential_5/conv2d_transpose_7/strided_slice/stack:output:0>sequential_5/conv2d_transpose_7/strided_slice/stack_1:output:0>sequential_5/conv2d_transpose_7/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2/
-sequential_5/conv2d_transpose_7/strided_slice?
'sequential_5/conv2d_transpose_7/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2)
'sequential_5/conv2d_transpose_7/stack/1?
'sequential_5/conv2d_transpose_7/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2)
'sequential_5/conv2d_transpose_7/stack/2?
'sequential_5/conv2d_transpose_7/stack/3Const*
_output_shapes
: *
dtype0*
value	B : 2)
'sequential_5/conv2d_transpose_7/stack/3?
%sequential_5/conv2d_transpose_7/stackPack6sequential_5/conv2d_transpose_7/strided_slice:output:00sequential_5/conv2d_transpose_7/stack/1:output:00sequential_5/conv2d_transpose_7/stack/2:output:00sequential_5/conv2d_transpose_7/stack/3:output:0*
N*
T0*
_output_shapes
:2'
%sequential_5/conv2d_transpose_7/stack?
5sequential_5/conv2d_transpose_7/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 27
5sequential_5/conv2d_transpose_7/strided_slice_1/stack?
7sequential_5/conv2d_transpose_7/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:29
7sequential_5/conv2d_transpose_7/strided_slice_1/stack_1?
7sequential_5/conv2d_transpose_7/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:29
7sequential_5/conv2d_transpose_7/strided_slice_1/stack_2?
/sequential_5/conv2d_transpose_7/strided_slice_1StridedSlice.sequential_5/conv2d_transpose_7/stack:output:0>sequential_5/conv2d_transpose_7/strided_slice_1/stack:output:0@sequential_5/conv2d_transpose_7/strided_slice_1/stack_1:output:0@sequential_5/conv2d_transpose_7/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask21
/sequential_5/conv2d_transpose_7/strided_slice_1?
?sequential_5/conv2d_transpose_7/conv2d_transpose/ReadVariableOpReadVariableOpHsequential_5_conv2d_transpose_7_conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype02A
?sequential_5/conv2d_transpose_7/conv2d_transpose/ReadVariableOp?
0sequential_5/conv2d_transpose_7/conv2d_transposeConv2DBackpropInput.sequential_5/conv2d_transpose_7/stack:output:0Gsequential_5/conv2d_transpose_7/conv2d_transpose/ReadVariableOp:value:02sequential_5/conv2d_transpose_6/Relu:activations:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
22
0sequential_5/conv2d_transpose_7/conv2d_transpose?
6sequential_5/conv2d_transpose_7/BiasAdd/ReadVariableOpReadVariableOp?sequential_5_conv2d_transpose_7_biasadd_readvariableop_resource*
_output_shapes
: *
dtype028
6sequential_5/conv2d_transpose_7/BiasAdd/ReadVariableOp?
'sequential_5/conv2d_transpose_7/BiasAddBiasAdd9sequential_5/conv2d_transpose_7/conv2d_transpose:output:0>sequential_5/conv2d_transpose_7/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2)
'sequential_5/conv2d_transpose_7/BiasAdd?
$sequential_5/conv2d_transpose_7/ReluRelu0sequential_5/conv2d_transpose_7/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2&
$sequential_5/conv2d_transpose_7/Relu?
%sequential_5/conv2d_transpose_8/ShapeShape2sequential_5/conv2d_transpose_7/Relu:activations:0*
T0*
_output_shapes
:2'
%sequential_5/conv2d_transpose_8/Shape?
3sequential_5/conv2d_transpose_8/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 25
3sequential_5/conv2d_transpose_8/strided_slice/stack?
5sequential_5/conv2d_transpose_8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential_5/conv2d_transpose_8/strided_slice/stack_1?
5sequential_5/conv2d_transpose_8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential_5/conv2d_transpose_8/strided_slice/stack_2?
-sequential_5/conv2d_transpose_8/strided_sliceStridedSlice.sequential_5/conv2d_transpose_8/Shape:output:0<sequential_5/conv2d_transpose_8/strided_slice/stack:output:0>sequential_5/conv2d_transpose_8/strided_slice/stack_1:output:0>sequential_5/conv2d_transpose_8/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2/
-sequential_5/conv2d_transpose_8/strided_slice?
'sequential_5/conv2d_transpose_8/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2)
'sequential_5/conv2d_transpose_8/stack/1?
'sequential_5/conv2d_transpose_8/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2)
'sequential_5/conv2d_transpose_8/stack/2?
'sequential_5/conv2d_transpose_8/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2)
'sequential_5/conv2d_transpose_8/stack/3?
%sequential_5/conv2d_transpose_8/stackPack6sequential_5/conv2d_transpose_8/strided_slice:output:00sequential_5/conv2d_transpose_8/stack/1:output:00sequential_5/conv2d_transpose_8/stack/2:output:00sequential_5/conv2d_transpose_8/stack/3:output:0*
N*
T0*
_output_shapes
:2'
%sequential_5/conv2d_transpose_8/stack?
5sequential_5/conv2d_transpose_8/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 27
5sequential_5/conv2d_transpose_8/strided_slice_1/stack?
7sequential_5/conv2d_transpose_8/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:29
7sequential_5/conv2d_transpose_8/strided_slice_1/stack_1?
7sequential_5/conv2d_transpose_8/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:29
7sequential_5/conv2d_transpose_8/strided_slice_1/stack_2?
/sequential_5/conv2d_transpose_8/strided_slice_1StridedSlice.sequential_5/conv2d_transpose_8/stack:output:0>sequential_5/conv2d_transpose_8/strided_slice_1/stack:output:0@sequential_5/conv2d_transpose_8/strided_slice_1/stack_1:output:0@sequential_5/conv2d_transpose_8/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask21
/sequential_5/conv2d_transpose_8/strided_slice_1?
?sequential_5/conv2d_transpose_8/conv2d_transpose/ReadVariableOpReadVariableOpHsequential_5_conv2d_transpose_8_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype02A
?sequential_5/conv2d_transpose_8/conv2d_transpose/ReadVariableOp?
0sequential_5/conv2d_transpose_8/conv2d_transposeConv2DBackpropInput.sequential_5/conv2d_transpose_8/stack:output:0Gsequential_5/conv2d_transpose_8/conv2d_transpose/ReadVariableOp:value:02sequential_5/conv2d_transpose_7/Relu:activations:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
22
0sequential_5/conv2d_transpose_8/conv2d_transpose?
6sequential_5/conv2d_transpose_8/BiasAdd/ReadVariableOpReadVariableOp?sequential_5_conv2d_transpose_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype028
6sequential_5/conv2d_transpose_8/BiasAdd/ReadVariableOp?
'sequential_5/conv2d_transpose_8/BiasAddBiasAdd9sequential_5/conv2d_transpose_8/conv2d_transpose:output:0>sequential_5/conv2d_transpose_8/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2)
'sequential_5/conv2d_transpose_8/BiasAdd?
'sequential_5/conv2d_transpose_8/SigmoidSigmoid0sequential_5/conv2d_transpose_8/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2)
'sequential_5/conv2d_transpose_8/Sigmoid?
sequential_5/reshape_11/ShapeShape+sequential_5/conv2d_transpose_8/Sigmoid:y:0*
T0*
_output_shapes
:2
sequential_5/reshape_11/Shape?
+sequential_5/reshape_11/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2-
+sequential_5/reshape_11/strided_slice/stack?
-sequential_5/reshape_11/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_5/reshape_11/strided_slice/stack_1?
-sequential_5/reshape_11/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_5/reshape_11/strided_slice/stack_2?
%sequential_5/reshape_11/strided_sliceStridedSlice&sequential_5/reshape_11/Shape:output:04sequential_5/reshape_11/strided_slice/stack:output:06sequential_5/reshape_11/strided_slice/stack_1:output:06sequential_5/reshape_11/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%sequential_5/reshape_11/strided_slice?
'sequential_5/reshape_11/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2)
'sequential_5/reshape_11/Reshape/shape/1?
'sequential_5/reshape_11/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2)
'sequential_5/reshape_11/Reshape/shape/2?
'sequential_5/reshape_11/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2)
'sequential_5/reshape_11/Reshape/shape/3?
%sequential_5/reshape_11/Reshape/shapePack.sequential_5/reshape_11/strided_slice:output:00sequential_5/reshape_11/Reshape/shape/1:output:00sequential_5/reshape_11/Reshape/shape/2:output:00sequential_5/reshape_11/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2'
%sequential_5/reshape_11/Reshape/shape?
sequential_5/reshape_11/ReshapeReshape+sequential_5/conv2d_transpose_8/Sigmoid:y:0.sequential_5/reshape_11/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????2!
sequential_5/reshape_11/Reshape?
IdentityIdentity(sequential_5/reshape_11/Reshape:output:0^NoOp*
T0*/
_output_shapes
:?????????2

Identity?
NoOpNoOp-^sequential_4/conv2d_4/BiasAdd/ReadVariableOp,^sequential_4/conv2d_4/Conv2D/ReadVariableOp-^sequential_4/conv2d_5/BiasAdd/ReadVariableOp,^sequential_4/conv2d_5/Conv2D/ReadVariableOp,^sequential_4/dense_4/BiasAdd/ReadVariableOp+^sequential_4/dense_4/MatMul/ReadVariableOp7^sequential_5/conv2d_transpose_6/BiasAdd/ReadVariableOp@^sequential_5/conv2d_transpose_6/conv2d_transpose/ReadVariableOp7^sequential_5/conv2d_transpose_7/BiasAdd/ReadVariableOp@^sequential_5/conv2d_transpose_7/conv2d_transpose/ReadVariableOp7^sequential_5/conv2d_transpose_8/BiasAdd/ReadVariableOp@^sequential_5/conv2d_transpose_8/conv2d_transpose/ReadVariableOp,^sequential_5/dense_5/BiasAdd/ReadVariableOp+^sequential_5/dense_5/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:?????????: : : : : : : : : : : : : : 2\
,sequential_4/conv2d_4/BiasAdd/ReadVariableOp,sequential_4/conv2d_4/BiasAdd/ReadVariableOp2Z
+sequential_4/conv2d_4/Conv2D/ReadVariableOp+sequential_4/conv2d_4/Conv2D/ReadVariableOp2\
,sequential_4/conv2d_5/BiasAdd/ReadVariableOp,sequential_4/conv2d_5/BiasAdd/ReadVariableOp2Z
+sequential_4/conv2d_5/Conv2D/ReadVariableOp+sequential_4/conv2d_5/Conv2D/ReadVariableOp2Z
+sequential_4/dense_4/BiasAdd/ReadVariableOp+sequential_4/dense_4/BiasAdd/ReadVariableOp2X
*sequential_4/dense_4/MatMul/ReadVariableOp*sequential_4/dense_4/MatMul/ReadVariableOp2p
6sequential_5/conv2d_transpose_6/BiasAdd/ReadVariableOp6sequential_5/conv2d_transpose_6/BiasAdd/ReadVariableOp2?
?sequential_5/conv2d_transpose_6/conv2d_transpose/ReadVariableOp?sequential_5/conv2d_transpose_6/conv2d_transpose/ReadVariableOp2p
6sequential_5/conv2d_transpose_7/BiasAdd/ReadVariableOp6sequential_5/conv2d_transpose_7/BiasAdd/ReadVariableOp2?
?sequential_5/conv2d_transpose_7/conv2d_transpose/ReadVariableOp?sequential_5/conv2d_transpose_7/conv2d_transpose/ReadVariableOp2p
6sequential_5/conv2d_transpose_8/BiasAdd/ReadVariableOp6sequential_5/conv2d_transpose_8/BiasAdd/ReadVariableOp2?
?sequential_5/conv2d_transpose_8/conv2d_transpose/ReadVariableOp?sequential_5/conv2d_transpose_8/conv2d_transpose/ReadVariableOp2Z
+sequential_5/dense_5/BiasAdd/ReadVariableOp+sequential_5/dense_5/BiasAdd/ReadVariableOp2X
*sequential_5/dense_5/MatMul/ReadVariableOp*sequential_5/dense_5/MatMul/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?9
?
G__inference_sequential_4_layer_call_and_return_conditional_losses_46657

inputsA
'conv2d_4_conv2d_readvariableop_resource: 6
(conv2d_4_biasadd_readvariableop_resource: A
'conv2d_5_conv2d_readvariableop_resource: @6
(conv2d_5_biasadd_readvariableop_resource:@9
&dense_4_matmul_readvariableop_resource:	?5
'dense_4_biasadd_readvariableop_resource:
identity??conv2d_4/BiasAdd/ReadVariableOp?conv2d_4/Conv2D/ReadVariableOp?conv2d_5/BiasAdd/ReadVariableOp?conv2d_5/Conv2D/ReadVariableOp?dense_4/BiasAdd/ReadVariableOp?dense_4/MatMul/ReadVariableOpX
reshape_8/ShapeShapeinputs*
T0*
_output_shapes
:2
reshape_8/Shape?
reshape_8/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape_8/strided_slice/stack?
reshape_8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_8/strided_slice/stack_1?
reshape_8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_8/strided_slice/stack_2?
reshape_8/strided_sliceStridedSlicereshape_8/Shape:output:0&reshape_8/strided_slice/stack:output:0(reshape_8/strided_slice/stack_1:output:0(reshape_8/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_8/strided_slicex
reshape_8/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_8/Reshape/shape/1x
reshape_8/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_8/Reshape/shape/2x
reshape_8/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_8/Reshape/shape/3?
reshape_8/Reshape/shapePack reshape_8/strided_slice:output:0"reshape_8/Reshape/shape/1:output:0"reshape_8/Reshape/shape/2:output:0"reshape_8/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
reshape_8/Reshape/shape?
reshape_8/ReshapeReshapeinputs reshape_8/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????2
reshape_8/Reshape?
conv2d_4/Conv2D/ReadVariableOpReadVariableOp'conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02 
conv2d_4/Conv2D/ReadVariableOp?
conv2d_4/Conv2DConv2Dreshape_8/Reshape:output:0&conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
2
conv2d_4/Conv2D?
conv2d_4/BiasAdd/ReadVariableOpReadVariableOp(conv2d_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv2d_4/BiasAdd/ReadVariableOp?
conv2d_4/BiasAddBiasAddconv2d_4/Conv2D:output:0'conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2
conv2d_4/BiasAdd{
conv2d_4/ReluReluconv2d_4/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
conv2d_4/Relu?
conv2d_5/Conv2D/ReadVariableOpReadVariableOp'conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02 
conv2d_5/Conv2D/ReadVariableOp?
conv2d_5/Conv2DConv2Dconv2d_4/Relu:activations:0&conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
2
conv2d_5/Conv2D?
conv2d_5/BiasAdd/ReadVariableOpReadVariableOp(conv2d_5_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_5/BiasAdd/ReadVariableOp?
conv2d_5/BiasAddBiasAddconv2d_5/Conv2D:output:0'conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
conv2d_5/BiasAdd{
conv2d_5/ReluReluconv2d_5/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
conv2d_5/Relus
flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"???? 	  2
flatten_2/Const?
flatten_2/ReshapeReshapeconv2d_5/Relu:activations:0flatten_2/Const:output:0*
T0*(
_output_shapes
:??????????2
flatten_2/Reshape?
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
dense_4/MatMul/ReadVariableOp?
dense_4/MatMulMatMulflatten_2/Reshape:output:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_4/MatMul?
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_4/BiasAdd/ReadVariableOp?
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_4/BiasAddj
reshape_9/ShapeShapedense_4/BiasAdd:output:0*
T0*
_output_shapes
:2
reshape_9/Shape?
reshape_9/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape_9/strided_slice/stack?
reshape_9/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_9/strided_slice/stack_1?
reshape_9/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_9/strided_slice/stack_2?
reshape_9/strided_sliceStridedSlicereshape_9/Shape:output:0&reshape_9/strided_slice/stack:output:0(reshape_9/strided_slice/stack_1:output:0(reshape_9/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_9/strided_slicex
reshape_9/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_9/Reshape/shape/1x
reshape_9/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_9/Reshape/shape/2?
reshape_9/Reshape/shapePack reshape_9/strided_slice:output:0"reshape_9/Reshape/shape/1:output:0"reshape_9/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
reshape_9/Reshape/shape?
reshape_9/ReshapeReshapedense_4/BiasAdd:output:0 reshape_9/Reshape/shape:output:0*
T0*+
_output_shapes
:?????????2
reshape_9/Reshapey
IdentityIdentityreshape_9/Reshape:output:0^NoOp*
T0*+
_output_shapes
:?????????2

Identity?
NoOpNoOp ^conv2d_4/BiasAdd/ReadVariableOp^conv2d_4/Conv2D/ReadVariableOp ^conv2d_5/BiasAdd/ReadVariableOp^conv2d_5/Conv2D/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : 2B
conv2d_4/BiasAdd/ReadVariableOpconv2d_4/BiasAdd/ReadVariableOp2@
conv2d_4/Conv2D/ReadVariableOpconv2d_4/Conv2D/ReadVariableOp2B
conv2d_5/BiasAdd/ReadVariableOpconv2d_5/BiasAdd/ReadVariableOp2@
conv2d_5/Conv2D/ReadVariableOpconv2d_5/Conv2D/ReadVariableOp2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
M__inference_conv2d_transpose_8_layer_call_and_return_conditional_losses_47314

inputsB
(conv2d_transpose_readvariableop_resource: -
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliceT
stack/1Const*
_output_shapes
: *
dtype0*
value	B :2	
stack/1T
stack/2Const*
_output_shapes
: *
dtype0*
value	B :2	
stack/2T
stack/3Const*
_output_shapes
: *
dtype0*
value	B :2	
stack/3?
stackPackstrided_slice:output:0stack/1:output:0stack/2:output:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSlicestack:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_transpose/ReadVariableOp?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
conv2d_transpose?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2	
BiasAddi
SigmoidSigmoidBiasAdd:output:0*
T0*/
_output_shapes
:?????????2	
Sigmoidn
IdentityIdentitySigmoid:y:0^NoOp*
T0*/
_output_shapes
:?????????2

Identity?
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
F
*__inference_reshape_11_layer_call_fn_47351

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_reshape_11_layer_call_and_return_conditional_losses_456402
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
C__inference_conv2d_4_layer_call_and_return_conditional_losses_46988

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
Reluu
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:????????? 2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
??
?
 __inference__wrapped_model_44958
input_1h
Nvariational_autoencoder_2_sequential_4_conv2d_4_conv2d_readvariableop_resource: ]
Ovariational_autoencoder_2_sequential_4_conv2d_4_biasadd_readvariableop_resource: h
Nvariational_autoencoder_2_sequential_4_conv2d_5_conv2d_readvariableop_resource: @]
Ovariational_autoencoder_2_sequential_4_conv2d_5_biasadd_readvariableop_resource:@`
Mvariational_autoencoder_2_sequential_4_dense_4_matmul_readvariableop_resource:	?\
Nvariational_autoencoder_2_sequential_4_dense_4_biasadd_readvariableop_resource:`
Mvariational_autoencoder_2_sequential_5_dense_5_matmul_readvariableop_resource:	?]
Nvariational_autoencoder_2_sequential_5_dense_5_biasadd_readvariableop_resource:	?|
bvariational_autoencoder_2_sequential_5_conv2d_transpose_6_conv2d_transpose_readvariableop_resource:@ g
Yvariational_autoencoder_2_sequential_5_conv2d_transpose_6_biasadd_readvariableop_resource:@|
bvariational_autoencoder_2_sequential_5_conv2d_transpose_7_conv2d_transpose_readvariableop_resource: @g
Yvariational_autoencoder_2_sequential_5_conv2d_transpose_7_biasadd_readvariableop_resource: |
bvariational_autoencoder_2_sequential_5_conv2d_transpose_8_conv2d_transpose_readvariableop_resource: g
Yvariational_autoencoder_2_sequential_5_conv2d_transpose_8_biasadd_readvariableop_resource:
identity??Fvariational_autoencoder_2/sequential_4/conv2d_4/BiasAdd/ReadVariableOp?Evariational_autoencoder_2/sequential_4/conv2d_4/Conv2D/ReadVariableOp?Fvariational_autoencoder_2/sequential_4/conv2d_5/BiasAdd/ReadVariableOp?Evariational_autoencoder_2/sequential_4/conv2d_5/Conv2D/ReadVariableOp?Evariational_autoencoder_2/sequential_4/dense_4/BiasAdd/ReadVariableOp?Dvariational_autoencoder_2/sequential_4/dense_4/MatMul/ReadVariableOp?Pvariational_autoencoder_2/sequential_5/conv2d_transpose_6/BiasAdd/ReadVariableOp?Yvariational_autoencoder_2/sequential_5/conv2d_transpose_6/conv2d_transpose/ReadVariableOp?Pvariational_autoencoder_2/sequential_5/conv2d_transpose_7/BiasAdd/ReadVariableOp?Yvariational_autoencoder_2/sequential_5/conv2d_transpose_7/conv2d_transpose/ReadVariableOp?Pvariational_autoencoder_2/sequential_5/conv2d_transpose_8/BiasAdd/ReadVariableOp?Yvariational_autoencoder_2/sequential_5/conv2d_transpose_8/conv2d_transpose/ReadVariableOp?Evariational_autoencoder_2/sequential_5/dense_5/BiasAdd/ReadVariableOp?Dvariational_autoencoder_2/sequential_5/dense_5/MatMul/ReadVariableOp?
6variational_autoencoder_2/sequential_4/reshape_8/ShapeShapeinput_1*
T0*
_output_shapes
:28
6variational_autoencoder_2/sequential_4/reshape_8/Shape?
Dvariational_autoencoder_2/sequential_4/reshape_8/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2F
Dvariational_autoencoder_2/sequential_4/reshape_8/strided_slice/stack?
Fvariational_autoencoder_2/sequential_4/reshape_8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2H
Fvariational_autoencoder_2/sequential_4/reshape_8/strided_slice/stack_1?
Fvariational_autoencoder_2/sequential_4/reshape_8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2H
Fvariational_autoencoder_2/sequential_4/reshape_8/strided_slice/stack_2?
>variational_autoencoder_2/sequential_4/reshape_8/strided_sliceStridedSlice?variational_autoencoder_2/sequential_4/reshape_8/Shape:output:0Mvariational_autoencoder_2/sequential_4/reshape_8/strided_slice/stack:output:0Ovariational_autoencoder_2/sequential_4/reshape_8/strided_slice/stack_1:output:0Ovariational_autoencoder_2/sequential_4/reshape_8/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2@
>variational_autoencoder_2/sequential_4/reshape_8/strided_slice?
@variational_autoencoder_2/sequential_4/reshape_8/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2B
@variational_autoencoder_2/sequential_4/reshape_8/Reshape/shape/1?
@variational_autoencoder_2/sequential_4/reshape_8/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2B
@variational_autoencoder_2/sequential_4/reshape_8/Reshape/shape/2?
@variational_autoencoder_2/sequential_4/reshape_8/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2B
@variational_autoencoder_2/sequential_4/reshape_8/Reshape/shape/3?
>variational_autoencoder_2/sequential_4/reshape_8/Reshape/shapePackGvariational_autoencoder_2/sequential_4/reshape_8/strided_slice:output:0Ivariational_autoencoder_2/sequential_4/reshape_8/Reshape/shape/1:output:0Ivariational_autoencoder_2/sequential_4/reshape_8/Reshape/shape/2:output:0Ivariational_autoencoder_2/sequential_4/reshape_8/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2@
>variational_autoencoder_2/sequential_4/reshape_8/Reshape/shape?
8variational_autoencoder_2/sequential_4/reshape_8/ReshapeReshapeinput_1Gvariational_autoencoder_2/sequential_4/reshape_8/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????2:
8variational_autoencoder_2/sequential_4/reshape_8/Reshape?
Evariational_autoencoder_2/sequential_4/conv2d_4/Conv2D/ReadVariableOpReadVariableOpNvariational_autoencoder_2_sequential_4_conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02G
Evariational_autoencoder_2/sequential_4/conv2d_4/Conv2D/ReadVariableOp?
6variational_autoencoder_2/sequential_4/conv2d_4/Conv2DConv2DAvariational_autoencoder_2/sequential_4/reshape_8/Reshape:output:0Mvariational_autoencoder_2/sequential_4/conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
28
6variational_autoencoder_2/sequential_4/conv2d_4/Conv2D?
Fvariational_autoencoder_2/sequential_4/conv2d_4/BiasAdd/ReadVariableOpReadVariableOpOvariational_autoencoder_2_sequential_4_conv2d_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02H
Fvariational_autoencoder_2/sequential_4/conv2d_4/BiasAdd/ReadVariableOp?
7variational_autoencoder_2/sequential_4/conv2d_4/BiasAddBiasAdd?variational_autoencoder_2/sequential_4/conv2d_4/Conv2D:output:0Nvariational_autoencoder_2/sequential_4/conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 29
7variational_autoencoder_2/sequential_4/conv2d_4/BiasAdd?
4variational_autoencoder_2/sequential_4/conv2d_4/ReluRelu@variational_autoencoder_2/sequential_4/conv2d_4/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 26
4variational_autoencoder_2/sequential_4/conv2d_4/Relu?
Evariational_autoencoder_2/sequential_4/conv2d_5/Conv2D/ReadVariableOpReadVariableOpNvariational_autoencoder_2_sequential_4_conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02G
Evariational_autoencoder_2/sequential_4/conv2d_5/Conv2D/ReadVariableOp?
6variational_autoencoder_2/sequential_4/conv2d_5/Conv2DConv2DBvariational_autoencoder_2/sequential_4/conv2d_4/Relu:activations:0Mvariational_autoencoder_2/sequential_4/conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
28
6variational_autoencoder_2/sequential_4/conv2d_5/Conv2D?
Fvariational_autoencoder_2/sequential_4/conv2d_5/BiasAdd/ReadVariableOpReadVariableOpOvariational_autoencoder_2_sequential_4_conv2d_5_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02H
Fvariational_autoencoder_2/sequential_4/conv2d_5/BiasAdd/ReadVariableOp?
7variational_autoencoder_2/sequential_4/conv2d_5/BiasAddBiasAdd?variational_autoencoder_2/sequential_4/conv2d_5/Conv2D:output:0Nvariational_autoencoder_2/sequential_4/conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@29
7variational_autoencoder_2/sequential_4/conv2d_5/BiasAdd?
4variational_autoencoder_2/sequential_4/conv2d_5/ReluRelu@variational_autoencoder_2/sequential_4/conv2d_5/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@26
4variational_autoencoder_2/sequential_4/conv2d_5/Relu?
6variational_autoencoder_2/sequential_4/flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"???? 	  28
6variational_autoencoder_2/sequential_4/flatten_2/Const?
8variational_autoencoder_2/sequential_4/flatten_2/ReshapeReshapeBvariational_autoencoder_2/sequential_4/conv2d_5/Relu:activations:0?variational_autoencoder_2/sequential_4/flatten_2/Const:output:0*
T0*(
_output_shapes
:??????????2:
8variational_autoencoder_2/sequential_4/flatten_2/Reshape?
Dvariational_autoencoder_2/sequential_4/dense_4/MatMul/ReadVariableOpReadVariableOpMvariational_autoencoder_2_sequential_4_dense_4_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02F
Dvariational_autoencoder_2/sequential_4/dense_4/MatMul/ReadVariableOp?
5variational_autoencoder_2/sequential_4/dense_4/MatMulMatMulAvariational_autoencoder_2/sequential_4/flatten_2/Reshape:output:0Lvariational_autoencoder_2/sequential_4/dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????27
5variational_autoencoder_2/sequential_4/dense_4/MatMul?
Evariational_autoencoder_2/sequential_4/dense_4/BiasAdd/ReadVariableOpReadVariableOpNvariational_autoencoder_2_sequential_4_dense_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02G
Evariational_autoencoder_2/sequential_4/dense_4/BiasAdd/ReadVariableOp?
6variational_autoencoder_2/sequential_4/dense_4/BiasAddBiasAdd?variational_autoencoder_2/sequential_4/dense_4/MatMul:product:0Mvariational_autoencoder_2/sequential_4/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????28
6variational_autoencoder_2/sequential_4/dense_4/BiasAdd?
6variational_autoencoder_2/sequential_4/reshape_9/ShapeShape?variational_autoencoder_2/sequential_4/dense_4/BiasAdd:output:0*
T0*
_output_shapes
:28
6variational_autoencoder_2/sequential_4/reshape_9/Shape?
Dvariational_autoencoder_2/sequential_4/reshape_9/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2F
Dvariational_autoencoder_2/sequential_4/reshape_9/strided_slice/stack?
Fvariational_autoencoder_2/sequential_4/reshape_9/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2H
Fvariational_autoencoder_2/sequential_4/reshape_9/strided_slice/stack_1?
Fvariational_autoencoder_2/sequential_4/reshape_9/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2H
Fvariational_autoencoder_2/sequential_4/reshape_9/strided_slice/stack_2?
>variational_autoencoder_2/sequential_4/reshape_9/strided_sliceStridedSlice?variational_autoencoder_2/sequential_4/reshape_9/Shape:output:0Mvariational_autoencoder_2/sequential_4/reshape_9/strided_slice/stack:output:0Ovariational_autoencoder_2/sequential_4/reshape_9/strided_slice/stack_1:output:0Ovariational_autoencoder_2/sequential_4/reshape_9/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2@
>variational_autoencoder_2/sequential_4/reshape_9/strided_slice?
@variational_autoencoder_2/sequential_4/reshape_9/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2B
@variational_autoencoder_2/sequential_4/reshape_9/Reshape/shape/1?
@variational_autoencoder_2/sequential_4/reshape_9/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2B
@variational_autoencoder_2/sequential_4/reshape_9/Reshape/shape/2?
>variational_autoencoder_2/sequential_4/reshape_9/Reshape/shapePackGvariational_autoencoder_2/sequential_4/reshape_9/strided_slice:output:0Ivariational_autoencoder_2/sequential_4/reshape_9/Reshape/shape/1:output:0Ivariational_autoencoder_2/sequential_4/reshape_9/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2@
>variational_autoencoder_2/sequential_4/reshape_9/Reshape/shape?
8variational_autoencoder_2/sequential_4/reshape_9/ReshapeReshape?variational_autoencoder_2/sequential_4/dense_4/BiasAdd:output:0Gvariational_autoencoder_2/sequential_4/reshape_9/Reshape/shape:output:0*
T0*+
_output_shapes
:?????????2:
8variational_autoencoder_2/sequential_4/reshape_9/Reshape?
9variational_autoencoder_2/normal_sampling_layer_2/unstackUnpackAvariational_autoencoder_2/sequential_4/reshape_9/Reshape:output:0*
T0*:
_output_shapes(
&:?????????:?????????*

axis*	
num2;
9variational_autoencoder_2/normal_sampling_layer_2/unstack?
Dvariational_autoencoder_2/sequential_5/dense_5/MatMul/ReadVariableOpReadVariableOpMvariational_autoencoder_2_sequential_5_dense_5_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02F
Dvariational_autoencoder_2/sequential_5/dense_5/MatMul/ReadVariableOp?
5variational_autoencoder_2/sequential_5/dense_5/MatMulMatMulBvariational_autoencoder_2/normal_sampling_layer_2/unstack:output:0Lvariational_autoencoder_2/sequential_5/dense_5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????27
5variational_autoencoder_2/sequential_5/dense_5/MatMul?
Evariational_autoencoder_2/sequential_5/dense_5/BiasAdd/ReadVariableOpReadVariableOpNvariational_autoencoder_2_sequential_5_dense_5_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02G
Evariational_autoencoder_2/sequential_5/dense_5/BiasAdd/ReadVariableOp?
6variational_autoencoder_2/sequential_5/dense_5/BiasAddBiasAdd?variational_autoencoder_2/sequential_5/dense_5/MatMul:product:0Mvariational_autoencoder_2/sequential_5/dense_5/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????28
6variational_autoencoder_2/sequential_5/dense_5/BiasAdd?
3variational_autoencoder_2/sequential_5/dense_5/ReluRelu?variational_autoencoder_2/sequential_5/dense_5/BiasAdd:output:0*
T0*(
_output_shapes
:??????????25
3variational_autoencoder_2/sequential_5/dense_5/Relu?
7variational_autoencoder_2/sequential_5/reshape_10/ShapeShapeAvariational_autoencoder_2/sequential_5/dense_5/Relu:activations:0*
T0*
_output_shapes
:29
7variational_autoencoder_2/sequential_5/reshape_10/Shape?
Evariational_autoencoder_2/sequential_5/reshape_10/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2G
Evariational_autoencoder_2/sequential_5/reshape_10/strided_slice/stack?
Gvariational_autoencoder_2/sequential_5/reshape_10/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2I
Gvariational_autoencoder_2/sequential_5/reshape_10/strided_slice/stack_1?
Gvariational_autoencoder_2/sequential_5/reshape_10/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2I
Gvariational_autoencoder_2/sequential_5/reshape_10/strided_slice/stack_2?
?variational_autoencoder_2/sequential_5/reshape_10/strided_sliceStridedSlice@variational_autoencoder_2/sequential_5/reshape_10/Shape:output:0Nvariational_autoencoder_2/sequential_5/reshape_10/strided_slice/stack:output:0Pvariational_autoencoder_2/sequential_5/reshape_10/strided_slice/stack_1:output:0Pvariational_autoencoder_2/sequential_5/reshape_10/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2A
?variational_autoencoder_2/sequential_5/reshape_10/strided_slice?
Avariational_autoencoder_2/sequential_5/reshape_10/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2C
Avariational_autoencoder_2/sequential_5/reshape_10/Reshape/shape/1?
Avariational_autoencoder_2/sequential_5/reshape_10/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2C
Avariational_autoencoder_2/sequential_5/reshape_10/Reshape/shape/2?
Avariational_autoencoder_2/sequential_5/reshape_10/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B : 2C
Avariational_autoencoder_2/sequential_5/reshape_10/Reshape/shape/3?
?variational_autoencoder_2/sequential_5/reshape_10/Reshape/shapePackHvariational_autoencoder_2/sequential_5/reshape_10/strided_slice:output:0Jvariational_autoencoder_2/sequential_5/reshape_10/Reshape/shape/1:output:0Jvariational_autoencoder_2/sequential_5/reshape_10/Reshape/shape/2:output:0Jvariational_autoencoder_2/sequential_5/reshape_10/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2A
?variational_autoencoder_2/sequential_5/reshape_10/Reshape/shape?
9variational_autoencoder_2/sequential_5/reshape_10/ReshapeReshapeAvariational_autoencoder_2/sequential_5/dense_5/Relu:activations:0Hvariational_autoencoder_2/sequential_5/reshape_10/Reshape/shape:output:0*
T0*/
_output_shapes
:????????? 2;
9variational_autoencoder_2/sequential_5/reshape_10/Reshape?
?variational_autoencoder_2/sequential_5/conv2d_transpose_6/ShapeShapeBvariational_autoencoder_2/sequential_5/reshape_10/Reshape:output:0*
T0*
_output_shapes
:2A
?variational_autoencoder_2/sequential_5/conv2d_transpose_6/Shape?
Mvariational_autoencoder_2/sequential_5/conv2d_transpose_6/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2O
Mvariational_autoencoder_2/sequential_5/conv2d_transpose_6/strided_slice/stack?
Ovariational_autoencoder_2/sequential_5/conv2d_transpose_6/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2Q
Ovariational_autoencoder_2/sequential_5/conv2d_transpose_6/strided_slice/stack_1?
Ovariational_autoencoder_2/sequential_5/conv2d_transpose_6/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2Q
Ovariational_autoencoder_2/sequential_5/conv2d_transpose_6/strided_slice/stack_2?
Gvariational_autoencoder_2/sequential_5/conv2d_transpose_6/strided_sliceStridedSliceHvariational_autoencoder_2/sequential_5/conv2d_transpose_6/Shape:output:0Vvariational_autoencoder_2/sequential_5/conv2d_transpose_6/strided_slice/stack:output:0Xvariational_autoencoder_2/sequential_5/conv2d_transpose_6/strided_slice/stack_1:output:0Xvariational_autoencoder_2/sequential_5/conv2d_transpose_6/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2I
Gvariational_autoencoder_2/sequential_5/conv2d_transpose_6/strided_slice?
Avariational_autoencoder_2/sequential_5/conv2d_transpose_6/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2C
Avariational_autoencoder_2/sequential_5/conv2d_transpose_6/stack/1?
Avariational_autoencoder_2/sequential_5/conv2d_transpose_6/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2C
Avariational_autoencoder_2/sequential_5/conv2d_transpose_6/stack/2?
Avariational_autoencoder_2/sequential_5/conv2d_transpose_6/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@2C
Avariational_autoencoder_2/sequential_5/conv2d_transpose_6/stack/3?
?variational_autoencoder_2/sequential_5/conv2d_transpose_6/stackPackPvariational_autoencoder_2/sequential_5/conv2d_transpose_6/strided_slice:output:0Jvariational_autoencoder_2/sequential_5/conv2d_transpose_6/stack/1:output:0Jvariational_autoencoder_2/sequential_5/conv2d_transpose_6/stack/2:output:0Jvariational_autoencoder_2/sequential_5/conv2d_transpose_6/stack/3:output:0*
N*
T0*
_output_shapes
:2A
?variational_autoencoder_2/sequential_5/conv2d_transpose_6/stack?
Ovariational_autoencoder_2/sequential_5/conv2d_transpose_6/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2Q
Ovariational_autoencoder_2/sequential_5/conv2d_transpose_6/strided_slice_1/stack?
Qvariational_autoencoder_2/sequential_5/conv2d_transpose_6/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2S
Qvariational_autoencoder_2/sequential_5/conv2d_transpose_6/strided_slice_1/stack_1?
Qvariational_autoencoder_2/sequential_5/conv2d_transpose_6/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2S
Qvariational_autoencoder_2/sequential_5/conv2d_transpose_6/strided_slice_1/stack_2?
Ivariational_autoencoder_2/sequential_5/conv2d_transpose_6/strided_slice_1StridedSliceHvariational_autoencoder_2/sequential_5/conv2d_transpose_6/stack:output:0Xvariational_autoencoder_2/sequential_5/conv2d_transpose_6/strided_slice_1/stack:output:0Zvariational_autoencoder_2/sequential_5/conv2d_transpose_6/strided_slice_1/stack_1:output:0Zvariational_autoencoder_2/sequential_5/conv2d_transpose_6/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2K
Ivariational_autoencoder_2/sequential_5/conv2d_transpose_6/strided_slice_1?
Yvariational_autoencoder_2/sequential_5/conv2d_transpose_6/conv2d_transpose/ReadVariableOpReadVariableOpbvariational_autoencoder_2_sequential_5_conv2d_transpose_6_conv2d_transpose_readvariableop_resource*&
_output_shapes
:@ *
dtype02[
Yvariational_autoencoder_2/sequential_5/conv2d_transpose_6/conv2d_transpose/ReadVariableOp?
Jvariational_autoencoder_2/sequential_5/conv2d_transpose_6/conv2d_transposeConv2DBackpropInputHvariational_autoencoder_2/sequential_5/conv2d_transpose_6/stack:output:0avariational_autoencoder_2/sequential_5/conv2d_transpose_6/conv2d_transpose/ReadVariableOp:value:0Bvariational_autoencoder_2/sequential_5/reshape_10/Reshape:output:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2L
Jvariational_autoencoder_2/sequential_5/conv2d_transpose_6/conv2d_transpose?
Pvariational_autoencoder_2/sequential_5/conv2d_transpose_6/BiasAdd/ReadVariableOpReadVariableOpYvariational_autoencoder_2_sequential_5_conv2d_transpose_6_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02R
Pvariational_autoencoder_2/sequential_5/conv2d_transpose_6/BiasAdd/ReadVariableOp?
Avariational_autoencoder_2/sequential_5/conv2d_transpose_6/BiasAddBiasAddSvariational_autoencoder_2/sequential_5/conv2d_transpose_6/conv2d_transpose:output:0Xvariational_autoencoder_2/sequential_5/conv2d_transpose_6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2C
Avariational_autoencoder_2/sequential_5/conv2d_transpose_6/BiasAdd?
>variational_autoencoder_2/sequential_5/conv2d_transpose_6/ReluReluJvariational_autoencoder_2/sequential_5/conv2d_transpose_6/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2@
>variational_autoencoder_2/sequential_5/conv2d_transpose_6/Relu?
?variational_autoencoder_2/sequential_5/conv2d_transpose_7/ShapeShapeLvariational_autoencoder_2/sequential_5/conv2d_transpose_6/Relu:activations:0*
T0*
_output_shapes
:2A
?variational_autoencoder_2/sequential_5/conv2d_transpose_7/Shape?
Mvariational_autoencoder_2/sequential_5/conv2d_transpose_7/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2O
Mvariational_autoencoder_2/sequential_5/conv2d_transpose_7/strided_slice/stack?
Ovariational_autoencoder_2/sequential_5/conv2d_transpose_7/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2Q
Ovariational_autoencoder_2/sequential_5/conv2d_transpose_7/strided_slice/stack_1?
Ovariational_autoencoder_2/sequential_5/conv2d_transpose_7/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2Q
Ovariational_autoencoder_2/sequential_5/conv2d_transpose_7/strided_slice/stack_2?
Gvariational_autoencoder_2/sequential_5/conv2d_transpose_7/strided_sliceStridedSliceHvariational_autoencoder_2/sequential_5/conv2d_transpose_7/Shape:output:0Vvariational_autoencoder_2/sequential_5/conv2d_transpose_7/strided_slice/stack:output:0Xvariational_autoencoder_2/sequential_5/conv2d_transpose_7/strided_slice/stack_1:output:0Xvariational_autoencoder_2/sequential_5/conv2d_transpose_7/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2I
Gvariational_autoencoder_2/sequential_5/conv2d_transpose_7/strided_slice?
Avariational_autoencoder_2/sequential_5/conv2d_transpose_7/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2C
Avariational_autoencoder_2/sequential_5/conv2d_transpose_7/stack/1?
Avariational_autoencoder_2/sequential_5/conv2d_transpose_7/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2C
Avariational_autoencoder_2/sequential_5/conv2d_transpose_7/stack/2?
Avariational_autoencoder_2/sequential_5/conv2d_transpose_7/stack/3Const*
_output_shapes
: *
dtype0*
value	B : 2C
Avariational_autoencoder_2/sequential_5/conv2d_transpose_7/stack/3?
?variational_autoencoder_2/sequential_5/conv2d_transpose_7/stackPackPvariational_autoencoder_2/sequential_5/conv2d_transpose_7/strided_slice:output:0Jvariational_autoencoder_2/sequential_5/conv2d_transpose_7/stack/1:output:0Jvariational_autoencoder_2/sequential_5/conv2d_transpose_7/stack/2:output:0Jvariational_autoencoder_2/sequential_5/conv2d_transpose_7/stack/3:output:0*
N*
T0*
_output_shapes
:2A
?variational_autoencoder_2/sequential_5/conv2d_transpose_7/stack?
Ovariational_autoencoder_2/sequential_5/conv2d_transpose_7/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2Q
Ovariational_autoencoder_2/sequential_5/conv2d_transpose_7/strided_slice_1/stack?
Qvariational_autoencoder_2/sequential_5/conv2d_transpose_7/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2S
Qvariational_autoencoder_2/sequential_5/conv2d_transpose_7/strided_slice_1/stack_1?
Qvariational_autoencoder_2/sequential_5/conv2d_transpose_7/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2S
Qvariational_autoencoder_2/sequential_5/conv2d_transpose_7/strided_slice_1/stack_2?
Ivariational_autoencoder_2/sequential_5/conv2d_transpose_7/strided_slice_1StridedSliceHvariational_autoencoder_2/sequential_5/conv2d_transpose_7/stack:output:0Xvariational_autoencoder_2/sequential_5/conv2d_transpose_7/strided_slice_1/stack:output:0Zvariational_autoencoder_2/sequential_5/conv2d_transpose_7/strided_slice_1/stack_1:output:0Zvariational_autoencoder_2/sequential_5/conv2d_transpose_7/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2K
Ivariational_autoencoder_2/sequential_5/conv2d_transpose_7/strided_slice_1?
Yvariational_autoencoder_2/sequential_5/conv2d_transpose_7/conv2d_transpose/ReadVariableOpReadVariableOpbvariational_autoencoder_2_sequential_5_conv2d_transpose_7_conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype02[
Yvariational_autoencoder_2/sequential_5/conv2d_transpose_7/conv2d_transpose/ReadVariableOp?
Jvariational_autoencoder_2/sequential_5/conv2d_transpose_7/conv2d_transposeConv2DBackpropInputHvariational_autoencoder_2/sequential_5/conv2d_transpose_7/stack:output:0avariational_autoencoder_2/sequential_5/conv2d_transpose_7/conv2d_transpose/ReadVariableOp:value:0Lvariational_autoencoder_2/sequential_5/conv2d_transpose_6/Relu:activations:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
2L
Jvariational_autoencoder_2/sequential_5/conv2d_transpose_7/conv2d_transpose?
Pvariational_autoencoder_2/sequential_5/conv2d_transpose_7/BiasAdd/ReadVariableOpReadVariableOpYvariational_autoencoder_2_sequential_5_conv2d_transpose_7_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02R
Pvariational_autoencoder_2/sequential_5/conv2d_transpose_7/BiasAdd/ReadVariableOp?
Avariational_autoencoder_2/sequential_5/conv2d_transpose_7/BiasAddBiasAddSvariational_autoencoder_2/sequential_5/conv2d_transpose_7/conv2d_transpose:output:0Xvariational_autoencoder_2/sequential_5/conv2d_transpose_7/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2C
Avariational_autoencoder_2/sequential_5/conv2d_transpose_7/BiasAdd?
>variational_autoencoder_2/sequential_5/conv2d_transpose_7/ReluReluJvariational_autoencoder_2/sequential_5/conv2d_transpose_7/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2@
>variational_autoencoder_2/sequential_5/conv2d_transpose_7/Relu?
?variational_autoencoder_2/sequential_5/conv2d_transpose_8/ShapeShapeLvariational_autoencoder_2/sequential_5/conv2d_transpose_7/Relu:activations:0*
T0*
_output_shapes
:2A
?variational_autoencoder_2/sequential_5/conv2d_transpose_8/Shape?
Mvariational_autoencoder_2/sequential_5/conv2d_transpose_8/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2O
Mvariational_autoencoder_2/sequential_5/conv2d_transpose_8/strided_slice/stack?
Ovariational_autoencoder_2/sequential_5/conv2d_transpose_8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2Q
Ovariational_autoencoder_2/sequential_5/conv2d_transpose_8/strided_slice/stack_1?
Ovariational_autoencoder_2/sequential_5/conv2d_transpose_8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2Q
Ovariational_autoencoder_2/sequential_5/conv2d_transpose_8/strided_slice/stack_2?
Gvariational_autoencoder_2/sequential_5/conv2d_transpose_8/strided_sliceStridedSliceHvariational_autoencoder_2/sequential_5/conv2d_transpose_8/Shape:output:0Vvariational_autoencoder_2/sequential_5/conv2d_transpose_8/strided_slice/stack:output:0Xvariational_autoencoder_2/sequential_5/conv2d_transpose_8/strided_slice/stack_1:output:0Xvariational_autoencoder_2/sequential_5/conv2d_transpose_8/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2I
Gvariational_autoencoder_2/sequential_5/conv2d_transpose_8/strided_slice?
Avariational_autoencoder_2/sequential_5/conv2d_transpose_8/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2C
Avariational_autoencoder_2/sequential_5/conv2d_transpose_8/stack/1?
Avariational_autoencoder_2/sequential_5/conv2d_transpose_8/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2C
Avariational_autoencoder_2/sequential_5/conv2d_transpose_8/stack/2?
Avariational_autoencoder_2/sequential_5/conv2d_transpose_8/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2C
Avariational_autoencoder_2/sequential_5/conv2d_transpose_8/stack/3?
?variational_autoencoder_2/sequential_5/conv2d_transpose_8/stackPackPvariational_autoencoder_2/sequential_5/conv2d_transpose_8/strided_slice:output:0Jvariational_autoencoder_2/sequential_5/conv2d_transpose_8/stack/1:output:0Jvariational_autoencoder_2/sequential_5/conv2d_transpose_8/stack/2:output:0Jvariational_autoencoder_2/sequential_5/conv2d_transpose_8/stack/3:output:0*
N*
T0*
_output_shapes
:2A
?variational_autoencoder_2/sequential_5/conv2d_transpose_8/stack?
Ovariational_autoencoder_2/sequential_5/conv2d_transpose_8/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2Q
Ovariational_autoencoder_2/sequential_5/conv2d_transpose_8/strided_slice_1/stack?
Qvariational_autoencoder_2/sequential_5/conv2d_transpose_8/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2S
Qvariational_autoencoder_2/sequential_5/conv2d_transpose_8/strided_slice_1/stack_1?
Qvariational_autoencoder_2/sequential_5/conv2d_transpose_8/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2S
Qvariational_autoencoder_2/sequential_5/conv2d_transpose_8/strided_slice_1/stack_2?
Ivariational_autoencoder_2/sequential_5/conv2d_transpose_8/strided_slice_1StridedSliceHvariational_autoencoder_2/sequential_5/conv2d_transpose_8/stack:output:0Xvariational_autoencoder_2/sequential_5/conv2d_transpose_8/strided_slice_1/stack:output:0Zvariational_autoencoder_2/sequential_5/conv2d_transpose_8/strided_slice_1/stack_1:output:0Zvariational_autoencoder_2/sequential_5/conv2d_transpose_8/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2K
Ivariational_autoencoder_2/sequential_5/conv2d_transpose_8/strided_slice_1?
Yvariational_autoencoder_2/sequential_5/conv2d_transpose_8/conv2d_transpose/ReadVariableOpReadVariableOpbvariational_autoencoder_2_sequential_5_conv2d_transpose_8_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype02[
Yvariational_autoencoder_2/sequential_5/conv2d_transpose_8/conv2d_transpose/ReadVariableOp?
Jvariational_autoencoder_2/sequential_5/conv2d_transpose_8/conv2d_transposeConv2DBackpropInputHvariational_autoencoder_2/sequential_5/conv2d_transpose_8/stack:output:0avariational_autoencoder_2/sequential_5/conv2d_transpose_8/conv2d_transpose/ReadVariableOp:value:0Lvariational_autoencoder_2/sequential_5/conv2d_transpose_7/Relu:activations:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2L
Jvariational_autoencoder_2/sequential_5/conv2d_transpose_8/conv2d_transpose?
Pvariational_autoencoder_2/sequential_5/conv2d_transpose_8/BiasAdd/ReadVariableOpReadVariableOpYvariational_autoencoder_2_sequential_5_conv2d_transpose_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02R
Pvariational_autoencoder_2/sequential_5/conv2d_transpose_8/BiasAdd/ReadVariableOp?
Avariational_autoencoder_2/sequential_5/conv2d_transpose_8/BiasAddBiasAddSvariational_autoencoder_2/sequential_5/conv2d_transpose_8/conv2d_transpose:output:0Xvariational_autoencoder_2/sequential_5/conv2d_transpose_8/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2C
Avariational_autoencoder_2/sequential_5/conv2d_transpose_8/BiasAdd?
Avariational_autoencoder_2/sequential_5/conv2d_transpose_8/SigmoidSigmoidJvariational_autoencoder_2/sequential_5/conv2d_transpose_8/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2C
Avariational_autoencoder_2/sequential_5/conv2d_transpose_8/Sigmoid?
7variational_autoencoder_2/sequential_5/reshape_11/ShapeShapeEvariational_autoencoder_2/sequential_5/conv2d_transpose_8/Sigmoid:y:0*
T0*
_output_shapes
:29
7variational_autoencoder_2/sequential_5/reshape_11/Shape?
Evariational_autoencoder_2/sequential_5/reshape_11/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2G
Evariational_autoencoder_2/sequential_5/reshape_11/strided_slice/stack?
Gvariational_autoencoder_2/sequential_5/reshape_11/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2I
Gvariational_autoencoder_2/sequential_5/reshape_11/strided_slice/stack_1?
Gvariational_autoencoder_2/sequential_5/reshape_11/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2I
Gvariational_autoencoder_2/sequential_5/reshape_11/strided_slice/stack_2?
?variational_autoencoder_2/sequential_5/reshape_11/strided_sliceStridedSlice@variational_autoencoder_2/sequential_5/reshape_11/Shape:output:0Nvariational_autoencoder_2/sequential_5/reshape_11/strided_slice/stack:output:0Pvariational_autoencoder_2/sequential_5/reshape_11/strided_slice/stack_1:output:0Pvariational_autoencoder_2/sequential_5/reshape_11/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2A
?variational_autoencoder_2/sequential_5/reshape_11/strided_slice?
Avariational_autoencoder_2/sequential_5/reshape_11/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2C
Avariational_autoencoder_2/sequential_5/reshape_11/Reshape/shape/1?
Avariational_autoencoder_2/sequential_5/reshape_11/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2C
Avariational_autoencoder_2/sequential_5/reshape_11/Reshape/shape/2?
Avariational_autoencoder_2/sequential_5/reshape_11/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2C
Avariational_autoencoder_2/sequential_5/reshape_11/Reshape/shape/3?
?variational_autoencoder_2/sequential_5/reshape_11/Reshape/shapePackHvariational_autoencoder_2/sequential_5/reshape_11/strided_slice:output:0Jvariational_autoencoder_2/sequential_5/reshape_11/Reshape/shape/1:output:0Jvariational_autoencoder_2/sequential_5/reshape_11/Reshape/shape/2:output:0Jvariational_autoencoder_2/sequential_5/reshape_11/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2A
?variational_autoencoder_2/sequential_5/reshape_11/Reshape/shape?
9variational_autoencoder_2/sequential_5/reshape_11/ReshapeReshapeEvariational_autoencoder_2/sequential_5/conv2d_transpose_8/Sigmoid:y:0Hvariational_autoencoder_2/sequential_5/reshape_11/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????2;
9variational_autoencoder_2/sequential_5/reshape_11/Reshape?
IdentityIdentityBvariational_autoencoder_2/sequential_5/reshape_11/Reshape:output:0^NoOp*
T0*/
_output_shapes
:?????????2

Identity?	
NoOpNoOpG^variational_autoencoder_2/sequential_4/conv2d_4/BiasAdd/ReadVariableOpF^variational_autoencoder_2/sequential_4/conv2d_4/Conv2D/ReadVariableOpG^variational_autoencoder_2/sequential_4/conv2d_5/BiasAdd/ReadVariableOpF^variational_autoencoder_2/sequential_4/conv2d_5/Conv2D/ReadVariableOpF^variational_autoencoder_2/sequential_4/dense_4/BiasAdd/ReadVariableOpE^variational_autoencoder_2/sequential_4/dense_4/MatMul/ReadVariableOpQ^variational_autoencoder_2/sequential_5/conv2d_transpose_6/BiasAdd/ReadVariableOpZ^variational_autoencoder_2/sequential_5/conv2d_transpose_6/conv2d_transpose/ReadVariableOpQ^variational_autoencoder_2/sequential_5/conv2d_transpose_7/BiasAdd/ReadVariableOpZ^variational_autoencoder_2/sequential_5/conv2d_transpose_7/conv2d_transpose/ReadVariableOpQ^variational_autoencoder_2/sequential_5/conv2d_transpose_8/BiasAdd/ReadVariableOpZ^variational_autoencoder_2/sequential_5/conv2d_transpose_8/conv2d_transpose/ReadVariableOpF^variational_autoencoder_2/sequential_5/dense_5/BiasAdd/ReadVariableOpE^variational_autoencoder_2/sequential_5/dense_5/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:?????????: : : : : : : : : : : : : : 2?
Fvariational_autoencoder_2/sequential_4/conv2d_4/BiasAdd/ReadVariableOpFvariational_autoencoder_2/sequential_4/conv2d_4/BiasAdd/ReadVariableOp2?
Evariational_autoencoder_2/sequential_4/conv2d_4/Conv2D/ReadVariableOpEvariational_autoencoder_2/sequential_4/conv2d_4/Conv2D/ReadVariableOp2?
Fvariational_autoencoder_2/sequential_4/conv2d_5/BiasAdd/ReadVariableOpFvariational_autoencoder_2/sequential_4/conv2d_5/BiasAdd/ReadVariableOp2?
Evariational_autoencoder_2/sequential_4/conv2d_5/Conv2D/ReadVariableOpEvariational_autoencoder_2/sequential_4/conv2d_5/Conv2D/ReadVariableOp2?
Evariational_autoencoder_2/sequential_4/dense_4/BiasAdd/ReadVariableOpEvariational_autoencoder_2/sequential_4/dense_4/BiasAdd/ReadVariableOp2?
Dvariational_autoencoder_2/sequential_4/dense_4/MatMul/ReadVariableOpDvariational_autoencoder_2/sequential_4/dense_4/MatMul/ReadVariableOp2?
Pvariational_autoencoder_2/sequential_5/conv2d_transpose_6/BiasAdd/ReadVariableOpPvariational_autoencoder_2/sequential_5/conv2d_transpose_6/BiasAdd/ReadVariableOp2?
Yvariational_autoencoder_2/sequential_5/conv2d_transpose_6/conv2d_transpose/ReadVariableOpYvariational_autoencoder_2/sequential_5/conv2d_transpose_6/conv2d_transpose/ReadVariableOp2?
Pvariational_autoencoder_2/sequential_5/conv2d_transpose_7/BiasAdd/ReadVariableOpPvariational_autoencoder_2/sequential_5/conv2d_transpose_7/BiasAdd/ReadVariableOp2?
Yvariational_autoencoder_2/sequential_5/conv2d_transpose_7/conv2d_transpose/ReadVariableOpYvariational_autoencoder_2/sequential_5/conv2d_transpose_7/conv2d_transpose/ReadVariableOp2?
Pvariational_autoencoder_2/sequential_5/conv2d_transpose_8/BiasAdd/ReadVariableOpPvariational_autoencoder_2/sequential_5/conv2d_transpose_8/BiasAdd/ReadVariableOp2?
Yvariational_autoencoder_2/sequential_5/conv2d_transpose_8/conv2d_transpose/ReadVariableOpYvariational_autoencoder_2/sequential_5/conv2d_transpose_8/conv2d_transpose/ReadVariableOp2?
Evariational_autoencoder_2/sequential_5/dense_5/BiasAdd/ReadVariableOpEvariational_autoencoder_2/sequential_5/dense_5/BiasAdd/ReadVariableOp2?
Dvariational_autoencoder_2/sequential_5/dense_5/MatMul/ReadVariableOpDvariational_autoencoder_2/sequential_5/dense_5/MatMul/ReadVariableOp:X T
/
_output_shapes
:?????????
!
_user_specified_name	input_1
?
?
#__inference_signature_wrapper_46212
input_1!
unknown: 
	unknown_0: #
	unknown_1: @
	unknown_2:@
	unknown_3:	?
	unknown_4:
	unknown_5:	?
	unknown_6:	?#
	unknown_7:@ 
	unknown_8:@#
	unknown_9: @

unknown_10: $

unknown_11: 

unknown_12:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *)
f$R"
 __inference__wrapped_model_449582
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:?????????: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:?????????
!
_user_specified_name	input_1
?!
?
G__inference_sequential_5_layer_call_and_return_conditional_losses_45852
input_10 
dense_5_45829:	?
dense_5_45831:	?2
conv2d_transpose_6_45835:@ &
conv2d_transpose_6_45837:@2
conv2d_transpose_7_45840: @&
conv2d_transpose_7_45842: 2
conv2d_transpose_8_45845: &
conv2d_transpose_8_45847:
identity??*conv2d_transpose_6/StatefulPartitionedCall?*conv2d_transpose_7/StatefulPartitionedCall?*conv2d_transpose_8/StatefulPartitionedCall?dense_5/StatefulPartitionedCall?
dense_5/StatefulPartitionedCallStatefulPartitionedCallinput_10dense_5_45829dense_5_45831*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dense_5_layer_call_and_return_conditional_losses_455172!
dense_5/StatefulPartitionedCall?
reshape_10/PartitionedCallPartitionedCall(dense_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_reshape_10_layer_call_and_return_conditional_losses_455372
reshape_10/PartitionedCall?
*conv2d_transpose_6/StatefulPartitionedCallStatefulPartitionedCall#reshape_10/PartitionedCall:output:0conv2d_transpose_6_45835conv2d_transpose_6_45837*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *V
fQRO
M__inference_conv2d_transpose_6_layer_call_and_return_conditional_losses_455622,
*conv2d_transpose_6/StatefulPartitionedCall?
*conv2d_transpose_7/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_6/StatefulPartitionedCall:output:0conv2d_transpose_7_45840conv2d_transpose_7_45842*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *V
fQRO
M__inference_conv2d_transpose_7_layer_call_and_return_conditional_losses_455912,
*conv2d_transpose_7/StatefulPartitionedCall?
*conv2d_transpose_8/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_7/StatefulPartitionedCall:output:0conv2d_transpose_8_45845conv2d_transpose_8_45847*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *V
fQRO
M__inference_conv2d_transpose_8_layer_call_and_return_conditional_losses_456202,
*conv2d_transpose_8/StatefulPartitionedCall?
reshape_11/PartitionedCallPartitionedCall3conv2d_transpose_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_reshape_11_layer_call_and_return_conditional_losses_456402
reshape_11/PartitionedCall?
IdentityIdentity#reshape_11/PartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????2

Identity?
NoOpNoOp+^conv2d_transpose_6/StatefulPartitionedCall+^conv2d_transpose_7/StatefulPartitionedCall+^conv2d_transpose_8/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : : : : : 2X
*conv2d_transpose_6/StatefulPartitionedCall*conv2d_transpose_6/StatefulPartitionedCall2X
*conv2d_transpose_7/StatefulPartitionedCall*conv2d_transpose_7/StatefulPartitionedCall2X
*conv2d_transpose_8/StatefulPartitionedCall*conv2d_transpose_8/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall:Q M
'
_output_shapes
:?????????
"
_user_specified_name
input_10
?
?
T__inference_variational_autoencoder_2_layer_call_and_return_conditional_losses_45898

inputs,
sequential_4_45859:  
sequential_4_45861: ,
sequential_4_45863: @ 
sequential_4_45865:@%
sequential_4_45867:	? 
sequential_4_45869:%
sequential_5_45880:	?!
sequential_5_45882:	?,
sequential_5_45884:@  
sequential_5_45886:@,
sequential_5_45888: @ 
sequential_5_45890: ,
sequential_5_45892:  
sequential_5_45894:
identity??$sequential_4/StatefulPartitionedCall?$sequential_5/StatefulPartitionedCall?
$sequential_4/StatefulPartitionedCallStatefulPartitionedCallinputssequential_4_45859sequential_4_45861sequential_4_45863sequential_4_45865sequential_4_45867sequential_4_45869*
Tin
	2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_sequential_4_layer_call_and_return_conditional_losses_450552&
$sequential_4/StatefulPartitionedCall?
'normal_sampling_layer_2/PartitionedCallPartitionedCall-sequential_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_normal_sampling_layer_2_layer_call_and_return_conditional_losses_458782)
'normal_sampling_layer_2/PartitionedCall?
$sequential_5/StatefulPartitionedCallStatefulPartitionedCall0normal_sampling_layer_2/PartitionedCall:output:0sequential_5_45880sequential_5_45882sequential_5_45884sequential_5_45886sequential_5_45888sequential_5_45890sequential_5_45892sequential_5_45894*
Tin
2	*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_sequential_5_layer_call_and_return_conditional_losses_456432&
$sequential_5/StatefulPartitionedCall?
IdentityIdentity-sequential_5/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????2

Identity?
NoOpNoOp%^sequential_4/StatefulPartitionedCall%^sequential_5/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:?????????: : : : : : : : : : : : : : 2L
$sequential_4/StatefulPartitionedCall$sequential_4/StatefulPartitionedCall2L
$sequential_5/StatefulPartitionedCall$sequential_5/StatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
9__inference_variational_autoencoder_2_layer_call_fn_46567

inputs!
unknown: 
	unknown_0: #
	unknown_1: @
	unknown_2:@
	unknown_3:	?
	unknown_4:
	unknown_5:	?
	unknown_6:	?#
	unknown_7:@ 
	unknown_8:@#
	unknown_9: @

unknown_10: $

unknown_11: 

unknown_12:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *]
fXRV
T__inference_variational_autoencoder_2_layer_call_and_return_conditional_losses_460372
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:?????????: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
G__inference_sequential_4_layer_call_and_return_conditional_losses_45159

inputs(
conv2d_4_45141: 
conv2d_4_45143: (
conv2d_5_45146: @
conv2d_5_45148:@ 
dense_4_45152:	?
dense_4_45154:
identity?? conv2d_4/StatefulPartitionedCall? conv2d_5/StatefulPartitionedCall?dense_4/StatefulPartitionedCall?
reshape_8/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_reshape_8_layer_call_and_return_conditional_losses_449792
reshape_8/PartitionedCall?
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall"reshape_8/PartitionedCall:output:0conv2d_4_45141conv2d_4_45143*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_conv2d_4_layer_call_and_return_conditional_losses_449922"
 conv2d_4/StatefulPartitionedCall?
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0conv2d_5_45146conv2d_5_45148*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_conv2d_5_layer_call_and_return_conditional_losses_450092"
 conv2d_5/StatefulPartitionedCall?
flatten_2/PartitionedCallPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_flatten_2_layer_call_and_return_conditional_losses_450212
flatten_2/PartitionedCall?
dense_4/StatefulPartitionedCallStatefulPartitionedCall"flatten_2/PartitionedCall:output:0dense_4_45152dense_4_45154*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dense_4_layer_call_and_return_conditional_losses_450332!
dense_4/StatefulPartitionedCall?
reshape_9/PartitionedCallPartitionedCall(dense_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_reshape_9_layer_call_and_return_conditional_losses_450522
reshape_9/PartitionedCall?
IdentityIdentity"reshape_9/PartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????2

Identity?
NoOpNoOp!^conv2d_4/StatefulPartitionedCall!^conv2d_5/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : 2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
G__inference_sequential_4_layer_call_and_return_conditional_losses_45235
input_9(
conv2d_4_45217: 
conv2d_4_45219: (
conv2d_5_45222: @
conv2d_5_45224:@ 
dense_4_45228:	?
dense_4_45230:
identity?? conv2d_4/StatefulPartitionedCall? conv2d_5/StatefulPartitionedCall?dense_4/StatefulPartitionedCall?
reshape_8/PartitionedCallPartitionedCallinput_9*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_reshape_8_layer_call_and_return_conditional_losses_449792
reshape_8/PartitionedCall?
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall"reshape_8/PartitionedCall:output:0conv2d_4_45217conv2d_4_45219*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_conv2d_4_layer_call_and_return_conditional_losses_449922"
 conv2d_4/StatefulPartitionedCall?
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0conv2d_5_45222conv2d_5_45224*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_conv2d_5_layer_call_and_return_conditional_losses_450092"
 conv2d_5/StatefulPartitionedCall?
flatten_2/PartitionedCallPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_flatten_2_layer_call_and_return_conditional_losses_450212
flatten_2/PartitionedCall?
dense_4/StatefulPartitionedCallStatefulPartitionedCall"flatten_2/PartitionedCall:output:0dense_4_45228dense_4_45230*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dense_4_layer_call_and_return_conditional_losses_450332!
dense_4/StatefulPartitionedCall?
reshape_9/PartitionedCallPartitionedCall(dense_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_reshape_9_layer_call_and_return_conditional_losses_450522
reshape_9/PartitionedCall?
IdentityIdentity"reshape_9/PartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????2

Identity?
NoOpNoOp!^conv2d_4/StatefulPartitionedCall!^conv2d_5/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : 2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall:X T
/
_output_shapes
:?????????
!
_user_specified_name	input_9
?!
?
G__inference_sequential_5_layer_call_and_return_conditional_losses_45643

inputs 
dense_5_45518:	?
dense_5_45520:	?2
conv2d_transpose_6_45563:@ &
conv2d_transpose_6_45565:@2
conv2d_transpose_7_45592: @&
conv2d_transpose_7_45594: 2
conv2d_transpose_8_45621: &
conv2d_transpose_8_45623:
identity??*conv2d_transpose_6/StatefulPartitionedCall?*conv2d_transpose_7/StatefulPartitionedCall?*conv2d_transpose_8/StatefulPartitionedCall?dense_5/StatefulPartitionedCall?
dense_5/StatefulPartitionedCallStatefulPartitionedCallinputsdense_5_45518dense_5_45520*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dense_5_layer_call_and_return_conditional_losses_455172!
dense_5/StatefulPartitionedCall?
reshape_10/PartitionedCallPartitionedCall(dense_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_reshape_10_layer_call_and_return_conditional_losses_455372
reshape_10/PartitionedCall?
*conv2d_transpose_6/StatefulPartitionedCallStatefulPartitionedCall#reshape_10/PartitionedCall:output:0conv2d_transpose_6_45563conv2d_transpose_6_45565*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *V
fQRO
M__inference_conv2d_transpose_6_layer_call_and_return_conditional_losses_455622,
*conv2d_transpose_6/StatefulPartitionedCall?
*conv2d_transpose_7/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_6/StatefulPartitionedCall:output:0conv2d_transpose_7_45592conv2d_transpose_7_45594*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *V
fQRO
M__inference_conv2d_transpose_7_layer_call_and_return_conditional_losses_455912,
*conv2d_transpose_7/StatefulPartitionedCall?
*conv2d_transpose_8/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_7/StatefulPartitionedCall:output:0conv2d_transpose_8_45621conv2d_transpose_8_45623*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *V
fQRO
M__inference_conv2d_transpose_8_layer_call_and_return_conditional_losses_456202,
*conv2d_transpose_8/StatefulPartitionedCall?
reshape_11/PartitionedCallPartitionedCall3conv2d_transpose_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_reshape_11_layer_call_and_return_conditional_losses_456402
reshape_11/PartitionedCall?
IdentityIdentity#reshape_11/PartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????2

Identity?
NoOpNoOp+^conv2d_transpose_6/StatefulPartitionedCall+^conv2d_transpose_7/StatefulPartitionedCall+^conv2d_transpose_8/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : : : : : 2X
*conv2d_transpose_6/StatefulPartitionedCall*conv2d_transpose_6/StatefulPartitionedCall2X
*conv2d_transpose_7/StatefulPartitionedCall*conv2d_transpose_7/StatefulPartitionedCall2X
*conv2d_transpose_8/StatefulPartitionedCall*conv2d_transpose_8/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
,__inference_sequential_4_layer_call_fn_45191
input_9!
unknown: 
	unknown_0: #
	unknown_1: @
	unknown_2:@
	unknown_3:	?
	unknown_4:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_9unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_sequential_4_layer_call_and_return_conditional_losses_451592
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:?????????
!
_user_specified_name	input_9
?
`
D__inference_reshape_9_layer_call_and_return_conditional_losses_45052

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliced
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/1d
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/2?
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapes
ReshapeReshapeinputsReshape/shape:output:0*
T0*+
_output_shapes
:?????????2	
Reshapeh
IdentityIdentityReshape:output:0*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
,__inference_sequential_4_layer_call_fn_46691

inputs!
unknown: 
	unknown_0: #
	unknown_1: @
	unknown_2:@
	unknown_3:	?
	unknown_4:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_sequential_4_layer_call_and_return_conditional_losses_451592
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
'__inference_dense_4_layer_call_fn_47047

inputs
unknown:	?
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dense_4_layer_call_and_return_conditional_losses_450332
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
E
)__inference_flatten_2_layer_call_fn_47028

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_flatten_2_layer_call_and_return_conditional_losses_450212
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
q
R__inference_normal_sampling_layer_2_layer_call_and_return_conditional_losses_46948

inputs
identity??
unstackUnpackinputs*
T0*:
_output_shapes(
&:?????????:?????????*

axis*	
num2	
unstackN
ShapeShapeunstack:output:0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliceR
Shape_1Shapeunstack:output:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
random_normal/shapePackstrided_slice:output:0strided_slice_1:output:0*
N*
T0*
_output_shapes
:2
random_normal/shapem
random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2
random_normal/meanq
random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
random_normal/stddev?
"random_normal/RandomStandardNormalRandomStandardNormalrandom_normal/shape:output:0*
T0*0
_output_shapes
:??????????????????*
dtype0*
seed???)*
seed2??22$
"random_normal/RandomStandardNormal?
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*0
_output_shapes
:??????????????????2
random_normal/mul?
random_normalAddV2random_normal/mul:z:0random_normal/mean:output:0*
T0*0
_output_shapes
:??????????????????2
random_normalS
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
mul/xe
mulMulmul/x:output:0unstack:output:1*
T0*'
_output_shapes
:?????????2
mulL
ExpExpmul:z:0*
T0*'
_output_shapes
:?????????2
Expc
mul_1MulExp:y:0random_normal:z:0*
T0*'
_output_shapes
:?????????2
mul_1b
addAddV2unstack:output:0	mul_1:z:0*
T0*'
_output_shapes
:?????????2
add[
IdentityIdentityadd:z:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
(__inference_conv2d_4_layer_call_fn_46997

inputs!
unknown: 
	unknown_0: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_conv2d_4_layer_call_and_return_conditional_losses_449922
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:????????? 2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
,__inference_sequential_4_layer_call_fn_46674

inputs!
unknown: 
	unknown_0: #
	unknown_1: @
	unknown_2:@
	unknown_3:	?
	unknown_4:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_sequential_4_layer_call_and_return_conditional_losses_450552
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
??
?
G__inference_sequential_5_layer_call_and_return_conditional_losses_46873

inputs9
&dense_5_matmul_readvariableop_resource:	?6
'dense_5_biasadd_readvariableop_resource:	?U
;conv2d_transpose_6_conv2d_transpose_readvariableop_resource:@ @
2conv2d_transpose_6_biasadd_readvariableop_resource:@U
;conv2d_transpose_7_conv2d_transpose_readvariableop_resource: @@
2conv2d_transpose_7_biasadd_readvariableop_resource: U
;conv2d_transpose_8_conv2d_transpose_readvariableop_resource: @
2conv2d_transpose_8_biasadd_readvariableop_resource:
identity??)conv2d_transpose_6/BiasAdd/ReadVariableOp?2conv2d_transpose_6/conv2d_transpose/ReadVariableOp?)conv2d_transpose_7/BiasAdd/ReadVariableOp?2conv2d_transpose_7/conv2d_transpose/ReadVariableOp?)conv2d_transpose_8/BiasAdd/ReadVariableOp?2conv2d_transpose_8/conv2d_transpose/ReadVariableOp?dense_5/BiasAdd/ReadVariableOp?dense_5/MatMul/ReadVariableOp?
dense_5/MatMul/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
dense_5/MatMul/ReadVariableOp?
dense_5/MatMulMatMulinputs%dense_5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_5/MatMul?
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
dense_5/BiasAdd/ReadVariableOp?
dense_5/BiasAddBiasAdddense_5/MatMul:product:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_5/BiasAddq
dense_5/ReluReludense_5/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_5/Relun
reshape_10/ShapeShapedense_5/Relu:activations:0*
T0*
_output_shapes
:2
reshape_10/Shape?
reshape_10/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
reshape_10/strided_slice/stack?
 reshape_10/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_10/strided_slice/stack_1?
 reshape_10/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_10/strided_slice/stack_2?
reshape_10/strided_sliceStridedSlicereshape_10/Shape:output:0'reshape_10/strided_slice/stack:output:0)reshape_10/strided_slice/stack_1:output:0)reshape_10/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_10/strided_slicez
reshape_10/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_10/Reshape/shape/1z
reshape_10/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_10/Reshape/shape/2z
reshape_10/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B : 2
reshape_10/Reshape/shape/3?
reshape_10/Reshape/shapePack!reshape_10/strided_slice:output:0#reshape_10/Reshape/shape/1:output:0#reshape_10/Reshape/shape/2:output:0#reshape_10/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
reshape_10/Reshape/shape?
reshape_10/ReshapeReshapedense_5/Relu:activations:0!reshape_10/Reshape/shape:output:0*
T0*/
_output_shapes
:????????? 2
reshape_10/Reshape
conv2d_transpose_6/ShapeShapereshape_10/Reshape:output:0*
T0*
_output_shapes
:2
conv2d_transpose_6/Shape?
&conv2d_transpose_6/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv2d_transpose_6/strided_slice/stack?
(conv2d_transpose_6/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_6/strided_slice/stack_1?
(conv2d_transpose_6/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_6/strided_slice/stack_2?
 conv2d_transpose_6/strided_sliceStridedSlice!conv2d_transpose_6/Shape:output:0/conv2d_transpose_6/strided_slice/stack:output:01conv2d_transpose_6/strided_slice/stack_1:output:01conv2d_transpose_6/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv2d_transpose_6/strided_slicez
conv2d_transpose_6/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_6/stack/1z
conv2d_transpose_6/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_6/stack/2z
conv2d_transpose_6/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@2
conv2d_transpose_6/stack/3?
conv2d_transpose_6/stackPack)conv2d_transpose_6/strided_slice:output:0#conv2d_transpose_6/stack/1:output:0#conv2d_transpose_6/stack/2:output:0#conv2d_transpose_6/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_6/stack?
(conv2d_transpose_6/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(conv2d_transpose_6/strided_slice_1/stack?
*conv2d_transpose_6/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_6/strided_slice_1/stack_1?
*conv2d_transpose_6/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_6/strided_slice_1/stack_2?
"conv2d_transpose_6/strided_slice_1StridedSlice!conv2d_transpose_6/stack:output:01conv2d_transpose_6/strided_slice_1/stack:output:03conv2d_transpose_6/strided_slice_1/stack_1:output:03conv2d_transpose_6/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv2d_transpose_6/strided_slice_1?
2conv2d_transpose_6/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_6_conv2d_transpose_readvariableop_resource*&
_output_shapes
:@ *
dtype024
2conv2d_transpose_6/conv2d_transpose/ReadVariableOp?
#conv2d_transpose_6/conv2d_transposeConv2DBackpropInput!conv2d_transpose_6/stack:output:0:conv2d_transpose_6/conv2d_transpose/ReadVariableOp:value:0reshape_10/Reshape:output:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2%
#conv2d_transpose_6/conv2d_transpose?
)conv2d_transpose_6/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_6_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02+
)conv2d_transpose_6/BiasAdd/ReadVariableOp?
conv2d_transpose_6/BiasAddBiasAdd,conv2d_transpose_6/conv2d_transpose:output:01conv2d_transpose_6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
conv2d_transpose_6/BiasAdd?
conv2d_transpose_6/ReluRelu#conv2d_transpose_6/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
conv2d_transpose_6/Relu?
conv2d_transpose_7/ShapeShape%conv2d_transpose_6/Relu:activations:0*
T0*
_output_shapes
:2
conv2d_transpose_7/Shape?
&conv2d_transpose_7/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv2d_transpose_7/strided_slice/stack?
(conv2d_transpose_7/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_7/strided_slice/stack_1?
(conv2d_transpose_7/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_7/strided_slice/stack_2?
 conv2d_transpose_7/strided_sliceStridedSlice!conv2d_transpose_7/Shape:output:0/conv2d_transpose_7/strided_slice/stack:output:01conv2d_transpose_7/strided_slice/stack_1:output:01conv2d_transpose_7/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv2d_transpose_7/strided_slicez
conv2d_transpose_7/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_7/stack/1z
conv2d_transpose_7/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_7/stack/2z
conv2d_transpose_7/stack/3Const*
_output_shapes
: *
dtype0*
value	B : 2
conv2d_transpose_7/stack/3?
conv2d_transpose_7/stackPack)conv2d_transpose_7/strided_slice:output:0#conv2d_transpose_7/stack/1:output:0#conv2d_transpose_7/stack/2:output:0#conv2d_transpose_7/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_7/stack?
(conv2d_transpose_7/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(conv2d_transpose_7/strided_slice_1/stack?
*conv2d_transpose_7/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_7/strided_slice_1/stack_1?
*conv2d_transpose_7/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_7/strided_slice_1/stack_2?
"conv2d_transpose_7/strided_slice_1StridedSlice!conv2d_transpose_7/stack:output:01conv2d_transpose_7/strided_slice_1/stack:output:03conv2d_transpose_7/strided_slice_1/stack_1:output:03conv2d_transpose_7/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv2d_transpose_7/strided_slice_1?
2conv2d_transpose_7/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_7_conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype024
2conv2d_transpose_7/conv2d_transpose/ReadVariableOp?
#conv2d_transpose_7/conv2d_transposeConv2DBackpropInput!conv2d_transpose_7/stack:output:0:conv2d_transpose_7/conv2d_transpose/ReadVariableOp:value:0%conv2d_transpose_6/Relu:activations:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
2%
#conv2d_transpose_7/conv2d_transpose?
)conv2d_transpose_7/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_7_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02+
)conv2d_transpose_7/BiasAdd/ReadVariableOp?
conv2d_transpose_7/BiasAddBiasAdd,conv2d_transpose_7/conv2d_transpose:output:01conv2d_transpose_7/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2
conv2d_transpose_7/BiasAdd?
conv2d_transpose_7/ReluRelu#conv2d_transpose_7/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
conv2d_transpose_7/Relu?
conv2d_transpose_8/ShapeShape%conv2d_transpose_7/Relu:activations:0*
T0*
_output_shapes
:2
conv2d_transpose_8/Shape?
&conv2d_transpose_8/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv2d_transpose_8/strided_slice/stack?
(conv2d_transpose_8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_8/strided_slice/stack_1?
(conv2d_transpose_8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_8/strided_slice/stack_2?
 conv2d_transpose_8/strided_sliceStridedSlice!conv2d_transpose_8/Shape:output:0/conv2d_transpose_8/strided_slice/stack:output:01conv2d_transpose_8/strided_slice/stack_1:output:01conv2d_transpose_8/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv2d_transpose_8/strided_slicez
conv2d_transpose_8/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_8/stack/1z
conv2d_transpose_8/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_8/stack/2z
conv2d_transpose_8/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_8/stack/3?
conv2d_transpose_8/stackPack)conv2d_transpose_8/strided_slice:output:0#conv2d_transpose_8/stack/1:output:0#conv2d_transpose_8/stack/2:output:0#conv2d_transpose_8/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_8/stack?
(conv2d_transpose_8/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(conv2d_transpose_8/strided_slice_1/stack?
*conv2d_transpose_8/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_8/strided_slice_1/stack_1?
*conv2d_transpose_8/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_8/strided_slice_1/stack_2?
"conv2d_transpose_8/strided_slice_1StridedSlice!conv2d_transpose_8/stack:output:01conv2d_transpose_8/strided_slice_1/stack:output:03conv2d_transpose_8/strided_slice_1/stack_1:output:03conv2d_transpose_8/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv2d_transpose_8/strided_slice_1?
2conv2d_transpose_8/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_8_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype024
2conv2d_transpose_8/conv2d_transpose/ReadVariableOp?
#conv2d_transpose_8/conv2d_transposeConv2DBackpropInput!conv2d_transpose_8/stack:output:0:conv2d_transpose_8/conv2d_transpose/ReadVariableOp:value:0%conv2d_transpose_7/Relu:activations:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2%
#conv2d_transpose_8/conv2d_transpose?
)conv2d_transpose_8/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)conv2d_transpose_8/BiasAdd/ReadVariableOp?
conv2d_transpose_8/BiasAddBiasAdd,conv2d_transpose_8/conv2d_transpose:output:01conv2d_transpose_8/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
conv2d_transpose_8/BiasAdd?
conv2d_transpose_8/SigmoidSigmoid#conv2d_transpose_8/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
conv2d_transpose_8/Sigmoidr
reshape_11/ShapeShapeconv2d_transpose_8/Sigmoid:y:0*
T0*
_output_shapes
:2
reshape_11/Shape?
reshape_11/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
reshape_11/strided_slice/stack?
 reshape_11/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_11/strided_slice/stack_1?
 reshape_11/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_11/strided_slice/stack_2?
reshape_11/strided_sliceStridedSlicereshape_11/Shape:output:0'reshape_11/strided_slice/stack:output:0)reshape_11/strided_slice/stack_1:output:0)reshape_11/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_11/strided_slicez
reshape_11/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_11/Reshape/shape/1z
reshape_11/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_11/Reshape/shape/2z
reshape_11/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_11/Reshape/shape/3?
reshape_11/Reshape/shapePack!reshape_11/strided_slice:output:0#reshape_11/Reshape/shape/1:output:0#reshape_11/Reshape/shape/2:output:0#reshape_11/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
reshape_11/Reshape/shape?
reshape_11/ReshapeReshapeconv2d_transpose_8/Sigmoid:y:0!reshape_11/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????2
reshape_11/Reshape~
IdentityIdentityreshape_11/Reshape:output:0^NoOp*
T0*/
_output_shapes
:?????????2

Identity?
NoOpNoOp*^conv2d_transpose_6/BiasAdd/ReadVariableOp3^conv2d_transpose_6/conv2d_transpose/ReadVariableOp*^conv2d_transpose_7/BiasAdd/ReadVariableOp3^conv2d_transpose_7/conv2d_transpose/ReadVariableOp*^conv2d_transpose_8/BiasAdd/ReadVariableOp3^conv2d_transpose_8/conv2d_transpose/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : : : : : 2V
)conv2d_transpose_6/BiasAdd/ReadVariableOp)conv2d_transpose_6/BiasAdd/ReadVariableOp2h
2conv2d_transpose_6/conv2d_transpose/ReadVariableOp2conv2d_transpose_6/conv2d_transpose/ReadVariableOp2V
)conv2d_transpose_7/BiasAdd/ReadVariableOp)conv2d_transpose_7/BiasAdd/ReadVariableOp2h
2conv2d_transpose_7/conv2d_transpose/ReadVariableOp2conv2d_transpose_7/conv2d_transpose/ReadVariableOp2V
)conv2d_transpose_8/BiasAdd/ReadVariableOp)conv2d_transpose_8/BiasAdd/ReadVariableOp2h
2conv2d_transpose_8/conv2d_transpose/ReadVariableOp2conv2d_transpose_8/conv2d_transpose/ReadVariableOp2@
dense_5/BiasAdd/ReadVariableOpdense_5/BiasAdd/ReadVariableOp2>
dense_5/MatMul/ReadVariableOpdense_5/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
2__inference_conv2d_transpose_7_layer_call_fn_47256

inputs!
unknown: @
	unknown_0: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *V
fQRO
M__inference_conv2d_transpose_7_layer_call_and_return_conditional_losses_455912
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:????????? 2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????@: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
2__inference_conv2d_transpose_7_layer_call_fn_47247

inputs!
unknown: @
	unknown_0: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *V
fQRO
M__inference_conv2d_transpose_7_layer_call_and_return_conditional_losses_453612
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????@: : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
E
)__inference_reshape_9_layer_call_fn_47065

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_reshape_9_layer_call_and_return_conditional_losses_450522
PartitionedCallp
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
B__inference_dense_5_layer_call_and_return_conditional_losses_47076

inputs1
matmul_readvariableop_resource:	?.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relun
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
a
E__inference_reshape_10_layer_call_and_return_conditional_losses_45537

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliced
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/1d
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/2d
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B : 2
Reshape/shape/3?
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapew
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:????????? 2	
Reshapel
IdentityIdentityReshape:output:0*
T0*/
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
`
D__inference_reshape_8_layer_call_and_return_conditional_losses_44979

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliced
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/1d
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/2d
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/3?
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapew
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:?????????2	
Reshapel
IdentityIdentityReshape:output:0*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
M__inference_conv2d_transpose_6_layer_call_and_return_conditional_losses_45562

inputsB
(conv2d_transpose_readvariableop_resource:@ -
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliceT
stack/1Const*
_output_shapes
: *
dtype0*
value	B :2	
stack/1T
stack/2Const*
_output_shapes
: *
dtype0*
value	B :2	
stack/2T
stack/3Const*
_output_shapes
: *
dtype0*
value	B :@2	
stack/3?
stackPackstrided_slice:output:0stack/1:output:0stack/2:output:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSlicestack:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:@ *
dtype02!
conv2d_transpose/ReadVariableOp?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
conv2d_transpose?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
Reluu
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????@2

Identity?
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?

?
,__inference_sequential_5_layer_call_fn_46915

inputs
unknown:	?
	unknown_0:	?#
	unknown_1:@ 
	unknown_2:@#
	unknown_3: @
	unknown_4: #
	unknown_5: 
	unknown_6:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_sequential_5_layer_call_and_return_conditional_losses_457602
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
q
R__inference_normal_sampling_layer_2_layer_call_and_return_conditional_losses_45964

inputs
identity??
unstackUnpackinputs*
T0*:
_output_shapes(
&:?????????:?????????*

axis*	
num2	
unstackN
ShapeShapeunstack:output:0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliceR
Shape_1Shapeunstack:output:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
random_normal/shapePackstrided_slice:output:0strided_slice_1:output:0*
N*
T0*
_output_shapes
:2
random_normal/shapem
random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2
random_normal/meanq
random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
random_normal/stddev?
"random_normal/RandomStandardNormalRandomStandardNormalrandom_normal/shape:output:0*
T0*0
_output_shapes
:??????????????????*
dtype0*
seed???)*
seed2??2$
"random_normal/RandomStandardNormal?
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*0
_output_shapes
:??????????????????2
random_normal/mul?
random_normalAddV2random_normal/mul:z:0random_normal/mean:output:0*
T0*0
_output_shapes
:??????????????????2
random_normalS
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
mul/xe
mulMulmul/x:output:0unstack:output:1*
T0*'
_output_shapes
:?????????2
mulL
ExpExpmul:z:0*
T0*'
_output_shapes
:?????????2
Expc
mul_1MulExp:y:0random_normal:z:0*
T0*'
_output_shapes
:?????????2
mul_1b
addAddV2unstack:output:0	mul_1:z:0*
T0*'
_output_shapes
:?????????2
add[
IdentityIdentityadd:z:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
,__inference_sequential_5_layer_call_fn_45800
input_10
unknown:	?
	unknown_0:	?#
	unknown_1:@ 
	unknown_2:@#
	unknown_3: @
	unknown_4: #
	unknown_5: 
	unknown_6:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_10unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_sequential_5_layer_call_and_return_conditional_losses_457602
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:?????????
"
_user_specified_name
input_10
?
S
7__inference_normal_sampling_layer_2_layer_call_fn_46953

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_normal_sampling_layer_2_layer_call_and_return_conditional_losses_458782
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
??
?
T__inference_variational_autoencoder_2_layer_call_and_return_conditional_losses_46501

inputsN
4sequential_4_conv2d_4_conv2d_readvariableop_resource: C
5sequential_4_conv2d_4_biasadd_readvariableop_resource: N
4sequential_4_conv2d_5_conv2d_readvariableop_resource: @C
5sequential_4_conv2d_5_biasadd_readvariableop_resource:@F
3sequential_4_dense_4_matmul_readvariableop_resource:	?B
4sequential_4_dense_4_biasadd_readvariableop_resource:F
3sequential_5_dense_5_matmul_readvariableop_resource:	?C
4sequential_5_dense_5_biasadd_readvariableop_resource:	?b
Hsequential_5_conv2d_transpose_6_conv2d_transpose_readvariableop_resource:@ M
?sequential_5_conv2d_transpose_6_biasadd_readvariableop_resource:@b
Hsequential_5_conv2d_transpose_7_conv2d_transpose_readvariableop_resource: @M
?sequential_5_conv2d_transpose_7_biasadd_readvariableop_resource: b
Hsequential_5_conv2d_transpose_8_conv2d_transpose_readvariableop_resource: M
?sequential_5_conv2d_transpose_8_biasadd_readvariableop_resource:
identity??,sequential_4/conv2d_4/BiasAdd/ReadVariableOp?+sequential_4/conv2d_4/Conv2D/ReadVariableOp?,sequential_4/conv2d_5/BiasAdd/ReadVariableOp?+sequential_4/conv2d_5/Conv2D/ReadVariableOp?+sequential_4/dense_4/BiasAdd/ReadVariableOp?*sequential_4/dense_4/MatMul/ReadVariableOp?6sequential_5/conv2d_transpose_6/BiasAdd/ReadVariableOp??sequential_5/conv2d_transpose_6/conv2d_transpose/ReadVariableOp?6sequential_5/conv2d_transpose_7/BiasAdd/ReadVariableOp??sequential_5/conv2d_transpose_7/conv2d_transpose/ReadVariableOp?6sequential_5/conv2d_transpose_8/BiasAdd/ReadVariableOp??sequential_5/conv2d_transpose_8/conv2d_transpose/ReadVariableOp?+sequential_5/dense_5/BiasAdd/ReadVariableOp?*sequential_5/dense_5/MatMul/ReadVariableOpr
sequential_4/reshape_8/ShapeShapeinputs*
T0*
_output_shapes
:2
sequential_4/reshape_8/Shape?
*sequential_4/reshape_8/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*sequential_4/reshape_8/strided_slice/stack?
,sequential_4/reshape_8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_4/reshape_8/strided_slice/stack_1?
,sequential_4/reshape_8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_4/reshape_8/strided_slice/stack_2?
$sequential_4/reshape_8/strided_sliceStridedSlice%sequential_4/reshape_8/Shape:output:03sequential_4/reshape_8/strided_slice/stack:output:05sequential_4/reshape_8/strided_slice/stack_1:output:05sequential_4/reshape_8/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2&
$sequential_4/reshape_8/strided_slice?
&sequential_4/reshape_8/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2(
&sequential_4/reshape_8/Reshape/shape/1?
&sequential_4/reshape_8/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2(
&sequential_4/reshape_8/Reshape/shape/2?
&sequential_4/reshape_8/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2(
&sequential_4/reshape_8/Reshape/shape/3?
$sequential_4/reshape_8/Reshape/shapePack-sequential_4/reshape_8/strided_slice:output:0/sequential_4/reshape_8/Reshape/shape/1:output:0/sequential_4/reshape_8/Reshape/shape/2:output:0/sequential_4/reshape_8/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2&
$sequential_4/reshape_8/Reshape/shape?
sequential_4/reshape_8/ReshapeReshapeinputs-sequential_4/reshape_8/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????2 
sequential_4/reshape_8/Reshape?
+sequential_4/conv2d_4/Conv2D/ReadVariableOpReadVariableOp4sequential_4_conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02-
+sequential_4/conv2d_4/Conv2D/ReadVariableOp?
sequential_4/conv2d_4/Conv2DConv2D'sequential_4/reshape_8/Reshape:output:03sequential_4/conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
2
sequential_4/conv2d_4/Conv2D?
,sequential_4/conv2d_4/BiasAdd/ReadVariableOpReadVariableOp5sequential_4_conv2d_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02.
,sequential_4/conv2d_4/BiasAdd/ReadVariableOp?
sequential_4/conv2d_4/BiasAddBiasAdd%sequential_4/conv2d_4/Conv2D:output:04sequential_4/conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2
sequential_4/conv2d_4/BiasAdd?
sequential_4/conv2d_4/ReluRelu&sequential_4/conv2d_4/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
sequential_4/conv2d_4/Relu?
+sequential_4/conv2d_5/Conv2D/ReadVariableOpReadVariableOp4sequential_4_conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02-
+sequential_4/conv2d_5/Conv2D/ReadVariableOp?
sequential_4/conv2d_5/Conv2DConv2D(sequential_4/conv2d_4/Relu:activations:03sequential_4/conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
2
sequential_4/conv2d_5/Conv2D?
,sequential_4/conv2d_5/BiasAdd/ReadVariableOpReadVariableOp5sequential_4_conv2d_5_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02.
,sequential_4/conv2d_5/BiasAdd/ReadVariableOp?
sequential_4/conv2d_5/BiasAddBiasAdd%sequential_4/conv2d_5/Conv2D:output:04sequential_4/conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
sequential_4/conv2d_5/BiasAdd?
sequential_4/conv2d_5/ReluRelu&sequential_4/conv2d_5/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
sequential_4/conv2d_5/Relu?
sequential_4/flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"???? 	  2
sequential_4/flatten_2/Const?
sequential_4/flatten_2/ReshapeReshape(sequential_4/conv2d_5/Relu:activations:0%sequential_4/flatten_2/Const:output:0*
T0*(
_output_shapes
:??????????2 
sequential_4/flatten_2/Reshape?
*sequential_4/dense_4/MatMul/ReadVariableOpReadVariableOp3sequential_4_dense_4_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02,
*sequential_4/dense_4/MatMul/ReadVariableOp?
sequential_4/dense_4/MatMulMatMul'sequential_4/flatten_2/Reshape:output:02sequential_4/dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential_4/dense_4/MatMul?
+sequential_4/dense_4/BiasAdd/ReadVariableOpReadVariableOp4sequential_4_dense_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02-
+sequential_4/dense_4/BiasAdd/ReadVariableOp?
sequential_4/dense_4/BiasAddBiasAdd%sequential_4/dense_4/MatMul:product:03sequential_4/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential_4/dense_4/BiasAdd?
sequential_4/reshape_9/ShapeShape%sequential_4/dense_4/BiasAdd:output:0*
T0*
_output_shapes
:2
sequential_4/reshape_9/Shape?
*sequential_4/reshape_9/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*sequential_4/reshape_9/strided_slice/stack?
,sequential_4/reshape_9/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_4/reshape_9/strided_slice/stack_1?
,sequential_4/reshape_9/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_4/reshape_9/strided_slice/stack_2?
$sequential_4/reshape_9/strided_sliceStridedSlice%sequential_4/reshape_9/Shape:output:03sequential_4/reshape_9/strided_slice/stack:output:05sequential_4/reshape_9/strided_slice/stack_1:output:05sequential_4/reshape_9/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2&
$sequential_4/reshape_9/strided_slice?
&sequential_4/reshape_9/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2(
&sequential_4/reshape_9/Reshape/shape/1?
&sequential_4/reshape_9/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2(
&sequential_4/reshape_9/Reshape/shape/2?
$sequential_4/reshape_9/Reshape/shapePack-sequential_4/reshape_9/strided_slice:output:0/sequential_4/reshape_9/Reshape/shape/1:output:0/sequential_4/reshape_9/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2&
$sequential_4/reshape_9/Reshape/shape?
sequential_4/reshape_9/ReshapeReshape%sequential_4/dense_4/BiasAdd:output:0-sequential_4/reshape_9/Reshape/shape:output:0*
T0*+
_output_shapes
:?????????2 
sequential_4/reshape_9/Reshape?
normal_sampling_layer_2/unstackUnpack'sequential_4/reshape_9/Reshape:output:0*
T0*:
_output_shapes(
&:?????????:?????????*

axis*	
num2!
normal_sampling_layer_2/unstack?
normal_sampling_layer_2/ShapeShape(normal_sampling_layer_2/unstack:output:0*
T0*
_output_shapes
:2
normal_sampling_layer_2/Shape?
+normal_sampling_layer_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2-
+normal_sampling_layer_2/strided_slice/stack?
-normal_sampling_layer_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-normal_sampling_layer_2/strided_slice/stack_1?
-normal_sampling_layer_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-normal_sampling_layer_2/strided_slice/stack_2?
%normal_sampling_layer_2/strided_sliceStridedSlice&normal_sampling_layer_2/Shape:output:04normal_sampling_layer_2/strided_slice/stack:output:06normal_sampling_layer_2/strided_slice/stack_1:output:06normal_sampling_layer_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%normal_sampling_layer_2/strided_slice?
normal_sampling_layer_2/Shape_1Shape(normal_sampling_layer_2/unstack:output:0*
T0*
_output_shapes
:2!
normal_sampling_layer_2/Shape_1?
-normal_sampling_layer_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2/
-normal_sampling_layer_2/strided_slice_1/stack?
/normal_sampling_layer_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:21
/normal_sampling_layer_2/strided_slice_1/stack_1?
/normal_sampling_layer_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:21
/normal_sampling_layer_2/strided_slice_1/stack_2?
'normal_sampling_layer_2/strided_slice_1StridedSlice(normal_sampling_layer_2/Shape_1:output:06normal_sampling_layer_2/strided_slice_1/stack:output:08normal_sampling_layer_2/strided_slice_1/stack_1:output:08normal_sampling_layer_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2)
'normal_sampling_layer_2/strided_slice_1?
+normal_sampling_layer_2/random_normal/shapePack.normal_sampling_layer_2/strided_slice:output:00normal_sampling_layer_2/strided_slice_1:output:0*
N*
T0*
_output_shapes
:2-
+normal_sampling_layer_2/random_normal/shape?
*normal_sampling_layer_2/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2,
*normal_sampling_layer_2/random_normal/mean?
,normal_sampling_layer_2/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2.
,normal_sampling_layer_2/random_normal/stddev?
:normal_sampling_layer_2/random_normal/RandomStandardNormalRandomStandardNormal4normal_sampling_layer_2/random_normal/shape:output:0*
T0*0
_output_shapes
:??????????????????*
dtype0*
seed???)*
seed2?ʩ2<
:normal_sampling_layer_2/random_normal/RandomStandardNormal?
)normal_sampling_layer_2/random_normal/mulMulCnormal_sampling_layer_2/random_normal/RandomStandardNormal:output:05normal_sampling_layer_2/random_normal/stddev:output:0*
T0*0
_output_shapes
:??????????????????2+
)normal_sampling_layer_2/random_normal/mul?
%normal_sampling_layer_2/random_normalAddV2-normal_sampling_layer_2/random_normal/mul:z:03normal_sampling_layer_2/random_normal/mean:output:0*
T0*0
_output_shapes
:??????????????????2'
%normal_sampling_layer_2/random_normal?
normal_sampling_layer_2/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
normal_sampling_layer_2/mul/x?
normal_sampling_layer_2/mulMul&normal_sampling_layer_2/mul/x:output:0(normal_sampling_layer_2/unstack:output:1*
T0*'
_output_shapes
:?????????2
normal_sampling_layer_2/mul?
normal_sampling_layer_2/ExpExpnormal_sampling_layer_2/mul:z:0*
T0*'
_output_shapes
:?????????2
normal_sampling_layer_2/Exp?
normal_sampling_layer_2/mul_1Mulnormal_sampling_layer_2/Exp:y:0)normal_sampling_layer_2/random_normal:z:0*
T0*'
_output_shapes
:?????????2
normal_sampling_layer_2/mul_1?
normal_sampling_layer_2/addAddV2(normal_sampling_layer_2/unstack:output:0!normal_sampling_layer_2/mul_1:z:0*
T0*'
_output_shapes
:?????????2
normal_sampling_layer_2/add?
*sequential_5/dense_5/MatMul/ReadVariableOpReadVariableOp3sequential_5_dense_5_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02,
*sequential_5/dense_5/MatMul/ReadVariableOp?
sequential_5/dense_5/MatMulMatMulnormal_sampling_layer_2/add:z:02sequential_5/dense_5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential_5/dense_5/MatMul?
+sequential_5/dense_5/BiasAdd/ReadVariableOpReadVariableOp4sequential_5_dense_5_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02-
+sequential_5/dense_5/BiasAdd/ReadVariableOp?
sequential_5/dense_5/BiasAddBiasAdd%sequential_5/dense_5/MatMul:product:03sequential_5/dense_5/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential_5/dense_5/BiasAdd?
sequential_5/dense_5/ReluRelu%sequential_5/dense_5/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
sequential_5/dense_5/Relu?
sequential_5/reshape_10/ShapeShape'sequential_5/dense_5/Relu:activations:0*
T0*
_output_shapes
:2
sequential_5/reshape_10/Shape?
+sequential_5/reshape_10/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2-
+sequential_5/reshape_10/strided_slice/stack?
-sequential_5/reshape_10/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_5/reshape_10/strided_slice/stack_1?
-sequential_5/reshape_10/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_5/reshape_10/strided_slice/stack_2?
%sequential_5/reshape_10/strided_sliceStridedSlice&sequential_5/reshape_10/Shape:output:04sequential_5/reshape_10/strided_slice/stack:output:06sequential_5/reshape_10/strided_slice/stack_1:output:06sequential_5/reshape_10/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%sequential_5/reshape_10/strided_slice?
'sequential_5/reshape_10/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2)
'sequential_5/reshape_10/Reshape/shape/1?
'sequential_5/reshape_10/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2)
'sequential_5/reshape_10/Reshape/shape/2?
'sequential_5/reshape_10/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B : 2)
'sequential_5/reshape_10/Reshape/shape/3?
%sequential_5/reshape_10/Reshape/shapePack.sequential_5/reshape_10/strided_slice:output:00sequential_5/reshape_10/Reshape/shape/1:output:00sequential_5/reshape_10/Reshape/shape/2:output:00sequential_5/reshape_10/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2'
%sequential_5/reshape_10/Reshape/shape?
sequential_5/reshape_10/ReshapeReshape'sequential_5/dense_5/Relu:activations:0.sequential_5/reshape_10/Reshape/shape:output:0*
T0*/
_output_shapes
:????????? 2!
sequential_5/reshape_10/Reshape?
%sequential_5/conv2d_transpose_6/ShapeShape(sequential_5/reshape_10/Reshape:output:0*
T0*
_output_shapes
:2'
%sequential_5/conv2d_transpose_6/Shape?
3sequential_5/conv2d_transpose_6/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 25
3sequential_5/conv2d_transpose_6/strided_slice/stack?
5sequential_5/conv2d_transpose_6/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential_5/conv2d_transpose_6/strided_slice/stack_1?
5sequential_5/conv2d_transpose_6/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential_5/conv2d_transpose_6/strided_slice/stack_2?
-sequential_5/conv2d_transpose_6/strided_sliceStridedSlice.sequential_5/conv2d_transpose_6/Shape:output:0<sequential_5/conv2d_transpose_6/strided_slice/stack:output:0>sequential_5/conv2d_transpose_6/strided_slice/stack_1:output:0>sequential_5/conv2d_transpose_6/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2/
-sequential_5/conv2d_transpose_6/strided_slice?
'sequential_5/conv2d_transpose_6/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2)
'sequential_5/conv2d_transpose_6/stack/1?
'sequential_5/conv2d_transpose_6/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2)
'sequential_5/conv2d_transpose_6/stack/2?
'sequential_5/conv2d_transpose_6/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@2)
'sequential_5/conv2d_transpose_6/stack/3?
%sequential_5/conv2d_transpose_6/stackPack6sequential_5/conv2d_transpose_6/strided_slice:output:00sequential_5/conv2d_transpose_6/stack/1:output:00sequential_5/conv2d_transpose_6/stack/2:output:00sequential_5/conv2d_transpose_6/stack/3:output:0*
N*
T0*
_output_shapes
:2'
%sequential_5/conv2d_transpose_6/stack?
5sequential_5/conv2d_transpose_6/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 27
5sequential_5/conv2d_transpose_6/strided_slice_1/stack?
7sequential_5/conv2d_transpose_6/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:29
7sequential_5/conv2d_transpose_6/strided_slice_1/stack_1?
7sequential_5/conv2d_transpose_6/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:29
7sequential_5/conv2d_transpose_6/strided_slice_1/stack_2?
/sequential_5/conv2d_transpose_6/strided_slice_1StridedSlice.sequential_5/conv2d_transpose_6/stack:output:0>sequential_5/conv2d_transpose_6/strided_slice_1/stack:output:0@sequential_5/conv2d_transpose_6/strided_slice_1/stack_1:output:0@sequential_5/conv2d_transpose_6/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask21
/sequential_5/conv2d_transpose_6/strided_slice_1?
?sequential_5/conv2d_transpose_6/conv2d_transpose/ReadVariableOpReadVariableOpHsequential_5_conv2d_transpose_6_conv2d_transpose_readvariableop_resource*&
_output_shapes
:@ *
dtype02A
?sequential_5/conv2d_transpose_6/conv2d_transpose/ReadVariableOp?
0sequential_5/conv2d_transpose_6/conv2d_transposeConv2DBackpropInput.sequential_5/conv2d_transpose_6/stack:output:0Gsequential_5/conv2d_transpose_6/conv2d_transpose/ReadVariableOp:value:0(sequential_5/reshape_10/Reshape:output:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
22
0sequential_5/conv2d_transpose_6/conv2d_transpose?
6sequential_5/conv2d_transpose_6/BiasAdd/ReadVariableOpReadVariableOp?sequential_5_conv2d_transpose_6_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype028
6sequential_5/conv2d_transpose_6/BiasAdd/ReadVariableOp?
'sequential_5/conv2d_transpose_6/BiasAddBiasAdd9sequential_5/conv2d_transpose_6/conv2d_transpose:output:0>sequential_5/conv2d_transpose_6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2)
'sequential_5/conv2d_transpose_6/BiasAdd?
$sequential_5/conv2d_transpose_6/ReluRelu0sequential_5/conv2d_transpose_6/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2&
$sequential_5/conv2d_transpose_6/Relu?
%sequential_5/conv2d_transpose_7/ShapeShape2sequential_5/conv2d_transpose_6/Relu:activations:0*
T0*
_output_shapes
:2'
%sequential_5/conv2d_transpose_7/Shape?
3sequential_5/conv2d_transpose_7/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 25
3sequential_5/conv2d_transpose_7/strided_slice/stack?
5sequential_5/conv2d_transpose_7/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential_5/conv2d_transpose_7/strided_slice/stack_1?
5sequential_5/conv2d_transpose_7/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential_5/conv2d_transpose_7/strided_slice/stack_2?
-sequential_5/conv2d_transpose_7/strided_sliceStridedSlice.sequential_5/conv2d_transpose_7/Shape:output:0<sequential_5/conv2d_transpose_7/strided_slice/stack:output:0>sequential_5/conv2d_transpose_7/strided_slice/stack_1:output:0>sequential_5/conv2d_transpose_7/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2/
-sequential_5/conv2d_transpose_7/strided_slice?
'sequential_5/conv2d_transpose_7/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2)
'sequential_5/conv2d_transpose_7/stack/1?
'sequential_5/conv2d_transpose_7/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2)
'sequential_5/conv2d_transpose_7/stack/2?
'sequential_5/conv2d_transpose_7/stack/3Const*
_output_shapes
: *
dtype0*
value	B : 2)
'sequential_5/conv2d_transpose_7/stack/3?
%sequential_5/conv2d_transpose_7/stackPack6sequential_5/conv2d_transpose_7/strided_slice:output:00sequential_5/conv2d_transpose_7/stack/1:output:00sequential_5/conv2d_transpose_7/stack/2:output:00sequential_5/conv2d_transpose_7/stack/3:output:0*
N*
T0*
_output_shapes
:2'
%sequential_5/conv2d_transpose_7/stack?
5sequential_5/conv2d_transpose_7/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 27
5sequential_5/conv2d_transpose_7/strided_slice_1/stack?
7sequential_5/conv2d_transpose_7/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:29
7sequential_5/conv2d_transpose_7/strided_slice_1/stack_1?
7sequential_5/conv2d_transpose_7/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:29
7sequential_5/conv2d_transpose_7/strided_slice_1/stack_2?
/sequential_5/conv2d_transpose_7/strided_slice_1StridedSlice.sequential_5/conv2d_transpose_7/stack:output:0>sequential_5/conv2d_transpose_7/strided_slice_1/stack:output:0@sequential_5/conv2d_transpose_7/strided_slice_1/stack_1:output:0@sequential_5/conv2d_transpose_7/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask21
/sequential_5/conv2d_transpose_7/strided_slice_1?
?sequential_5/conv2d_transpose_7/conv2d_transpose/ReadVariableOpReadVariableOpHsequential_5_conv2d_transpose_7_conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype02A
?sequential_5/conv2d_transpose_7/conv2d_transpose/ReadVariableOp?
0sequential_5/conv2d_transpose_7/conv2d_transposeConv2DBackpropInput.sequential_5/conv2d_transpose_7/stack:output:0Gsequential_5/conv2d_transpose_7/conv2d_transpose/ReadVariableOp:value:02sequential_5/conv2d_transpose_6/Relu:activations:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
22
0sequential_5/conv2d_transpose_7/conv2d_transpose?
6sequential_5/conv2d_transpose_7/BiasAdd/ReadVariableOpReadVariableOp?sequential_5_conv2d_transpose_7_biasadd_readvariableop_resource*
_output_shapes
: *
dtype028
6sequential_5/conv2d_transpose_7/BiasAdd/ReadVariableOp?
'sequential_5/conv2d_transpose_7/BiasAddBiasAdd9sequential_5/conv2d_transpose_7/conv2d_transpose:output:0>sequential_5/conv2d_transpose_7/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2)
'sequential_5/conv2d_transpose_7/BiasAdd?
$sequential_5/conv2d_transpose_7/ReluRelu0sequential_5/conv2d_transpose_7/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2&
$sequential_5/conv2d_transpose_7/Relu?
%sequential_5/conv2d_transpose_8/ShapeShape2sequential_5/conv2d_transpose_7/Relu:activations:0*
T0*
_output_shapes
:2'
%sequential_5/conv2d_transpose_8/Shape?
3sequential_5/conv2d_transpose_8/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 25
3sequential_5/conv2d_transpose_8/strided_slice/stack?
5sequential_5/conv2d_transpose_8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential_5/conv2d_transpose_8/strided_slice/stack_1?
5sequential_5/conv2d_transpose_8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential_5/conv2d_transpose_8/strided_slice/stack_2?
-sequential_5/conv2d_transpose_8/strided_sliceStridedSlice.sequential_5/conv2d_transpose_8/Shape:output:0<sequential_5/conv2d_transpose_8/strided_slice/stack:output:0>sequential_5/conv2d_transpose_8/strided_slice/stack_1:output:0>sequential_5/conv2d_transpose_8/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2/
-sequential_5/conv2d_transpose_8/strided_slice?
'sequential_5/conv2d_transpose_8/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2)
'sequential_5/conv2d_transpose_8/stack/1?
'sequential_5/conv2d_transpose_8/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2)
'sequential_5/conv2d_transpose_8/stack/2?
'sequential_5/conv2d_transpose_8/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2)
'sequential_5/conv2d_transpose_8/stack/3?
%sequential_5/conv2d_transpose_8/stackPack6sequential_5/conv2d_transpose_8/strided_slice:output:00sequential_5/conv2d_transpose_8/stack/1:output:00sequential_5/conv2d_transpose_8/stack/2:output:00sequential_5/conv2d_transpose_8/stack/3:output:0*
N*
T0*
_output_shapes
:2'
%sequential_5/conv2d_transpose_8/stack?
5sequential_5/conv2d_transpose_8/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 27
5sequential_5/conv2d_transpose_8/strided_slice_1/stack?
7sequential_5/conv2d_transpose_8/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:29
7sequential_5/conv2d_transpose_8/strided_slice_1/stack_1?
7sequential_5/conv2d_transpose_8/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:29
7sequential_5/conv2d_transpose_8/strided_slice_1/stack_2?
/sequential_5/conv2d_transpose_8/strided_slice_1StridedSlice.sequential_5/conv2d_transpose_8/stack:output:0>sequential_5/conv2d_transpose_8/strided_slice_1/stack:output:0@sequential_5/conv2d_transpose_8/strided_slice_1/stack_1:output:0@sequential_5/conv2d_transpose_8/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask21
/sequential_5/conv2d_transpose_8/strided_slice_1?
?sequential_5/conv2d_transpose_8/conv2d_transpose/ReadVariableOpReadVariableOpHsequential_5_conv2d_transpose_8_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype02A
?sequential_5/conv2d_transpose_8/conv2d_transpose/ReadVariableOp?
0sequential_5/conv2d_transpose_8/conv2d_transposeConv2DBackpropInput.sequential_5/conv2d_transpose_8/stack:output:0Gsequential_5/conv2d_transpose_8/conv2d_transpose/ReadVariableOp:value:02sequential_5/conv2d_transpose_7/Relu:activations:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
22
0sequential_5/conv2d_transpose_8/conv2d_transpose?
6sequential_5/conv2d_transpose_8/BiasAdd/ReadVariableOpReadVariableOp?sequential_5_conv2d_transpose_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype028
6sequential_5/conv2d_transpose_8/BiasAdd/ReadVariableOp?
'sequential_5/conv2d_transpose_8/BiasAddBiasAdd9sequential_5/conv2d_transpose_8/conv2d_transpose:output:0>sequential_5/conv2d_transpose_8/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2)
'sequential_5/conv2d_transpose_8/BiasAdd?
'sequential_5/conv2d_transpose_8/SigmoidSigmoid0sequential_5/conv2d_transpose_8/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2)
'sequential_5/conv2d_transpose_8/Sigmoid?
sequential_5/reshape_11/ShapeShape+sequential_5/conv2d_transpose_8/Sigmoid:y:0*
T0*
_output_shapes
:2
sequential_5/reshape_11/Shape?
+sequential_5/reshape_11/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2-
+sequential_5/reshape_11/strided_slice/stack?
-sequential_5/reshape_11/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_5/reshape_11/strided_slice/stack_1?
-sequential_5/reshape_11/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_5/reshape_11/strided_slice/stack_2?
%sequential_5/reshape_11/strided_sliceStridedSlice&sequential_5/reshape_11/Shape:output:04sequential_5/reshape_11/strided_slice/stack:output:06sequential_5/reshape_11/strided_slice/stack_1:output:06sequential_5/reshape_11/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%sequential_5/reshape_11/strided_slice?
'sequential_5/reshape_11/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2)
'sequential_5/reshape_11/Reshape/shape/1?
'sequential_5/reshape_11/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2)
'sequential_5/reshape_11/Reshape/shape/2?
'sequential_5/reshape_11/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2)
'sequential_5/reshape_11/Reshape/shape/3?
%sequential_5/reshape_11/Reshape/shapePack.sequential_5/reshape_11/strided_slice:output:00sequential_5/reshape_11/Reshape/shape/1:output:00sequential_5/reshape_11/Reshape/shape/2:output:00sequential_5/reshape_11/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2'
%sequential_5/reshape_11/Reshape/shape?
sequential_5/reshape_11/ReshapeReshape+sequential_5/conv2d_transpose_8/Sigmoid:y:0.sequential_5/reshape_11/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????2!
sequential_5/reshape_11/Reshape?
IdentityIdentity(sequential_5/reshape_11/Reshape:output:0^NoOp*
T0*/
_output_shapes
:?????????2

Identity?
NoOpNoOp-^sequential_4/conv2d_4/BiasAdd/ReadVariableOp,^sequential_4/conv2d_4/Conv2D/ReadVariableOp-^sequential_4/conv2d_5/BiasAdd/ReadVariableOp,^sequential_4/conv2d_5/Conv2D/ReadVariableOp,^sequential_4/dense_4/BiasAdd/ReadVariableOp+^sequential_4/dense_4/MatMul/ReadVariableOp7^sequential_5/conv2d_transpose_6/BiasAdd/ReadVariableOp@^sequential_5/conv2d_transpose_6/conv2d_transpose/ReadVariableOp7^sequential_5/conv2d_transpose_7/BiasAdd/ReadVariableOp@^sequential_5/conv2d_transpose_7/conv2d_transpose/ReadVariableOp7^sequential_5/conv2d_transpose_8/BiasAdd/ReadVariableOp@^sequential_5/conv2d_transpose_8/conv2d_transpose/ReadVariableOp,^sequential_5/dense_5/BiasAdd/ReadVariableOp+^sequential_5/dense_5/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:?????????: : : : : : : : : : : : : : 2\
,sequential_4/conv2d_4/BiasAdd/ReadVariableOp,sequential_4/conv2d_4/BiasAdd/ReadVariableOp2Z
+sequential_4/conv2d_4/Conv2D/ReadVariableOp+sequential_4/conv2d_4/Conv2D/ReadVariableOp2\
,sequential_4/conv2d_5/BiasAdd/ReadVariableOp,sequential_4/conv2d_5/BiasAdd/ReadVariableOp2Z
+sequential_4/conv2d_5/Conv2D/ReadVariableOp+sequential_4/conv2d_5/Conv2D/ReadVariableOp2Z
+sequential_4/dense_4/BiasAdd/ReadVariableOp+sequential_4/dense_4/BiasAdd/ReadVariableOp2X
*sequential_4/dense_4/MatMul/ReadVariableOp*sequential_4/dense_4/MatMul/ReadVariableOp2p
6sequential_5/conv2d_transpose_6/BiasAdd/ReadVariableOp6sequential_5/conv2d_transpose_6/BiasAdd/ReadVariableOp2?
?sequential_5/conv2d_transpose_6/conv2d_transpose/ReadVariableOp?sequential_5/conv2d_transpose_6/conv2d_transpose/ReadVariableOp2p
6sequential_5/conv2d_transpose_7/BiasAdd/ReadVariableOp6sequential_5/conv2d_transpose_7/BiasAdd/ReadVariableOp2?
?sequential_5/conv2d_transpose_7/conv2d_transpose/ReadVariableOp?sequential_5/conv2d_transpose_7/conv2d_transpose/ReadVariableOp2p
6sequential_5/conv2d_transpose_8/BiasAdd/ReadVariableOp6sequential_5/conv2d_transpose_8/BiasAdd/ReadVariableOp2?
?sequential_5/conv2d_transpose_8/conv2d_transpose/ReadVariableOp?sequential_5/conv2d_transpose_8/conv2d_transpose/ReadVariableOp2Z
+sequential_5/dense_5/BiasAdd/ReadVariableOp+sequential_5/dense_5/BiasAdd/ReadVariableOp2X
*sequential_5/dense_5/MatMul/ReadVariableOp*sequential_5/dense_5/MatMul/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?%
?
M__inference_conv2d_transpose_7_layer_call_and_return_conditional_losses_47214

inputsB
(conv2d_transpose_readvariableop_resource: @-
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulT
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/yb
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1T
stack/3Const*
_output_shapes
: *
dtype0*
value	B : 2	
stack/3?
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype02!
conv2d_transpose/ReadVariableOp?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+??????????????????????????? *
paddingSAME*
strides
2
conv2d_transpose?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2
Relu?
IdentityIdentityRelu:activations:0^NoOp*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity?
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?l
?
__inference__traced_save_47533
file_prefix$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop&
"savev2_total_2_read_readvariableop&
"savev2_count_2_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop.
*savev2_conv2d_4_kernel_read_readvariableop,
(savev2_conv2d_4_bias_read_readvariableop.
*savev2_conv2d_5_kernel_read_readvariableop,
(savev2_conv2d_5_bias_read_readvariableop-
)savev2_dense_4_kernel_read_readvariableop+
'savev2_dense_4_bias_read_readvariableop-
)savev2_dense_5_kernel_read_readvariableop+
'savev2_dense_5_bias_read_readvariableop8
4savev2_conv2d_transpose_6_kernel_read_readvariableop6
2savev2_conv2d_transpose_6_bias_read_readvariableop8
4savev2_conv2d_transpose_7_kernel_read_readvariableop6
2savev2_conv2d_transpose_7_bias_read_readvariableop8
4savev2_conv2d_transpose_8_kernel_read_readvariableop6
2savev2_conv2d_transpose_8_bias_read_readvariableop5
1savev2_adam_conv2d_4_kernel_m_read_readvariableop3
/savev2_adam_conv2d_4_bias_m_read_readvariableop5
1savev2_adam_conv2d_5_kernel_m_read_readvariableop3
/savev2_adam_conv2d_5_bias_m_read_readvariableop4
0savev2_adam_dense_4_kernel_m_read_readvariableop2
.savev2_adam_dense_4_bias_m_read_readvariableop4
0savev2_adam_dense_5_kernel_m_read_readvariableop2
.savev2_adam_dense_5_bias_m_read_readvariableop?
;savev2_adam_conv2d_transpose_6_kernel_m_read_readvariableop=
9savev2_adam_conv2d_transpose_6_bias_m_read_readvariableop?
;savev2_adam_conv2d_transpose_7_kernel_m_read_readvariableop=
9savev2_adam_conv2d_transpose_7_bias_m_read_readvariableop?
;savev2_adam_conv2d_transpose_8_kernel_m_read_readvariableop=
9savev2_adam_conv2d_transpose_8_bias_m_read_readvariableop5
1savev2_adam_conv2d_4_kernel_v_read_readvariableop3
/savev2_adam_conv2d_4_bias_v_read_readvariableop5
1savev2_adam_conv2d_5_kernel_v_read_readvariableop3
/savev2_adam_conv2d_5_bias_v_read_readvariableop4
0savev2_adam_dense_4_kernel_v_read_readvariableop2
.savev2_adam_dense_4_bias_v_read_readvariableop4
0savev2_adam_dense_5_kernel_v_read_readvariableop2
.savev2_adam_dense_5_bias_v_read_readvariableop?
;savev2_adam_conv2d_transpose_6_kernel_v_read_readvariableop=
9savev2_adam_conv2d_transpose_6_bias_v_read_readvariableop?
;savev2_adam_conv2d_transpose_7_kernel_v_read_readvariableop=
9savev2_adam_conv2d_transpose_7_bias_v_read_readvariableop?
;savev2_adam_conv2d_transpose_8_kernel_v_read_readvariableop=
9savev2_adam_conv2d_transpose_8_bias_v_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:6*
dtype0*?
value?B?6B3loss_tracker_total/total/.ATTRIBUTES/VARIABLE_VALUEB3loss_tracker_total/count/.ATTRIBUTES/VARIABLE_VALUEB<loss_tracker_reconstruction/total/.ATTRIBUTES/VARIABLE_VALUEB<loss_tracker_reconstruction/count/.ATTRIBUTES/VARIABLE_VALUEB4loss_tracker_latent/total/.ATTRIBUTES/VARIABLE_VALUEB4loss_tracker_latent/count/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/12/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/13/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:6*
dtype0*
valuevBt6B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0 savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop"savev2_total_2_read_readvariableop"savev2_count_2_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop*savev2_conv2d_4_kernel_read_readvariableop(savev2_conv2d_4_bias_read_readvariableop*savev2_conv2d_5_kernel_read_readvariableop(savev2_conv2d_5_bias_read_readvariableop)savev2_dense_4_kernel_read_readvariableop'savev2_dense_4_bias_read_readvariableop)savev2_dense_5_kernel_read_readvariableop'savev2_dense_5_bias_read_readvariableop4savev2_conv2d_transpose_6_kernel_read_readvariableop2savev2_conv2d_transpose_6_bias_read_readvariableop4savev2_conv2d_transpose_7_kernel_read_readvariableop2savev2_conv2d_transpose_7_bias_read_readvariableop4savev2_conv2d_transpose_8_kernel_read_readvariableop2savev2_conv2d_transpose_8_bias_read_readvariableop1savev2_adam_conv2d_4_kernel_m_read_readvariableop/savev2_adam_conv2d_4_bias_m_read_readvariableop1savev2_adam_conv2d_5_kernel_m_read_readvariableop/savev2_adam_conv2d_5_bias_m_read_readvariableop0savev2_adam_dense_4_kernel_m_read_readvariableop.savev2_adam_dense_4_bias_m_read_readvariableop0savev2_adam_dense_5_kernel_m_read_readvariableop.savev2_adam_dense_5_bias_m_read_readvariableop;savev2_adam_conv2d_transpose_6_kernel_m_read_readvariableop9savev2_adam_conv2d_transpose_6_bias_m_read_readvariableop;savev2_adam_conv2d_transpose_7_kernel_m_read_readvariableop9savev2_adam_conv2d_transpose_7_bias_m_read_readvariableop;savev2_adam_conv2d_transpose_8_kernel_m_read_readvariableop9savev2_adam_conv2d_transpose_8_bias_m_read_readvariableop1savev2_adam_conv2d_4_kernel_v_read_readvariableop/savev2_adam_conv2d_4_bias_v_read_readvariableop1savev2_adam_conv2d_5_kernel_v_read_readvariableop/savev2_adam_conv2d_5_bias_v_read_readvariableop0savev2_adam_dense_4_kernel_v_read_readvariableop.savev2_adam_dense_4_bias_v_read_readvariableop0savev2_adam_dense_5_kernel_v_read_readvariableop.savev2_adam_dense_5_bias_v_read_readvariableop;savev2_adam_conv2d_transpose_6_kernel_v_read_readvariableop9savev2_adam_conv2d_transpose_6_bias_v_read_readvariableop;savev2_adam_conv2d_transpose_7_kernel_v_read_readvariableop9savev2_adam_conv2d_transpose_7_bias_v_read_readvariableop;savev2_adam_conv2d_transpose_8_kernel_v_read_readvariableop9savev2_adam_conv2d_transpose_8_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *D
dtypes:
826	2
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: 2

Identity_1c
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"!

identity_1Identity_1:output:0*?
_input_shapes?
?: : : : : : : : : : : : : : : @:@:	?::	?:?:@ :@: @: : :: : : @:@:	?::	?:?:@ :@: @: : :: : : @:@:	?::	?:?:@ :@: @: : :: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :,(
&
_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
: @: 

_output_shapes
:@:%!

_output_shapes
:	?: 

_output_shapes
::%!

_output_shapes
:	?:!

_output_shapes	
:?:,(
&
_output_shapes
:@ : 

_output_shapes
:@:,(
&
_output_shapes
: @: 

_output_shapes
: :,(
&
_output_shapes
: : 

_output_shapes
::,(
&
_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
: @: 

_output_shapes
:@:%!

_output_shapes
:	?: 

_output_shapes
::% !

_output_shapes
:	?:!!

_output_shapes	
:?:,"(
&
_output_shapes
:@ : #

_output_shapes
:@:,$(
&
_output_shapes
: @: %

_output_shapes
: :,&(
&
_output_shapes
: : '

_output_shapes
::,((
&
_output_shapes
: : )

_output_shapes
: :,*(
&
_output_shapes
: @: +

_output_shapes
:@:%,!

_output_shapes
:	?: -

_output_shapes
::%.!

_output_shapes
:	?:!/

_output_shapes	
:?:,0(
&
_output_shapes
:@ : 1

_output_shapes
:@:,2(
&
_output_shapes
: @: 3

_output_shapes
: :,4(
&
_output_shapes
: : 5

_output_shapes
::6

_output_shapes
: "?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
C
input_18
serving_default_input_1:0?????????D
output_18
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
?
encoder
decoder
sampler
loss_tracker_total
loss_tracker_reconstruction
loss_tracker_latent
	optimizer
loss
	trainable_variables

regularization_losses
	variables
	keras_api

signatures
?_default_save_signature
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_model
?
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer-5
trainable_variables
regularization_losses
	variables
	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_sequential
?
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer-5
trainable_variables
regularization_losses
 	variables
!	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_sequential
?
"regularization_losses
#trainable_variables
$	variables
%	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
N
	&total
	'count
(	variables
)	keras_api"
_tf_keras_metric
N
	*total
	+count
,	variables
-	keras_api"
_tf_keras_metric
N
	.total
	/count
0	variables
1	keras_api"
_tf_keras_metric
?
2iter

3beta_1

4beta_2
	5decay
6learning_rate7m?8m?9m?:m?;m?<m?=m?>m??m?@m?Am?Bm?Cm?Dm?7v?8v?9v?:v?;v?<v?=v?>v??v?@v?Av?Bv?Cv?Dv?"
	optimizer
 "
trackable_dict_wrapper
?
70
81
92
:3
;4
<5
=6
>7
?8
@9
A10
B11
C12
D13"
trackable_list_wrapper
 "
trackable_list_wrapper
?
70
81
92
:3
;4
<5
=6
>7
?8
@9
A10
B11
C12
D13
&14
'15
*16
+17
.18
/19"
trackable_list_wrapper
?
Elayer_metrics

Flayers
Glayer_regularization_losses
	trainable_variables

regularization_losses
	variables
Hnon_trainable_variables
Imetrics
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
?
Jregularization_losses
Ktrainable_variables
L	variables
M	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?

7kernel
8bias
Nregularization_losses
Otrainable_variables
P	variables
Q	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?

9kernel
:bias
Rregularization_losses
Strainable_variables
T	variables
U	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
Vregularization_losses
Wtrainable_variables
X	variables
Y	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?

;kernel
<bias
Zregularization_losses
[trainable_variables
\	variables
]	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
^regularization_losses
_trainable_variables
`	variables
a	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
J
70
81
92
:3
;4
<5"
trackable_list_wrapper
 "
trackable_list_wrapper
J
70
81
92
:3
;4
<5"
trackable_list_wrapper
?
blayer_metrics

clayers
dlayer_regularization_losses
trainable_variables
regularization_losses
	variables
enon_trainable_variables
fmetrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?

=kernel
>bias
gregularization_losses
htrainable_variables
i	variables
j	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
kregularization_losses
ltrainable_variables
m	variables
n	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?

?kernel
@bias
oregularization_losses
ptrainable_variables
q	variables
r	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?

Akernel
Bbias
sregularization_losses
ttrainable_variables
u	variables
v	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?

Ckernel
Dbias
wregularization_losses
xtrainable_variables
y	variables
z	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
{regularization_losses
|trainable_variables
}	variables
~	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
X
=0
>1
?2
@3
A4
B5
C6
D7"
trackable_list_wrapper
 "
trackable_list_wrapper
X
=0
>1
?2
@3
A4
B5
C6
D7"
trackable_list_wrapper
?
layer_metrics
?layers
 ?layer_regularization_losses
trainable_variables
regularization_losses
 	variables
?non_trainable_variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
?non_trainable_variables
 ?layer_regularization_losses
"regularization_losses
#trainable_variables
$	variables
?layers
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:  (2total
:  (2count
.
&0
'1"
trackable_list_wrapper
-
(	variables"
_generic_user_object
:  (2total
:  (2count
.
*0
+1"
trackable_list_wrapper
-
,	variables"
_generic_user_object
:  (2total
:  (2count
.
.0
/1"
trackable_list_wrapper
-
0	variables"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
):' 2conv2d_4/kernel
: 2conv2d_4/bias
):' @2conv2d_5/kernel
:@2conv2d_5/bias
!:	?2dense_4/kernel
:2dense_4/bias
!:	?2dense_5/kernel
:?2dense_5/bias
3:1@ 2conv2d_transpose_6/kernel
%:#@2conv2d_transpose_6/bias
3:1 @2conv2d_transpose_7/kernel
%:# 2conv2d_transpose_7/bias
3:1 2conv2d_transpose_8/kernel
%:#2conv2d_transpose_8/bias
Z

total_loss
reconstruction_loss
latent_loss"
trackable_dict_wrapper
5
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
J
&0
'1
*2
+3
.4
/5"
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
?non_trainable_variables
 ?layer_regularization_losses
Jregularization_losses
Ktrainable_variables
L	variables
?layers
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
70
81"
trackable_list_wrapper
.
70
81"
trackable_list_wrapper
?
?layer_metrics
?non_trainable_variables
 ?layer_regularization_losses
Nregularization_losses
Otrainable_variables
P	variables
?layers
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
90
:1"
trackable_list_wrapper
.
90
:1"
trackable_list_wrapper
?
?layer_metrics
?non_trainable_variables
 ?layer_regularization_losses
Rregularization_losses
Strainable_variables
T	variables
?layers
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
?non_trainable_variables
 ?layer_regularization_losses
Vregularization_losses
Wtrainable_variables
X	variables
?layers
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
;0
<1"
trackable_list_wrapper
.
;0
<1"
trackable_list_wrapper
?
?layer_metrics
?non_trainable_variables
 ?layer_regularization_losses
Zregularization_losses
[trainable_variables
\	variables
?layers
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
?non_trainable_variables
 ?layer_regularization_losses
^regularization_losses
_trainable_variables
`	variables
?layers
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
J
0
1
2
3
4
5"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
=0
>1"
trackable_list_wrapper
.
=0
>1"
trackable_list_wrapper
?
?layer_metrics
?non_trainable_variables
 ?layer_regularization_losses
gregularization_losses
htrainable_variables
i	variables
?layers
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
?non_trainable_variables
 ?layer_regularization_losses
kregularization_losses
ltrainable_variables
m	variables
?layers
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
?0
@1"
trackable_list_wrapper
.
?0
@1"
trackable_list_wrapper
?
?layer_metrics
?non_trainable_variables
 ?layer_regularization_losses
oregularization_losses
ptrainable_variables
q	variables
?layers
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
A0
B1"
trackable_list_wrapper
.
A0
B1"
trackable_list_wrapper
?
?layer_metrics
?non_trainable_variables
 ?layer_regularization_losses
sregularization_losses
ttrainable_variables
u	variables
?layers
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
C0
D1"
trackable_list_wrapper
.
C0
D1"
trackable_list_wrapper
?
?layer_metrics
?non_trainable_variables
 ?layer_regularization_losses
wregularization_losses
xtrainable_variables
y	variables
?layers
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
?non_trainable_variables
 ?layer_regularization_losses
{regularization_losses
|trainable_variables
}	variables
?layers
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
J
0
1
2
3
4
5"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.:, 2Adam/conv2d_4/kernel/m
 : 2Adam/conv2d_4/bias/m
.:, @2Adam/conv2d_5/kernel/m
 :@2Adam/conv2d_5/bias/m
&:$	?2Adam/dense_4/kernel/m
:2Adam/dense_4/bias/m
&:$	?2Adam/dense_5/kernel/m
 :?2Adam/dense_5/bias/m
8:6@ 2 Adam/conv2d_transpose_6/kernel/m
*:(@2Adam/conv2d_transpose_6/bias/m
8:6 @2 Adam/conv2d_transpose_7/kernel/m
*:( 2Adam/conv2d_transpose_7/bias/m
8:6 2 Adam/conv2d_transpose_8/kernel/m
*:(2Adam/conv2d_transpose_8/bias/m
.:, 2Adam/conv2d_4/kernel/v
 : 2Adam/conv2d_4/bias/v
.:, @2Adam/conv2d_5/kernel/v
 :@2Adam/conv2d_5/bias/v
&:$	?2Adam/dense_4/kernel/v
:2Adam/dense_4/bias/v
&:$	?2Adam/dense_5/kernel/v
 :?2Adam/dense_5/bias/v
8:6@ 2 Adam/conv2d_transpose_6/kernel/v
*:(@2Adam/conv2d_transpose_6/bias/v
8:6 @2 Adam/conv2d_transpose_7/kernel/v
*:( 2Adam/conv2d_transpose_7/bias/v
8:6 2 Adam/conv2d_transpose_8/kernel/v
*:(2Adam/conv2d_transpose_8/bias/v
?B?
 __inference__wrapped_model_44958input_1"?
???
FullArgSpec
args? 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
T__inference_variational_autoencoder_2_layer_call_and_return_conditional_losses_46346
T__inference_variational_autoencoder_2_layer_call_and_return_conditional_losses_46501
T__inference_variational_autoencoder_2_layer_call_and_return_conditional_losses_46136
T__inference_variational_autoencoder_2_layer_call_and_return_conditional_losses_46171?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
9__inference_variational_autoencoder_2_layer_call_fn_45929
9__inference_variational_autoencoder_2_layer_call_fn_46534
9__inference_variational_autoencoder_2_layer_call_fn_46567
9__inference_variational_autoencoder_2_layer_call_fn_46101?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
G__inference_sequential_4_layer_call_and_return_conditional_losses_46612
G__inference_sequential_4_layer_call_and_return_conditional_losses_46657
G__inference_sequential_4_layer_call_and_return_conditional_losses_45213
G__inference_sequential_4_layer_call_and_return_conditional_losses_45235?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
,__inference_sequential_4_layer_call_fn_45070
,__inference_sequential_4_layer_call_fn_46674
,__inference_sequential_4_layer_call_fn_46691
,__inference_sequential_4_layer_call_fn_45191?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
G__inference_sequential_5_layer_call_and_return_conditional_losses_46782
G__inference_sequential_5_layer_call_and_return_conditional_losses_46873
G__inference_sequential_5_layer_call_and_return_conditional_losses_45826
G__inference_sequential_5_layer_call_and_return_conditional_losses_45852?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
,__inference_sequential_5_layer_call_fn_45662
,__inference_sequential_5_layer_call_fn_46894
,__inference_sequential_5_layer_call_fn_46915
,__inference_sequential_5_layer_call_fn_45800?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
R__inference_normal_sampling_layer_2_layer_call_and_return_conditional_losses_46921
R__inference_normal_sampling_layer_2_layer_call_and_return_conditional_losses_46948?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
7__inference_normal_sampling_layer_2_layer_call_fn_46953
7__inference_normal_sampling_layer_2_layer_call_fn_46958?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
#__inference_signature_wrapper_46212input_1"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_reshape_8_layer_call_and_return_conditional_losses_46972?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_reshape_8_layer_call_fn_46977?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_conv2d_4_layer_call_and_return_conditional_losses_46988?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_conv2d_4_layer_call_fn_46997?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_conv2d_5_layer_call_and_return_conditional_losses_47008?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_conv2d_5_layer_call_fn_47017?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_flatten_2_layer_call_and_return_conditional_losses_47023?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_flatten_2_layer_call_fn_47028?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_dense_4_layer_call_and_return_conditional_losses_47038?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
'__inference_dense_4_layer_call_fn_47047?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_reshape_9_layer_call_and_return_conditional_losses_47060?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_reshape_9_layer_call_fn_47065?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_dense_5_layer_call_and_return_conditional_losses_47076?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
'__inference_dense_5_layer_call_fn_47085?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_reshape_10_layer_call_and_return_conditional_losses_47099?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_reshape_10_layer_call_fn_47104?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
M__inference_conv2d_transpose_6_layer_call_and_return_conditional_losses_47138
M__inference_conv2d_transpose_6_layer_call_and_return_conditional_losses_47162?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
2__inference_conv2d_transpose_6_layer_call_fn_47171
2__inference_conv2d_transpose_6_layer_call_fn_47180?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
M__inference_conv2d_transpose_7_layer_call_and_return_conditional_losses_47214
M__inference_conv2d_transpose_7_layer_call_and_return_conditional_losses_47238?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
2__inference_conv2d_transpose_7_layer_call_fn_47247
2__inference_conv2d_transpose_7_layer_call_fn_47256?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
M__inference_conv2d_transpose_8_layer_call_and_return_conditional_losses_47290
M__inference_conv2d_transpose_8_layer_call_and_return_conditional_losses_47314?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
2__inference_conv2d_transpose_8_layer_call_fn_47323
2__inference_conv2d_transpose_8_layer_call_fn_47332?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_reshape_11_layer_call_and_return_conditional_losses_47346?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_reshape_11_layer_call_fn_47351?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
 __inference__wrapped_model_44958?789:;<=>?@ABCD8?5
.?+
)?&
input_1?????????
? ";?8
6
output_1*?'
output_1??????????
C__inference_conv2d_4_layer_call_and_return_conditional_losses_46988l787?4
-?*
(?%
inputs?????????
? "-?*
#? 
0????????? 
? ?
(__inference_conv2d_4_layer_call_fn_46997_787?4
-?*
(?%
inputs?????????
? " ?????????? ?
C__inference_conv2d_5_layer_call_and_return_conditional_losses_47008l9:7?4
-?*
(?%
inputs????????? 
? "-?*
#? 
0?????????@
? ?
(__inference_conv2d_5_layer_call_fn_47017_9:7?4
-?*
(?%
inputs????????? 
? " ??????????@?
M__inference_conv2d_transpose_6_layer_call_and_return_conditional_losses_47138??@I?F
??<
:?7
inputs+??????????????????????????? 
? "??<
5?2
0+???????????????????????????@
? ?
M__inference_conv2d_transpose_6_layer_call_and_return_conditional_losses_47162l?@7?4
-?*
(?%
inputs????????? 
? "-?*
#? 
0?????????@
? ?
2__inference_conv2d_transpose_6_layer_call_fn_47171??@I?F
??<
:?7
inputs+??????????????????????????? 
? "2?/+???????????????????????????@?
2__inference_conv2d_transpose_6_layer_call_fn_47180_?@7?4
-?*
(?%
inputs????????? 
? " ??????????@?
M__inference_conv2d_transpose_7_layer_call_and_return_conditional_losses_47214?ABI?F
??<
:?7
inputs+???????????????????????????@
? "??<
5?2
0+??????????????????????????? 
? ?
M__inference_conv2d_transpose_7_layer_call_and_return_conditional_losses_47238lAB7?4
-?*
(?%
inputs?????????@
? "-?*
#? 
0????????? 
? ?
2__inference_conv2d_transpose_7_layer_call_fn_47247?ABI?F
??<
:?7
inputs+???????????????????????????@
? "2?/+??????????????????????????? ?
2__inference_conv2d_transpose_7_layer_call_fn_47256_AB7?4
-?*
(?%
inputs?????????@
? " ?????????? ?
M__inference_conv2d_transpose_8_layer_call_and_return_conditional_losses_47290?CDI?F
??<
:?7
inputs+??????????????????????????? 
? "??<
5?2
0+???????????????????????????
? ?
M__inference_conv2d_transpose_8_layer_call_and_return_conditional_losses_47314lCD7?4
-?*
(?%
inputs????????? 
? "-?*
#? 
0?????????
? ?
2__inference_conv2d_transpose_8_layer_call_fn_47323?CDI?F
??<
:?7
inputs+??????????????????????????? 
? "2?/+????????????????????????????
2__inference_conv2d_transpose_8_layer_call_fn_47332_CD7?4
-?*
(?%
inputs????????? 
? " ???????????
B__inference_dense_4_layer_call_and_return_conditional_losses_47038];<0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????
? {
'__inference_dense_4_layer_call_fn_47047P;<0?-
&?#
!?
inputs??????????
? "???????????
B__inference_dense_5_layer_call_and_return_conditional_losses_47076]=>/?,
%?"
 ?
inputs?????????
? "&?#
?
0??????????
? {
'__inference_dense_5_layer_call_fn_47085P=>/?,
%?"
 ?
inputs?????????
? "????????????
D__inference_flatten_2_layer_call_and_return_conditional_losses_47023a7?4
-?*
(?%
inputs?????????@
? "&?#
?
0??????????
? ?
)__inference_flatten_2_layer_call_fn_47028T7?4
-?*
(?%
inputs?????????@
? "????????????
R__inference_normal_sampling_layer_2_layer_call_and_return_conditional_losses_46921`7?4
-?*
$?!
inputs?????????
p 
? "%?"
?
0?????????
? ?
R__inference_normal_sampling_layer_2_layer_call_and_return_conditional_losses_46948`7?4
-?*
$?!
inputs?????????
p
? "%?"
?
0?????????
? ?
7__inference_normal_sampling_layer_2_layer_call_fn_46953S7?4
-?*
$?!
inputs?????????
p 
? "???????????
7__inference_normal_sampling_layer_2_layer_call_fn_46958S7?4
-?*
$?!
inputs?????????
p
? "???????????
E__inference_reshape_10_layer_call_and_return_conditional_losses_47099a0?-
&?#
!?
inputs??????????
? "-?*
#? 
0????????? 
? ?
*__inference_reshape_10_layer_call_fn_47104T0?-
&?#
!?
inputs??????????
? " ?????????? ?
E__inference_reshape_11_layer_call_and_return_conditional_losses_47346h7?4
-?*
(?%
inputs?????????
? "-?*
#? 
0?????????
? ?
*__inference_reshape_11_layer_call_fn_47351[7?4
-?*
(?%
inputs?????????
? " ???????????
D__inference_reshape_8_layer_call_and_return_conditional_losses_46972h7?4
-?*
(?%
inputs?????????
? "-?*
#? 
0?????????
? ?
)__inference_reshape_8_layer_call_fn_46977[7?4
-?*
(?%
inputs?????????
? " ???????????
D__inference_reshape_9_layer_call_and_return_conditional_losses_47060\/?,
%?"
 ?
inputs?????????
? ")?&
?
0?????????
? |
)__inference_reshape_9_layer_call_fn_47065O/?,
%?"
 ?
inputs?????????
? "???????????
G__inference_sequential_4_layer_call_and_return_conditional_losses_45213u789:;<@?=
6?3
)?&
input_9?????????
p 

 
? ")?&
?
0?????????
? ?
G__inference_sequential_4_layer_call_and_return_conditional_losses_45235u789:;<@?=
6?3
)?&
input_9?????????
p

 
? ")?&
?
0?????????
? ?
G__inference_sequential_4_layer_call_and_return_conditional_losses_46612t789:;<??<
5?2
(?%
inputs?????????
p 

 
? ")?&
?
0?????????
? ?
G__inference_sequential_4_layer_call_and_return_conditional_losses_46657t789:;<??<
5?2
(?%
inputs?????????
p

 
? ")?&
?
0?????????
? ?
,__inference_sequential_4_layer_call_fn_45070h789:;<@?=
6?3
)?&
input_9?????????
p 

 
? "???????????
,__inference_sequential_4_layer_call_fn_45191h789:;<@?=
6?3
)?&
input_9?????????
p

 
? "???????????
,__inference_sequential_4_layer_call_fn_46674g789:;<??<
5?2
(?%
inputs?????????
p 

 
? "???????????
,__inference_sequential_4_layer_call_fn_46691g789:;<??<
5?2
(?%
inputs?????????
p

 
? "???????????
G__inference_sequential_5_layer_call_and_return_conditional_losses_45826t=>?@ABCD9?6
/?,
"?
input_10?????????
p 

 
? "-?*
#? 
0?????????
? ?
G__inference_sequential_5_layer_call_and_return_conditional_losses_45852t=>?@ABCD9?6
/?,
"?
input_10?????????
p

 
? "-?*
#? 
0?????????
? ?
G__inference_sequential_5_layer_call_and_return_conditional_losses_46782r=>?@ABCD7?4
-?*
 ?
inputs?????????
p 

 
? "-?*
#? 
0?????????
? ?
G__inference_sequential_5_layer_call_and_return_conditional_losses_46873r=>?@ABCD7?4
-?*
 ?
inputs?????????
p

 
? "-?*
#? 
0?????????
? ?
,__inference_sequential_5_layer_call_fn_45662g=>?@ABCD9?6
/?,
"?
input_10?????????
p 

 
? " ???????????
,__inference_sequential_5_layer_call_fn_45800g=>?@ABCD9?6
/?,
"?
input_10?????????
p

 
? " ???????????
,__inference_sequential_5_layer_call_fn_46894e=>?@ABCD7?4
-?*
 ?
inputs?????????
p 

 
? " ???????????
,__inference_sequential_5_layer_call_fn_46915e=>?@ABCD7?4
-?*
 ?
inputs?????????
p

 
? " ???????????
#__inference_signature_wrapper_46212?789:;<=>?@ABCDC?@
? 
9?6
4
input_1)?&
input_1?????????";?8
6
output_1*?'
output_1??????????
T__inference_variational_autoencoder_2_layer_call_and_return_conditional_losses_46136}789:;<=>?@ABCD<?9
2?/
)?&
input_1?????????
p 
? "-?*
#? 
0?????????
? ?
T__inference_variational_autoencoder_2_layer_call_and_return_conditional_losses_46171}789:;<=>?@ABCD<?9
2?/
)?&
input_1?????????
p
? "-?*
#? 
0?????????
? ?
T__inference_variational_autoencoder_2_layer_call_and_return_conditional_losses_46346|789:;<=>?@ABCD;?8
1?.
(?%
inputs?????????
p 
? "-?*
#? 
0?????????
? ?
T__inference_variational_autoencoder_2_layer_call_and_return_conditional_losses_46501|789:;<=>?@ABCD;?8
1?.
(?%
inputs?????????
p
? "-?*
#? 
0?????????
? ?
9__inference_variational_autoencoder_2_layer_call_fn_45929p789:;<=>?@ABCD<?9
2?/
)?&
input_1?????????
p 
? " ???????????
9__inference_variational_autoencoder_2_layer_call_fn_46101p789:;<=>?@ABCD<?9
2?/
)?&
input_1?????????
p
? " ???????????
9__inference_variational_autoencoder_2_layer_call_fn_46534o789:;<=>?@ABCD;?8
1?.
(?%
inputs?????????
p 
? " ???????????
9__inference_variational_autoencoder_2_layer_call_fn_46567o789:;<=>?@ABCD;?8
1?.
(?%
inputs?????????
p
? " ??????????