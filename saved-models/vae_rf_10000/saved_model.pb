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
conv2d_8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: * 
shared_nameconv2d_8/kernel
{
#conv2d_8/kernel/Read/ReadVariableOpReadVariableOpconv2d_8/kernel*&
_output_shapes
: *
dtype0
r
conv2d_8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_8/bias
k
!conv2d_8/bias/Read/ReadVariableOpReadVariableOpconv2d_8/bias*
_output_shapes
: *
dtype0
?
conv2d_9/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @* 
shared_nameconv2d_9/kernel
{
#conv2d_9/kernel/Read/ReadVariableOpReadVariableOpconv2d_9/kernel*&
_output_shapes
: @*
dtype0
r
conv2d_9/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_9/bias
k
!conv2d_9/bias/Read/ReadVariableOpReadVariableOpconv2d_9/bias*
_output_shapes
:@*
dtype0
y
dense_8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*
shared_namedense_8/kernel
r
"dense_8/kernel/Read/ReadVariableOpReadVariableOpdense_8/kernel*
_output_shapes
:	?*
dtype0
p
dense_8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_8/bias
i
 dense_8/bias/Read/ReadVariableOpReadVariableOpdense_8/bias*
_output_shapes
:*
dtype0
y
dense_9/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*
shared_namedense_9/kernel
r
"dense_9/kernel/Read/ReadVariableOpReadVariableOpdense_9/kernel*
_output_shapes
:	?*
dtype0
q
dense_9/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_9/bias
j
 dense_9/bias/Read/ReadVariableOpReadVariableOpdense_9/bias*
_output_shapes	
:?*
dtype0
?
conv2d_transpose_12/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@ *+
shared_nameconv2d_transpose_12/kernel
?
.conv2d_transpose_12/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_12/kernel*&
_output_shapes
:@ *
dtype0
?
conv2d_transpose_12/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*)
shared_nameconv2d_transpose_12/bias
?
,conv2d_transpose_12/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose_12/bias*
_output_shapes
:@*
dtype0
?
conv2d_transpose_13/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*+
shared_nameconv2d_transpose_13/kernel
?
.conv2d_transpose_13/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_13/kernel*&
_output_shapes
: @*
dtype0
?
conv2d_transpose_13/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *)
shared_nameconv2d_transpose_13/bias
?
,conv2d_transpose_13/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose_13/bias*
_output_shapes
: *
dtype0
?
conv2d_transpose_14/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *+
shared_nameconv2d_transpose_14/kernel
?
.conv2d_transpose_14/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_14/kernel*&
_output_shapes
: *
dtype0
?
conv2d_transpose_14/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameconv2d_transpose_14/bias
?
,conv2d_transpose_14/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose_14/bias*
_output_shapes
:*
dtype0
?
Adam/conv2d_8/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_nameAdam/conv2d_8/kernel/m
?
*Adam/conv2d_8/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_8/kernel/m*&
_output_shapes
: *
dtype0
?
Adam/conv2d_8/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/conv2d_8/bias/m
y
(Adam/conv2d_8/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_8/bias/m*
_output_shapes
: *
dtype0
?
Adam/conv2d_9/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*'
shared_nameAdam/conv2d_9/kernel/m
?
*Adam/conv2d_9/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_9/kernel/m*&
_output_shapes
: @*
dtype0
?
Adam/conv2d_9/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/conv2d_9/bias/m
y
(Adam/conv2d_9/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_9/bias/m*
_output_shapes
:@*
dtype0
?
Adam/dense_8/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*&
shared_nameAdam/dense_8/kernel/m
?
)Adam/dense_8/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_8/kernel/m*
_output_shapes
:	?*
dtype0
~
Adam/dense_8/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_8/bias/m
w
'Adam/dense_8/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_8/bias/m*
_output_shapes
:*
dtype0
?
Adam/dense_9/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*&
shared_nameAdam/dense_9/kernel/m
?
)Adam/dense_9/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_9/kernel/m*
_output_shapes
:	?*
dtype0

Adam/dense_9/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*$
shared_nameAdam/dense_9/bias/m
x
'Adam/dense_9/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_9/bias/m*
_output_shapes	
:?*
dtype0
?
!Adam/conv2d_transpose_12/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@ *2
shared_name#!Adam/conv2d_transpose_12/kernel/m
?
5Adam/conv2d_transpose_12/kernel/m/Read/ReadVariableOpReadVariableOp!Adam/conv2d_transpose_12/kernel/m*&
_output_shapes
:@ *
dtype0
?
Adam/conv2d_transpose_12/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*0
shared_name!Adam/conv2d_transpose_12/bias/m
?
3Adam/conv2d_transpose_12/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_transpose_12/bias/m*
_output_shapes
:@*
dtype0
?
!Adam/conv2d_transpose_13/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*2
shared_name#!Adam/conv2d_transpose_13/kernel/m
?
5Adam/conv2d_transpose_13/kernel/m/Read/ReadVariableOpReadVariableOp!Adam/conv2d_transpose_13/kernel/m*&
_output_shapes
: @*
dtype0
?
Adam/conv2d_transpose_13/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *0
shared_name!Adam/conv2d_transpose_13/bias/m
?
3Adam/conv2d_transpose_13/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_transpose_13/bias/m*
_output_shapes
: *
dtype0
?
!Adam/conv2d_transpose_14/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!Adam/conv2d_transpose_14/kernel/m
?
5Adam/conv2d_transpose_14/kernel/m/Read/ReadVariableOpReadVariableOp!Adam/conv2d_transpose_14/kernel/m*&
_output_shapes
: *
dtype0
?
Adam/conv2d_transpose_14/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!Adam/conv2d_transpose_14/bias/m
?
3Adam/conv2d_transpose_14/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_transpose_14/bias/m*
_output_shapes
:*
dtype0
?
Adam/conv2d_8/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_nameAdam/conv2d_8/kernel/v
?
*Adam/conv2d_8/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_8/kernel/v*&
_output_shapes
: *
dtype0
?
Adam/conv2d_8/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/conv2d_8/bias/v
y
(Adam/conv2d_8/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_8/bias/v*
_output_shapes
: *
dtype0
?
Adam/conv2d_9/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*'
shared_nameAdam/conv2d_9/kernel/v
?
*Adam/conv2d_9/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_9/kernel/v*&
_output_shapes
: @*
dtype0
?
Adam/conv2d_9/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/conv2d_9/bias/v
y
(Adam/conv2d_9/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_9/bias/v*
_output_shapes
:@*
dtype0
?
Adam/dense_8/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*&
shared_nameAdam/dense_8/kernel/v
?
)Adam/dense_8/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_8/kernel/v*
_output_shapes
:	?*
dtype0
~
Adam/dense_8/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_8/bias/v
w
'Adam/dense_8/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_8/bias/v*
_output_shapes
:*
dtype0
?
Adam/dense_9/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*&
shared_nameAdam/dense_9/kernel/v
?
)Adam/dense_9/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_9/kernel/v*
_output_shapes
:	?*
dtype0

Adam/dense_9/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*$
shared_nameAdam/dense_9/bias/v
x
'Adam/dense_9/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_9/bias/v*
_output_shapes	
:?*
dtype0
?
!Adam/conv2d_transpose_12/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@ *2
shared_name#!Adam/conv2d_transpose_12/kernel/v
?
5Adam/conv2d_transpose_12/kernel/v/Read/ReadVariableOpReadVariableOp!Adam/conv2d_transpose_12/kernel/v*&
_output_shapes
:@ *
dtype0
?
Adam/conv2d_transpose_12/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*0
shared_name!Adam/conv2d_transpose_12/bias/v
?
3Adam/conv2d_transpose_12/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_transpose_12/bias/v*
_output_shapes
:@*
dtype0
?
!Adam/conv2d_transpose_13/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*2
shared_name#!Adam/conv2d_transpose_13/kernel/v
?
5Adam/conv2d_transpose_13/kernel/v/Read/ReadVariableOpReadVariableOp!Adam/conv2d_transpose_13/kernel/v*&
_output_shapes
: @*
dtype0
?
Adam/conv2d_transpose_13/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *0
shared_name!Adam/conv2d_transpose_13/bias/v
?
3Adam/conv2d_transpose_13/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_transpose_13/bias/v*
_output_shapes
: *
dtype0
?
!Adam/conv2d_transpose_14/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!Adam/conv2d_transpose_14/kernel/v
?
5Adam/conv2d_transpose_14/kernel/v/Read/ReadVariableOpReadVariableOp!Adam/conv2d_transpose_14/kernel/v*&
_output_shapes
: *
dtype0
?
Adam/conv2d_transpose_14/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!Adam/conv2d_transpose_14/bias/v
?
3Adam/conv2d_transpose_14/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_transpose_14/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
?a
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
VARIABLE_VALUEconv2d_8/kernel0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEconv2d_8/bias0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEconv2d_9/kernel0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEconv2d_9/bias0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEdense_8/kernel0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUEdense_8/bias0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEdense_9/kernel0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUEdense_9/bias0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUE
`^
VARIABLE_VALUEconv2d_transpose_12/kernel0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUEconv2d_transpose_12/bias0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUEconv2d_transpose_13/kernel1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUEconv2d_transpose_13/bias1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUEconv2d_transpose_14/kernel1trainable_variables/12/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUEconv2d_transpose_14/bias1trainable_variables/13/.ATTRIBUTES/VARIABLE_VALUE
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
VARIABLE_VALUEAdam/conv2d_8/kernel/mLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/conv2d_8/bias/mLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/conv2d_9/kernel/mLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/conv2d_9/bias/mLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense_8/kernel/mLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUEAdam/dense_8/bias/mLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense_9/kernel/mLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUEAdam/dense_9/bias/mLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE!Adam/conv2d_transpose_12/kernel/mLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUEAdam/conv2d_transpose_12/bias/mLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE!Adam/conv2d_transpose_13/kernel/mMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/conv2d_transpose_13/bias/mMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE!Adam/conv2d_transpose_14/kernel/mMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/conv2d_transpose_14/bias/mMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/conv2d_8/kernel/vLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/conv2d_8/bias/vLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/conv2d_9/kernel/vLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/conv2d_9/bias/vLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense_8/kernel/vLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUEAdam/dense_8/bias/vLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense_9/kernel/vLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUEAdam/dense_9/bias/vLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE!Adam/conv2d_transpose_12/kernel/vLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUEAdam/conv2d_transpose_12/bias/vLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE!Adam/conv2d_transpose_13/kernel/vMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/conv2d_transpose_13/bias/vMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE!Adam/conv2d_transpose_14/kernel/vMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/conv2d_transpose_14/bias/vMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
serving_default_input_1Placeholder*/
_output_shapes
:?????????*
dtype0*$
shape:?????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1conv2d_8/kernelconv2d_8/biasconv2d_9/kernelconv2d_9/biasdense_8/kerneldense_8/biasdense_9/kerneldense_9/biasconv2d_transpose_12/kernelconv2d_transpose_12/biasconv2d_transpose_13/kernelconv2d_transpose_13/biasconv2d_transpose_14/kernelconv2d_transpose_14/bias*
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
#__inference_signature_wrapper_78168
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenametotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal_2/Read/ReadVariableOpcount_2/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp#conv2d_8/kernel/Read/ReadVariableOp!conv2d_8/bias/Read/ReadVariableOp#conv2d_9/kernel/Read/ReadVariableOp!conv2d_9/bias/Read/ReadVariableOp"dense_8/kernel/Read/ReadVariableOp dense_8/bias/Read/ReadVariableOp"dense_9/kernel/Read/ReadVariableOp dense_9/bias/Read/ReadVariableOp.conv2d_transpose_12/kernel/Read/ReadVariableOp,conv2d_transpose_12/bias/Read/ReadVariableOp.conv2d_transpose_13/kernel/Read/ReadVariableOp,conv2d_transpose_13/bias/Read/ReadVariableOp.conv2d_transpose_14/kernel/Read/ReadVariableOp,conv2d_transpose_14/bias/Read/ReadVariableOp*Adam/conv2d_8/kernel/m/Read/ReadVariableOp(Adam/conv2d_8/bias/m/Read/ReadVariableOp*Adam/conv2d_9/kernel/m/Read/ReadVariableOp(Adam/conv2d_9/bias/m/Read/ReadVariableOp)Adam/dense_8/kernel/m/Read/ReadVariableOp'Adam/dense_8/bias/m/Read/ReadVariableOp)Adam/dense_9/kernel/m/Read/ReadVariableOp'Adam/dense_9/bias/m/Read/ReadVariableOp5Adam/conv2d_transpose_12/kernel/m/Read/ReadVariableOp3Adam/conv2d_transpose_12/bias/m/Read/ReadVariableOp5Adam/conv2d_transpose_13/kernel/m/Read/ReadVariableOp3Adam/conv2d_transpose_13/bias/m/Read/ReadVariableOp5Adam/conv2d_transpose_14/kernel/m/Read/ReadVariableOp3Adam/conv2d_transpose_14/bias/m/Read/ReadVariableOp*Adam/conv2d_8/kernel/v/Read/ReadVariableOp(Adam/conv2d_8/bias/v/Read/ReadVariableOp*Adam/conv2d_9/kernel/v/Read/ReadVariableOp(Adam/conv2d_9/bias/v/Read/ReadVariableOp)Adam/dense_8/kernel/v/Read/ReadVariableOp'Adam/dense_8/bias/v/Read/ReadVariableOp)Adam/dense_9/kernel/v/Read/ReadVariableOp'Adam/dense_9/bias/v/Read/ReadVariableOp5Adam/conv2d_transpose_12/kernel/v/Read/ReadVariableOp3Adam/conv2d_transpose_12/bias/v/Read/ReadVariableOp5Adam/conv2d_transpose_13/kernel/v/Read/ReadVariableOp3Adam/conv2d_transpose_13/bias/v/Read/ReadVariableOp5Adam/conv2d_transpose_14/kernel/v/Read/ReadVariableOp3Adam/conv2d_transpose_14/bias/v/Read/ReadVariableOpConst*B
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
__inference__traced_save_79489
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenametotalcounttotal_1count_1total_2count_2	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_rateconv2d_8/kernelconv2d_8/biasconv2d_9/kernelconv2d_9/biasdense_8/kerneldense_8/biasdense_9/kerneldense_9/biasconv2d_transpose_12/kernelconv2d_transpose_12/biasconv2d_transpose_13/kernelconv2d_transpose_13/biasconv2d_transpose_14/kernelconv2d_transpose_14/biasAdam/conv2d_8/kernel/mAdam/conv2d_8/bias/mAdam/conv2d_9/kernel/mAdam/conv2d_9/bias/mAdam/dense_8/kernel/mAdam/dense_8/bias/mAdam/dense_9/kernel/mAdam/dense_9/bias/m!Adam/conv2d_transpose_12/kernel/mAdam/conv2d_transpose_12/bias/m!Adam/conv2d_transpose_13/kernel/mAdam/conv2d_transpose_13/bias/m!Adam/conv2d_transpose_14/kernel/mAdam/conv2d_transpose_14/bias/mAdam/conv2d_8/kernel/vAdam/conv2d_8/bias/vAdam/conv2d_9/kernel/vAdam/conv2d_9/bias/vAdam/dense_8/kernel/vAdam/dense_8/bias/vAdam/dense_9/kernel/vAdam/dense_9/bias/v!Adam/conv2d_transpose_12/kernel/vAdam/conv2d_transpose_12/bias/v!Adam/conv2d_transpose_13/kernel/vAdam/conv2d_transpose_13/bias/v!Adam/conv2d_transpose_14/kernel/vAdam/conv2d_transpose_14/bias/v*A
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
!__inference__traced_restore_79658??
?
?
C__inference_conv2d_9_layer_call_and_return_conditional_losses_76965

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

?
B__inference_dense_8_layer_call_and_return_conditional_losses_76989

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
?&
?
N__inference_conv2d_transpose_12_layer_call_and_return_conditional_losses_77229

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
?
?
N__inference_conv2d_transpose_12_layer_call_and_return_conditional_losses_77518

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
?
?
N__inference_conv2d_transpose_14_layer_call_and_return_conditional_losses_77576

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
?
?
N__inference_conv2d_transpose_13_layer_call_and_return_conditional_losses_79194

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
?
9__inference_variational_autoencoder_4_layer_call_fn_77885
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
T__inference_variational_autoencoder_4_layer_call_and_return_conditional_losses_778542
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
?&
?
N__inference_conv2d_transpose_13_layer_call_and_return_conditional_losses_77317

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
?&
?
N__inference_conv2d_transpose_14_layer_call_and_return_conditional_losses_79246

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
?
F
*__inference_reshape_18_layer_call_fn_79060

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
E__inference_reshape_18_layer_call_and_return_conditional_losses_774932
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
a
E__inference_reshape_16_layer_call_and_return_conditional_losses_76935

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
(__inference_conv2d_9_layer_call_fn_78973

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
C__inference_conv2d_9_layer_call_and_return_conditional_losses_769652
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
a
E__inference_reshape_19_layer_call_and_return_conditional_losses_77596

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
(__inference_conv2d_8_layer_call_fn_78953

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
C__inference_conv2d_8_layer_call_and_return_conditional_losses_769482
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
?
?
3__inference_conv2d_transpose_13_layer_call_fn_79212

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
GPU2*0J 8? *W
fRRP
N__inference_conv2d_transpose_13_layer_call_and_return_conditional_losses_775472
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
?	
?
,__inference_sequential_8_layer_call_fn_78647

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
G__inference_sequential_8_layer_call_and_return_conditional_losses_771152
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
?
F
*__inference_reshape_16_layer_call_fn_78933

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
E__inference_reshape_16_layer_call_and_return_conditional_losses_769352
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
a
E__inference_reshape_17_layer_call_and_return_conditional_losses_79016

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
?
p
7__inference_normal_sampling_layer_4_layer_call_fn_78914

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
R__inference_normal_sampling_layer_4_layer_call_and_return_conditional_losses_779202
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
E__inference_reshape_16_layer_call_and_return_conditional_losses_78928

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
?
?
'__inference_dense_9_layer_call_fn_79041

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
B__inference_dense_9_layer_call_and_return_conditional_losses_774732
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
?:
?
G__inference_sequential_8_layer_call_and_return_conditional_losses_78568

inputsA
'conv2d_8_conv2d_readvariableop_resource: 6
(conv2d_8_biasadd_readvariableop_resource: A
'conv2d_9_conv2d_readvariableop_resource: @6
(conv2d_9_biasadd_readvariableop_resource:@9
&dense_8_matmul_readvariableop_resource:	?5
'dense_8_biasadd_readvariableop_resource:
identity??conv2d_8/BiasAdd/ReadVariableOp?conv2d_8/Conv2D/ReadVariableOp?conv2d_9/BiasAdd/ReadVariableOp?conv2d_9/Conv2D/ReadVariableOp?dense_8/BiasAdd/ReadVariableOp?dense_8/MatMul/ReadVariableOpZ
reshape_16/ShapeShapeinputs*
T0*
_output_shapes
:2
reshape_16/Shape?
reshape_16/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
reshape_16/strided_slice/stack?
 reshape_16/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_16/strided_slice/stack_1?
 reshape_16/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_16/strided_slice/stack_2?
reshape_16/strided_sliceStridedSlicereshape_16/Shape:output:0'reshape_16/strided_slice/stack:output:0)reshape_16/strided_slice/stack_1:output:0)reshape_16/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_16/strided_slicez
reshape_16/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_16/Reshape/shape/1z
reshape_16/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_16/Reshape/shape/2z
reshape_16/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_16/Reshape/shape/3?
reshape_16/Reshape/shapePack!reshape_16/strided_slice:output:0#reshape_16/Reshape/shape/1:output:0#reshape_16/Reshape/shape/2:output:0#reshape_16/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
reshape_16/Reshape/shape?
reshape_16/ReshapeReshapeinputs!reshape_16/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????2
reshape_16/Reshape?
conv2d_8/Conv2D/ReadVariableOpReadVariableOp'conv2d_8_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02 
conv2d_8/Conv2D/ReadVariableOp?
conv2d_8/Conv2DConv2Dreshape_16/Reshape:output:0&conv2d_8/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
2
conv2d_8/Conv2D?
conv2d_8/BiasAdd/ReadVariableOpReadVariableOp(conv2d_8_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv2d_8/BiasAdd/ReadVariableOp?
conv2d_8/BiasAddBiasAddconv2d_8/Conv2D:output:0'conv2d_8/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2
conv2d_8/BiasAdd{
conv2d_8/ReluReluconv2d_8/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
conv2d_8/Relu?
conv2d_9/Conv2D/ReadVariableOpReadVariableOp'conv2d_9_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02 
conv2d_9/Conv2D/ReadVariableOp?
conv2d_9/Conv2DConv2Dconv2d_8/Relu:activations:0&conv2d_9/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
2
conv2d_9/Conv2D?
conv2d_9/BiasAdd/ReadVariableOpReadVariableOp(conv2d_9_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_9/BiasAdd/ReadVariableOp?
conv2d_9/BiasAddBiasAddconv2d_9/Conv2D:output:0'conv2d_9/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
conv2d_9/BiasAdd{
conv2d_9/ReluReluconv2d_9/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
conv2d_9/Relus
flatten_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"???? 	  2
flatten_4/Const?
flatten_4/ReshapeReshapeconv2d_9/Relu:activations:0flatten_4/Const:output:0*
T0*(
_output_shapes
:??????????2
flatten_4/Reshape?
dense_8/MatMul/ReadVariableOpReadVariableOp&dense_8_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
dense_8/MatMul/ReadVariableOp?
dense_8/MatMulMatMulflatten_4/Reshape:output:0%dense_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_8/MatMul?
dense_8/BiasAdd/ReadVariableOpReadVariableOp'dense_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_8/BiasAdd/ReadVariableOp?
dense_8/BiasAddBiasAdddense_8/MatMul:product:0&dense_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_8/BiasAddl
reshape_17/ShapeShapedense_8/BiasAdd:output:0*
T0*
_output_shapes
:2
reshape_17/Shape?
reshape_17/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
reshape_17/strided_slice/stack?
 reshape_17/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_17/strided_slice/stack_1?
 reshape_17/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_17/strided_slice/stack_2?
reshape_17/strided_sliceStridedSlicereshape_17/Shape:output:0'reshape_17/strided_slice/stack:output:0)reshape_17/strided_slice/stack_1:output:0)reshape_17/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_17/strided_slicez
reshape_17/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_17/Reshape/shape/1z
reshape_17/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_17/Reshape/shape/2?
reshape_17/Reshape/shapePack!reshape_17/strided_slice:output:0#reshape_17/Reshape/shape/1:output:0#reshape_17/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
reshape_17/Reshape/shape?
reshape_17/ReshapeReshapedense_8/BiasAdd:output:0!reshape_17/Reshape/shape:output:0*
T0*+
_output_shapes
:?????????2
reshape_17/Reshapez
IdentityIdentityreshape_17/Reshape:output:0^NoOp*
T0*+
_output_shapes
:?????????2

Identity?
NoOpNoOp ^conv2d_8/BiasAdd/ReadVariableOp^conv2d_8/Conv2D/ReadVariableOp ^conv2d_9/BiasAdd/ReadVariableOp^conv2d_9/Conv2D/ReadVariableOp^dense_8/BiasAdd/ReadVariableOp^dense_8/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : 2B
conv2d_8/BiasAdd/ReadVariableOpconv2d_8/BiasAdd/ReadVariableOp2@
conv2d_8/Conv2D/ReadVariableOpconv2d_8/Conv2D/ReadVariableOp2B
conv2d_9/BiasAdd/ReadVariableOpconv2d_9/BiasAdd/ReadVariableOp2@
conv2d_9/Conv2D/ReadVariableOpconv2d_9/Conv2D/ReadVariableOp2@
dense_8/BiasAdd/ReadVariableOpdense_8/BiasAdd/ReadVariableOp2>
dense_8/MatMul/ReadVariableOpdense_8/MatMul/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
9__inference_variational_autoencoder_4_layer_call_fn_78057
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
T__inference_variational_autoencoder_4_layer_call_and_return_conditional_losses_779932
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
?
n
R__inference_normal_sampling_layer_4_layer_call_and_return_conditional_losses_78877

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
?
`
D__inference_flatten_4_layer_call_and_return_conditional_losses_78979

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
?&
?
N__inference_conv2d_transpose_12_layer_call_and_return_conditional_losses_79094

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
?&
?
N__inference_conv2d_transpose_13_layer_call_and_return_conditional_losses_79170

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
?
?
#__inference_signature_wrapper_78168
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
 __inference__wrapped_model_769142
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
C__inference_conv2d_9_layer_call_and_return_conditional_losses_78964

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
?
C__inference_conv2d_8_layer_call_and_return_conditional_losses_78944

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
?!
?
G__inference_sequential_9_layer_call_and_return_conditional_losses_77808
input_18 
dense_9_77785:	?
dense_9_77787:	?3
conv2d_transpose_12_77791:@ '
conv2d_transpose_12_77793:@3
conv2d_transpose_13_77796: @'
conv2d_transpose_13_77798: 3
conv2d_transpose_14_77801: '
conv2d_transpose_14_77803:
identity??+conv2d_transpose_12/StatefulPartitionedCall?+conv2d_transpose_13/StatefulPartitionedCall?+conv2d_transpose_14/StatefulPartitionedCall?dense_9/StatefulPartitionedCall?
dense_9/StatefulPartitionedCallStatefulPartitionedCallinput_18dense_9_77785dense_9_77787*
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
B__inference_dense_9_layer_call_and_return_conditional_losses_774732!
dense_9/StatefulPartitionedCall?
reshape_18/PartitionedCallPartitionedCall(dense_9/StatefulPartitionedCall:output:0*
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
E__inference_reshape_18_layer_call_and_return_conditional_losses_774932
reshape_18/PartitionedCall?
+conv2d_transpose_12/StatefulPartitionedCallStatefulPartitionedCall#reshape_18/PartitionedCall:output:0conv2d_transpose_12_77791conv2d_transpose_12_77793*
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
GPU2*0J 8? *W
fRRP
N__inference_conv2d_transpose_12_layer_call_and_return_conditional_losses_775182-
+conv2d_transpose_12/StatefulPartitionedCall?
+conv2d_transpose_13/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_12/StatefulPartitionedCall:output:0conv2d_transpose_13_77796conv2d_transpose_13_77798*
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
GPU2*0J 8? *W
fRRP
N__inference_conv2d_transpose_13_layer_call_and_return_conditional_losses_775472-
+conv2d_transpose_13/StatefulPartitionedCall?
+conv2d_transpose_14/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_13/StatefulPartitionedCall:output:0conv2d_transpose_14_77801conv2d_transpose_14_77803*
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
GPU2*0J 8? *W
fRRP
N__inference_conv2d_transpose_14_layer_call_and_return_conditional_losses_775762-
+conv2d_transpose_14/StatefulPartitionedCall?
reshape_19/PartitionedCallPartitionedCall4conv2d_transpose_14/StatefulPartitionedCall:output:0*
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
E__inference_reshape_19_layer_call_and_return_conditional_losses_775962
reshape_19/PartitionedCall?
IdentityIdentity#reshape_19/PartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????2

Identity?
NoOpNoOp,^conv2d_transpose_12/StatefulPartitionedCall,^conv2d_transpose_13/StatefulPartitionedCall,^conv2d_transpose_14/StatefulPartitionedCall ^dense_9/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : : : : : 2Z
+conv2d_transpose_12/StatefulPartitionedCall+conv2d_transpose_12/StatefulPartitionedCall2Z
+conv2d_transpose_13/StatefulPartitionedCall+conv2d_transpose_13/StatefulPartitionedCall2Z
+conv2d_transpose_14/StatefulPartitionedCall+conv2d_transpose_14/StatefulPartitionedCall2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall:Q M
'
_output_shapes
:?????????
"
_user_specified_name
input_18
?
a
E__inference_reshape_17_layer_call_and_return_conditional_losses_77008

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
?
G__inference_sequential_8_layer_call_and_return_conditional_losses_77169
input_17(
conv2d_8_77151: 
conv2d_8_77153: (
conv2d_9_77156: @
conv2d_9_77158:@ 
dense_8_77162:	?
dense_8_77164:
identity?? conv2d_8/StatefulPartitionedCall? conv2d_9/StatefulPartitionedCall?dense_8/StatefulPartitionedCall?
reshape_16/PartitionedCallPartitionedCallinput_17*
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
E__inference_reshape_16_layer_call_and_return_conditional_losses_769352
reshape_16/PartitionedCall?
 conv2d_8/StatefulPartitionedCallStatefulPartitionedCall#reshape_16/PartitionedCall:output:0conv2d_8_77151conv2d_8_77153*
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
C__inference_conv2d_8_layer_call_and_return_conditional_losses_769482"
 conv2d_8/StatefulPartitionedCall?
 conv2d_9/StatefulPartitionedCallStatefulPartitionedCall)conv2d_8/StatefulPartitionedCall:output:0conv2d_9_77156conv2d_9_77158*
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
C__inference_conv2d_9_layer_call_and_return_conditional_losses_769652"
 conv2d_9/StatefulPartitionedCall?
flatten_4/PartitionedCallPartitionedCall)conv2d_9/StatefulPartitionedCall:output:0*
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
D__inference_flatten_4_layer_call_and_return_conditional_losses_769772
flatten_4/PartitionedCall?
dense_8/StatefulPartitionedCallStatefulPartitionedCall"flatten_4/PartitionedCall:output:0dense_8_77162dense_8_77164*
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
B__inference_dense_8_layer_call_and_return_conditional_losses_769892!
dense_8/StatefulPartitionedCall?
reshape_17/PartitionedCallPartitionedCall(dense_8/StatefulPartitionedCall:output:0*
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
GPU2*0J 8? *N
fIRG
E__inference_reshape_17_layer_call_and_return_conditional_losses_770082
reshape_17/PartitionedCall?
IdentityIdentity#reshape_17/PartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????2

Identity?
NoOpNoOp!^conv2d_8/StatefulPartitionedCall!^conv2d_9/StatefulPartitionedCall ^dense_8/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : 2D
 conv2d_8/StatefulPartitionedCall conv2d_8/StatefulPartitionedCall2D
 conv2d_9/StatefulPartitionedCall conv2d_9/StatefulPartitionedCall2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall:Y U
/
_output_shapes
:?????????
"
_user_specified_name
input_17
?
?
G__inference_sequential_8_layer_call_and_return_conditional_losses_77011

inputs(
conv2d_8_76949: 
conv2d_8_76951: (
conv2d_9_76966: @
conv2d_9_76968:@ 
dense_8_76990:	?
dense_8_76992:
identity?? conv2d_8/StatefulPartitionedCall? conv2d_9/StatefulPartitionedCall?dense_8/StatefulPartitionedCall?
reshape_16/PartitionedCallPartitionedCallinputs*
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
E__inference_reshape_16_layer_call_and_return_conditional_losses_769352
reshape_16/PartitionedCall?
 conv2d_8/StatefulPartitionedCallStatefulPartitionedCall#reshape_16/PartitionedCall:output:0conv2d_8_76949conv2d_8_76951*
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
C__inference_conv2d_8_layer_call_and_return_conditional_losses_769482"
 conv2d_8/StatefulPartitionedCall?
 conv2d_9/StatefulPartitionedCallStatefulPartitionedCall)conv2d_8/StatefulPartitionedCall:output:0conv2d_9_76966conv2d_9_76968*
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
C__inference_conv2d_9_layer_call_and_return_conditional_losses_769652"
 conv2d_9/StatefulPartitionedCall?
flatten_4/PartitionedCallPartitionedCall)conv2d_9/StatefulPartitionedCall:output:0*
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
D__inference_flatten_4_layer_call_and_return_conditional_losses_769772
flatten_4/PartitionedCall?
dense_8/StatefulPartitionedCallStatefulPartitionedCall"flatten_4/PartitionedCall:output:0dense_8_76990dense_8_76992*
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
B__inference_dense_8_layer_call_and_return_conditional_losses_769892!
dense_8/StatefulPartitionedCall?
reshape_17/PartitionedCallPartitionedCall(dense_8/StatefulPartitionedCall:output:0*
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
GPU2*0J 8? *N
fIRG
E__inference_reshape_17_layer_call_and_return_conditional_losses_770082
reshape_17/PartitionedCall?
IdentityIdentity#reshape_17/PartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????2

Identity?
NoOpNoOp!^conv2d_8/StatefulPartitionedCall!^conv2d_9/StatefulPartitionedCall ^dense_8/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : 2D
 conv2d_8/StatefulPartitionedCall conv2d_8/StatefulPartitionedCall2D
 conv2d_9/StatefulPartitionedCall conv2d_9/StatefulPartitionedCall2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
9__inference_variational_autoencoder_4_layer_call_fn_78490

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
T__inference_variational_autoencoder_4_layer_call_and_return_conditional_losses_778542
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
?
?
T__inference_variational_autoencoder_4_layer_call_and_return_conditional_losses_77993

inputs,
sequential_8_77961:  
sequential_8_77963: ,
sequential_8_77965: @ 
sequential_8_77967:@%
sequential_8_77969:	? 
sequential_8_77971:%
sequential_9_77975:	?!
sequential_9_77977:	?,
sequential_9_77979:@  
sequential_9_77981:@,
sequential_9_77983: @ 
sequential_9_77985: ,
sequential_9_77987:  
sequential_9_77989:
identity??/normal_sampling_layer_4/StatefulPartitionedCall?$sequential_8/StatefulPartitionedCall?$sequential_9/StatefulPartitionedCall?
$sequential_8/StatefulPartitionedCallStatefulPartitionedCallinputssequential_8_77961sequential_8_77963sequential_8_77965sequential_8_77967sequential_8_77969sequential_8_77971*
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
G__inference_sequential_8_layer_call_and_return_conditional_losses_771152&
$sequential_8/StatefulPartitionedCall?
/normal_sampling_layer_4/StatefulPartitionedCallStatefulPartitionedCall-sequential_8/StatefulPartitionedCall:output:0*
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
R__inference_normal_sampling_layer_4_layer_call_and_return_conditional_losses_7792021
/normal_sampling_layer_4/StatefulPartitionedCall?
$sequential_9/StatefulPartitionedCallStatefulPartitionedCall8normal_sampling_layer_4/StatefulPartitionedCall:output:0sequential_9_77975sequential_9_77977sequential_9_77979sequential_9_77981sequential_9_77983sequential_9_77985sequential_9_77987sequential_9_77989*
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
G__inference_sequential_9_layer_call_and_return_conditional_losses_777162&
$sequential_9/StatefulPartitionedCall?
IdentityIdentity-sequential_9/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????2

Identity?
NoOpNoOp0^normal_sampling_layer_4/StatefulPartitionedCall%^sequential_8/StatefulPartitionedCall%^sequential_9/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:?????????: : : : : : : : : : : : : : 2b
/normal_sampling_layer_4/StatefulPartitionedCall/normal_sampling_layer_4/StatefulPartitionedCall2L
$sequential_8/StatefulPartitionedCall$sequential_8/StatefulPartitionedCall2L
$sequential_9/StatefulPartitionedCall$sequential_9/StatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
`
D__inference_flatten_4_layer_call_and_return_conditional_losses_76977

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
?
q
R__inference_normal_sampling_layer_4_layer_call_and_return_conditional_losses_78904

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
seed???)*
seed2???2$
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
,__inference_sequential_9_layer_call_fn_78871

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
G__inference_sequential_9_layer_call_and_return_conditional_losses_777162
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
?
?
B__inference_dense_9_layer_call_and_return_conditional_losses_77473

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
?
?
3__inference_conv2d_transpose_12_layer_call_fn_79136

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
GPU2*0J 8? *W
fRRP
N__inference_conv2d_transpose_12_layer_call_and_return_conditional_losses_775182
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
?
?
C__inference_conv2d_8_layer_call_and_return_conditional_losses_76948

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
?
n
R__inference_normal_sampling_layer_4_layer_call_and_return_conditional_losses_77834

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
?
G__inference_sequential_8_layer_call_and_return_conditional_losses_77115

inputs(
conv2d_8_77097: 
conv2d_8_77099: (
conv2d_9_77102: @
conv2d_9_77104:@ 
dense_8_77108:	?
dense_8_77110:
identity?? conv2d_8/StatefulPartitionedCall? conv2d_9/StatefulPartitionedCall?dense_8/StatefulPartitionedCall?
reshape_16/PartitionedCallPartitionedCallinputs*
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
E__inference_reshape_16_layer_call_and_return_conditional_losses_769352
reshape_16/PartitionedCall?
 conv2d_8/StatefulPartitionedCallStatefulPartitionedCall#reshape_16/PartitionedCall:output:0conv2d_8_77097conv2d_8_77099*
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
C__inference_conv2d_8_layer_call_and_return_conditional_losses_769482"
 conv2d_8/StatefulPartitionedCall?
 conv2d_9/StatefulPartitionedCallStatefulPartitionedCall)conv2d_8/StatefulPartitionedCall:output:0conv2d_9_77102conv2d_9_77104*
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
C__inference_conv2d_9_layer_call_and_return_conditional_losses_769652"
 conv2d_9/StatefulPartitionedCall?
flatten_4/PartitionedCallPartitionedCall)conv2d_9/StatefulPartitionedCall:output:0*
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
D__inference_flatten_4_layer_call_and_return_conditional_losses_769772
flatten_4/PartitionedCall?
dense_8/StatefulPartitionedCallStatefulPartitionedCall"flatten_4/PartitionedCall:output:0dense_8_77108dense_8_77110*
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
B__inference_dense_8_layer_call_and_return_conditional_losses_769892!
dense_8/StatefulPartitionedCall?
reshape_17/PartitionedCallPartitionedCall(dense_8/StatefulPartitionedCall:output:0*
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
GPU2*0J 8? *N
fIRG
E__inference_reshape_17_layer_call_and_return_conditional_losses_770082
reshape_17/PartitionedCall?
IdentityIdentity#reshape_17/PartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????2

Identity?
NoOpNoOp!^conv2d_8/StatefulPartitionedCall!^conv2d_9/StatefulPartitionedCall ^dense_8/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : 2D
 conv2d_8/StatefulPartitionedCall conv2d_8/StatefulPartitionedCall2D
 conv2d_9/StatefulPartitionedCall conv2d_9/StatefulPartitionedCall2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
a
E__inference_reshape_18_layer_call_and_return_conditional_losses_77493

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
?
?
'__inference_dense_8_layer_call_fn_79003

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
B__inference_dense_8_layer_call_and_return_conditional_losses_769892
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
?:
?
G__inference_sequential_8_layer_call_and_return_conditional_losses_78613

inputsA
'conv2d_8_conv2d_readvariableop_resource: 6
(conv2d_8_biasadd_readvariableop_resource: A
'conv2d_9_conv2d_readvariableop_resource: @6
(conv2d_9_biasadd_readvariableop_resource:@9
&dense_8_matmul_readvariableop_resource:	?5
'dense_8_biasadd_readvariableop_resource:
identity??conv2d_8/BiasAdd/ReadVariableOp?conv2d_8/Conv2D/ReadVariableOp?conv2d_9/BiasAdd/ReadVariableOp?conv2d_9/Conv2D/ReadVariableOp?dense_8/BiasAdd/ReadVariableOp?dense_8/MatMul/ReadVariableOpZ
reshape_16/ShapeShapeinputs*
T0*
_output_shapes
:2
reshape_16/Shape?
reshape_16/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
reshape_16/strided_slice/stack?
 reshape_16/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_16/strided_slice/stack_1?
 reshape_16/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_16/strided_slice/stack_2?
reshape_16/strided_sliceStridedSlicereshape_16/Shape:output:0'reshape_16/strided_slice/stack:output:0)reshape_16/strided_slice/stack_1:output:0)reshape_16/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_16/strided_slicez
reshape_16/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_16/Reshape/shape/1z
reshape_16/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_16/Reshape/shape/2z
reshape_16/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_16/Reshape/shape/3?
reshape_16/Reshape/shapePack!reshape_16/strided_slice:output:0#reshape_16/Reshape/shape/1:output:0#reshape_16/Reshape/shape/2:output:0#reshape_16/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
reshape_16/Reshape/shape?
reshape_16/ReshapeReshapeinputs!reshape_16/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????2
reshape_16/Reshape?
conv2d_8/Conv2D/ReadVariableOpReadVariableOp'conv2d_8_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02 
conv2d_8/Conv2D/ReadVariableOp?
conv2d_8/Conv2DConv2Dreshape_16/Reshape:output:0&conv2d_8/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
2
conv2d_8/Conv2D?
conv2d_8/BiasAdd/ReadVariableOpReadVariableOp(conv2d_8_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv2d_8/BiasAdd/ReadVariableOp?
conv2d_8/BiasAddBiasAddconv2d_8/Conv2D:output:0'conv2d_8/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2
conv2d_8/BiasAdd{
conv2d_8/ReluReluconv2d_8/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
conv2d_8/Relu?
conv2d_9/Conv2D/ReadVariableOpReadVariableOp'conv2d_9_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02 
conv2d_9/Conv2D/ReadVariableOp?
conv2d_9/Conv2DConv2Dconv2d_8/Relu:activations:0&conv2d_9/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
2
conv2d_9/Conv2D?
conv2d_9/BiasAdd/ReadVariableOpReadVariableOp(conv2d_9_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_9/BiasAdd/ReadVariableOp?
conv2d_9/BiasAddBiasAddconv2d_9/Conv2D:output:0'conv2d_9/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
conv2d_9/BiasAdd{
conv2d_9/ReluReluconv2d_9/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
conv2d_9/Relus
flatten_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"???? 	  2
flatten_4/Const?
flatten_4/ReshapeReshapeconv2d_9/Relu:activations:0flatten_4/Const:output:0*
T0*(
_output_shapes
:??????????2
flatten_4/Reshape?
dense_8/MatMul/ReadVariableOpReadVariableOp&dense_8_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
dense_8/MatMul/ReadVariableOp?
dense_8/MatMulMatMulflatten_4/Reshape:output:0%dense_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_8/MatMul?
dense_8/BiasAdd/ReadVariableOpReadVariableOp'dense_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_8/BiasAdd/ReadVariableOp?
dense_8/BiasAddBiasAdddense_8/MatMul:product:0&dense_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_8/BiasAddl
reshape_17/ShapeShapedense_8/BiasAdd:output:0*
T0*
_output_shapes
:2
reshape_17/Shape?
reshape_17/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
reshape_17/strided_slice/stack?
 reshape_17/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_17/strided_slice/stack_1?
 reshape_17/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_17/strided_slice/stack_2?
reshape_17/strided_sliceStridedSlicereshape_17/Shape:output:0'reshape_17/strided_slice/stack:output:0)reshape_17/strided_slice/stack_1:output:0)reshape_17/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_17/strided_slicez
reshape_17/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_17/Reshape/shape/1z
reshape_17/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_17/Reshape/shape/2?
reshape_17/Reshape/shapePack!reshape_17/strided_slice:output:0#reshape_17/Reshape/shape/1:output:0#reshape_17/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
reshape_17/Reshape/shape?
reshape_17/ReshapeReshapedense_8/BiasAdd:output:0!reshape_17/Reshape/shape:output:0*
T0*+
_output_shapes
:?????????2
reshape_17/Reshapez
IdentityIdentityreshape_17/Reshape:output:0^NoOp*
T0*+
_output_shapes
:?????????2

Identity?
NoOpNoOp ^conv2d_8/BiasAdd/ReadVariableOp^conv2d_8/Conv2D/ReadVariableOp ^conv2d_9/BiasAdd/ReadVariableOp^conv2d_9/Conv2D/ReadVariableOp^dense_8/BiasAdd/ReadVariableOp^dense_8/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : 2B
conv2d_8/BiasAdd/ReadVariableOpconv2d_8/BiasAdd/ReadVariableOp2@
conv2d_8/Conv2D/ReadVariableOpconv2d_8/Conv2D/ReadVariableOp2B
conv2d_9/BiasAdd/ReadVariableOpconv2d_9/BiasAdd/ReadVariableOp2@
conv2d_9/Conv2D/ReadVariableOpconv2d_9/Conv2D/ReadVariableOp2@
dense_8/BiasAdd/ReadVariableOpdense_8/BiasAdd/ReadVariableOp2>
dense_8/MatMul/ReadVariableOpdense_8/MatMul/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
B__inference_dense_9_layer_call_and_return_conditional_losses_79032

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
?
?
N__inference_conv2d_transpose_12_layer_call_and_return_conditional_losses_79118

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
?m
?
__inference__traced_save_79489
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
*savev2_conv2d_8_kernel_read_readvariableop,
(savev2_conv2d_8_bias_read_readvariableop.
*savev2_conv2d_9_kernel_read_readvariableop,
(savev2_conv2d_9_bias_read_readvariableop-
)savev2_dense_8_kernel_read_readvariableop+
'savev2_dense_8_bias_read_readvariableop-
)savev2_dense_9_kernel_read_readvariableop+
'savev2_dense_9_bias_read_readvariableop9
5savev2_conv2d_transpose_12_kernel_read_readvariableop7
3savev2_conv2d_transpose_12_bias_read_readvariableop9
5savev2_conv2d_transpose_13_kernel_read_readvariableop7
3savev2_conv2d_transpose_13_bias_read_readvariableop9
5savev2_conv2d_transpose_14_kernel_read_readvariableop7
3savev2_conv2d_transpose_14_bias_read_readvariableop5
1savev2_adam_conv2d_8_kernel_m_read_readvariableop3
/savev2_adam_conv2d_8_bias_m_read_readvariableop5
1savev2_adam_conv2d_9_kernel_m_read_readvariableop3
/savev2_adam_conv2d_9_bias_m_read_readvariableop4
0savev2_adam_dense_8_kernel_m_read_readvariableop2
.savev2_adam_dense_8_bias_m_read_readvariableop4
0savev2_adam_dense_9_kernel_m_read_readvariableop2
.savev2_adam_dense_9_bias_m_read_readvariableop@
<savev2_adam_conv2d_transpose_12_kernel_m_read_readvariableop>
:savev2_adam_conv2d_transpose_12_bias_m_read_readvariableop@
<savev2_adam_conv2d_transpose_13_kernel_m_read_readvariableop>
:savev2_adam_conv2d_transpose_13_bias_m_read_readvariableop@
<savev2_adam_conv2d_transpose_14_kernel_m_read_readvariableop>
:savev2_adam_conv2d_transpose_14_bias_m_read_readvariableop5
1savev2_adam_conv2d_8_kernel_v_read_readvariableop3
/savev2_adam_conv2d_8_bias_v_read_readvariableop5
1savev2_adam_conv2d_9_kernel_v_read_readvariableop3
/savev2_adam_conv2d_9_bias_v_read_readvariableop4
0savev2_adam_dense_8_kernel_v_read_readvariableop2
.savev2_adam_dense_8_bias_v_read_readvariableop4
0savev2_adam_dense_9_kernel_v_read_readvariableop2
.savev2_adam_dense_9_bias_v_read_readvariableop@
<savev2_adam_conv2d_transpose_12_kernel_v_read_readvariableop>
:savev2_adam_conv2d_transpose_12_bias_v_read_readvariableop@
<savev2_adam_conv2d_transpose_13_kernel_v_read_readvariableop>
:savev2_adam_conv2d_transpose_13_bias_v_read_readvariableop@
<savev2_adam_conv2d_transpose_14_kernel_v_read_readvariableop>
:savev2_adam_conv2d_transpose_14_bias_v_read_readvariableop
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
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0 savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop"savev2_total_2_read_readvariableop"savev2_count_2_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop*savev2_conv2d_8_kernel_read_readvariableop(savev2_conv2d_8_bias_read_readvariableop*savev2_conv2d_9_kernel_read_readvariableop(savev2_conv2d_9_bias_read_readvariableop)savev2_dense_8_kernel_read_readvariableop'savev2_dense_8_bias_read_readvariableop)savev2_dense_9_kernel_read_readvariableop'savev2_dense_9_bias_read_readvariableop5savev2_conv2d_transpose_12_kernel_read_readvariableop3savev2_conv2d_transpose_12_bias_read_readvariableop5savev2_conv2d_transpose_13_kernel_read_readvariableop3savev2_conv2d_transpose_13_bias_read_readvariableop5savev2_conv2d_transpose_14_kernel_read_readvariableop3savev2_conv2d_transpose_14_bias_read_readvariableop1savev2_adam_conv2d_8_kernel_m_read_readvariableop/savev2_adam_conv2d_8_bias_m_read_readvariableop1savev2_adam_conv2d_9_kernel_m_read_readvariableop/savev2_adam_conv2d_9_bias_m_read_readvariableop0savev2_adam_dense_8_kernel_m_read_readvariableop.savev2_adam_dense_8_bias_m_read_readvariableop0savev2_adam_dense_9_kernel_m_read_readvariableop.savev2_adam_dense_9_bias_m_read_readvariableop<savev2_adam_conv2d_transpose_12_kernel_m_read_readvariableop:savev2_adam_conv2d_transpose_12_bias_m_read_readvariableop<savev2_adam_conv2d_transpose_13_kernel_m_read_readvariableop:savev2_adam_conv2d_transpose_13_bias_m_read_readvariableop<savev2_adam_conv2d_transpose_14_kernel_m_read_readvariableop:savev2_adam_conv2d_transpose_14_bias_m_read_readvariableop1savev2_adam_conv2d_8_kernel_v_read_readvariableop/savev2_adam_conv2d_8_bias_v_read_readvariableop1savev2_adam_conv2d_9_kernel_v_read_readvariableop/savev2_adam_conv2d_9_bias_v_read_readvariableop0savev2_adam_dense_8_kernel_v_read_readvariableop.savev2_adam_dense_8_bias_v_read_readvariableop0savev2_adam_dense_9_kernel_v_read_readvariableop.savev2_adam_dense_9_bias_v_read_readvariableop<savev2_adam_conv2d_transpose_12_kernel_v_read_readvariableop:savev2_adam_conv2d_transpose_12_bias_v_read_readvariableop<savev2_adam_conv2d_transpose_13_kernel_v_read_readvariableop:savev2_adam_conv2d_transpose_13_bias_v_read_readvariableop<savev2_adam_conv2d_transpose_14_kernel_v_read_readvariableop:savev2_adam_conv2d_transpose_14_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
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
: 
??
?"
!__inference__traced_restore_79658
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
#assignvariableop_11_conv2d_8_kernel: /
!assignvariableop_12_conv2d_8_bias: =
#assignvariableop_13_conv2d_9_kernel: @/
!assignvariableop_14_conv2d_9_bias:@5
"assignvariableop_15_dense_8_kernel:	?.
 assignvariableop_16_dense_8_bias:5
"assignvariableop_17_dense_9_kernel:	?/
 assignvariableop_18_dense_9_bias:	?H
.assignvariableop_19_conv2d_transpose_12_kernel:@ :
,assignvariableop_20_conv2d_transpose_12_bias:@H
.assignvariableop_21_conv2d_transpose_13_kernel: @:
,assignvariableop_22_conv2d_transpose_13_bias: H
.assignvariableop_23_conv2d_transpose_14_kernel: :
,assignvariableop_24_conv2d_transpose_14_bias:D
*assignvariableop_25_adam_conv2d_8_kernel_m: 6
(assignvariableop_26_adam_conv2d_8_bias_m: D
*assignvariableop_27_adam_conv2d_9_kernel_m: @6
(assignvariableop_28_adam_conv2d_9_bias_m:@<
)assignvariableop_29_adam_dense_8_kernel_m:	?5
'assignvariableop_30_adam_dense_8_bias_m:<
)assignvariableop_31_adam_dense_9_kernel_m:	?6
'assignvariableop_32_adam_dense_9_bias_m:	?O
5assignvariableop_33_adam_conv2d_transpose_12_kernel_m:@ A
3assignvariableop_34_adam_conv2d_transpose_12_bias_m:@O
5assignvariableop_35_adam_conv2d_transpose_13_kernel_m: @A
3assignvariableop_36_adam_conv2d_transpose_13_bias_m: O
5assignvariableop_37_adam_conv2d_transpose_14_kernel_m: A
3assignvariableop_38_adam_conv2d_transpose_14_bias_m:D
*assignvariableop_39_adam_conv2d_8_kernel_v: 6
(assignvariableop_40_adam_conv2d_8_bias_v: D
*assignvariableop_41_adam_conv2d_9_kernel_v: @6
(assignvariableop_42_adam_conv2d_9_bias_v:@<
)assignvariableop_43_adam_dense_8_kernel_v:	?5
'assignvariableop_44_adam_dense_8_bias_v:<
)assignvariableop_45_adam_dense_9_kernel_v:	?6
'assignvariableop_46_adam_dense_9_bias_v:	?O
5assignvariableop_47_adam_conv2d_transpose_12_kernel_v:@ A
3assignvariableop_48_adam_conv2d_transpose_12_bias_v:@O
5assignvariableop_49_adam_conv2d_transpose_13_kernel_v: @A
3assignvariableop_50_adam_conv2d_transpose_13_bias_v: O
5assignvariableop_51_adam_conv2d_transpose_14_kernel_v: A
3assignvariableop_52_adam_conv2d_transpose_14_bias_v:
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
AssignVariableOp_11AssignVariableOp#assignvariableop_11_conv2d_8_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp!assignvariableop_12_conv2d_8_biasIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOp#assignvariableop_13_conv2d_9_kernelIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOp!assignvariableop_14_conv2d_9_biasIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOp"assignvariableop_15_dense_8_kernelIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOp assignvariableop_16_dense_8_biasIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOp"assignvariableop_17_dense_9_kernelIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOp assignvariableop_18_dense_9_biasIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOp.assignvariableop_19_conv2d_transpose_12_kernelIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp,assignvariableop_20_conv2d_transpose_12_biasIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp.assignvariableop_21_conv2d_transpose_13_kernelIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp,assignvariableop_22_conv2d_transpose_13_biasIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp.assignvariableop_23_conv2d_transpose_14_kernelIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp,assignvariableop_24_conv2d_transpose_14_biasIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOp*assignvariableop_25_adam_conv2d_8_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOp(assignvariableop_26_adam_conv2d_8_bias_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOp*assignvariableop_27_adam_conv2d_9_kernel_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOp(assignvariableop_28_adam_conv2d_9_bias_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOp)assignvariableop_29_adam_dense_8_kernel_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOp'assignvariableop_30_adam_dense_8_bias_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOp)assignvariableop_31_adam_dense_9_kernel_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOp'assignvariableop_32_adam_dense_9_bias_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33?
AssignVariableOp_33AssignVariableOp5assignvariableop_33_adam_conv2d_transpose_12_kernel_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34?
AssignVariableOp_34AssignVariableOp3assignvariableop_34_adam_conv2d_transpose_12_bias_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35?
AssignVariableOp_35AssignVariableOp5assignvariableop_35_adam_conv2d_transpose_13_kernel_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36?
AssignVariableOp_36AssignVariableOp3assignvariableop_36_adam_conv2d_transpose_13_bias_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37?
AssignVariableOp_37AssignVariableOp5assignvariableop_37_adam_conv2d_transpose_14_kernel_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38?
AssignVariableOp_38AssignVariableOp3assignvariableop_38_adam_conv2d_transpose_14_bias_mIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39?
AssignVariableOp_39AssignVariableOp*assignvariableop_39_adam_conv2d_8_kernel_vIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40?
AssignVariableOp_40AssignVariableOp(assignvariableop_40_adam_conv2d_8_bias_vIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41?
AssignVariableOp_41AssignVariableOp*assignvariableop_41_adam_conv2d_9_kernel_vIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42?
AssignVariableOp_42AssignVariableOp(assignvariableop_42_adam_conv2d_9_bias_vIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43?
AssignVariableOp_43AssignVariableOp)assignvariableop_43_adam_dense_8_kernel_vIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44?
AssignVariableOp_44AssignVariableOp'assignvariableop_44_adam_dense_8_bias_vIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45?
AssignVariableOp_45AssignVariableOp)assignvariableop_45_adam_dense_9_kernel_vIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46?
AssignVariableOp_46AssignVariableOp'assignvariableop_46_adam_dense_9_bias_vIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47?
AssignVariableOp_47AssignVariableOp5assignvariableop_47_adam_conv2d_transpose_12_kernel_vIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48?
AssignVariableOp_48AssignVariableOp3assignvariableop_48_adam_conv2d_transpose_12_bias_vIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49?
AssignVariableOp_49AssignVariableOp5assignvariableop_49_adam_conv2d_transpose_13_kernel_vIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50?
AssignVariableOp_50AssignVariableOp3assignvariableop_50_adam_conv2d_transpose_13_bias_vIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51?
AssignVariableOp_51AssignVariableOp5assignvariableop_51_adam_conv2d_transpose_14_kernel_vIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52?
AssignVariableOp_52AssignVariableOp3assignvariableop_52_adam_conv2d_transpose_14_bias_vIdentity_52:output:0"/device:CPU:0*
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
?
?
3__inference_conv2d_transpose_13_layer_call_fn_79203

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
GPU2*0J 8? *W
fRRP
N__inference_conv2d_transpose_13_layer_call_and_return_conditional_losses_773172
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
?	
?
,__inference_sequential_8_layer_call_fn_77026
input_17!
unknown: 
	unknown_0: #
	unknown_1: @
	unknown_2:@
	unknown_3:	?
	unknown_4:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_17unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
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
G__inference_sequential_8_layer_call_and_return_conditional_losses_770112
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
StatefulPartitionedCallStatefulPartitionedCall:Y U
/
_output_shapes
:?????????
"
_user_specified_name
input_17
?
a
E__inference_reshape_18_layer_call_and_return_conditional_losses_79055

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
?
?
N__inference_conv2d_transpose_14_layer_call_and_return_conditional_losses_79270

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
?

?
B__inference_dense_8_layer_call_and_return_conditional_losses_78994

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
?
a
E__inference_reshape_19_layer_call_and_return_conditional_losses_79302

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
?

?
,__inference_sequential_9_layer_call_fn_77618
input_18
unknown:	?
	unknown_0:	?#
	unknown_1:@ 
	unknown_2:@#
	unknown_3: @
	unknown_4: #
	unknown_5: 
	unknown_6:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_18unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
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
G__inference_sequential_9_layer_call_and_return_conditional_losses_775992
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
input_18
?
?
T__inference_variational_autoencoder_4_layer_call_and_return_conditional_losses_78127
input_1,
sequential_8_78095:  
sequential_8_78097: ,
sequential_8_78099: @ 
sequential_8_78101:@%
sequential_8_78103:	? 
sequential_8_78105:%
sequential_9_78109:	?!
sequential_9_78111:	?,
sequential_9_78113:@  
sequential_9_78115:@,
sequential_9_78117: @ 
sequential_9_78119: ,
sequential_9_78121:  
sequential_9_78123:
identity??/normal_sampling_layer_4/StatefulPartitionedCall?$sequential_8/StatefulPartitionedCall?$sequential_9/StatefulPartitionedCall?
$sequential_8/StatefulPartitionedCallStatefulPartitionedCallinput_1sequential_8_78095sequential_8_78097sequential_8_78099sequential_8_78101sequential_8_78103sequential_8_78105*
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
G__inference_sequential_8_layer_call_and_return_conditional_losses_771152&
$sequential_8/StatefulPartitionedCall?
/normal_sampling_layer_4/StatefulPartitionedCallStatefulPartitionedCall-sequential_8/StatefulPartitionedCall:output:0*
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
R__inference_normal_sampling_layer_4_layer_call_and_return_conditional_losses_7792021
/normal_sampling_layer_4/StatefulPartitionedCall?
$sequential_9/StatefulPartitionedCallStatefulPartitionedCall8normal_sampling_layer_4/StatefulPartitionedCall:output:0sequential_9_78109sequential_9_78111sequential_9_78113sequential_9_78115sequential_9_78117sequential_9_78119sequential_9_78121sequential_9_78123*
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
G__inference_sequential_9_layer_call_and_return_conditional_losses_777162&
$sequential_9/StatefulPartitionedCall?
IdentityIdentity-sequential_9/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????2

Identity?
NoOpNoOp0^normal_sampling_layer_4/StatefulPartitionedCall%^sequential_8/StatefulPartitionedCall%^sequential_9/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:?????????: : : : : : : : : : : : : : 2b
/normal_sampling_layer_4/StatefulPartitionedCall/normal_sampling_layer_4/StatefulPartitionedCall2L
$sequential_8/StatefulPartitionedCall$sequential_8/StatefulPartitionedCall2L
$sequential_9/StatefulPartitionedCall$sequential_9/StatefulPartitionedCall:X T
/
_output_shapes
:?????????
!
_user_specified_name	input_1
?
q
R__inference_normal_sampling_layer_4_layer_call_and_return_conditional_losses_77920

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
seed2??V2$
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
?
G__inference_sequential_8_layer_call_and_return_conditional_losses_77191
input_17(
conv2d_8_77173: 
conv2d_8_77175: (
conv2d_9_77178: @
conv2d_9_77180:@ 
dense_8_77184:	?
dense_8_77186:
identity?? conv2d_8/StatefulPartitionedCall? conv2d_9/StatefulPartitionedCall?dense_8/StatefulPartitionedCall?
reshape_16/PartitionedCallPartitionedCallinput_17*
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
E__inference_reshape_16_layer_call_and_return_conditional_losses_769352
reshape_16/PartitionedCall?
 conv2d_8/StatefulPartitionedCallStatefulPartitionedCall#reshape_16/PartitionedCall:output:0conv2d_8_77173conv2d_8_77175*
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
C__inference_conv2d_8_layer_call_and_return_conditional_losses_769482"
 conv2d_8/StatefulPartitionedCall?
 conv2d_9/StatefulPartitionedCallStatefulPartitionedCall)conv2d_8/StatefulPartitionedCall:output:0conv2d_9_77178conv2d_9_77180*
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
C__inference_conv2d_9_layer_call_and_return_conditional_losses_769652"
 conv2d_9/StatefulPartitionedCall?
flatten_4/PartitionedCallPartitionedCall)conv2d_9/StatefulPartitionedCall:output:0*
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
D__inference_flatten_4_layer_call_and_return_conditional_losses_769772
flatten_4/PartitionedCall?
dense_8/StatefulPartitionedCallStatefulPartitionedCall"flatten_4/PartitionedCall:output:0dense_8_77184dense_8_77186*
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
B__inference_dense_8_layer_call_and_return_conditional_losses_769892!
dense_8/StatefulPartitionedCall?
reshape_17/PartitionedCallPartitionedCall(dense_8/StatefulPartitionedCall:output:0*
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
GPU2*0J 8? *N
fIRG
E__inference_reshape_17_layer_call_and_return_conditional_losses_770082
reshape_17/PartitionedCall?
IdentityIdentity#reshape_17/PartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????2

Identity?
NoOpNoOp!^conv2d_8/StatefulPartitionedCall!^conv2d_9/StatefulPartitionedCall ^dense_8/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : 2D
 conv2d_8/StatefulPartitionedCall conv2d_8/StatefulPartitionedCall2D
 conv2d_9/StatefulPartitionedCall conv2d_9/StatefulPartitionedCall2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall:Y U
/
_output_shapes
:?????????
"
_user_specified_name
input_17
?
?
3__inference_conv2d_transpose_14_layer_call_fn_79288

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
GPU2*0J 8? *W
fRRP
N__inference_conv2d_transpose_14_layer_call_and_return_conditional_losses_775762
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
?
F
*__inference_reshape_19_layer_call_fn_79307

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
E__inference_reshape_19_layer_call_and_return_conditional_losses_775962
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
?!
?
G__inference_sequential_9_layer_call_and_return_conditional_losses_77599

inputs 
dense_9_77474:	?
dense_9_77476:	?3
conv2d_transpose_12_77519:@ '
conv2d_transpose_12_77521:@3
conv2d_transpose_13_77548: @'
conv2d_transpose_13_77550: 3
conv2d_transpose_14_77577: '
conv2d_transpose_14_77579:
identity??+conv2d_transpose_12/StatefulPartitionedCall?+conv2d_transpose_13/StatefulPartitionedCall?+conv2d_transpose_14/StatefulPartitionedCall?dense_9/StatefulPartitionedCall?
dense_9/StatefulPartitionedCallStatefulPartitionedCallinputsdense_9_77474dense_9_77476*
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
B__inference_dense_9_layer_call_and_return_conditional_losses_774732!
dense_9/StatefulPartitionedCall?
reshape_18/PartitionedCallPartitionedCall(dense_9/StatefulPartitionedCall:output:0*
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
E__inference_reshape_18_layer_call_and_return_conditional_losses_774932
reshape_18/PartitionedCall?
+conv2d_transpose_12/StatefulPartitionedCallStatefulPartitionedCall#reshape_18/PartitionedCall:output:0conv2d_transpose_12_77519conv2d_transpose_12_77521*
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
GPU2*0J 8? *W
fRRP
N__inference_conv2d_transpose_12_layer_call_and_return_conditional_losses_775182-
+conv2d_transpose_12/StatefulPartitionedCall?
+conv2d_transpose_13/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_12/StatefulPartitionedCall:output:0conv2d_transpose_13_77548conv2d_transpose_13_77550*
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
GPU2*0J 8? *W
fRRP
N__inference_conv2d_transpose_13_layer_call_and_return_conditional_losses_775472-
+conv2d_transpose_13/StatefulPartitionedCall?
+conv2d_transpose_14/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_13/StatefulPartitionedCall:output:0conv2d_transpose_14_77577conv2d_transpose_14_77579*
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
GPU2*0J 8? *W
fRRP
N__inference_conv2d_transpose_14_layer_call_and_return_conditional_losses_775762-
+conv2d_transpose_14/StatefulPartitionedCall?
reshape_19/PartitionedCallPartitionedCall4conv2d_transpose_14/StatefulPartitionedCall:output:0*
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
E__inference_reshape_19_layer_call_and_return_conditional_losses_775962
reshape_19/PartitionedCall?
IdentityIdentity#reshape_19/PartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????2

Identity?
NoOpNoOp,^conv2d_transpose_12/StatefulPartitionedCall,^conv2d_transpose_13/StatefulPartitionedCall,^conv2d_transpose_14/StatefulPartitionedCall ^dense_9/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : : : : : 2Z
+conv2d_transpose_12/StatefulPartitionedCall+conv2d_transpose_12/StatefulPartitionedCall2Z
+conv2d_transpose_13/StatefulPartitionedCall+conv2d_transpose_13/StatefulPartitionedCall2Z
+conv2d_transpose_14/StatefulPartitionedCall+conv2d_transpose_14/StatefulPartitionedCall2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
??
?
T__inference_variational_autoencoder_4_layer_call_and_return_conditional_losses_78302

inputsN
4sequential_8_conv2d_8_conv2d_readvariableop_resource: C
5sequential_8_conv2d_8_biasadd_readvariableop_resource: N
4sequential_8_conv2d_9_conv2d_readvariableop_resource: @C
5sequential_8_conv2d_9_biasadd_readvariableop_resource:@F
3sequential_8_dense_8_matmul_readvariableop_resource:	?B
4sequential_8_dense_8_biasadd_readvariableop_resource:F
3sequential_9_dense_9_matmul_readvariableop_resource:	?C
4sequential_9_dense_9_biasadd_readvariableop_resource:	?c
Isequential_9_conv2d_transpose_12_conv2d_transpose_readvariableop_resource:@ N
@sequential_9_conv2d_transpose_12_biasadd_readvariableop_resource:@c
Isequential_9_conv2d_transpose_13_conv2d_transpose_readvariableop_resource: @N
@sequential_9_conv2d_transpose_13_biasadd_readvariableop_resource: c
Isequential_9_conv2d_transpose_14_conv2d_transpose_readvariableop_resource: N
@sequential_9_conv2d_transpose_14_biasadd_readvariableop_resource:
identity??,sequential_8/conv2d_8/BiasAdd/ReadVariableOp?+sequential_8/conv2d_8/Conv2D/ReadVariableOp?,sequential_8/conv2d_9/BiasAdd/ReadVariableOp?+sequential_8/conv2d_9/Conv2D/ReadVariableOp?+sequential_8/dense_8/BiasAdd/ReadVariableOp?*sequential_8/dense_8/MatMul/ReadVariableOp?7sequential_9/conv2d_transpose_12/BiasAdd/ReadVariableOp?@sequential_9/conv2d_transpose_12/conv2d_transpose/ReadVariableOp?7sequential_9/conv2d_transpose_13/BiasAdd/ReadVariableOp?@sequential_9/conv2d_transpose_13/conv2d_transpose/ReadVariableOp?7sequential_9/conv2d_transpose_14/BiasAdd/ReadVariableOp?@sequential_9/conv2d_transpose_14/conv2d_transpose/ReadVariableOp?+sequential_9/dense_9/BiasAdd/ReadVariableOp?*sequential_9/dense_9/MatMul/ReadVariableOpt
sequential_8/reshape_16/ShapeShapeinputs*
T0*
_output_shapes
:2
sequential_8/reshape_16/Shape?
+sequential_8/reshape_16/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2-
+sequential_8/reshape_16/strided_slice/stack?
-sequential_8/reshape_16/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_8/reshape_16/strided_slice/stack_1?
-sequential_8/reshape_16/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_8/reshape_16/strided_slice/stack_2?
%sequential_8/reshape_16/strided_sliceStridedSlice&sequential_8/reshape_16/Shape:output:04sequential_8/reshape_16/strided_slice/stack:output:06sequential_8/reshape_16/strided_slice/stack_1:output:06sequential_8/reshape_16/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%sequential_8/reshape_16/strided_slice?
'sequential_8/reshape_16/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2)
'sequential_8/reshape_16/Reshape/shape/1?
'sequential_8/reshape_16/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2)
'sequential_8/reshape_16/Reshape/shape/2?
'sequential_8/reshape_16/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2)
'sequential_8/reshape_16/Reshape/shape/3?
%sequential_8/reshape_16/Reshape/shapePack.sequential_8/reshape_16/strided_slice:output:00sequential_8/reshape_16/Reshape/shape/1:output:00sequential_8/reshape_16/Reshape/shape/2:output:00sequential_8/reshape_16/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2'
%sequential_8/reshape_16/Reshape/shape?
sequential_8/reshape_16/ReshapeReshapeinputs.sequential_8/reshape_16/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????2!
sequential_8/reshape_16/Reshape?
+sequential_8/conv2d_8/Conv2D/ReadVariableOpReadVariableOp4sequential_8_conv2d_8_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02-
+sequential_8/conv2d_8/Conv2D/ReadVariableOp?
sequential_8/conv2d_8/Conv2DConv2D(sequential_8/reshape_16/Reshape:output:03sequential_8/conv2d_8/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
2
sequential_8/conv2d_8/Conv2D?
,sequential_8/conv2d_8/BiasAdd/ReadVariableOpReadVariableOp5sequential_8_conv2d_8_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02.
,sequential_8/conv2d_8/BiasAdd/ReadVariableOp?
sequential_8/conv2d_8/BiasAddBiasAdd%sequential_8/conv2d_8/Conv2D:output:04sequential_8/conv2d_8/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2
sequential_8/conv2d_8/BiasAdd?
sequential_8/conv2d_8/ReluRelu&sequential_8/conv2d_8/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
sequential_8/conv2d_8/Relu?
+sequential_8/conv2d_9/Conv2D/ReadVariableOpReadVariableOp4sequential_8_conv2d_9_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02-
+sequential_8/conv2d_9/Conv2D/ReadVariableOp?
sequential_8/conv2d_9/Conv2DConv2D(sequential_8/conv2d_8/Relu:activations:03sequential_8/conv2d_9/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
2
sequential_8/conv2d_9/Conv2D?
,sequential_8/conv2d_9/BiasAdd/ReadVariableOpReadVariableOp5sequential_8_conv2d_9_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02.
,sequential_8/conv2d_9/BiasAdd/ReadVariableOp?
sequential_8/conv2d_9/BiasAddBiasAdd%sequential_8/conv2d_9/Conv2D:output:04sequential_8/conv2d_9/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
sequential_8/conv2d_9/BiasAdd?
sequential_8/conv2d_9/ReluRelu&sequential_8/conv2d_9/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
sequential_8/conv2d_9/Relu?
sequential_8/flatten_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"???? 	  2
sequential_8/flatten_4/Const?
sequential_8/flatten_4/ReshapeReshape(sequential_8/conv2d_9/Relu:activations:0%sequential_8/flatten_4/Const:output:0*
T0*(
_output_shapes
:??????????2 
sequential_8/flatten_4/Reshape?
*sequential_8/dense_8/MatMul/ReadVariableOpReadVariableOp3sequential_8_dense_8_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02,
*sequential_8/dense_8/MatMul/ReadVariableOp?
sequential_8/dense_8/MatMulMatMul'sequential_8/flatten_4/Reshape:output:02sequential_8/dense_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential_8/dense_8/MatMul?
+sequential_8/dense_8/BiasAdd/ReadVariableOpReadVariableOp4sequential_8_dense_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02-
+sequential_8/dense_8/BiasAdd/ReadVariableOp?
sequential_8/dense_8/BiasAddBiasAdd%sequential_8/dense_8/MatMul:product:03sequential_8/dense_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential_8/dense_8/BiasAdd?
sequential_8/reshape_17/ShapeShape%sequential_8/dense_8/BiasAdd:output:0*
T0*
_output_shapes
:2
sequential_8/reshape_17/Shape?
+sequential_8/reshape_17/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2-
+sequential_8/reshape_17/strided_slice/stack?
-sequential_8/reshape_17/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_8/reshape_17/strided_slice/stack_1?
-sequential_8/reshape_17/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_8/reshape_17/strided_slice/stack_2?
%sequential_8/reshape_17/strided_sliceStridedSlice&sequential_8/reshape_17/Shape:output:04sequential_8/reshape_17/strided_slice/stack:output:06sequential_8/reshape_17/strided_slice/stack_1:output:06sequential_8/reshape_17/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%sequential_8/reshape_17/strided_slice?
'sequential_8/reshape_17/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2)
'sequential_8/reshape_17/Reshape/shape/1?
'sequential_8/reshape_17/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2)
'sequential_8/reshape_17/Reshape/shape/2?
%sequential_8/reshape_17/Reshape/shapePack.sequential_8/reshape_17/strided_slice:output:00sequential_8/reshape_17/Reshape/shape/1:output:00sequential_8/reshape_17/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2'
%sequential_8/reshape_17/Reshape/shape?
sequential_8/reshape_17/ReshapeReshape%sequential_8/dense_8/BiasAdd:output:0.sequential_8/reshape_17/Reshape/shape:output:0*
T0*+
_output_shapes
:?????????2!
sequential_8/reshape_17/Reshape?
normal_sampling_layer_4/unstackUnpack(sequential_8/reshape_17/Reshape:output:0*
T0*:
_output_shapes(
&:?????????:?????????*

axis*	
num2!
normal_sampling_layer_4/unstack?
*sequential_9/dense_9/MatMul/ReadVariableOpReadVariableOp3sequential_9_dense_9_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02,
*sequential_9/dense_9/MatMul/ReadVariableOp?
sequential_9/dense_9/MatMulMatMul(normal_sampling_layer_4/unstack:output:02sequential_9/dense_9/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential_9/dense_9/MatMul?
+sequential_9/dense_9/BiasAdd/ReadVariableOpReadVariableOp4sequential_9_dense_9_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02-
+sequential_9/dense_9/BiasAdd/ReadVariableOp?
sequential_9/dense_9/BiasAddBiasAdd%sequential_9/dense_9/MatMul:product:03sequential_9/dense_9/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential_9/dense_9/BiasAdd?
sequential_9/dense_9/ReluRelu%sequential_9/dense_9/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
sequential_9/dense_9/Relu?
sequential_9/reshape_18/ShapeShape'sequential_9/dense_9/Relu:activations:0*
T0*
_output_shapes
:2
sequential_9/reshape_18/Shape?
+sequential_9/reshape_18/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2-
+sequential_9/reshape_18/strided_slice/stack?
-sequential_9/reshape_18/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_9/reshape_18/strided_slice/stack_1?
-sequential_9/reshape_18/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_9/reshape_18/strided_slice/stack_2?
%sequential_9/reshape_18/strided_sliceStridedSlice&sequential_9/reshape_18/Shape:output:04sequential_9/reshape_18/strided_slice/stack:output:06sequential_9/reshape_18/strided_slice/stack_1:output:06sequential_9/reshape_18/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%sequential_9/reshape_18/strided_slice?
'sequential_9/reshape_18/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2)
'sequential_9/reshape_18/Reshape/shape/1?
'sequential_9/reshape_18/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2)
'sequential_9/reshape_18/Reshape/shape/2?
'sequential_9/reshape_18/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B : 2)
'sequential_9/reshape_18/Reshape/shape/3?
%sequential_9/reshape_18/Reshape/shapePack.sequential_9/reshape_18/strided_slice:output:00sequential_9/reshape_18/Reshape/shape/1:output:00sequential_9/reshape_18/Reshape/shape/2:output:00sequential_9/reshape_18/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2'
%sequential_9/reshape_18/Reshape/shape?
sequential_9/reshape_18/ReshapeReshape'sequential_9/dense_9/Relu:activations:0.sequential_9/reshape_18/Reshape/shape:output:0*
T0*/
_output_shapes
:????????? 2!
sequential_9/reshape_18/Reshape?
&sequential_9/conv2d_transpose_12/ShapeShape(sequential_9/reshape_18/Reshape:output:0*
T0*
_output_shapes
:2(
&sequential_9/conv2d_transpose_12/Shape?
4sequential_9/conv2d_transpose_12/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 26
4sequential_9/conv2d_transpose_12/strided_slice/stack?
6sequential_9/conv2d_transpose_12/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:28
6sequential_9/conv2d_transpose_12/strided_slice/stack_1?
6sequential_9/conv2d_transpose_12/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:28
6sequential_9/conv2d_transpose_12/strided_slice/stack_2?
.sequential_9/conv2d_transpose_12/strided_sliceStridedSlice/sequential_9/conv2d_transpose_12/Shape:output:0=sequential_9/conv2d_transpose_12/strided_slice/stack:output:0?sequential_9/conv2d_transpose_12/strided_slice/stack_1:output:0?sequential_9/conv2d_transpose_12/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask20
.sequential_9/conv2d_transpose_12/strided_slice?
(sequential_9/conv2d_transpose_12/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2*
(sequential_9/conv2d_transpose_12/stack/1?
(sequential_9/conv2d_transpose_12/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2*
(sequential_9/conv2d_transpose_12/stack/2?
(sequential_9/conv2d_transpose_12/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@2*
(sequential_9/conv2d_transpose_12/stack/3?
&sequential_9/conv2d_transpose_12/stackPack7sequential_9/conv2d_transpose_12/strided_slice:output:01sequential_9/conv2d_transpose_12/stack/1:output:01sequential_9/conv2d_transpose_12/stack/2:output:01sequential_9/conv2d_transpose_12/stack/3:output:0*
N*
T0*
_output_shapes
:2(
&sequential_9/conv2d_transpose_12/stack?
6sequential_9/conv2d_transpose_12/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 28
6sequential_9/conv2d_transpose_12/strided_slice_1/stack?
8sequential_9/conv2d_transpose_12/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2:
8sequential_9/conv2d_transpose_12/strided_slice_1/stack_1?
8sequential_9/conv2d_transpose_12/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2:
8sequential_9/conv2d_transpose_12/strided_slice_1/stack_2?
0sequential_9/conv2d_transpose_12/strided_slice_1StridedSlice/sequential_9/conv2d_transpose_12/stack:output:0?sequential_9/conv2d_transpose_12/strided_slice_1/stack:output:0Asequential_9/conv2d_transpose_12/strided_slice_1/stack_1:output:0Asequential_9/conv2d_transpose_12/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask22
0sequential_9/conv2d_transpose_12/strided_slice_1?
@sequential_9/conv2d_transpose_12/conv2d_transpose/ReadVariableOpReadVariableOpIsequential_9_conv2d_transpose_12_conv2d_transpose_readvariableop_resource*&
_output_shapes
:@ *
dtype02B
@sequential_9/conv2d_transpose_12/conv2d_transpose/ReadVariableOp?
1sequential_9/conv2d_transpose_12/conv2d_transposeConv2DBackpropInput/sequential_9/conv2d_transpose_12/stack:output:0Hsequential_9/conv2d_transpose_12/conv2d_transpose/ReadVariableOp:value:0(sequential_9/reshape_18/Reshape:output:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
23
1sequential_9/conv2d_transpose_12/conv2d_transpose?
7sequential_9/conv2d_transpose_12/BiasAdd/ReadVariableOpReadVariableOp@sequential_9_conv2d_transpose_12_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype029
7sequential_9/conv2d_transpose_12/BiasAdd/ReadVariableOp?
(sequential_9/conv2d_transpose_12/BiasAddBiasAdd:sequential_9/conv2d_transpose_12/conv2d_transpose:output:0?sequential_9/conv2d_transpose_12/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2*
(sequential_9/conv2d_transpose_12/BiasAdd?
%sequential_9/conv2d_transpose_12/ReluRelu1sequential_9/conv2d_transpose_12/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2'
%sequential_9/conv2d_transpose_12/Relu?
&sequential_9/conv2d_transpose_13/ShapeShape3sequential_9/conv2d_transpose_12/Relu:activations:0*
T0*
_output_shapes
:2(
&sequential_9/conv2d_transpose_13/Shape?
4sequential_9/conv2d_transpose_13/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 26
4sequential_9/conv2d_transpose_13/strided_slice/stack?
6sequential_9/conv2d_transpose_13/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:28
6sequential_9/conv2d_transpose_13/strided_slice/stack_1?
6sequential_9/conv2d_transpose_13/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:28
6sequential_9/conv2d_transpose_13/strided_slice/stack_2?
.sequential_9/conv2d_transpose_13/strided_sliceStridedSlice/sequential_9/conv2d_transpose_13/Shape:output:0=sequential_9/conv2d_transpose_13/strided_slice/stack:output:0?sequential_9/conv2d_transpose_13/strided_slice/stack_1:output:0?sequential_9/conv2d_transpose_13/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask20
.sequential_9/conv2d_transpose_13/strided_slice?
(sequential_9/conv2d_transpose_13/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2*
(sequential_9/conv2d_transpose_13/stack/1?
(sequential_9/conv2d_transpose_13/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2*
(sequential_9/conv2d_transpose_13/stack/2?
(sequential_9/conv2d_transpose_13/stack/3Const*
_output_shapes
: *
dtype0*
value	B : 2*
(sequential_9/conv2d_transpose_13/stack/3?
&sequential_9/conv2d_transpose_13/stackPack7sequential_9/conv2d_transpose_13/strided_slice:output:01sequential_9/conv2d_transpose_13/stack/1:output:01sequential_9/conv2d_transpose_13/stack/2:output:01sequential_9/conv2d_transpose_13/stack/3:output:0*
N*
T0*
_output_shapes
:2(
&sequential_9/conv2d_transpose_13/stack?
6sequential_9/conv2d_transpose_13/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 28
6sequential_9/conv2d_transpose_13/strided_slice_1/stack?
8sequential_9/conv2d_transpose_13/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2:
8sequential_9/conv2d_transpose_13/strided_slice_1/stack_1?
8sequential_9/conv2d_transpose_13/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2:
8sequential_9/conv2d_transpose_13/strided_slice_1/stack_2?
0sequential_9/conv2d_transpose_13/strided_slice_1StridedSlice/sequential_9/conv2d_transpose_13/stack:output:0?sequential_9/conv2d_transpose_13/strided_slice_1/stack:output:0Asequential_9/conv2d_transpose_13/strided_slice_1/stack_1:output:0Asequential_9/conv2d_transpose_13/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask22
0sequential_9/conv2d_transpose_13/strided_slice_1?
@sequential_9/conv2d_transpose_13/conv2d_transpose/ReadVariableOpReadVariableOpIsequential_9_conv2d_transpose_13_conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype02B
@sequential_9/conv2d_transpose_13/conv2d_transpose/ReadVariableOp?
1sequential_9/conv2d_transpose_13/conv2d_transposeConv2DBackpropInput/sequential_9/conv2d_transpose_13/stack:output:0Hsequential_9/conv2d_transpose_13/conv2d_transpose/ReadVariableOp:value:03sequential_9/conv2d_transpose_12/Relu:activations:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
23
1sequential_9/conv2d_transpose_13/conv2d_transpose?
7sequential_9/conv2d_transpose_13/BiasAdd/ReadVariableOpReadVariableOp@sequential_9_conv2d_transpose_13_biasadd_readvariableop_resource*
_output_shapes
: *
dtype029
7sequential_9/conv2d_transpose_13/BiasAdd/ReadVariableOp?
(sequential_9/conv2d_transpose_13/BiasAddBiasAdd:sequential_9/conv2d_transpose_13/conv2d_transpose:output:0?sequential_9/conv2d_transpose_13/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2*
(sequential_9/conv2d_transpose_13/BiasAdd?
%sequential_9/conv2d_transpose_13/ReluRelu1sequential_9/conv2d_transpose_13/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2'
%sequential_9/conv2d_transpose_13/Relu?
&sequential_9/conv2d_transpose_14/ShapeShape3sequential_9/conv2d_transpose_13/Relu:activations:0*
T0*
_output_shapes
:2(
&sequential_9/conv2d_transpose_14/Shape?
4sequential_9/conv2d_transpose_14/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 26
4sequential_9/conv2d_transpose_14/strided_slice/stack?
6sequential_9/conv2d_transpose_14/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:28
6sequential_9/conv2d_transpose_14/strided_slice/stack_1?
6sequential_9/conv2d_transpose_14/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:28
6sequential_9/conv2d_transpose_14/strided_slice/stack_2?
.sequential_9/conv2d_transpose_14/strided_sliceStridedSlice/sequential_9/conv2d_transpose_14/Shape:output:0=sequential_9/conv2d_transpose_14/strided_slice/stack:output:0?sequential_9/conv2d_transpose_14/strided_slice/stack_1:output:0?sequential_9/conv2d_transpose_14/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask20
.sequential_9/conv2d_transpose_14/strided_slice?
(sequential_9/conv2d_transpose_14/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2*
(sequential_9/conv2d_transpose_14/stack/1?
(sequential_9/conv2d_transpose_14/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2*
(sequential_9/conv2d_transpose_14/stack/2?
(sequential_9/conv2d_transpose_14/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2*
(sequential_9/conv2d_transpose_14/stack/3?
&sequential_9/conv2d_transpose_14/stackPack7sequential_9/conv2d_transpose_14/strided_slice:output:01sequential_9/conv2d_transpose_14/stack/1:output:01sequential_9/conv2d_transpose_14/stack/2:output:01sequential_9/conv2d_transpose_14/stack/3:output:0*
N*
T0*
_output_shapes
:2(
&sequential_9/conv2d_transpose_14/stack?
6sequential_9/conv2d_transpose_14/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 28
6sequential_9/conv2d_transpose_14/strided_slice_1/stack?
8sequential_9/conv2d_transpose_14/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2:
8sequential_9/conv2d_transpose_14/strided_slice_1/stack_1?
8sequential_9/conv2d_transpose_14/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2:
8sequential_9/conv2d_transpose_14/strided_slice_1/stack_2?
0sequential_9/conv2d_transpose_14/strided_slice_1StridedSlice/sequential_9/conv2d_transpose_14/stack:output:0?sequential_9/conv2d_transpose_14/strided_slice_1/stack:output:0Asequential_9/conv2d_transpose_14/strided_slice_1/stack_1:output:0Asequential_9/conv2d_transpose_14/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask22
0sequential_9/conv2d_transpose_14/strided_slice_1?
@sequential_9/conv2d_transpose_14/conv2d_transpose/ReadVariableOpReadVariableOpIsequential_9_conv2d_transpose_14_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype02B
@sequential_9/conv2d_transpose_14/conv2d_transpose/ReadVariableOp?
1sequential_9/conv2d_transpose_14/conv2d_transposeConv2DBackpropInput/sequential_9/conv2d_transpose_14/stack:output:0Hsequential_9/conv2d_transpose_14/conv2d_transpose/ReadVariableOp:value:03sequential_9/conv2d_transpose_13/Relu:activations:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
23
1sequential_9/conv2d_transpose_14/conv2d_transpose?
7sequential_9/conv2d_transpose_14/BiasAdd/ReadVariableOpReadVariableOp@sequential_9_conv2d_transpose_14_biasadd_readvariableop_resource*
_output_shapes
:*
dtype029
7sequential_9/conv2d_transpose_14/BiasAdd/ReadVariableOp?
(sequential_9/conv2d_transpose_14/BiasAddBiasAdd:sequential_9/conv2d_transpose_14/conv2d_transpose:output:0?sequential_9/conv2d_transpose_14/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2*
(sequential_9/conv2d_transpose_14/BiasAdd?
(sequential_9/conv2d_transpose_14/SigmoidSigmoid1sequential_9/conv2d_transpose_14/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2*
(sequential_9/conv2d_transpose_14/Sigmoid?
sequential_9/reshape_19/ShapeShape,sequential_9/conv2d_transpose_14/Sigmoid:y:0*
T0*
_output_shapes
:2
sequential_9/reshape_19/Shape?
+sequential_9/reshape_19/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2-
+sequential_9/reshape_19/strided_slice/stack?
-sequential_9/reshape_19/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_9/reshape_19/strided_slice/stack_1?
-sequential_9/reshape_19/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_9/reshape_19/strided_slice/stack_2?
%sequential_9/reshape_19/strided_sliceStridedSlice&sequential_9/reshape_19/Shape:output:04sequential_9/reshape_19/strided_slice/stack:output:06sequential_9/reshape_19/strided_slice/stack_1:output:06sequential_9/reshape_19/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%sequential_9/reshape_19/strided_slice?
'sequential_9/reshape_19/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2)
'sequential_9/reshape_19/Reshape/shape/1?
'sequential_9/reshape_19/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2)
'sequential_9/reshape_19/Reshape/shape/2?
'sequential_9/reshape_19/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2)
'sequential_9/reshape_19/Reshape/shape/3?
%sequential_9/reshape_19/Reshape/shapePack.sequential_9/reshape_19/strided_slice:output:00sequential_9/reshape_19/Reshape/shape/1:output:00sequential_9/reshape_19/Reshape/shape/2:output:00sequential_9/reshape_19/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2'
%sequential_9/reshape_19/Reshape/shape?
sequential_9/reshape_19/ReshapeReshape,sequential_9/conv2d_transpose_14/Sigmoid:y:0.sequential_9/reshape_19/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????2!
sequential_9/reshape_19/Reshape?
IdentityIdentity(sequential_9/reshape_19/Reshape:output:0^NoOp*
T0*/
_output_shapes
:?????????2

Identity?
NoOpNoOp-^sequential_8/conv2d_8/BiasAdd/ReadVariableOp,^sequential_8/conv2d_8/Conv2D/ReadVariableOp-^sequential_8/conv2d_9/BiasAdd/ReadVariableOp,^sequential_8/conv2d_9/Conv2D/ReadVariableOp,^sequential_8/dense_8/BiasAdd/ReadVariableOp+^sequential_8/dense_8/MatMul/ReadVariableOp8^sequential_9/conv2d_transpose_12/BiasAdd/ReadVariableOpA^sequential_9/conv2d_transpose_12/conv2d_transpose/ReadVariableOp8^sequential_9/conv2d_transpose_13/BiasAdd/ReadVariableOpA^sequential_9/conv2d_transpose_13/conv2d_transpose/ReadVariableOp8^sequential_9/conv2d_transpose_14/BiasAdd/ReadVariableOpA^sequential_9/conv2d_transpose_14/conv2d_transpose/ReadVariableOp,^sequential_9/dense_9/BiasAdd/ReadVariableOp+^sequential_9/dense_9/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:?????????: : : : : : : : : : : : : : 2\
,sequential_8/conv2d_8/BiasAdd/ReadVariableOp,sequential_8/conv2d_8/BiasAdd/ReadVariableOp2Z
+sequential_8/conv2d_8/Conv2D/ReadVariableOp+sequential_8/conv2d_8/Conv2D/ReadVariableOp2\
,sequential_8/conv2d_9/BiasAdd/ReadVariableOp,sequential_8/conv2d_9/BiasAdd/ReadVariableOp2Z
+sequential_8/conv2d_9/Conv2D/ReadVariableOp+sequential_8/conv2d_9/Conv2D/ReadVariableOp2Z
+sequential_8/dense_8/BiasAdd/ReadVariableOp+sequential_8/dense_8/BiasAdd/ReadVariableOp2X
*sequential_8/dense_8/MatMul/ReadVariableOp*sequential_8/dense_8/MatMul/ReadVariableOp2r
7sequential_9/conv2d_transpose_12/BiasAdd/ReadVariableOp7sequential_9/conv2d_transpose_12/BiasAdd/ReadVariableOp2?
@sequential_9/conv2d_transpose_12/conv2d_transpose/ReadVariableOp@sequential_9/conv2d_transpose_12/conv2d_transpose/ReadVariableOp2r
7sequential_9/conv2d_transpose_13/BiasAdd/ReadVariableOp7sequential_9/conv2d_transpose_13/BiasAdd/ReadVariableOp2?
@sequential_9/conv2d_transpose_13/conv2d_transpose/ReadVariableOp@sequential_9/conv2d_transpose_13/conv2d_transpose/ReadVariableOp2r
7sequential_9/conv2d_transpose_14/BiasAdd/ReadVariableOp7sequential_9/conv2d_transpose_14/BiasAdd/ReadVariableOp2?
@sequential_9/conv2d_transpose_14/conv2d_transpose/ReadVariableOp@sequential_9/conv2d_transpose_14/conv2d_transpose/ReadVariableOp2Z
+sequential_9/dense_9/BiasAdd/ReadVariableOp+sequential_9/dense_9/BiasAdd/ReadVariableOp2X
*sequential_9/dense_9/MatMul/ReadVariableOp*sequential_9/dense_9/MatMul/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
9__inference_variational_autoencoder_4_layer_call_fn_78523

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
T__inference_variational_autoencoder_4_layer_call_and_return_conditional_losses_779932
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
??
?
G__inference_sequential_9_layer_call_and_return_conditional_losses_78829

inputs9
&dense_9_matmul_readvariableop_resource:	?6
'dense_9_biasadd_readvariableop_resource:	?V
<conv2d_transpose_12_conv2d_transpose_readvariableop_resource:@ A
3conv2d_transpose_12_biasadd_readvariableop_resource:@V
<conv2d_transpose_13_conv2d_transpose_readvariableop_resource: @A
3conv2d_transpose_13_biasadd_readvariableop_resource: V
<conv2d_transpose_14_conv2d_transpose_readvariableop_resource: A
3conv2d_transpose_14_biasadd_readvariableop_resource:
identity??*conv2d_transpose_12/BiasAdd/ReadVariableOp?3conv2d_transpose_12/conv2d_transpose/ReadVariableOp?*conv2d_transpose_13/BiasAdd/ReadVariableOp?3conv2d_transpose_13/conv2d_transpose/ReadVariableOp?*conv2d_transpose_14/BiasAdd/ReadVariableOp?3conv2d_transpose_14/conv2d_transpose/ReadVariableOp?dense_9/BiasAdd/ReadVariableOp?dense_9/MatMul/ReadVariableOp?
dense_9/MatMul/ReadVariableOpReadVariableOp&dense_9_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
dense_9/MatMul/ReadVariableOp?
dense_9/MatMulMatMulinputs%dense_9/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_9/MatMul?
dense_9/BiasAdd/ReadVariableOpReadVariableOp'dense_9_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
dense_9/BiasAdd/ReadVariableOp?
dense_9/BiasAddBiasAdddense_9/MatMul:product:0&dense_9/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_9/BiasAddq
dense_9/ReluReludense_9/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_9/Relun
reshape_18/ShapeShapedense_9/Relu:activations:0*
T0*
_output_shapes
:2
reshape_18/Shape?
reshape_18/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
reshape_18/strided_slice/stack?
 reshape_18/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_18/strided_slice/stack_1?
 reshape_18/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_18/strided_slice/stack_2?
reshape_18/strided_sliceStridedSlicereshape_18/Shape:output:0'reshape_18/strided_slice/stack:output:0)reshape_18/strided_slice/stack_1:output:0)reshape_18/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_18/strided_slicez
reshape_18/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_18/Reshape/shape/1z
reshape_18/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_18/Reshape/shape/2z
reshape_18/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B : 2
reshape_18/Reshape/shape/3?
reshape_18/Reshape/shapePack!reshape_18/strided_slice:output:0#reshape_18/Reshape/shape/1:output:0#reshape_18/Reshape/shape/2:output:0#reshape_18/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
reshape_18/Reshape/shape?
reshape_18/ReshapeReshapedense_9/Relu:activations:0!reshape_18/Reshape/shape:output:0*
T0*/
_output_shapes
:????????? 2
reshape_18/Reshape?
conv2d_transpose_12/ShapeShapereshape_18/Reshape:output:0*
T0*
_output_shapes
:2
conv2d_transpose_12/Shape?
'conv2d_transpose_12/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'conv2d_transpose_12/strided_slice/stack?
)conv2d_transpose_12/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_12/strided_slice/stack_1?
)conv2d_transpose_12/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_12/strided_slice/stack_2?
!conv2d_transpose_12/strided_sliceStridedSlice"conv2d_transpose_12/Shape:output:00conv2d_transpose_12/strided_slice/stack:output:02conv2d_transpose_12/strided_slice/stack_1:output:02conv2d_transpose_12/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!conv2d_transpose_12/strided_slice|
conv2d_transpose_12/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_12/stack/1|
conv2d_transpose_12/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_12/stack/2|
conv2d_transpose_12/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@2
conv2d_transpose_12/stack/3?
conv2d_transpose_12/stackPack*conv2d_transpose_12/strided_slice:output:0$conv2d_transpose_12/stack/1:output:0$conv2d_transpose_12/stack/2:output:0$conv2d_transpose_12/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_12/stack?
)conv2d_transpose_12/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)conv2d_transpose_12/strided_slice_1/stack?
+conv2d_transpose_12/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_12/strided_slice_1/stack_1?
+conv2d_transpose_12/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_12/strided_slice_1/stack_2?
#conv2d_transpose_12/strided_slice_1StridedSlice"conv2d_transpose_12/stack:output:02conv2d_transpose_12/strided_slice_1/stack:output:04conv2d_transpose_12/strided_slice_1/stack_1:output:04conv2d_transpose_12/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#conv2d_transpose_12/strided_slice_1?
3conv2d_transpose_12/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_12_conv2d_transpose_readvariableop_resource*&
_output_shapes
:@ *
dtype025
3conv2d_transpose_12/conv2d_transpose/ReadVariableOp?
$conv2d_transpose_12/conv2d_transposeConv2DBackpropInput"conv2d_transpose_12/stack:output:0;conv2d_transpose_12/conv2d_transpose/ReadVariableOp:value:0reshape_18/Reshape:output:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2&
$conv2d_transpose_12/conv2d_transpose?
*conv2d_transpose_12/BiasAdd/ReadVariableOpReadVariableOp3conv2d_transpose_12_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02,
*conv2d_transpose_12/BiasAdd/ReadVariableOp?
conv2d_transpose_12/BiasAddBiasAdd-conv2d_transpose_12/conv2d_transpose:output:02conv2d_transpose_12/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
conv2d_transpose_12/BiasAdd?
conv2d_transpose_12/ReluRelu$conv2d_transpose_12/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
conv2d_transpose_12/Relu?
conv2d_transpose_13/ShapeShape&conv2d_transpose_12/Relu:activations:0*
T0*
_output_shapes
:2
conv2d_transpose_13/Shape?
'conv2d_transpose_13/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'conv2d_transpose_13/strided_slice/stack?
)conv2d_transpose_13/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_13/strided_slice/stack_1?
)conv2d_transpose_13/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_13/strided_slice/stack_2?
!conv2d_transpose_13/strided_sliceStridedSlice"conv2d_transpose_13/Shape:output:00conv2d_transpose_13/strided_slice/stack:output:02conv2d_transpose_13/strided_slice/stack_1:output:02conv2d_transpose_13/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!conv2d_transpose_13/strided_slice|
conv2d_transpose_13/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_13/stack/1|
conv2d_transpose_13/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_13/stack/2|
conv2d_transpose_13/stack/3Const*
_output_shapes
: *
dtype0*
value	B : 2
conv2d_transpose_13/stack/3?
conv2d_transpose_13/stackPack*conv2d_transpose_13/strided_slice:output:0$conv2d_transpose_13/stack/1:output:0$conv2d_transpose_13/stack/2:output:0$conv2d_transpose_13/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_13/stack?
)conv2d_transpose_13/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)conv2d_transpose_13/strided_slice_1/stack?
+conv2d_transpose_13/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_13/strided_slice_1/stack_1?
+conv2d_transpose_13/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_13/strided_slice_1/stack_2?
#conv2d_transpose_13/strided_slice_1StridedSlice"conv2d_transpose_13/stack:output:02conv2d_transpose_13/strided_slice_1/stack:output:04conv2d_transpose_13/strided_slice_1/stack_1:output:04conv2d_transpose_13/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#conv2d_transpose_13/strided_slice_1?
3conv2d_transpose_13/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_13_conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype025
3conv2d_transpose_13/conv2d_transpose/ReadVariableOp?
$conv2d_transpose_13/conv2d_transposeConv2DBackpropInput"conv2d_transpose_13/stack:output:0;conv2d_transpose_13/conv2d_transpose/ReadVariableOp:value:0&conv2d_transpose_12/Relu:activations:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
2&
$conv2d_transpose_13/conv2d_transpose?
*conv2d_transpose_13/BiasAdd/ReadVariableOpReadVariableOp3conv2d_transpose_13_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02,
*conv2d_transpose_13/BiasAdd/ReadVariableOp?
conv2d_transpose_13/BiasAddBiasAdd-conv2d_transpose_13/conv2d_transpose:output:02conv2d_transpose_13/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2
conv2d_transpose_13/BiasAdd?
conv2d_transpose_13/ReluRelu$conv2d_transpose_13/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
conv2d_transpose_13/Relu?
conv2d_transpose_14/ShapeShape&conv2d_transpose_13/Relu:activations:0*
T0*
_output_shapes
:2
conv2d_transpose_14/Shape?
'conv2d_transpose_14/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'conv2d_transpose_14/strided_slice/stack?
)conv2d_transpose_14/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_14/strided_slice/stack_1?
)conv2d_transpose_14/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_14/strided_slice/stack_2?
!conv2d_transpose_14/strided_sliceStridedSlice"conv2d_transpose_14/Shape:output:00conv2d_transpose_14/strided_slice/stack:output:02conv2d_transpose_14/strided_slice/stack_1:output:02conv2d_transpose_14/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!conv2d_transpose_14/strided_slice|
conv2d_transpose_14/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_14/stack/1|
conv2d_transpose_14/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_14/stack/2|
conv2d_transpose_14/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_14/stack/3?
conv2d_transpose_14/stackPack*conv2d_transpose_14/strided_slice:output:0$conv2d_transpose_14/stack/1:output:0$conv2d_transpose_14/stack/2:output:0$conv2d_transpose_14/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_14/stack?
)conv2d_transpose_14/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)conv2d_transpose_14/strided_slice_1/stack?
+conv2d_transpose_14/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_14/strided_slice_1/stack_1?
+conv2d_transpose_14/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_14/strided_slice_1/stack_2?
#conv2d_transpose_14/strided_slice_1StridedSlice"conv2d_transpose_14/stack:output:02conv2d_transpose_14/strided_slice_1/stack:output:04conv2d_transpose_14/strided_slice_1/stack_1:output:04conv2d_transpose_14/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#conv2d_transpose_14/strided_slice_1?
3conv2d_transpose_14/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_14_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype025
3conv2d_transpose_14/conv2d_transpose/ReadVariableOp?
$conv2d_transpose_14/conv2d_transposeConv2DBackpropInput"conv2d_transpose_14/stack:output:0;conv2d_transpose_14/conv2d_transpose/ReadVariableOp:value:0&conv2d_transpose_13/Relu:activations:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2&
$conv2d_transpose_14/conv2d_transpose?
*conv2d_transpose_14/BiasAdd/ReadVariableOpReadVariableOp3conv2d_transpose_14_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02,
*conv2d_transpose_14/BiasAdd/ReadVariableOp?
conv2d_transpose_14/BiasAddBiasAdd-conv2d_transpose_14/conv2d_transpose:output:02conv2d_transpose_14/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
conv2d_transpose_14/BiasAdd?
conv2d_transpose_14/SigmoidSigmoid$conv2d_transpose_14/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
conv2d_transpose_14/Sigmoids
reshape_19/ShapeShapeconv2d_transpose_14/Sigmoid:y:0*
T0*
_output_shapes
:2
reshape_19/Shape?
reshape_19/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
reshape_19/strided_slice/stack?
 reshape_19/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_19/strided_slice/stack_1?
 reshape_19/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_19/strided_slice/stack_2?
reshape_19/strided_sliceStridedSlicereshape_19/Shape:output:0'reshape_19/strided_slice/stack:output:0)reshape_19/strided_slice/stack_1:output:0)reshape_19/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_19/strided_slicez
reshape_19/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_19/Reshape/shape/1z
reshape_19/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_19/Reshape/shape/2z
reshape_19/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_19/Reshape/shape/3?
reshape_19/Reshape/shapePack!reshape_19/strided_slice:output:0#reshape_19/Reshape/shape/1:output:0#reshape_19/Reshape/shape/2:output:0#reshape_19/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
reshape_19/Reshape/shape?
reshape_19/ReshapeReshapeconv2d_transpose_14/Sigmoid:y:0!reshape_19/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????2
reshape_19/Reshape~
IdentityIdentityreshape_19/Reshape:output:0^NoOp*
T0*/
_output_shapes
:?????????2

Identity?
NoOpNoOp+^conv2d_transpose_12/BiasAdd/ReadVariableOp4^conv2d_transpose_12/conv2d_transpose/ReadVariableOp+^conv2d_transpose_13/BiasAdd/ReadVariableOp4^conv2d_transpose_13/conv2d_transpose/ReadVariableOp+^conv2d_transpose_14/BiasAdd/ReadVariableOp4^conv2d_transpose_14/conv2d_transpose/ReadVariableOp^dense_9/BiasAdd/ReadVariableOp^dense_9/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : : : : : 2X
*conv2d_transpose_12/BiasAdd/ReadVariableOp*conv2d_transpose_12/BiasAdd/ReadVariableOp2j
3conv2d_transpose_12/conv2d_transpose/ReadVariableOp3conv2d_transpose_12/conv2d_transpose/ReadVariableOp2X
*conv2d_transpose_13/BiasAdd/ReadVariableOp*conv2d_transpose_13/BiasAdd/ReadVariableOp2j
3conv2d_transpose_13/conv2d_transpose/ReadVariableOp3conv2d_transpose_13/conv2d_transpose/ReadVariableOp2X
*conv2d_transpose_14/BiasAdd/ReadVariableOp*conv2d_transpose_14/BiasAdd/ReadVariableOp2j
3conv2d_transpose_14/conv2d_transpose/ReadVariableOp3conv2d_transpose_14/conv2d_transpose/ReadVariableOp2@
dense_9/BiasAdd/ReadVariableOpdense_9/BiasAdd/ReadVariableOp2>
dense_9/MatMul/ReadVariableOpdense_9/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
,__inference_sequential_9_layer_call_fn_78850

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
G__inference_sequential_9_layer_call_and_return_conditional_losses_775992
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
?!
?
G__inference_sequential_9_layer_call_and_return_conditional_losses_77716

inputs 
dense_9_77693:	?
dense_9_77695:	?3
conv2d_transpose_12_77699:@ '
conv2d_transpose_12_77701:@3
conv2d_transpose_13_77704: @'
conv2d_transpose_13_77706: 3
conv2d_transpose_14_77709: '
conv2d_transpose_14_77711:
identity??+conv2d_transpose_12/StatefulPartitionedCall?+conv2d_transpose_13/StatefulPartitionedCall?+conv2d_transpose_14/StatefulPartitionedCall?dense_9/StatefulPartitionedCall?
dense_9/StatefulPartitionedCallStatefulPartitionedCallinputsdense_9_77693dense_9_77695*
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
B__inference_dense_9_layer_call_and_return_conditional_losses_774732!
dense_9/StatefulPartitionedCall?
reshape_18/PartitionedCallPartitionedCall(dense_9/StatefulPartitionedCall:output:0*
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
E__inference_reshape_18_layer_call_and_return_conditional_losses_774932
reshape_18/PartitionedCall?
+conv2d_transpose_12/StatefulPartitionedCallStatefulPartitionedCall#reshape_18/PartitionedCall:output:0conv2d_transpose_12_77699conv2d_transpose_12_77701*
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
GPU2*0J 8? *W
fRRP
N__inference_conv2d_transpose_12_layer_call_and_return_conditional_losses_775182-
+conv2d_transpose_12/StatefulPartitionedCall?
+conv2d_transpose_13/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_12/StatefulPartitionedCall:output:0conv2d_transpose_13_77704conv2d_transpose_13_77706*
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
GPU2*0J 8? *W
fRRP
N__inference_conv2d_transpose_13_layer_call_and_return_conditional_losses_775472-
+conv2d_transpose_13/StatefulPartitionedCall?
+conv2d_transpose_14/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_13/StatefulPartitionedCall:output:0conv2d_transpose_14_77709conv2d_transpose_14_77711*
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
GPU2*0J 8? *W
fRRP
N__inference_conv2d_transpose_14_layer_call_and_return_conditional_losses_775762-
+conv2d_transpose_14/StatefulPartitionedCall?
reshape_19/PartitionedCallPartitionedCall4conv2d_transpose_14/StatefulPartitionedCall:output:0*
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
E__inference_reshape_19_layer_call_and_return_conditional_losses_775962
reshape_19/PartitionedCall?
IdentityIdentity#reshape_19/PartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????2

Identity?
NoOpNoOp,^conv2d_transpose_12/StatefulPartitionedCall,^conv2d_transpose_13/StatefulPartitionedCall,^conv2d_transpose_14/StatefulPartitionedCall ^dense_9/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : : : : : 2Z
+conv2d_transpose_12/StatefulPartitionedCall+conv2d_transpose_12/StatefulPartitionedCall2Z
+conv2d_transpose_13/StatefulPartitionedCall+conv2d_transpose_13/StatefulPartitionedCall2Z
+conv2d_transpose_14/StatefulPartitionedCall+conv2d_transpose_14/StatefulPartitionedCall2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
S
7__inference_normal_sampling_layer_4_layer_call_fn_78909

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
R__inference_normal_sampling_layer_4_layer_call_and_return_conditional_losses_778342
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
?&
?
N__inference_conv2d_transpose_14_layer_call_and_return_conditional_losses_77405

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
?	
?
,__inference_sequential_8_layer_call_fn_77147
input_17!
unknown: 
	unknown_0: #
	unknown_1: @
	unknown_2:@
	unknown_3:	?
	unknown_4:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_17unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
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
G__inference_sequential_8_layer_call_and_return_conditional_losses_771152
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
StatefulPartitionedCallStatefulPartitionedCall:Y U
/
_output_shapes
:?????????
"
_user_specified_name
input_17
?
?
3__inference_conv2d_transpose_12_layer_call_fn_79127

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
GPU2*0J 8? *W
fRRP
N__inference_conv2d_transpose_12_layer_call_and_return_conditional_losses_772292
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
?

?
,__inference_sequential_9_layer_call_fn_77756
input_18
unknown:	?
	unknown_0:	?#
	unknown_1:@ 
	unknown_2:@#
	unknown_3: @
	unknown_4: #
	unknown_5: 
	unknown_6:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_18unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
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
G__inference_sequential_9_layer_call_and_return_conditional_losses_777162
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
input_18
?!
?
G__inference_sequential_9_layer_call_and_return_conditional_losses_77782
input_18 
dense_9_77759:	?
dense_9_77761:	?3
conv2d_transpose_12_77765:@ '
conv2d_transpose_12_77767:@3
conv2d_transpose_13_77770: @'
conv2d_transpose_13_77772: 3
conv2d_transpose_14_77775: '
conv2d_transpose_14_77777:
identity??+conv2d_transpose_12/StatefulPartitionedCall?+conv2d_transpose_13/StatefulPartitionedCall?+conv2d_transpose_14/StatefulPartitionedCall?dense_9/StatefulPartitionedCall?
dense_9/StatefulPartitionedCallStatefulPartitionedCallinput_18dense_9_77759dense_9_77761*
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
B__inference_dense_9_layer_call_and_return_conditional_losses_774732!
dense_9/StatefulPartitionedCall?
reshape_18/PartitionedCallPartitionedCall(dense_9/StatefulPartitionedCall:output:0*
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
E__inference_reshape_18_layer_call_and_return_conditional_losses_774932
reshape_18/PartitionedCall?
+conv2d_transpose_12/StatefulPartitionedCallStatefulPartitionedCall#reshape_18/PartitionedCall:output:0conv2d_transpose_12_77765conv2d_transpose_12_77767*
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
GPU2*0J 8? *W
fRRP
N__inference_conv2d_transpose_12_layer_call_and_return_conditional_losses_775182-
+conv2d_transpose_12/StatefulPartitionedCall?
+conv2d_transpose_13/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_12/StatefulPartitionedCall:output:0conv2d_transpose_13_77770conv2d_transpose_13_77772*
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
GPU2*0J 8? *W
fRRP
N__inference_conv2d_transpose_13_layer_call_and_return_conditional_losses_775472-
+conv2d_transpose_13/StatefulPartitionedCall?
+conv2d_transpose_14/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_13/StatefulPartitionedCall:output:0conv2d_transpose_14_77775conv2d_transpose_14_77777*
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
GPU2*0J 8? *W
fRRP
N__inference_conv2d_transpose_14_layer_call_and_return_conditional_losses_775762-
+conv2d_transpose_14/StatefulPartitionedCall?
reshape_19/PartitionedCallPartitionedCall4conv2d_transpose_14/StatefulPartitionedCall:output:0*
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
E__inference_reshape_19_layer_call_and_return_conditional_losses_775962
reshape_19/PartitionedCall?
IdentityIdentity#reshape_19/PartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????2

Identity?
NoOpNoOp,^conv2d_transpose_12/StatefulPartitionedCall,^conv2d_transpose_13/StatefulPartitionedCall,^conv2d_transpose_14/StatefulPartitionedCall ^dense_9/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : : : : : 2Z
+conv2d_transpose_12/StatefulPartitionedCall+conv2d_transpose_12/StatefulPartitionedCall2Z
+conv2d_transpose_13/StatefulPartitionedCall+conv2d_transpose_13/StatefulPartitionedCall2Z
+conv2d_transpose_14/StatefulPartitionedCall+conv2d_transpose_14/StatefulPartitionedCall2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall:Q M
'
_output_shapes
:?????????
"
_user_specified_name
input_18
??
?
T__inference_variational_autoencoder_4_layer_call_and_return_conditional_losses_78457

inputsN
4sequential_8_conv2d_8_conv2d_readvariableop_resource: C
5sequential_8_conv2d_8_biasadd_readvariableop_resource: N
4sequential_8_conv2d_9_conv2d_readvariableop_resource: @C
5sequential_8_conv2d_9_biasadd_readvariableop_resource:@F
3sequential_8_dense_8_matmul_readvariableop_resource:	?B
4sequential_8_dense_8_biasadd_readvariableop_resource:F
3sequential_9_dense_9_matmul_readvariableop_resource:	?C
4sequential_9_dense_9_biasadd_readvariableop_resource:	?c
Isequential_9_conv2d_transpose_12_conv2d_transpose_readvariableop_resource:@ N
@sequential_9_conv2d_transpose_12_biasadd_readvariableop_resource:@c
Isequential_9_conv2d_transpose_13_conv2d_transpose_readvariableop_resource: @N
@sequential_9_conv2d_transpose_13_biasadd_readvariableop_resource: c
Isequential_9_conv2d_transpose_14_conv2d_transpose_readvariableop_resource: N
@sequential_9_conv2d_transpose_14_biasadd_readvariableop_resource:
identity??,sequential_8/conv2d_8/BiasAdd/ReadVariableOp?+sequential_8/conv2d_8/Conv2D/ReadVariableOp?,sequential_8/conv2d_9/BiasAdd/ReadVariableOp?+sequential_8/conv2d_9/Conv2D/ReadVariableOp?+sequential_8/dense_8/BiasAdd/ReadVariableOp?*sequential_8/dense_8/MatMul/ReadVariableOp?7sequential_9/conv2d_transpose_12/BiasAdd/ReadVariableOp?@sequential_9/conv2d_transpose_12/conv2d_transpose/ReadVariableOp?7sequential_9/conv2d_transpose_13/BiasAdd/ReadVariableOp?@sequential_9/conv2d_transpose_13/conv2d_transpose/ReadVariableOp?7sequential_9/conv2d_transpose_14/BiasAdd/ReadVariableOp?@sequential_9/conv2d_transpose_14/conv2d_transpose/ReadVariableOp?+sequential_9/dense_9/BiasAdd/ReadVariableOp?*sequential_9/dense_9/MatMul/ReadVariableOpt
sequential_8/reshape_16/ShapeShapeinputs*
T0*
_output_shapes
:2
sequential_8/reshape_16/Shape?
+sequential_8/reshape_16/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2-
+sequential_8/reshape_16/strided_slice/stack?
-sequential_8/reshape_16/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_8/reshape_16/strided_slice/stack_1?
-sequential_8/reshape_16/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_8/reshape_16/strided_slice/stack_2?
%sequential_8/reshape_16/strided_sliceStridedSlice&sequential_8/reshape_16/Shape:output:04sequential_8/reshape_16/strided_slice/stack:output:06sequential_8/reshape_16/strided_slice/stack_1:output:06sequential_8/reshape_16/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%sequential_8/reshape_16/strided_slice?
'sequential_8/reshape_16/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2)
'sequential_8/reshape_16/Reshape/shape/1?
'sequential_8/reshape_16/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2)
'sequential_8/reshape_16/Reshape/shape/2?
'sequential_8/reshape_16/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2)
'sequential_8/reshape_16/Reshape/shape/3?
%sequential_8/reshape_16/Reshape/shapePack.sequential_8/reshape_16/strided_slice:output:00sequential_8/reshape_16/Reshape/shape/1:output:00sequential_8/reshape_16/Reshape/shape/2:output:00sequential_8/reshape_16/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2'
%sequential_8/reshape_16/Reshape/shape?
sequential_8/reshape_16/ReshapeReshapeinputs.sequential_8/reshape_16/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????2!
sequential_8/reshape_16/Reshape?
+sequential_8/conv2d_8/Conv2D/ReadVariableOpReadVariableOp4sequential_8_conv2d_8_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02-
+sequential_8/conv2d_8/Conv2D/ReadVariableOp?
sequential_8/conv2d_8/Conv2DConv2D(sequential_8/reshape_16/Reshape:output:03sequential_8/conv2d_8/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
2
sequential_8/conv2d_8/Conv2D?
,sequential_8/conv2d_8/BiasAdd/ReadVariableOpReadVariableOp5sequential_8_conv2d_8_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02.
,sequential_8/conv2d_8/BiasAdd/ReadVariableOp?
sequential_8/conv2d_8/BiasAddBiasAdd%sequential_8/conv2d_8/Conv2D:output:04sequential_8/conv2d_8/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2
sequential_8/conv2d_8/BiasAdd?
sequential_8/conv2d_8/ReluRelu&sequential_8/conv2d_8/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
sequential_8/conv2d_8/Relu?
+sequential_8/conv2d_9/Conv2D/ReadVariableOpReadVariableOp4sequential_8_conv2d_9_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02-
+sequential_8/conv2d_9/Conv2D/ReadVariableOp?
sequential_8/conv2d_9/Conv2DConv2D(sequential_8/conv2d_8/Relu:activations:03sequential_8/conv2d_9/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
2
sequential_8/conv2d_9/Conv2D?
,sequential_8/conv2d_9/BiasAdd/ReadVariableOpReadVariableOp5sequential_8_conv2d_9_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02.
,sequential_8/conv2d_9/BiasAdd/ReadVariableOp?
sequential_8/conv2d_9/BiasAddBiasAdd%sequential_8/conv2d_9/Conv2D:output:04sequential_8/conv2d_9/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
sequential_8/conv2d_9/BiasAdd?
sequential_8/conv2d_9/ReluRelu&sequential_8/conv2d_9/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
sequential_8/conv2d_9/Relu?
sequential_8/flatten_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"???? 	  2
sequential_8/flatten_4/Const?
sequential_8/flatten_4/ReshapeReshape(sequential_8/conv2d_9/Relu:activations:0%sequential_8/flatten_4/Const:output:0*
T0*(
_output_shapes
:??????????2 
sequential_8/flatten_4/Reshape?
*sequential_8/dense_8/MatMul/ReadVariableOpReadVariableOp3sequential_8_dense_8_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02,
*sequential_8/dense_8/MatMul/ReadVariableOp?
sequential_8/dense_8/MatMulMatMul'sequential_8/flatten_4/Reshape:output:02sequential_8/dense_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential_8/dense_8/MatMul?
+sequential_8/dense_8/BiasAdd/ReadVariableOpReadVariableOp4sequential_8_dense_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02-
+sequential_8/dense_8/BiasAdd/ReadVariableOp?
sequential_8/dense_8/BiasAddBiasAdd%sequential_8/dense_8/MatMul:product:03sequential_8/dense_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential_8/dense_8/BiasAdd?
sequential_8/reshape_17/ShapeShape%sequential_8/dense_8/BiasAdd:output:0*
T0*
_output_shapes
:2
sequential_8/reshape_17/Shape?
+sequential_8/reshape_17/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2-
+sequential_8/reshape_17/strided_slice/stack?
-sequential_8/reshape_17/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_8/reshape_17/strided_slice/stack_1?
-sequential_8/reshape_17/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_8/reshape_17/strided_slice/stack_2?
%sequential_8/reshape_17/strided_sliceStridedSlice&sequential_8/reshape_17/Shape:output:04sequential_8/reshape_17/strided_slice/stack:output:06sequential_8/reshape_17/strided_slice/stack_1:output:06sequential_8/reshape_17/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%sequential_8/reshape_17/strided_slice?
'sequential_8/reshape_17/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2)
'sequential_8/reshape_17/Reshape/shape/1?
'sequential_8/reshape_17/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2)
'sequential_8/reshape_17/Reshape/shape/2?
%sequential_8/reshape_17/Reshape/shapePack.sequential_8/reshape_17/strided_slice:output:00sequential_8/reshape_17/Reshape/shape/1:output:00sequential_8/reshape_17/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2'
%sequential_8/reshape_17/Reshape/shape?
sequential_8/reshape_17/ReshapeReshape%sequential_8/dense_8/BiasAdd:output:0.sequential_8/reshape_17/Reshape/shape:output:0*
T0*+
_output_shapes
:?????????2!
sequential_8/reshape_17/Reshape?
normal_sampling_layer_4/unstackUnpack(sequential_8/reshape_17/Reshape:output:0*
T0*:
_output_shapes(
&:?????????:?????????*

axis*	
num2!
normal_sampling_layer_4/unstack?
normal_sampling_layer_4/ShapeShape(normal_sampling_layer_4/unstack:output:0*
T0*
_output_shapes
:2
normal_sampling_layer_4/Shape?
+normal_sampling_layer_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2-
+normal_sampling_layer_4/strided_slice/stack?
-normal_sampling_layer_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-normal_sampling_layer_4/strided_slice/stack_1?
-normal_sampling_layer_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-normal_sampling_layer_4/strided_slice/stack_2?
%normal_sampling_layer_4/strided_sliceStridedSlice&normal_sampling_layer_4/Shape:output:04normal_sampling_layer_4/strided_slice/stack:output:06normal_sampling_layer_4/strided_slice/stack_1:output:06normal_sampling_layer_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%normal_sampling_layer_4/strided_slice?
normal_sampling_layer_4/Shape_1Shape(normal_sampling_layer_4/unstack:output:0*
T0*
_output_shapes
:2!
normal_sampling_layer_4/Shape_1?
-normal_sampling_layer_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2/
-normal_sampling_layer_4/strided_slice_1/stack?
/normal_sampling_layer_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:21
/normal_sampling_layer_4/strided_slice_1/stack_1?
/normal_sampling_layer_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:21
/normal_sampling_layer_4/strided_slice_1/stack_2?
'normal_sampling_layer_4/strided_slice_1StridedSlice(normal_sampling_layer_4/Shape_1:output:06normal_sampling_layer_4/strided_slice_1/stack:output:08normal_sampling_layer_4/strided_slice_1/stack_1:output:08normal_sampling_layer_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2)
'normal_sampling_layer_4/strided_slice_1?
+normal_sampling_layer_4/random_normal/shapePack.normal_sampling_layer_4/strided_slice:output:00normal_sampling_layer_4/strided_slice_1:output:0*
N*
T0*
_output_shapes
:2-
+normal_sampling_layer_4/random_normal/shape?
*normal_sampling_layer_4/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2,
*normal_sampling_layer_4/random_normal/mean?
,normal_sampling_layer_4/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2.
,normal_sampling_layer_4/random_normal/stddev?
:normal_sampling_layer_4/random_normal/RandomStandardNormalRandomStandardNormal4normal_sampling_layer_4/random_normal/shape:output:0*
T0*0
_output_shapes
:??????????????????*
dtype0*
seed???)*
seed2???2<
:normal_sampling_layer_4/random_normal/RandomStandardNormal?
)normal_sampling_layer_4/random_normal/mulMulCnormal_sampling_layer_4/random_normal/RandomStandardNormal:output:05normal_sampling_layer_4/random_normal/stddev:output:0*
T0*0
_output_shapes
:??????????????????2+
)normal_sampling_layer_4/random_normal/mul?
%normal_sampling_layer_4/random_normalAddV2-normal_sampling_layer_4/random_normal/mul:z:03normal_sampling_layer_4/random_normal/mean:output:0*
T0*0
_output_shapes
:??????????????????2'
%normal_sampling_layer_4/random_normal?
normal_sampling_layer_4/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
normal_sampling_layer_4/mul/x?
normal_sampling_layer_4/mulMul&normal_sampling_layer_4/mul/x:output:0(normal_sampling_layer_4/unstack:output:1*
T0*'
_output_shapes
:?????????2
normal_sampling_layer_4/mul?
normal_sampling_layer_4/ExpExpnormal_sampling_layer_4/mul:z:0*
T0*'
_output_shapes
:?????????2
normal_sampling_layer_4/Exp?
normal_sampling_layer_4/mul_1Mulnormal_sampling_layer_4/Exp:y:0)normal_sampling_layer_4/random_normal:z:0*
T0*'
_output_shapes
:?????????2
normal_sampling_layer_4/mul_1?
normal_sampling_layer_4/addAddV2(normal_sampling_layer_4/unstack:output:0!normal_sampling_layer_4/mul_1:z:0*
T0*'
_output_shapes
:?????????2
normal_sampling_layer_4/add?
*sequential_9/dense_9/MatMul/ReadVariableOpReadVariableOp3sequential_9_dense_9_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02,
*sequential_9/dense_9/MatMul/ReadVariableOp?
sequential_9/dense_9/MatMulMatMulnormal_sampling_layer_4/add:z:02sequential_9/dense_9/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential_9/dense_9/MatMul?
+sequential_9/dense_9/BiasAdd/ReadVariableOpReadVariableOp4sequential_9_dense_9_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02-
+sequential_9/dense_9/BiasAdd/ReadVariableOp?
sequential_9/dense_9/BiasAddBiasAdd%sequential_9/dense_9/MatMul:product:03sequential_9/dense_9/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential_9/dense_9/BiasAdd?
sequential_9/dense_9/ReluRelu%sequential_9/dense_9/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
sequential_9/dense_9/Relu?
sequential_9/reshape_18/ShapeShape'sequential_9/dense_9/Relu:activations:0*
T0*
_output_shapes
:2
sequential_9/reshape_18/Shape?
+sequential_9/reshape_18/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2-
+sequential_9/reshape_18/strided_slice/stack?
-sequential_9/reshape_18/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_9/reshape_18/strided_slice/stack_1?
-sequential_9/reshape_18/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_9/reshape_18/strided_slice/stack_2?
%sequential_9/reshape_18/strided_sliceStridedSlice&sequential_9/reshape_18/Shape:output:04sequential_9/reshape_18/strided_slice/stack:output:06sequential_9/reshape_18/strided_slice/stack_1:output:06sequential_9/reshape_18/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%sequential_9/reshape_18/strided_slice?
'sequential_9/reshape_18/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2)
'sequential_9/reshape_18/Reshape/shape/1?
'sequential_9/reshape_18/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2)
'sequential_9/reshape_18/Reshape/shape/2?
'sequential_9/reshape_18/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B : 2)
'sequential_9/reshape_18/Reshape/shape/3?
%sequential_9/reshape_18/Reshape/shapePack.sequential_9/reshape_18/strided_slice:output:00sequential_9/reshape_18/Reshape/shape/1:output:00sequential_9/reshape_18/Reshape/shape/2:output:00sequential_9/reshape_18/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2'
%sequential_9/reshape_18/Reshape/shape?
sequential_9/reshape_18/ReshapeReshape'sequential_9/dense_9/Relu:activations:0.sequential_9/reshape_18/Reshape/shape:output:0*
T0*/
_output_shapes
:????????? 2!
sequential_9/reshape_18/Reshape?
&sequential_9/conv2d_transpose_12/ShapeShape(sequential_9/reshape_18/Reshape:output:0*
T0*
_output_shapes
:2(
&sequential_9/conv2d_transpose_12/Shape?
4sequential_9/conv2d_transpose_12/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 26
4sequential_9/conv2d_transpose_12/strided_slice/stack?
6sequential_9/conv2d_transpose_12/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:28
6sequential_9/conv2d_transpose_12/strided_slice/stack_1?
6sequential_9/conv2d_transpose_12/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:28
6sequential_9/conv2d_transpose_12/strided_slice/stack_2?
.sequential_9/conv2d_transpose_12/strided_sliceStridedSlice/sequential_9/conv2d_transpose_12/Shape:output:0=sequential_9/conv2d_transpose_12/strided_slice/stack:output:0?sequential_9/conv2d_transpose_12/strided_slice/stack_1:output:0?sequential_9/conv2d_transpose_12/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask20
.sequential_9/conv2d_transpose_12/strided_slice?
(sequential_9/conv2d_transpose_12/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2*
(sequential_9/conv2d_transpose_12/stack/1?
(sequential_9/conv2d_transpose_12/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2*
(sequential_9/conv2d_transpose_12/stack/2?
(sequential_9/conv2d_transpose_12/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@2*
(sequential_9/conv2d_transpose_12/stack/3?
&sequential_9/conv2d_transpose_12/stackPack7sequential_9/conv2d_transpose_12/strided_slice:output:01sequential_9/conv2d_transpose_12/stack/1:output:01sequential_9/conv2d_transpose_12/stack/2:output:01sequential_9/conv2d_transpose_12/stack/3:output:0*
N*
T0*
_output_shapes
:2(
&sequential_9/conv2d_transpose_12/stack?
6sequential_9/conv2d_transpose_12/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 28
6sequential_9/conv2d_transpose_12/strided_slice_1/stack?
8sequential_9/conv2d_transpose_12/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2:
8sequential_9/conv2d_transpose_12/strided_slice_1/stack_1?
8sequential_9/conv2d_transpose_12/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2:
8sequential_9/conv2d_transpose_12/strided_slice_1/stack_2?
0sequential_9/conv2d_transpose_12/strided_slice_1StridedSlice/sequential_9/conv2d_transpose_12/stack:output:0?sequential_9/conv2d_transpose_12/strided_slice_1/stack:output:0Asequential_9/conv2d_transpose_12/strided_slice_1/stack_1:output:0Asequential_9/conv2d_transpose_12/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask22
0sequential_9/conv2d_transpose_12/strided_slice_1?
@sequential_9/conv2d_transpose_12/conv2d_transpose/ReadVariableOpReadVariableOpIsequential_9_conv2d_transpose_12_conv2d_transpose_readvariableop_resource*&
_output_shapes
:@ *
dtype02B
@sequential_9/conv2d_transpose_12/conv2d_transpose/ReadVariableOp?
1sequential_9/conv2d_transpose_12/conv2d_transposeConv2DBackpropInput/sequential_9/conv2d_transpose_12/stack:output:0Hsequential_9/conv2d_transpose_12/conv2d_transpose/ReadVariableOp:value:0(sequential_9/reshape_18/Reshape:output:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
23
1sequential_9/conv2d_transpose_12/conv2d_transpose?
7sequential_9/conv2d_transpose_12/BiasAdd/ReadVariableOpReadVariableOp@sequential_9_conv2d_transpose_12_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype029
7sequential_9/conv2d_transpose_12/BiasAdd/ReadVariableOp?
(sequential_9/conv2d_transpose_12/BiasAddBiasAdd:sequential_9/conv2d_transpose_12/conv2d_transpose:output:0?sequential_9/conv2d_transpose_12/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2*
(sequential_9/conv2d_transpose_12/BiasAdd?
%sequential_9/conv2d_transpose_12/ReluRelu1sequential_9/conv2d_transpose_12/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2'
%sequential_9/conv2d_transpose_12/Relu?
&sequential_9/conv2d_transpose_13/ShapeShape3sequential_9/conv2d_transpose_12/Relu:activations:0*
T0*
_output_shapes
:2(
&sequential_9/conv2d_transpose_13/Shape?
4sequential_9/conv2d_transpose_13/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 26
4sequential_9/conv2d_transpose_13/strided_slice/stack?
6sequential_9/conv2d_transpose_13/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:28
6sequential_9/conv2d_transpose_13/strided_slice/stack_1?
6sequential_9/conv2d_transpose_13/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:28
6sequential_9/conv2d_transpose_13/strided_slice/stack_2?
.sequential_9/conv2d_transpose_13/strided_sliceStridedSlice/sequential_9/conv2d_transpose_13/Shape:output:0=sequential_9/conv2d_transpose_13/strided_slice/stack:output:0?sequential_9/conv2d_transpose_13/strided_slice/stack_1:output:0?sequential_9/conv2d_transpose_13/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask20
.sequential_9/conv2d_transpose_13/strided_slice?
(sequential_9/conv2d_transpose_13/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2*
(sequential_9/conv2d_transpose_13/stack/1?
(sequential_9/conv2d_transpose_13/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2*
(sequential_9/conv2d_transpose_13/stack/2?
(sequential_9/conv2d_transpose_13/stack/3Const*
_output_shapes
: *
dtype0*
value	B : 2*
(sequential_9/conv2d_transpose_13/stack/3?
&sequential_9/conv2d_transpose_13/stackPack7sequential_9/conv2d_transpose_13/strided_slice:output:01sequential_9/conv2d_transpose_13/stack/1:output:01sequential_9/conv2d_transpose_13/stack/2:output:01sequential_9/conv2d_transpose_13/stack/3:output:0*
N*
T0*
_output_shapes
:2(
&sequential_9/conv2d_transpose_13/stack?
6sequential_9/conv2d_transpose_13/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 28
6sequential_9/conv2d_transpose_13/strided_slice_1/stack?
8sequential_9/conv2d_transpose_13/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2:
8sequential_9/conv2d_transpose_13/strided_slice_1/stack_1?
8sequential_9/conv2d_transpose_13/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2:
8sequential_9/conv2d_transpose_13/strided_slice_1/stack_2?
0sequential_9/conv2d_transpose_13/strided_slice_1StridedSlice/sequential_9/conv2d_transpose_13/stack:output:0?sequential_9/conv2d_transpose_13/strided_slice_1/stack:output:0Asequential_9/conv2d_transpose_13/strided_slice_1/stack_1:output:0Asequential_9/conv2d_transpose_13/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask22
0sequential_9/conv2d_transpose_13/strided_slice_1?
@sequential_9/conv2d_transpose_13/conv2d_transpose/ReadVariableOpReadVariableOpIsequential_9_conv2d_transpose_13_conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype02B
@sequential_9/conv2d_transpose_13/conv2d_transpose/ReadVariableOp?
1sequential_9/conv2d_transpose_13/conv2d_transposeConv2DBackpropInput/sequential_9/conv2d_transpose_13/stack:output:0Hsequential_9/conv2d_transpose_13/conv2d_transpose/ReadVariableOp:value:03sequential_9/conv2d_transpose_12/Relu:activations:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
23
1sequential_9/conv2d_transpose_13/conv2d_transpose?
7sequential_9/conv2d_transpose_13/BiasAdd/ReadVariableOpReadVariableOp@sequential_9_conv2d_transpose_13_biasadd_readvariableop_resource*
_output_shapes
: *
dtype029
7sequential_9/conv2d_transpose_13/BiasAdd/ReadVariableOp?
(sequential_9/conv2d_transpose_13/BiasAddBiasAdd:sequential_9/conv2d_transpose_13/conv2d_transpose:output:0?sequential_9/conv2d_transpose_13/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2*
(sequential_9/conv2d_transpose_13/BiasAdd?
%sequential_9/conv2d_transpose_13/ReluRelu1sequential_9/conv2d_transpose_13/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2'
%sequential_9/conv2d_transpose_13/Relu?
&sequential_9/conv2d_transpose_14/ShapeShape3sequential_9/conv2d_transpose_13/Relu:activations:0*
T0*
_output_shapes
:2(
&sequential_9/conv2d_transpose_14/Shape?
4sequential_9/conv2d_transpose_14/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 26
4sequential_9/conv2d_transpose_14/strided_slice/stack?
6sequential_9/conv2d_transpose_14/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:28
6sequential_9/conv2d_transpose_14/strided_slice/stack_1?
6sequential_9/conv2d_transpose_14/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:28
6sequential_9/conv2d_transpose_14/strided_slice/stack_2?
.sequential_9/conv2d_transpose_14/strided_sliceStridedSlice/sequential_9/conv2d_transpose_14/Shape:output:0=sequential_9/conv2d_transpose_14/strided_slice/stack:output:0?sequential_9/conv2d_transpose_14/strided_slice/stack_1:output:0?sequential_9/conv2d_transpose_14/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask20
.sequential_9/conv2d_transpose_14/strided_slice?
(sequential_9/conv2d_transpose_14/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2*
(sequential_9/conv2d_transpose_14/stack/1?
(sequential_9/conv2d_transpose_14/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2*
(sequential_9/conv2d_transpose_14/stack/2?
(sequential_9/conv2d_transpose_14/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2*
(sequential_9/conv2d_transpose_14/stack/3?
&sequential_9/conv2d_transpose_14/stackPack7sequential_9/conv2d_transpose_14/strided_slice:output:01sequential_9/conv2d_transpose_14/stack/1:output:01sequential_9/conv2d_transpose_14/stack/2:output:01sequential_9/conv2d_transpose_14/stack/3:output:0*
N*
T0*
_output_shapes
:2(
&sequential_9/conv2d_transpose_14/stack?
6sequential_9/conv2d_transpose_14/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 28
6sequential_9/conv2d_transpose_14/strided_slice_1/stack?
8sequential_9/conv2d_transpose_14/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2:
8sequential_9/conv2d_transpose_14/strided_slice_1/stack_1?
8sequential_9/conv2d_transpose_14/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2:
8sequential_9/conv2d_transpose_14/strided_slice_1/stack_2?
0sequential_9/conv2d_transpose_14/strided_slice_1StridedSlice/sequential_9/conv2d_transpose_14/stack:output:0?sequential_9/conv2d_transpose_14/strided_slice_1/stack:output:0Asequential_9/conv2d_transpose_14/strided_slice_1/stack_1:output:0Asequential_9/conv2d_transpose_14/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask22
0sequential_9/conv2d_transpose_14/strided_slice_1?
@sequential_9/conv2d_transpose_14/conv2d_transpose/ReadVariableOpReadVariableOpIsequential_9_conv2d_transpose_14_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype02B
@sequential_9/conv2d_transpose_14/conv2d_transpose/ReadVariableOp?
1sequential_9/conv2d_transpose_14/conv2d_transposeConv2DBackpropInput/sequential_9/conv2d_transpose_14/stack:output:0Hsequential_9/conv2d_transpose_14/conv2d_transpose/ReadVariableOp:value:03sequential_9/conv2d_transpose_13/Relu:activations:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
23
1sequential_9/conv2d_transpose_14/conv2d_transpose?
7sequential_9/conv2d_transpose_14/BiasAdd/ReadVariableOpReadVariableOp@sequential_9_conv2d_transpose_14_biasadd_readvariableop_resource*
_output_shapes
:*
dtype029
7sequential_9/conv2d_transpose_14/BiasAdd/ReadVariableOp?
(sequential_9/conv2d_transpose_14/BiasAddBiasAdd:sequential_9/conv2d_transpose_14/conv2d_transpose:output:0?sequential_9/conv2d_transpose_14/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2*
(sequential_9/conv2d_transpose_14/BiasAdd?
(sequential_9/conv2d_transpose_14/SigmoidSigmoid1sequential_9/conv2d_transpose_14/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2*
(sequential_9/conv2d_transpose_14/Sigmoid?
sequential_9/reshape_19/ShapeShape,sequential_9/conv2d_transpose_14/Sigmoid:y:0*
T0*
_output_shapes
:2
sequential_9/reshape_19/Shape?
+sequential_9/reshape_19/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2-
+sequential_9/reshape_19/strided_slice/stack?
-sequential_9/reshape_19/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_9/reshape_19/strided_slice/stack_1?
-sequential_9/reshape_19/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_9/reshape_19/strided_slice/stack_2?
%sequential_9/reshape_19/strided_sliceStridedSlice&sequential_9/reshape_19/Shape:output:04sequential_9/reshape_19/strided_slice/stack:output:06sequential_9/reshape_19/strided_slice/stack_1:output:06sequential_9/reshape_19/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%sequential_9/reshape_19/strided_slice?
'sequential_9/reshape_19/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2)
'sequential_9/reshape_19/Reshape/shape/1?
'sequential_9/reshape_19/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2)
'sequential_9/reshape_19/Reshape/shape/2?
'sequential_9/reshape_19/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2)
'sequential_9/reshape_19/Reshape/shape/3?
%sequential_9/reshape_19/Reshape/shapePack.sequential_9/reshape_19/strided_slice:output:00sequential_9/reshape_19/Reshape/shape/1:output:00sequential_9/reshape_19/Reshape/shape/2:output:00sequential_9/reshape_19/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2'
%sequential_9/reshape_19/Reshape/shape?
sequential_9/reshape_19/ReshapeReshape,sequential_9/conv2d_transpose_14/Sigmoid:y:0.sequential_9/reshape_19/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????2!
sequential_9/reshape_19/Reshape?
IdentityIdentity(sequential_9/reshape_19/Reshape:output:0^NoOp*
T0*/
_output_shapes
:?????????2

Identity?
NoOpNoOp-^sequential_8/conv2d_8/BiasAdd/ReadVariableOp,^sequential_8/conv2d_8/Conv2D/ReadVariableOp-^sequential_8/conv2d_9/BiasAdd/ReadVariableOp,^sequential_8/conv2d_9/Conv2D/ReadVariableOp,^sequential_8/dense_8/BiasAdd/ReadVariableOp+^sequential_8/dense_8/MatMul/ReadVariableOp8^sequential_9/conv2d_transpose_12/BiasAdd/ReadVariableOpA^sequential_9/conv2d_transpose_12/conv2d_transpose/ReadVariableOp8^sequential_9/conv2d_transpose_13/BiasAdd/ReadVariableOpA^sequential_9/conv2d_transpose_13/conv2d_transpose/ReadVariableOp8^sequential_9/conv2d_transpose_14/BiasAdd/ReadVariableOpA^sequential_9/conv2d_transpose_14/conv2d_transpose/ReadVariableOp,^sequential_9/dense_9/BiasAdd/ReadVariableOp+^sequential_9/dense_9/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:?????????: : : : : : : : : : : : : : 2\
,sequential_8/conv2d_8/BiasAdd/ReadVariableOp,sequential_8/conv2d_8/BiasAdd/ReadVariableOp2Z
+sequential_8/conv2d_8/Conv2D/ReadVariableOp+sequential_8/conv2d_8/Conv2D/ReadVariableOp2\
,sequential_8/conv2d_9/BiasAdd/ReadVariableOp,sequential_8/conv2d_9/BiasAdd/ReadVariableOp2Z
+sequential_8/conv2d_9/Conv2D/ReadVariableOp+sequential_8/conv2d_9/Conv2D/ReadVariableOp2Z
+sequential_8/dense_8/BiasAdd/ReadVariableOp+sequential_8/dense_8/BiasAdd/ReadVariableOp2X
*sequential_8/dense_8/MatMul/ReadVariableOp*sequential_8/dense_8/MatMul/ReadVariableOp2r
7sequential_9/conv2d_transpose_12/BiasAdd/ReadVariableOp7sequential_9/conv2d_transpose_12/BiasAdd/ReadVariableOp2?
@sequential_9/conv2d_transpose_12/conv2d_transpose/ReadVariableOp@sequential_9/conv2d_transpose_12/conv2d_transpose/ReadVariableOp2r
7sequential_9/conv2d_transpose_13/BiasAdd/ReadVariableOp7sequential_9/conv2d_transpose_13/BiasAdd/ReadVariableOp2?
@sequential_9/conv2d_transpose_13/conv2d_transpose/ReadVariableOp@sequential_9/conv2d_transpose_13/conv2d_transpose/ReadVariableOp2r
7sequential_9/conv2d_transpose_14/BiasAdd/ReadVariableOp7sequential_9/conv2d_transpose_14/BiasAdd/ReadVariableOp2?
@sequential_9/conv2d_transpose_14/conv2d_transpose/ReadVariableOp@sequential_9/conv2d_transpose_14/conv2d_transpose/ReadVariableOp2Z
+sequential_9/dense_9/BiasAdd/ReadVariableOp+sequential_9/dense_9/BiasAdd/ReadVariableOp2X
*sequential_9/dense_9/MatMul/ReadVariableOp*sequential_9/dense_9/MatMul/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
T__inference_variational_autoencoder_4_layer_call_and_return_conditional_losses_77854

inputs,
sequential_8_77815:  
sequential_8_77817: ,
sequential_8_77819: @ 
sequential_8_77821:@%
sequential_8_77823:	? 
sequential_8_77825:%
sequential_9_77836:	?!
sequential_9_77838:	?,
sequential_9_77840:@  
sequential_9_77842:@,
sequential_9_77844: @ 
sequential_9_77846: ,
sequential_9_77848:  
sequential_9_77850:
identity??$sequential_8/StatefulPartitionedCall?$sequential_9/StatefulPartitionedCall?
$sequential_8/StatefulPartitionedCallStatefulPartitionedCallinputssequential_8_77815sequential_8_77817sequential_8_77819sequential_8_77821sequential_8_77823sequential_8_77825*
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
G__inference_sequential_8_layer_call_and_return_conditional_losses_770112&
$sequential_8/StatefulPartitionedCall?
'normal_sampling_layer_4/PartitionedCallPartitionedCall-sequential_8/StatefulPartitionedCall:output:0*
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
R__inference_normal_sampling_layer_4_layer_call_and_return_conditional_losses_778342)
'normal_sampling_layer_4/PartitionedCall?
$sequential_9/StatefulPartitionedCallStatefulPartitionedCall0normal_sampling_layer_4/PartitionedCall:output:0sequential_9_77836sequential_9_77838sequential_9_77840sequential_9_77842sequential_9_77844sequential_9_77846sequential_9_77848sequential_9_77850*
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
G__inference_sequential_9_layer_call_and_return_conditional_losses_775992&
$sequential_9/StatefulPartitionedCall?
IdentityIdentity-sequential_9/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????2

Identity?
NoOpNoOp%^sequential_8/StatefulPartitionedCall%^sequential_9/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:?????????: : : : : : : : : : : : : : 2L
$sequential_8/StatefulPartitionedCall$sequential_8/StatefulPartitionedCall2L
$sequential_9/StatefulPartitionedCall$sequential_9/StatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
3__inference_conv2d_transpose_14_layer_call_fn_79279

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
GPU2*0J 8? *W
fRRP
N__inference_conv2d_transpose_14_layer_call_and_return_conditional_losses_774052
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
?
F
*__inference_reshape_17_layer_call_fn_79021

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
GPU2*0J 8? *N
fIRG
E__inference_reshape_17_layer_call_and_return_conditional_losses_770082
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
??
?
G__inference_sequential_9_layer_call_and_return_conditional_losses_78738

inputs9
&dense_9_matmul_readvariableop_resource:	?6
'dense_9_biasadd_readvariableop_resource:	?V
<conv2d_transpose_12_conv2d_transpose_readvariableop_resource:@ A
3conv2d_transpose_12_biasadd_readvariableop_resource:@V
<conv2d_transpose_13_conv2d_transpose_readvariableop_resource: @A
3conv2d_transpose_13_biasadd_readvariableop_resource: V
<conv2d_transpose_14_conv2d_transpose_readvariableop_resource: A
3conv2d_transpose_14_biasadd_readvariableop_resource:
identity??*conv2d_transpose_12/BiasAdd/ReadVariableOp?3conv2d_transpose_12/conv2d_transpose/ReadVariableOp?*conv2d_transpose_13/BiasAdd/ReadVariableOp?3conv2d_transpose_13/conv2d_transpose/ReadVariableOp?*conv2d_transpose_14/BiasAdd/ReadVariableOp?3conv2d_transpose_14/conv2d_transpose/ReadVariableOp?dense_9/BiasAdd/ReadVariableOp?dense_9/MatMul/ReadVariableOp?
dense_9/MatMul/ReadVariableOpReadVariableOp&dense_9_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
dense_9/MatMul/ReadVariableOp?
dense_9/MatMulMatMulinputs%dense_9/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_9/MatMul?
dense_9/BiasAdd/ReadVariableOpReadVariableOp'dense_9_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
dense_9/BiasAdd/ReadVariableOp?
dense_9/BiasAddBiasAdddense_9/MatMul:product:0&dense_9/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_9/BiasAddq
dense_9/ReluReludense_9/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_9/Relun
reshape_18/ShapeShapedense_9/Relu:activations:0*
T0*
_output_shapes
:2
reshape_18/Shape?
reshape_18/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
reshape_18/strided_slice/stack?
 reshape_18/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_18/strided_slice/stack_1?
 reshape_18/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_18/strided_slice/stack_2?
reshape_18/strided_sliceStridedSlicereshape_18/Shape:output:0'reshape_18/strided_slice/stack:output:0)reshape_18/strided_slice/stack_1:output:0)reshape_18/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_18/strided_slicez
reshape_18/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_18/Reshape/shape/1z
reshape_18/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_18/Reshape/shape/2z
reshape_18/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B : 2
reshape_18/Reshape/shape/3?
reshape_18/Reshape/shapePack!reshape_18/strided_slice:output:0#reshape_18/Reshape/shape/1:output:0#reshape_18/Reshape/shape/2:output:0#reshape_18/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
reshape_18/Reshape/shape?
reshape_18/ReshapeReshapedense_9/Relu:activations:0!reshape_18/Reshape/shape:output:0*
T0*/
_output_shapes
:????????? 2
reshape_18/Reshape?
conv2d_transpose_12/ShapeShapereshape_18/Reshape:output:0*
T0*
_output_shapes
:2
conv2d_transpose_12/Shape?
'conv2d_transpose_12/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'conv2d_transpose_12/strided_slice/stack?
)conv2d_transpose_12/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_12/strided_slice/stack_1?
)conv2d_transpose_12/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_12/strided_slice/stack_2?
!conv2d_transpose_12/strided_sliceStridedSlice"conv2d_transpose_12/Shape:output:00conv2d_transpose_12/strided_slice/stack:output:02conv2d_transpose_12/strided_slice/stack_1:output:02conv2d_transpose_12/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!conv2d_transpose_12/strided_slice|
conv2d_transpose_12/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_12/stack/1|
conv2d_transpose_12/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_12/stack/2|
conv2d_transpose_12/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@2
conv2d_transpose_12/stack/3?
conv2d_transpose_12/stackPack*conv2d_transpose_12/strided_slice:output:0$conv2d_transpose_12/stack/1:output:0$conv2d_transpose_12/stack/2:output:0$conv2d_transpose_12/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_12/stack?
)conv2d_transpose_12/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)conv2d_transpose_12/strided_slice_1/stack?
+conv2d_transpose_12/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_12/strided_slice_1/stack_1?
+conv2d_transpose_12/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_12/strided_slice_1/stack_2?
#conv2d_transpose_12/strided_slice_1StridedSlice"conv2d_transpose_12/stack:output:02conv2d_transpose_12/strided_slice_1/stack:output:04conv2d_transpose_12/strided_slice_1/stack_1:output:04conv2d_transpose_12/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#conv2d_transpose_12/strided_slice_1?
3conv2d_transpose_12/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_12_conv2d_transpose_readvariableop_resource*&
_output_shapes
:@ *
dtype025
3conv2d_transpose_12/conv2d_transpose/ReadVariableOp?
$conv2d_transpose_12/conv2d_transposeConv2DBackpropInput"conv2d_transpose_12/stack:output:0;conv2d_transpose_12/conv2d_transpose/ReadVariableOp:value:0reshape_18/Reshape:output:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2&
$conv2d_transpose_12/conv2d_transpose?
*conv2d_transpose_12/BiasAdd/ReadVariableOpReadVariableOp3conv2d_transpose_12_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02,
*conv2d_transpose_12/BiasAdd/ReadVariableOp?
conv2d_transpose_12/BiasAddBiasAdd-conv2d_transpose_12/conv2d_transpose:output:02conv2d_transpose_12/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
conv2d_transpose_12/BiasAdd?
conv2d_transpose_12/ReluRelu$conv2d_transpose_12/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
conv2d_transpose_12/Relu?
conv2d_transpose_13/ShapeShape&conv2d_transpose_12/Relu:activations:0*
T0*
_output_shapes
:2
conv2d_transpose_13/Shape?
'conv2d_transpose_13/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'conv2d_transpose_13/strided_slice/stack?
)conv2d_transpose_13/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_13/strided_slice/stack_1?
)conv2d_transpose_13/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_13/strided_slice/stack_2?
!conv2d_transpose_13/strided_sliceStridedSlice"conv2d_transpose_13/Shape:output:00conv2d_transpose_13/strided_slice/stack:output:02conv2d_transpose_13/strided_slice/stack_1:output:02conv2d_transpose_13/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!conv2d_transpose_13/strided_slice|
conv2d_transpose_13/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_13/stack/1|
conv2d_transpose_13/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_13/stack/2|
conv2d_transpose_13/stack/3Const*
_output_shapes
: *
dtype0*
value	B : 2
conv2d_transpose_13/stack/3?
conv2d_transpose_13/stackPack*conv2d_transpose_13/strided_slice:output:0$conv2d_transpose_13/stack/1:output:0$conv2d_transpose_13/stack/2:output:0$conv2d_transpose_13/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_13/stack?
)conv2d_transpose_13/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)conv2d_transpose_13/strided_slice_1/stack?
+conv2d_transpose_13/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_13/strided_slice_1/stack_1?
+conv2d_transpose_13/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_13/strided_slice_1/stack_2?
#conv2d_transpose_13/strided_slice_1StridedSlice"conv2d_transpose_13/stack:output:02conv2d_transpose_13/strided_slice_1/stack:output:04conv2d_transpose_13/strided_slice_1/stack_1:output:04conv2d_transpose_13/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#conv2d_transpose_13/strided_slice_1?
3conv2d_transpose_13/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_13_conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype025
3conv2d_transpose_13/conv2d_transpose/ReadVariableOp?
$conv2d_transpose_13/conv2d_transposeConv2DBackpropInput"conv2d_transpose_13/stack:output:0;conv2d_transpose_13/conv2d_transpose/ReadVariableOp:value:0&conv2d_transpose_12/Relu:activations:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
2&
$conv2d_transpose_13/conv2d_transpose?
*conv2d_transpose_13/BiasAdd/ReadVariableOpReadVariableOp3conv2d_transpose_13_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02,
*conv2d_transpose_13/BiasAdd/ReadVariableOp?
conv2d_transpose_13/BiasAddBiasAdd-conv2d_transpose_13/conv2d_transpose:output:02conv2d_transpose_13/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2
conv2d_transpose_13/BiasAdd?
conv2d_transpose_13/ReluRelu$conv2d_transpose_13/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
conv2d_transpose_13/Relu?
conv2d_transpose_14/ShapeShape&conv2d_transpose_13/Relu:activations:0*
T0*
_output_shapes
:2
conv2d_transpose_14/Shape?
'conv2d_transpose_14/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'conv2d_transpose_14/strided_slice/stack?
)conv2d_transpose_14/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_14/strided_slice/stack_1?
)conv2d_transpose_14/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_14/strided_slice/stack_2?
!conv2d_transpose_14/strided_sliceStridedSlice"conv2d_transpose_14/Shape:output:00conv2d_transpose_14/strided_slice/stack:output:02conv2d_transpose_14/strided_slice/stack_1:output:02conv2d_transpose_14/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!conv2d_transpose_14/strided_slice|
conv2d_transpose_14/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_14/stack/1|
conv2d_transpose_14/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_14/stack/2|
conv2d_transpose_14/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_14/stack/3?
conv2d_transpose_14/stackPack*conv2d_transpose_14/strided_slice:output:0$conv2d_transpose_14/stack/1:output:0$conv2d_transpose_14/stack/2:output:0$conv2d_transpose_14/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_14/stack?
)conv2d_transpose_14/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)conv2d_transpose_14/strided_slice_1/stack?
+conv2d_transpose_14/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_14/strided_slice_1/stack_1?
+conv2d_transpose_14/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_14/strided_slice_1/stack_2?
#conv2d_transpose_14/strided_slice_1StridedSlice"conv2d_transpose_14/stack:output:02conv2d_transpose_14/strided_slice_1/stack:output:04conv2d_transpose_14/strided_slice_1/stack_1:output:04conv2d_transpose_14/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#conv2d_transpose_14/strided_slice_1?
3conv2d_transpose_14/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_14_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype025
3conv2d_transpose_14/conv2d_transpose/ReadVariableOp?
$conv2d_transpose_14/conv2d_transposeConv2DBackpropInput"conv2d_transpose_14/stack:output:0;conv2d_transpose_14/conv2d_transpose/ReadVariableOp:value:0&conv2d_transpose_13/Relu:activations:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2&
$conv2d_transpose_14/conv2d_transpose?
*conv2d_transpose_14/BiasAdd/ReadVariableOpReadVariableOp3conv2d_transpose_14_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02,
*conv2d_transpose_14/BiasAdd/ReadVariableOp?
conv2d_transpose_14/BiasAddBiasAdd-conv2d_transpose_14/conv2d_transpose:output:02conv2d_transpose_14/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
conv2d_transpose_14/BiasAdd?
conv2d_transpose_14/SigmoidSigmoid$conv2d_transpose_14/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
conv2d_transpose_14/Sigmoids
reshape_19/ShapeShapeconv2d_transpose_14/Sigmoid:y:0*
T0*
_output_shapes
:2
reshape_19/Shape?
reshape_19/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
reshape_19/strided_slice/stack?
 reshape_19/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_19/strided_slice/stack_1?
 reshape_19/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_19/strided_slice/stack_2?
reshape_19/strided_sliceStridedSlicereshape_19/Shape:output:0'reshape_19/strided_slice/stack:output:0)reshape_19/strided_slice/stack_1:output:0)reshape_19/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_19/strided_slicez
reshape_19/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_19/Reshape/shape/1z
reshape_19/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_19/Reshape/shape/2z
reshape_19/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_19/Reshape/shape/3?
reshape_19/Reshape/shapePack!reshape_19/strided_slice:output:0#reshape_19/Reshape/shape/1:output:0#reshape_19/Reshape/shape/2:output:0#reshape_19/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
reshape_19/Reshape/shape?
reshape_19/ReshapeReshapeconv2d_transpose_14/Sigmoid:y:0!reshape_19/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????2
reshape_19/Reshape~
IdentityIdentityreshape_19/Reshape:output:0^NoOp*
T0*/
_output_shapes
:?????????2

Identity?
NoOpNoOp+^conv2d_transpose_12/BiasAdd/ReadVariableOp4^conv2d_transpose_12/conv2d_transpose/ReadVariableOp+^conv2d_transpose_13/BiasAdd/ReadVariableOp4^conv2d_transpose_13/conv2d_transpose/ReadVariableOp+^conv2d_transpose_14/BiasAdd/ReadVariableOp4^conv2d_transpose_14/conv2d_transpose/ReadVariableOp^dense_9/BiasAdd/ReadVariableOp^dense_9/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : : : : : 2X
*conv2d_transpose_12/BiasAdd/ReadVariableOp*conv2d_transpose_12/BiasAdd/ReadVariableOp2j
3conv2d_transpose_12/conv2d_transpose/ReadVariableOp3conv2d_transpose_12/conv2d_transpose/ReadVariableOp2X
*conv2d_transpose_13/BiasAdd/ReadVariableOp*conv2d_transpose_13/BiasAdd/ReadVariableOp2j
3conv2d_transpose_13/conv2d_transpose/ReadVariableOp3conv2d_transpose_13/conv2d_transpose/ReadVariableOp2X
*conv2d_transpose_14/BiasAdd/ReadVariableOp*conv2d_transpose_14/BiasAdd/ReadVariableOp2j
3conv2d_transpose_14/conv2d_transpose/ReadVariableOp3conv2d_transpose_14/conv2d_transpose/ReadVariableOp2@
dense_9/BiasAdd/ReadVariableOpdense_9/BiasAdd/ReadVariableOp2>
dense_9/MatMul/ReadVariableOpdense_9/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
??
?
 __inference__wrapped_model_76914
input_1h
Nvariational_autoencoder_4_sequential_8_conv2d_8_conv2d_readvariableop_resource: ]
Ovariational_autoencoder_4_sequential_8_conv2d_8_biasadd_readvariableop_resource: h
Nvariational_autoencoder_4_sequential_8_conv2d_9_conv2d_readvariableop_resource: @]
Ovariational_autoencoder_4_sequential_8_conv2d_9_biasadd_readvariableop_resource:@`
Mvariational_autoencoder_4_sequential_8_dense_8_matmul_readvariableop_resource:	?\
Nvariational_autoencoder_4_sequential_8_dense_8_biasadd_readvariableop_resource:`
Mvariational_autoencoder_4_sequential_9_dense_9_matmul_readvariableop_resource:	?]
Nvariational_autoencoder_4_sequential_9_dense_9_biasadd_readvariableop_resource:	?}
cvariational_autoencoder_4_sequential_9_conv2d_transpose_12_conv2d_transpose_readvariableop_resource:@ h
Zvariational_autoencoder_4_sequential_9_conv2d_transpose_12_biasadd_readvariableop_resource:@}
cvariational_autoencoder_4_sequential_9_conv2d_transpose_13_conv2d_transpose_readvariableop_resource: @h
Zvariational_autoencoder_4_sequential_9_conv2d_transpose_13_biasadd_readvariableop_resource: }
cvariational_autoencoder_4_sequential_9_conv2d_transpose_14_conv2d_transpose_readvariableop_resource: h
Zvariational_autoencoder_4_sequential_9_conv2d_transpose_14_biasadd_readvariableop_resource:
identity??Fvariational_autoencoder_4/sequential_8/conv2d_8/BiasAdd/ReadVariableOp?Evariational_autoencoder_4/sequential_8/conv2d_8/Conv2D/ReadVariableOp?Fvariational_autoencoder_4/sequential_8/conv2d_9/BiasAdd/ReadVariableOp?Evariational_autoencoder_4/sequential_8/conv2d_9/Conv2D/ReadVariableOp?Evariational_autoencoder_4/sequential_8/dense_8/BiasAdd/ReadVariableOp?Dvariational_autoencoder_4/sequential_8/dense_8/MatMul/ReadVariableOp?Qvariational_autoencoder_4/sequential_9/conv2d_transpose_12/BiasAdd/ReadVariableOp?Zvariational_autoencoder_4/sequential_9/conv2d_transpose_12/conv2d_transpose/ReadVariableOp?Qvariational_autoencoder_4/sequential_9/conv2d_transpose_13/BiasAdd/ReadVariableOp?Zvariational_autoencoder_4/sequential_9/conv2d_transpose_13/conv2d_transpose/ReadVariableOp?Qvariational_autoencoder_4/sequential_9/conv2d_transpose_14/BiasAdd/ReadVariableOp?Zvariational_autoencoder_4/sequential_9/conv2d_transpose_14/conv2d_transpose/ReadVariableOp?Evariational_autoencoder_4/sequential_9/dense_9/BiasAdd/ReadVariableOp?Dvariational_autoencoder_4/sequential_9/dense_9/MatMul/ReadVariableOp?
7variational_autoencoder_4/sequential_8/reshape_16/ShapeShapeinput_1*
T0*
_output_shapes
:29
7variational_autoencoder_4/sequential_8/reshape_16/Shape?
Evariational_autoencoder_4/sequential_8/reshape_16/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2G
Evariational_autoencoder_4/sequential_8/reshape_16/strided_slice/stack?
Gvariational_autoencoder_4/sequential_8/reshape_16/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2I
Gvariational_autoencoder_4/sequential_8/reshape_16/strided_slice/stack_1?
Gvariational_autoencoder_4/sequential_8/reshape_16/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2I
Gvariational_autoencoder_4/sequential_8/reshape_16/strided_slice/stack_2?
?variational_autoencoder_4/sequential_8/reshape_16/strided_sliceStridedSlice@variational_autoencoder_4/sequential_8/reshape_16/Shape:output:0Nvariational_autoencoder_4/sequential_8/reshape_16/strided_slice/stack:output:0Pvariational_autoencoder_4/sequential_8/reshape_16/strided_slice/stack_1:output:0Pvariational_autoencoder_4/sequential_8/reshape_16/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2A
?variational_autoencoder_4/sequential_8/reshape_16/strided_slice?
Avariational_autoencoder_4/sequential_8/reshape_16/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2C
Avariational_autoencoder_4/sequential_8/reshape_16/Reshape/shape/1?
Avariational_autoencoder_4/sequential_8/reshape_16/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2C
Avariational_autoencoder_4/sequential_8/reshape_16/Reshape/shape/2?
Avariational_autoencoder_4/sequential_8/reshape_16/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2C
Avariational_autoencoder_4/sequential_8/reshape_16/Reshape/shape/3?
?variational_autoencoder_4/sequential_8/reshape_16/Reshape/shapePackHvariational_autoencoder_4/sequential_8/reshape_16/strided_slice:output:0Jvariational_autoencoder_4/sequential_8/reshape_16/Reshape/shape/1:output:0Jvariational_autoencoder_4/sequential_8/reshape_16/Reshape/shape/2:output:0Jvariational_autoencoder_4/sequential_8/reshape_16/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2A
?variational_autoencoder_4/sequential_8/reshape_16/Reshape/shape?
9variational_autoencoder_4/sequential_8/reshape_16/ReshapeReshapeinput_1Hvariational_autoencoder_4/sequential_8/reshape_16/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????2;
9variational_autoencoder_4/sequential_8/reshape_16/Reshape?
Evariational_autoencoder_4/sequential_8/conv2d_8/Conv2D/ReadVariableOpReadVariableOpNvariational_autoencoder_4_sequential_8_conv2d_8_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02G
Evariational_autoencoder_4/sequential_8/conv2d_8/Conv2D/ReadVariableOp?
6variational_autoencoder_4/sequential_8/conv2d_8/Conv2DConv2DBvariational_autoencoder_4/sequential_8/reshape_16/Reshape:output:0Mvariational_autoencoder_4/sequential_8/conv2d_8/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
28
6variational_autoencoder_4/sequential_8/conv2d_8/Conv2D?
Fvariational_autoencoder_4/sequential_8/conv2d_8/BiasAdd/ReadVariableOpReadVariableOpOvariational_autoencoder_4_sequential_8_conv2d_8_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02H
Fvariational_autoencoder_4/sequential_8/conv2d_8/BiasAdd/ReadVariableOp?
7variational_autoencoder_4/sequential_8/conv2d_8/BiasAddBiasAdd?variational_autoencoder_4/sequential_8/conv2d_8/Conv2D:output:0Nvariational_autoencoder_4/sequential_8/conv2d_8/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 29
7variational_autoencoder_4/sequential_8/conv2d_8/BiasAdd?
4variational_autoencoder_4/sequential_8/conv2d_8/ReluRelu@variational_autoencoder_4/sequential_8/conv2d_8/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 26
4variational_autoencoder_4/sequential_8/conv2d_8/Relu?
Evariational_autoencoder_4/sequential_8/conv2d_9/Conv2D/ReadVariableOpReadVariableOpNvariational_autoencoder_4_sequential_8_conv2d_9_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02G
Evariational_autoencoder_4/sequential_8/conv2d_9/Conv2D/ReadVariableOp?
6variational_autoencoder_4/sequential_8/conv2d_9/Conv2DConv2DBvariational_autoencoder_4/sequential_8/conv2d_8/Relu:activations:0Mvariational_autoencoder_4/sequential_8/conv2d_9/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
28
6variational_autoencoder_4/sequential_8/conv2d_9/Conv2D?
Fvariational_autoencoder_4/sequential_8/conv2d_9/BiasAdd/ReadVariableOpReadVariableOpOvariational_autoencoder_4_sequential_8_conv2d_9_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02H
Fvariational_autoencoder_4/sequential_8/conv2d_9/BiasAdd/ReadVariableOp?
7variational_autoencoder_4/sequential_8/conv2d_9/BiasAddBiasAdd?variational_autoencoder_4/sequential_8/conv2d_9/Conv2D:output:0Nvariational_autoencoder_4/sequential_8/conv2d_9/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@29
7variational_autoencoder_4/sequential_8/conv2d_9/BiasAdd?
4variational_autoencoder_4/sequential_8/conv2d_9/ReluRelu@variational_autoencoder_4/sequential_8/conv2d_9/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@26
4variational_autoencoder_4/sequential_8/conv2d_9/Relu?
6variational_autoencoder_4/sequential_8/flatten_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"???? 	  28
6variational_autoencoder_4/sequential_8/flatten_4/Const?
8variational_autoencoder_4/sequential_8/flatten_4/ReshapeReshapeBvariational_autoencoder_4/sequential_8/conv2d_9/Relu:activations:0?variational_autoencoder_4/sequential_8/flatten_4/Const:output:0*
T0*(
_output_shapes
:??????????2:
8variational_autoencoder_4/sequential_8/flatten_4/Reshape?
Dvariational_autoencoder_4/sequential_8/dense_8/MatMul/ReadVariableOpReadVariableOpMvariational_autoencoder_4_sequential_8_dense_8_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02F
Dvariational_autoencoder_4/sequential_8/dense_8/MatMul/ReadVariableOp?
5variational_autoencoder_4/sequential_8/dense_8/MatMulMatMulAvariational_autoencoder_4/sequential_8/flatten_4/Reshape:output:0Lvariational_autoencoder_4/sequential_8/dense_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????27
5variational_autoencoder_4/sequential_8/dense_8/MatMul?
Evariational_autoencoder_4/sequential_8/dense_8/BiasAdd/ReadVariableOpReadVariableOpNvariational_autoencoder_4_sequential_8_dense_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02G
Evariational_autoencoder_4/sequential_8/dense_8/BiasAdd/ReadVariableOp?
6variational_autoencoder_4/sequential_8/dense_8/BiasAddBiasAdd?variational_autoencoder_4/sequential_8/dense_8/MatMul:product:0Mvariational_autoencoder_4/sequential_8/dense_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????28
6variational_autoencoder_4/sequential_8/dense_8/BiasAdd?
7variational_autoencoder_4/sequential_8/reshape_17/ShapeShape?variational_autoencoder_4/sequential_8/dense_8/BiasAdd:output:0*
T0*
_output_shapes
:29
7variational_autoencoder_4/sequential_8/reshape_17/Shape?
Evariational_autoencoder_4/sequential_8/reshape_17/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2G
Evariational_autoencoder_4/sequential_8/reshape_17/strided_slice/stack?
Gvariational_autoencoder_4/sequential_8/reshape_17/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2I
Gvariational_autoencoder_4/sequential_8/reshape_17/strided_slice/stack_1?
Gvariational_autoencoder_4/sequential_8/reshape_17/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2I
Gvariational_autoencoder_4/sequential_8/reshape_17/strided_slice/stack_2?
?variational_autoencoder_4/sequential_8/reshape_17/strided_sliceStridedSlice@variational_autoencoder_4/sequential_8/reshape_17/Shape:output:0Nvariational_autoencoder_4/sequential_8/reshape_17/strided_slice/stack:output:0Pvariational_autoencoder_4/sequential_8/reshape_17/strided_slice/stack_1:output:0Pvariational_autoencoder_4/sequential_8/reshape_17/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2A
?variational_autoencoder_4/sequential_8/reshape_17/strided_slice?
Avariational_autoencoder_4/sequential_8/reshape_17/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2C
Avariational_autoencoder_4/sequential_8/reshape_17/Reshape/shape/1?
Avariational_autoencoder_4/sequential_8/reshape_17/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2C
Avariational_autoencoder_4/sequential_8/reshape_17/Reshape/shape/2?
?variational_autoencoder_4/sequential_8/reshape_17/Reshape/shapePackHvariational_autoencoder_4/sequential_8/reshape_17/strided_slice:output:0Jvariational_autoencoder_4/sequential_8/reshape_17/Reshape/shape/1:output:0Jvariational_autoencoder_4/sequential_8/reshape_17/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2A
?variational_autoencoder_4/sequential_8/reshape_17/Reshape/shape?
9variational_autoencoder_4/sequential_8/reshape_17/ReshapeReshape?variational_autoencoder_4/sequential_8/dense_8/BiasAdd:output:0Hvariational_autoencoder_4/sequential_8/reshape_17/Reshape/shape:output:0*
T0*+
_output_shapes
:?????????2;
9variational_autoencoder_4/sequential_8/reshape_17/Reshape?
9variational_autoencoder_4/normal_sampling_layer_4/unstackUnpackBvariational_autoencoder_4/sequential_8/reshape_17/Reshape:output:0*
T0*:
_output_shapes(
&:?????????:?????????*

axis*	
num2;
9variational_autoencoder_4/normal_sampling_layer_4/unstack?
Dvariational_autoencoder_4/sequential_9/dense_9/MatMul/ReadVariableOpReadVariableOpMvariational_autoencoder_4_sequential_9_dense_9_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02F
Dvariational_autoencoder_4/sequential_9/dense_9/MatMul/ReadVariableOp?
5variational_autoencoder_4/sequential_9/dense_9/MatMulMatMulBvariational_autoencoder_4/normal_sampling_layer_4/unstack:output:0Lvariational_autoencoder_4/sequential_9/dense_9/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????27
5variational_autoencoder_4/sequential_9/dense_9/MatMul?
Evariational_autoencoder_4/sequential_9/dense_9/BiasAdd/ReadVariableOpReadVariableOpNvariational_autoencoder_4_sequential_9_dense_9_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02G
Evariational_autoencoder_4/sequential_9/dense_9/BiasAdd/ReadVariableOp?
6variational_autoencoder_4/sequential_9/dense_9/BiasAddBiasAdd?variational_autoencoder_4/sequential_9/dense_9/MatMul:product:0Mvariational_autoencoder_4/sequential_9/dense_9/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????28
6variational_autoencoder_4/sequential_9/dense_9/BiasAdd?
3variational_autoencoder_4/sequential_9/dense_9/ReluRelu?variational_autoencoder_4/sequential_9/dense_9/BiasAdd:output:0*
T0*(
_output_shapes
:??????????25
3variational_autoencoder_4/sequential_9/dense_9/Relu?
7variational_autoencoder_4/sequential_9/reshape_18/ShapeShapeAvariational_autoencoder_4/sequential_9/dense_9/Relu:activations:0*
T0*
_output_shapes
:29
7variational_autoencoder_4/sequential_9/reshape_18/Shape?
Evariational_autoencoder_4/sequential_9/reshape_18/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2G
Evariational_autoencoder_4/sequential_9/reshape_18/strided_slice/stack?
Gvariational_autoencoder_4/sequential_9/reshape_18/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2I
Gvariational_autoencoder_4/sequential_9/reshape_18/strided_slice/stack_1?
Gvariational_autoencoder_4/sequential_9/reshape_18/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2I
Gvariational_autoencoder_4/sequential_9/reshape_18/strided_slice/stack_2?
?variational_autoencoder_4/sequential_9/reshape_18/strided_sliceStridedSlice@variational_autoencoder_4/sequential_9/reshape_18/Shape:output:0Nvariational_autoencoder_4/sequential_9/reshape_18/strided_slice/stack:output:0Pvariational_autoencoder_4/sequential_9/reshape_18/strided_slice/stack_1:output:0Pvariational_autoencoder_4/sequential_9/reshape_18/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2A
?variational_autoencoder_4/sequential_9/reshape_18/strided_slice?
Avariational_autoencoder_4/sequential_9/reshape_18/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2C
Avariational_autoencoder_4/sequential_9/reshape_18/Reshape/shape/1?
Avariational_autoencoder_4/sequential_9/reshape_18/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2C
Avariational_autoencoder_4/sequential_9/reshape_18/Reshape/shape/2?
Avariational_autoencoder_4/sequential_9/reshape_18/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B : 2C
Avariational_autoencoder_4/sequential_9/reshape_18/Reshape/shape/3?
?variational_autoencoder_4/sequential_9/reshape_18/Reshape/shapePackHvariational_autoencoder_4/sequential_9/reshape_18/strided_slice:output:0Jvariational_autoencoder_4/sequential_9/reshape_18/Reshape/shape/1:output:0Jvariational_autoencoder_4/sequential_9/reshape_18/Reshape/shape/2:output:0Jvariational_autoencoder_4/sequential_9/reshape_18/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2A
?variational_autoencoder_4/sequential_9/reshape_18/Reshape/shape?
9variational_autoencoder_4/sequential_9/reshape_18/ReshapeReshapeAvariational_autoencoder_4/sequential_9/dense_9/Relu:activations:0Hvariational_autoencoder_4/sequential_9/reshape_18/Reshape/shape:output:0*
T0*/
_output_shapes
:????????? 2;
9variational_autoencoder_4/sequential_9/reshape_18/Reshape?
@variational_autoencoder_4/sequential_9/conv2d_transpose_12/ShapeShapeBvariational_autoencoder_4/sequential_9/reshape_18/Reshape:output:0*
T0*
_output_shapes
:2B
@variational_autoencoder_4/sequential_9/conv2d_transpose_12/Shape?
Nvariational_autoencoder_4/sequential_9/conv2d_transpose_12/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2P
Nvariational_autoencoder_4/sequential_9/conv2d_transpose_12/strided_slice/stack?
Pvariational_autoencoder_4/sequential_9/conv2d_transpose_12/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2R
Pvariational_autoencoder_4/sequential_9/conv2d_transpose_12/strided_slice/stack_1?
Pvariational_autoencoder_4/sequential_9/conv2d_transpose_12/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2R
Pvariational_autoencoder_4/sequential_9/conv2d_transpose_12/strided_slice/stack_2?
Hvariational_autoencoder_4/sequential_9/conv2d_transpose_12/strided_sliceStridedSliceIvariational_autoencoder_4/sequential_9/conv2d_transpose_12/Shape:output:0Wvariational_autoencoder_4/sequential_9/conv2d_transpose_12/strided_slice/stack:output:0Yvariational_autoencoder_4/sequential_9/conv2d_transpose_12/strided_slice/stack_1:output:0Yvariational_autoencoder_4/sequential_9/conv2d_transpose_12/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2J
Hvariational_autoencoder_4/sequential_9/conv2d_transpose_12/strided_slice?
Bvariational_autoencoder_4/sequential_9/conv2d_transpose_12/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2D
Bvariational_autoencoder_4/sequential_9/conv2d_transpose_12/stack/1?
Bvariational_autoencoder_4/sequential_9/conv2d_transpose_12/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2D
Bvariational_autoencoder_4/sequential_9/conv2d_transpose_12/stack/2?
Bvariational_autoencoder_4/sequential_9/conv2d_transpose_12/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@2D
Bvariational_autoencoder_4/sequential_9/conv2d_transpose_12/stack/3?
@variational_autoencoder_4/sequential_9/conv2d_transpose_12/stackPackQvariational_autoencoder_4/sequential_9/conv2d_transpose_12/strided_slice:output:0Kvariational_autoencoder_4/sequential_9/conv2d_transpose_12/stack/1:output:0Kvariational_autoencoder_4/sequential_9/conv2d_transpose_12/stack/2:output:0Kvariational_autoencoder_4/sequential_9/conv2d_transpose_12/stack/3:output:0*
N*
T0*
_output_shapes
:2B
@variational_autoencoder_4/sequential_9/conv2d_transpose_12/stack?
Pvariational_autoencoder_4/sequential_9/conv2d_transpose_12/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2R
Pvariational_autoencoder_4/sequential_9/conv2d_transpose_12/strided_slice_1/stack?
Rvariational_autoencoder_4/sequential_9/conv2d_transpose_12/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2T
Rvariational_autoencoder_4/sequential_9/conv2d_transpose_12/strided_slice_1/stack_1?
Rvariational_autoencoder_4/sequential_9/conv2d_transpose_12/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2T
Rvariational_autoencoder_4/sequential_9/conv2d_transpose_12/strided_slice_1/stack_2?
Jvariational_autoencoder_4/sequential_9/conv2d_transpose_12/strided_slice_1StridedSliceIvariational_autoencoder_4/sequential_9/conv2d_transpose_12/stack:output:0Yvariational_autoencoder_4/sequential_9/conv2d_transpose_12/strided_slice_1/stack:output:0[variational_autoencoder_4/sequential_9/conv2d_transpose_12/strided_slice_1/stack_1:output:0[variational_autoencoder_4/sequential_9/conv2d_transpose_12/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2L
Jvariational_autoencoder_4/sequential_9/conv2d_transpose_12/strided_slice_1?
Zvariational_autoencoder_4/sequential_9/conv2d_transpose_12/conv2d_transpose/ReadVariableOpReadVariableOpcvariational_autoencoder_4_sequential_9_conv2d_transpose_12_conv2d_transpose_readvariableop_resource*&
_output_shapes
:@ *
dtype02\
Zvariational_autoencoder_4/sequential_9/conv2d_transpose_12/conv2d_transpose/ReadVariableOp?
Kvariational_autoencoder_4/sequential_9/conv2d_transpose_12/conv2d_transposeConv2DBackpropInputIvariational_autoencoder_4/sequential_9/conv2d_transpose_12/stack:output:0bvariational_autoencoder_4/sequential_9/conv2d_transpose_12/conv2d_transpose/ReadVariableOp:value:0Bvariational_autoencoder_4/sequential_9/reshape_18/Reshape:output:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2M
Kvariational_autoencoder_4/sequential_9/conv2d_transpose_12/conv2d_transpose?
Qvariational_autoencoder_4/sequential_9/conv2d_transpose_12/BiasAdd/ReadVariableOpReadVariableOpZvariational_autoencoder_4_sequential_9_conv2d_transpose_12_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02S
Qvariational_autoencoder_4/sequential_9/conv2d_transpose_12/BiasAdd/ReadVariableOp?
Bvariational_autoencoder_4/sequential_9/conv2d_transpose_12/BiasAddBiasAddTvariational_autoencoder_4/sequential_9/conv2d_transpose_12/conv2d_transpose:output:0Yvariational_autoencoder_4/sequential_9/conv2d_transpose_12/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2D
Bvariational_autoencoder_4/sequential_9/conv2d_transpose_12/BiasAdd?
?variational_autoencoder_4/sequential_9/conv2d_transpose_12/ReluReluKvariational_autoencoder_4/sequential_9/conv2d_transpose_12/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2A
?variational_autoencoder_4/sequential_9/conv2d_transpose_12/Relu?
@variational_autoencoder_4/sequential_9/conv2d_transpose_13/ShapeShapeMvariational_autoencoder_4/sequential_9/conv2d_transpose_12/Relu:activations:0*
T0*
_output_shapes
:2B
@variational_autoencoder_4/sequential_9/conv2d_transpose_13/Shape?
Nvariational_autoencoder_4/sequential_9/conv2d_transpose_13/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2P
Nvariational_autoencoder_4/sequential_9/conv2d_transpose_13/strided_slice/stack?
Pvariational_autoencoder_4/sequential_9/conv2d_transpose_13/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2R
Pvariational_autoencoder_4/sequential_9/conv2d_transpose_13/strided_slice/stack_1?
Pvariational_autoencoder_4/sequential_9/conv2d_transpose_13/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2R
Pvariational_autoencoder_4/sequential_9/conv2d_transpose_13/strided_slice/stack_2?
Hvariational_autoencoder_4/sequential_9/conv2d_transpose_13/strided_sliceStridedSliceIvariational_autoencoder_4/sequential_9/conv2d_transpose_13/Shape:output:0Wvariational_autoencoder_4/sequential_9/conv2d_transpose_13/strided_slice/stack:output:0Yvariational_autoencoder_4/sequential_9/conv2d_transpose_13/strided_slice/stack_1:output:0Yvariational_autoencoder_4/sequential_9/conv2d_transpose_13/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2J
Hvariational_autoencoder_4/sequential_9/conv2d_transpose_13/strided_slice?
Bvariational_autoencoder_4/sequential_9/conv2d_transpose_13/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2D
Bvariational_autoencoder_4/sequential_9/conv2d_transpose_13/stack/1?
Bvariational_autoencoder_4/sequential_9/conv2d_transpose_13/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2D
Bvariational_autoencoder_4/sequential_9/conv2d_transpose_13/stack/2?
Bvariational_autoencoder_4/sequential_9/conv2d_transpose_13/stack/3Const*
_output_shapes
: *
dtype0*
value	B : 2D
Bvariational_autoencoder_4/sequential_9/conv2d_transpose_13/stack/3?
@variational_autoencoder_4/sequential_9/conv2d_transpose_13/stackPackQvariational_autoencoder_4/sequential_9/conv2d_transpose_13/strided_slice:output:0Kvariational_autoencoder_4/sequential_9/conv2d_transpose_13/stack/1:output:0Kvariational_autoencoder_4/sequential_9/conv2d_transpose_13/stack/2:output:0Kvariational_autoencoder_4/sequential_9/conv2d_transpose_13/stack/3:output:0*
N*
T0*
_output_shapes
:2B
@variational_autoencoder_4/sequential_9/conv2d_transpose_13/stack?
Pvariational_autoencoder_4/sequential_9/conv2d_transpose_13/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2R
Pvariational_autoencoder_4/sequential_9/conv2d_transpose_13/strided_slice_1/stack?
Rvariational_autoencoder_4/sequential_9/conv2d_transpose_13/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2T
Rvariational_autoencoder_4/sequential_9/conv2d_transpose_13/strided_slice_1/stack_1?
Rvariational_autoencoder_4/sequential_9/conv2d_transpose_13/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2T
Rvariational_autoencoder_4/sequential_9/conv2d_transpose_13/strided_slice_1/stack_2?
Jvariational_autoencoder_4/sequential_9/conv2d_transpose_13/strided_slice_1StridedSliceIvariational_autoencoder_4/sequential_9/conv2d_transpose_13/stack:output:0Yvariational_autoencoder_4/sequential_9/conv2d_transpose_13/strided_slice_1/stack:output:0[variational_autoencoder_4/sequential_9/conv2d_transpose_13/strided_slice_1/stack_1:output:0[variational_autoencoder_4/sequential_9/conv2d_transpose_13/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2L
Jvariational_autoencoder_4/sequential_9/conv2d_transpose_13/strided_slice_1?
Zvariational_autoencoder_4/sequential_9/conv2d_transpose_13/conv2d_transpose/ReadVariableOpReadVariableOpcvariational_autoencoder_4_sequential_9_conv2d_transpose_13_conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype02\
Zvariational_autoencoder_4/sequential_9/conv2d_transpose_13/conv2d_transpose/ReadVariableOp?
Kvariational_autoencoder_4/sequential_9/conv2d_transpose_13/conv2d_transposeConv2DBackpropInputIvariational_autoencoder_4/sequential_9/conv2d_transpose_13/stack:output:0bvariational_autoencoder_4/sequential_9/conv2d_transpose_13/conv2d_transpose/ReadVariableOp:value:0Mvariational_autoencoder_4/sequential_9/conv2d_transpose_12/Relu:activations:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
2M
Kvariational_autoencoder_4/sequential_9/conv2d_transpose_13/conv2d_transpose?
Qvariational_autoencoder_4/sequential_9/conv2d_transpose_13/BiasAdd/ReadVariableOpReadVariableOpZvariational_autoencoder_4_sequential_9_conv2d_transpose_13_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02S
Qvariational_autoencoder_4/sequential_9/conv2d_transpose_13/BiasAdd/ReadVariableOp?
Bvariational_autoencoder_4/sequential_9/conv2d_transpose_13/BiasAddBiasAddTvariational_autoencoder_4/sequential_9/conv2d_transpose_13/conv2d_transpose:output:0Yvariational_autoencoder_4/sequential_9/conv2d_transpose_13/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2D
Bvariational_autoencoder_4/sequential_9/conv2d_transpose_13/BiasAdd?
?variational_autoencoder_4/sequential_9/conv2d_transpose_13/ReluReluKvariational_autoencoder_4/sequential_9/conv2d_transpose_13/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2A
?variational_autoencoder_4/sequential_9/conv2d_transpose_13/Relu?
@variational_autoencoder_4/sequential_9/conv2d_transpose_14/ShapeShapeMvariational_autoencoder_4/sequential_9/conv2d_transpose_13/Relu:activations:0*
T0*
_output_shapes
:2B
@variational_autoencoder_4/sequential_9/conv2d_transpose_14/Shape?
Nvariational_autoencoder_4/sequential_9/conv2d_transpose_14/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2P
Nvariational_autoencoder_4/sequential_9/conv2d_transpose_14/strided_slice/stack?
Pvariational_autoencoder_4/sequential_9/conv2d_transpose_14/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2R
Pvariational_autoencoder_4/sequential_9/conv2d_transpose_14/strided_slice/stack_1?
Pvariational_autoencoder_4/sequential_9/conv2d_transpose_14/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2R
Pvariational_autoencoder_4/sequential_9/conv2d_transpose_14/strided_slice/stack_2?
Hvariational_autoencoder_4/sequential_9/conv2d_transpose_14/strided_sliceStridedSliceIvariational_autoencoder_4/sequential_9/conv2d_transpose_14/Shape:output:0Wvariational_autoencoder_4/sequential_9/conv2d_transpose_14/strided_slice/stack:output:0Yvariational_autoencoder_4/sequential_9/conv2d_transpose_14/strided_slice/stack_1:output:0Yvariational_autoencoder_4/sequential_9/conv2d_transpose_14/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2J
Hvariational_autoencoder_4/sequential_9/conv2d_transpose_14/strided_slice?
Bvariational_autoencoder_4/sequential_9/conv2d_transpose_14/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2D
Bvariational_autoencoder_4/sequential_9/conv2d_transpose_14/stack/1?
Bvariational_autoencoder_4/sequential_9/conv2d_transpose_14/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2D
Bvariational_autoencoder_4/sequential_9/conv2d_transpose_14/stack/2?
Bvariational_autoencoder_4/sequential_9/conv2d_transpose_14/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2D
Bvariational_autoencoder_4/sequential_9/conv2d_transpose_14/stack/3?
@variational_autoencoder_4/sequential_9/conv2d_transpose_14/stackPackQvariational_autoencoder_4/sequential_9/conv2d_transpose_14/strided_slice:output:0Kvariational_autoencoder_4/sequential_9/conv2d_transpose_14/stack/1:output:0Kvariational_autoencoder_4/sequential_9/conv2d_transpose_14/stack/2:output:0Kvariational_autoencoder_4/sequential_9/conv2d_transpose_14/stack/3:output:0*
N*
T0*
_output_shapes
:2B
@variational_autoencoder_4/sequential_9/conv2d_transpose_14/stack?
Pvariational_autoencoder_4/sequential_9/conv2d_transpose_14/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2R
Pvariational_autoencoder_4/sequential_9/conv2d_transpose_14/strided_slice_1/stack?
Rvariational_autoencoder_4/sequential_9/conv2d_transpose_14/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2T
Rvariational_autoencoder_4/sequential_9/conv2d_transpose_14/strided_slice_1/stack_1?
Rvariational_autoencoder_4/sequential_9/conv2d_transpose_14/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2T
Rvariational_autoencoder_4/sequential_9/conv2d_transpose_14/strided_slice_1/stack_2?
Jvariational_autoencoder_4/sequential_9/conv2d_transpose_14/strided_slice_1StridedSliceIvariational_autoencoder_4/sequential_9/conv2d_transpose_14/stack:output:0Yvariational_autoencoder_4/sequential_9/conv2d_transpose_14/strided_slice_1/stack:output:0[variational_autoencoder_4/sequential_9/conv2d_transpose_14/strided_slice_1/stack_1:output:0[variational_autoencoder_4/sequential_9/conv2d_transpose_14/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2L
Jvariational_autoencoder_4/sequential_9/conv2d_transpose_14/strided_slice_1?
Zvariational_autoencoder_4/sequential_9/conv2d_transpose_14/conv2d_transpose/ReadVariableOpReadVariableOpcvariational_autoencoder_4_sequential_9_conv2d_transpose_14_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype02\
Zvariational_autoencoder_4/sequential_9/conv2d_transpose_14/conv2d_transpose/ReadVariableOp?
Kvariational_autoencoder_4/sequential_9/conv2d_transpose_14/conv2d_transposeConv2DBackpropInputIvariational_autoencoder_4/sequential_9/conv2d_transpose_14/stack:output:0bvariational_autoencoder_4/sequential_9/conv2d_transpose_14/conv2d_transpose/ReadVariableOp:value:0Mvariational_autoencoder_4/sequential_9/conv2d_transpose_13/Relu:activations:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2M
Kvariational_autoencoder_4/sequential_9/conv2d_transpose_14/conv2d_transpose?
Qvariational_autoencoder_4/sequential_9/conv2d_transpose_14/BiasAdd/ReadVariableOpReadVariableOpZvariational_autoencoder_4_sequential_9_conv2d_transpose_14_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02S
Qvariational_autoencoder_4/sequential_9/conv2d_transpose_14/BiasAdd/ReadVariableOp?
Bvariational_autoencoder_4/sequential_9/conv2d_transpose_14/BiasAddBiasAddTvariational_autoencoder_4/sequential_9/conv2d_transpose_14/conv2d_transpose:output:0Yvariational_autoencoder_4/sequential_9/conv2d_transpose_14/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2D
Bvariational_autoencoder_4/sequential_9/conv2d_transpose_14/BiasAdd?
Bvariational_autoencoder_4/sequential_9/conv2d_transpose_14/SigmoidSigmoidKvariational_autoencoder_4/sequential_9/conv2d_transpose_14/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2D
Bvariational_autoencoder_4/sequential_9/conv2d_transpose_14/Sigmoid?
7variational_autoencoder_4/sequential_9/reshape_19/ShapeShapeFvariational_autoencoder_4/sequential_9/conv2d_transpose_14/Sigmoid:y:0*
T0*
_output_shapes
:29
7variational_autoencoder_4/sequential_9/reshape_19/Shape?
Evariational_autoencoder_4/sequential_9/reshape_19/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2G
Evariational_autoencoder_4/sequential_9/reshape_19/strided_slice/stack?
Gvariational_autoencoder_4/sequential_9/reshape_19/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2I
Gvariational_autoencoder_4/sequential_9/reshape_19/strided_slice/stack_1?
Gvariational_autoencoder_4/sequential_9/reshape_19/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2I
Gvariational_autoencoder_4/sequential_9/reshape_19/strided_slice/stack_2?
?variational_autoencoder_4/sequential_9/reshape_19/strided_sliceStridedSlice@variational_autoencoder_4/sequential_9/reshape_19/Shape:output:0Nvariational_autoencoder_4/sequential_9/reshape_19/strided_slice/stack:output:0Pvariational_autoencoder_4/sequential_9/reshape_19/strided_slice/stack_1:output:0Pvariational_autoencoder_4/sequential_9/reshape_19/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2A
?variational_autoencoder_4/sequential_9/reshape_19/strided_slice?
Avariational_autoencoder_4/sequential_9/reshape_19/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2C
Avariational_autoencoder_4/sequential_9/reshape_19/Reshape/shape/1?
Avariational_autoencoder_4/sequential_9/reshape_19/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2C
Avariational_autoencoder_4/sequential_9/reshape_19/Reshape/shape/2?
Avariational_autoencoder_4/sequential_9/reshape_19/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2C
Avariational_autoencoder_4/sequential_9/reshape_19/Reshape/shape/3?
?variational_autoencoder_4/sequential_9/reshape_19/Reshape/shapePackHvariational_autoencoder_4/sequential_9/reshape_19/strided_slice:output:0Jvariational_autoencoder_4/sequential_9/reshape_19/Reshape/shape/1:output:0Jvariational_autoencoder_4/sequential_9/reshape_19/Reshape/shape/2:output:0Jvariational_autoencoder_4/sequential_9/reshape_19/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2A
?variational_autoencoder_4/sequential_9/reshape_19/Reshape/shape?
9variational_autoencoder_4/sequential_9/reshape_19/ReshapeReshapeFvariational_autoencoder_4/sequential_9/conv2d_transpose_14/Sigmoid:y:0Hvariational_autoencoder_4/sequential_9/reshape_19/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????2;
9variational_autoencoder_4/sequential_9/reshape_19/Reshape?
IdentityIdentityBvariational_autoencoder_4/sequential_9/reshape_19/Reshape:output:0^NoOp*
T0*/
_output_shapes
:?????????2

Identity?	
NoOpNoOpG^variational_autoencoder_4/sequential_8/conv2d_8/BiasAdd/ReadVariableOpF^variational_autoencoder_4/sequential_8/conv2d_8/Conv2D/ReadVariableOpG^variational_autoencoder_4/sequential_8/conv2d_9/BiasAdd/ReadVariableOpF^variational_autoencoder_4/sequential_8/conv2d_9/Conv2D/ReadVariableOpF^variational_autoencoder_4/sequential_8/dense_8/BiasAdd/ReadVariableOpE^variational_autoencoder_4/sequential_8/dense_8/MatMul/ReadVariableOpR^variational_autoencoder_4/sequential_9/conv2d_transpose_12/BiasAdd/ReadVariableOp[^variational_autoencoder_4/sequential_9/conv2d_transpose_12/conv2d_transpose/ReadVariableOpR^variational_autoencoder_4/sequential_9/conv2d_transpose_13/BiasAdd/ReadVariableOp[^variational_autoencoder_4/sequential_9/conv2d_transpose_13/conv2d_transpose/ReadVariableOpR^variational_autoencoder_4/sequential_9/conv2d_transpose_14/BiasAdd/ReadVariableOp[^variational_autoencoder_4/sequential_9/conv2d_transpose_14/conv2d_transpose/ReadVariableOpF^variational_autoencoder_4/sequential_9/dense_9/BiasAdd/ReadVariableOpE^variational_autoencoder_4/sequential_9/dense_9/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:?????????: : : : : : : : : : : : : : 2?
Fvariational_autoencoder_4/sequential_8/conv2d_8/BiasAdd/ReadVariableOpFvariational_autoencoder_4/sequential_8/conv2d_8/BiasAdd/ReadVariableOp2?
Evariational_autoencoder_4/sequential_8/conv2d_8/Conv2D/ReadVariableOpEvariational_autoencoder_4/sequential_8/conv2d_8/Conv2D/ReadVariableOp2?
Fvariational_autoencoder_4/sequential_8/conv2d_9/BiasAdd/ReadVariableOpFvariational_autoencoder_4/sequential_8/conv2d_9/BiasAdd/ReadVariableOp2?
Evariational_autoencoder_4/sequential_8/conv2d_9/Conv2D/ReadVariableOpEvariational_autoencoder_4/sequential_8/conv2d_9/Conv2D/ReadVariableOp2?
Evariational_autoencoder_4/sequential_8/dense_8/BiasAdd/ReadVariableOpEvariational_autoencoder_4/sequential_8/dense_8/BiasAdd/ReadVariableOp2?
Dvariational_autoencoder_4/sequential_8/dense_8/MatMul/ReadVariableOpDvariational_autoencoder_4/sequential_8/dense_8/MatMul/ReadVariableOp2?
Qvariational_autoencoder_4/sequential_9/conv2d_transpose_12/BiasAdd/ReadVariableOpQvariational_autoencoder_4/sequential_9/conv2d_transpose_12/BiasAdd/ReadVariableOp2?
Zvariational_autoencoder_4/sequential_9/conv2d_transpose_12/conv2d_transpose/ReadVariableOpZvariational_autoencoder_4/sequential_9/conv2d_transpose_12/conv2d_transpose/ReadVariableOp2?
Qvariational_autoencoder_4/sequential_9/conv2d_transpose_13/BiasAdd/ReadVariableOpQvariational_autoencoder_4/sequential_9/conv2d_transpose_13/BiasAdd/ReadVariableOp2?
Zvariational_autoencoder_4/sequential_9/conv2d_transpose_13/conv2d_transpose/ReadVariableOpZvariational_autoencoder_4/sequential_9/conv2d_transpose_13/conv2d_transpose/ReadVariableOp2?
Qvariational_autoencoder_4/sequential_9/conv2d_transpose_14/BiasAdd/ReadVariableOpQvariational_autoencoder_4/sequential_9/conv2d_transpose_14/BiasAdd/ReadVariableOp2?
Zvariational_autoencoder_4/sequential_9/conv2d_transpose_14/conv2d_transpose/ReadVariableOpZvariational_autoencoder_4/sequential_9/conv2d_transpose_14/conv2d_transpose/ReadVariableOp2?
Evariational_autoencoder_4/sequential_9/dense_9/BiasAdd/ReadVariableOpEvariational_autoencoder_4/sequential_9/dense_9/BiasAdd/ReadVariableOp2?
Dvariational_autoencoder_4/sequential_9/dense_9/MatMul/ReadVariableOpDvariational_autoencoder_4/sequential_9/dense_9/MatMul/ReadVariableOp:X T
/
_output_shapes
:?????????
!
_user_specified_name	input_1
?
?
T__inference_variational_autoencoder_4_layer_call_and_return_conditional_losses_78092
input_1,
sequential_8_78060:  
sequential_8_78062: ,
sequential_8_78064: @ 
sequential_8_78066:@%
sequential_8_78068:	? 
sequential_8_78070:%
sequential_9_78074:	?!
sequential_9_78076:	?,
sequential_9_78078:@  
sequential_9_78080:@,
sequential_9_78082: @ 
sequential_9_78084: ,
sequential_9_78086:  
sequential_9_78088:
identity??$sequential_8/StatefulPartitionedCall?$sequential_9/StatefulPartitionedCall?
$sequential_8/StatefulPartitionedCallStatefulPartitionedCallinput_1sequential_8_78060sequential_8_78062sequential_8_78064sequential_8_78066sequential_8_78068sequential_8_78070*
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
G__inference_sequential_8_layer_call_and_return_conditional_losses_770112&
$sequential_8/StatefulPartitionedCall?
'normal_sampling_layer_4/PartitionedCallPartitionedCall-sequential_8/StatefulPartitionedCall:output:0*
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
R__inference_normal_sampling_layer_4_layer_call_and_return_conditional_losses_778342)
'normal_sampling_layer_4/PartitionedCall?
$sequential_9/StatefulPartitionedCallStatefulPartitionedCall0normal_sampling_layer_4/PartitionedCall:output:0sequential_9_78074sequential_9_78076sequential_9_78078sequential_9_78080sequential_9_78082sequential_9_78084sequential_9_78086sequential_9_78088*
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
G__inference_sequential_9_layer_call_and_return_conditional_losses_775992&
$sequential_9/StatefulPartitionedCall?
IdentityIdentity-sequential_9/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????2

Identity?
NoOpNoOp%^sequential_8/StatefulPartitionedCall%^sequential_9/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:?????????: : : : : : : : : : : : : : 2L
$sequential_8/StatefulPartitionedCall$sequential_8/StatefulPartitionedCall2L
$sequential_9/StatefulPartitionedCall$sequential_9/StatefulPartitionedCall:X T
/
_output_shapes
:?????????
!
_user_specified_name	input_1
?
E
)__inference_flatten_4_layer_call_fn_78984

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
D__inference_flatten_4_layer_call_and_return_conditional_losses_769772
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
?	
?
,__inference_sequential_8_layer_call_fn_78630

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
G__inference_sequential_8_layer_call_and_return_conditional_losses_770112
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
?
?
N__inference_conv2d_transpose_13_layer_call_and_return_conditional_losses_77547

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
 
_user_specified_nameinputs"?L
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
StatefulPartitionedCall:0?????????tensorflow/serving/predict:֟
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
):' 2conv2d_8/kernel
: 2conv2d_8/bias
):' @2conv2d_9/kernel
:@2conv2d_9/bias
!:	?2dense_8/kernel
:2dense_8/bias
!:	?2dense_9/kernel
:?2dense_9/bias
4:2@ 2conv2d_transpose_12/kernel
&:$@2conv2d_transpose_12/bias
4:2 @2conv2d_transpose_13/kernel
&:$ 2conv2d_transpose_13/bias
4:2 2conv2d_transpose_14/kernel
&:$2conv2d_transpose_14/bias
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
.:, 2Adam/conv2d_8/kernel/m
 : 2Adam/conv2d_8/bias/m
.:, @2Adam/conv2d_9/kernel/m
 :@2Adam/conv2d_9/bias/m
&:$	?2Adam/dense_8/kernel/m
:2Adam/dense_8/bias/m
&:$	?2Adam/dense_9/kernel/m
 :?2Adam/dense_9/bias/m
9:7@ 2!Adam/conv2d_transpose_12/kernel/m
+:)@2Adam/conv2d_transpose_12/bias/m
9:7 @2!Adam/conv2d_transpose_13/kernel/m
+:) 2Adam/conv2d_transpose_13/bias/m
9:7 2!Adam/conv2d_transpose_14/kernel/m
+:)2Adam/conv2d_transpose_14/bias/m
.:, 2Adam/conv2d_8/kernel/v
 : 2Adam/conv2d_8/bias/v
.:, @2Adam/conv2d_9/kernel/v
 :@2Adam/conv2d_9/bias/v
&:$	?2Adam/dense_8/kernel/v
:2Adam/dense_8/bias/v
&:$	?2Adam/dense_9/kernel/v
 :?2Adam/dense_9/bias/v
9:7@ 2!Adam/conv2d_transpose_12/kernel/v
+:)@2Adam/conv2d_transpose_12/bias/v
9:7 @2!Adam/conv2d_transpose_13/kernel/v
+:) 2Adam/conv2d_transpose_13/bias/v
9:7 2!Adam/conv2d_transpose_14/kernel/v
+:)2Adam/conv2d_transpose_14/bias/v
?B?
 __inference__wrapped_model_76914input_1"?
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
T__inference_variational_autoencoder_4_layer_call_and_return_conditional_losses_78302
T__inference_variational_autoencoder_4_layer_call_and_return_conditional_losses_78457
T__inference_variational_autoencoder_4_layer_call_and_return_conditional_losses_78092
T__inference_variational_autoencoder_4_layer_call_and_return_conditional_losses_78127?
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
9__inference_variational_autoencoder_4_layer_call_fn_77885
9__inference_variational_autoencoder_4_layer_call_fn_78490
9__inference_variational_autoencoder_4_layer_call_fn_78523
9__inference_variational_autoencoder_4_layer_call_fn_78057?
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
G__inference_sequential_8_layer_call_and_return_conditional_losses_78568
G__inference_sequential_8_layer_call_and_return_conditional_losses_78613
G__inference_sequential_8_layer_call_and_return_conditional_losses_77169
G__inference_sequential_8_layer_call_and_return_conditional_losses_77191?
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
,__inference_sequential_8_layer_call_fn_77026
,__inference_sequential_8_layer_call_fn_78630
,__inference_sequential_8_layer_call_fn_78647
,__inference_sequential_8_layer_call_fn_77147?
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
G__inference_sequential_9_layer_call_and_return_conditional_losses_78738
G__inference_sequential_9_layer_call_and_return_conditional_losses_78829
G__inference_sequential_9_layer_call_and_return_conditional_losses_77782
G__inference_sequential_9_layer_call_and_return_conditional_losses_77808?
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
,__inference_sequential_9_layer_call_fn_77618
,__inference_sequential_9_layer_call_fn_78850
,__inference_sequential_9_layer_call_fn_78871
,__inference_sequential_9_layer_call_fn_77756?
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
R__inference_normal_sampling_layer_4_layer_call_and_return_conditional_losses_78877
R__inference_normal_sampling_layer_4_layer_call_and_return_conditional_losses_78904?
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
7__inference_normal_sampling_layer_4_layer_call_fn_78909
7__inference_normal_sampling_layer_4_layer_call_fn_78914?
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
#__inference_signature_wrapper_78168input_1"?
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
E__inference_reshape_16_layer_call_and_return_conditional_losses_78928?
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
*__inference_reshape_16_layer_call_fn_78933?
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
C__inference_conv2d_8_layer_call_and_return_conditional_losses_78944?
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
(__inference_conv2d_8_layer_call_fn_78953?
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
C__inference_conv2d_9_layer_call_and_return_conditional_losses_78964?
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
(__inference_conv2d_9_layer_call_fn_78973?
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
D__inference_flatten_4_layer_call_and_return_conditional_losses_78979?
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
)__inference_flatten_4_layer_call_fn_78984?
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
B__inference_dense_8_layer_call_and_return_conditional_losses_78994?
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
'__inference_dense_8_layer_call_fn_79003?
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
E__inference_reshape_17_layer_call_and_return_conditional_losses_79016?
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
*__inference_reshape_17_layer_call_fn_79021?
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
B__inference_dense_9_layer_call_and_return_conditional_losses_79032?
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
'__inference_dense_9_layer_call_fn_79041?
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
E__inference_reshape_18_layer_call_and_return_conditional_losses_79055?
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
*__inference_reshape_18_layer_call_fn_79060?
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
N__inference_conv2d_transpose_12_layer_call_and_return_conditional_losses_79094
N__inference_conv2d_transpose_12_layer_call_and_return_conditional_losses_79118?
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
3__inference_conv2d_transpose_12_layer_call_fn_79127
3__inference_conv2d_transpose_12_layer_call_fn_79136?
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
N__inference_conv2d_transpose_13_layer_call_and_return_conditional_losses_79170
N__inference_conv2d_transpose_13_layer_call_and_return_conditional_losses_79194?
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
3__inference_conv2d_transpose_13_layer_call_fn_79203
3__inference_conv2d_transpose_13_layer_call_fn_79212?
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
N__inference_conv2d_transpose_14_layer_call_and_return_conditional_losses_79246
N__inference_conv2d_transpose_14_layer_call_and_return_conditional_losses_79270?
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
3__inference_conv2d_transpose_14_layer_call_fn_79279
3__inference_conv2d_transpose_14_layer_call_fn_79288?
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
E__inference_reshape_19_layer_call_and_return_conditional_losses_79302?
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
*__inference_reshape_19_layer_call_fn_79307?
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
 __inference__wrapped_model_76914?789:;<=>?@ABCD8?5
.?+
)?&
input_1?????????
? ";?8
6
output_1*?'
output_1??????????
C__inference_conv2d_8_layer_call_and_return_conditional_losses_78944l787?4
-?*
(?%
inputs?????????
? "-?*
#? 
0????????? 
? ?
(__inference_conv2d_8_layer_call_fn_78953_787?4
-?*
(?%
inputs?????????
? " ?????????? ?
C__inference_conv2d_9_layer_call_and_return_conditional_losses_78964l9:7?4
-?*
(?%
inputs????????? 
? "-?*
#? 
0?????????@
? ?
(__inference_conv2d_9_layer_call_fn_78973_9:7?4
-?*
(?%
inputs????????? 
? " ??????????@?
N__inference_conv2d_transpose_12_layer_call_and_return_conditional_losses_79094??@I?F
??<
:?7
inputs+??????????????????????????? 
? "??<
5?2
0+???????????????????????????@
? ?
N__inference_conv2d_transpose_12_layer_call_and_return_conditional_losses_79118l?@7?4
-?*
(?%
inputs????????? 
? "-?*
#? 
0?????????@
? ?
3__inference_conv2d_transpose_12_layer_call_fn_79127??@I?F
??<
:?7
inputs+??????????????????????????? 
? "2?/+???????????????????????????@?
3__inference_conv2d_transpose_12_layer_call_fn_79136_?@7?4
-?*
(?%
inputs????????? 
? " ??????????@?
N__inference_conv2d_transpose_13_layer_call_and_return_conditional_losses_79170?ABI?F
??<
:?7
inputs+???????????????????????????@
? "??<
5?2
0+??????????????????????????? 
? ?
N__inference_conv2d_transpose_13_layer_call_and_return_conditional_losses_79194lAB7?4
-?*
(?%
inputs?????????@
? "-?*
#? 
0????????? 
? ?
3__inference_conv2d_transpose_13_layer_call_fn_79203?ABI?F
??<
:?7
inputs+???????????????????????????@
? "2?/+??????????????????????????? ?
3__inference_conv2d_transpose_13_layer_call_fn_79212_AB7?4
-?*
(?%
inputs?????????@
? " ?????????? ?
N__inference_conv2d_transpose_14_layer_call_and_return_conditional_losses_79246?CDI?F
??<
:?7
inputs+??????????????????????????? 
? "??<
5?2
0+???????????????????????????
? ?
N__inference_conv2d_transpose_14_layer_call_and_return_conditional_losses_79270lCD7?4
-?*
(?%
inputs????????? 
? "-?*
#? 
0?????????
? ?
3__inference_conv2d_transpose_14_layer_call_fn_79279?CDI?F
??<
:?7
inputs+??????????????????????????? 
? "2?/+????????????????????????????
3__inference_conv2d_transpose_14_layer_call_fn_79288_CD7?4
-?*
(?%
inputs????????? 
? " ???????????
B__inference_dense_8_layer_call_and_return_conditional_losses_78994];<0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????
? {
'__inference_dense_8_layer_call_fn_79003P;<0?-
&?#
!?
inputs??????????
? "???????????
B__inference_dense_9_layer_call_and_return_conditional_losses_79032]=>/?,
%?"
 ?
inputs?????????
? "&?#
?
0??????????
? {
'__inference_dense_9_layer_call_fn_79041P=>/?,
%?"
 ?
inputs?????????
? "????????????
D__inference_flatten_4_layer_call_and_return_conditional_losses_78979a7?4
-?*
(?%
inputs?????????@
? "&?#
?
0??????????
? ?
)__inference_flatten_4_layer_call_fn_78984T7?4
-?*
(?%
inputs?????????@
? "????????????
R__inference_normal_sampling_layer_4_layer_call_and_return_conditional_losses_78877`7?4
-?*
$?!
inputs?????????
p 
? "%?"
?
0?????????
? ?
R__inference_normal_sampling_layer_4_layer_call_and_return_conditional_losses_78904`7?4
-?*
$?!
inputs?????????
p
? "%?"
?
0?????????
? ?
7__inference_normal_sampling_layer_4_layer_call_fn_78909S7?4
-?*
$?!
inputs?????????
p 
? "???????????
7__inference_normal_sampling_layer_4_layer_call_fn_78914S7?4
-?*
$?!
inputs?????????
p
? "???????????
E__inference_reshape_16_layer_call_and_return_conditional_losses_78928h7?4
-?*
(?%
inputs?????????
? "-?*
#? 
0?????????
? ?
*__inference_reshape_16_layer_call_fn_78933[7?4
-?*
(?%
inputs?????????
? " ???????????
E__inference_reshape_17_layer_call_and_return_conditional_losses_79016\/?,
%?"
 ?
inputs?????????
? ")?&
?
0?????????
? }
*__inference_reshape_17_layer_call_fn_79021O/?,
%?"
 ?
inputs?????????
? "???????????
E__inference_reshape_18_layer_call_and_return_conditional_losses_79055a0?-
&?#
!?
inputs??????????
? "-?*
#? 
0????????? 
? ?
*__inference_reshape_18_layer_call_fn_79060T0?-
&?#
!?
inputs??????????
? " ?????????? ?
E__inference_reshape_19_layer_call_and_return_conditional_losses_79302h7?4
-?*
(?%
inputs?????????
? "-?*
#? 
0?????????
? ?
*__inference_reshape_19_layer_call_fn_79307[7?4
-?*
(?%
inputs?????????
? " ???????????
G__inference_sequential_8_layer_call_and_return_conditional_losses_77169v789:;<A?>
7?4
*?'
input_17?????????
p 

 
? ")?&
?
0?????????
? ?
G__inference_sequential_8_layer_call_and_return_conditional_losses_77191v789:;<A?>
7?4
*?'
input_17?????????
p

 
? ")?&
?
0?????????
? ?
G__inference_sequential_8_layer_call_and_return_conditional_losses_78568t789:;<??<
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
G__inference_sequential_8_layer_call_and_return_conditional_losses_78613t789:;<??<
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
,__inference_sequential_8_layer_call_fn_77026i789:;<A?>
7?4
*?'
input_17?????????
p 

 
? "???????????
,__inference_sequential_8_layer_call_fn_77147i789:;<A?>
7?4
*?'
input_17?????????
p

 
? "???????????
,__inference_sequential_8_layer_call_fn_78630g789:;<??<
5?2
(?%
inputs?????????
p 

 
? "???????????
,__inference_sequential_8_layer_call_fn_78647g789:;<??<
5?2
(?%
inputs?????????
p

 
? "???????????
G__inference_sequential_9_layer_call_and_return_conditional_losses_77782t=>?@ABCD9?6
/?,
"?
input_18?????????
p 

 
? "-?*
#? 
0?????????
? ?
G__inference_sequential_9_layer_call_and_return_conditional_losses_77808t=>?@ABCD9?6
/?,
"?
input_18?????????
p

 
? "-?*
#? 
0?????????
? ?
G__inference_sequential_9_layer_call_and_return_conditional_losses_78738r=>?@ABCD7?4
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
G__inference_sequential_9_layer_call_and_return_conditional_losses_78829r=>?@ABCD7?4
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
,__inference_sequential_9_layer_call_fn_77618g=>?@ABCD9?6
/?,
"?
input_18?????????
p 

 
? " ???????????
,__inference_sequential_9_layer_call_fn_77756g=>?@ABCD9?6
/?,
"?
input_18?????????
p

 
? " ???????????
,__inference_sequential_9_layer_call_fn_78850e=>?@ABCD7?4
-?*
 ?
inputs?????????
p 

 
? " ???????????
,__inference_sequential_9_layer_call_fn_78871e=>?@ABCD7?4
-?*
 ?
inputs?????????
p

 
? " ???????????
#__inference_signature_wrapper_78168?789:;<=>?@ABCDC?@
? 
9?6
4
input_1)?&
input_1?????????";?8
6
output_1*?'
output_1??????????
T__inference_variational_autoencoder_4_layer_call_and_return_conditional_losses_78092}789:;<=>?@ABCD<?9
2?/
)?&
input_1?????????
p 
? "-?*
#? 
0?????????
? ?
T__inference_variational_autoencoder_4_layer_call_and_return_conditional_losses_78127}789:;<=>?@ABCD<?9
2?/
)?&
input_1?????????
p
? "-?*
#? 
0?????????
? ?
T__inference_variational_autoencoder_4_layer_call_and_return_conditional_losses_78302|789:;<=>?@ABCD;?8
1?.
(?%
inputs?????????
p 
? "-?*
#? 
0?????????
? ?
T__inference_variational_autoencoder_4_layer_call_and_return_conditional_losses_78457|789:;<=>?@ABCD;?8
1?.
(?%
inputs?????????
p
? "-?*
#? 
0?????????
? ?
9__inference_variational_autoencoder_4_layer_call_fn_77885p789:;<=>?@ABCD<?9
2?/
)?&
input_1?????????
p 
? " ???????????
9__inference_variational_autoencoder_4_layer_call_fn_78057p789:;<=>?@ABCD<?9
2?/
)?&
input_1?????????
p
? " ???????????
9__inference_variational_autoencoder_4_layer_call_fn_78490o789:;<=>?@ABCD;?8
1?.
(?%
inputs?????????
p 
? " ???????????
9__inference_variational_autoencoder_4_layer_call_fn_78523o789:;<=>?@ABCD;?8
1?.
(?%
inputs?????????
p
? " ??????????