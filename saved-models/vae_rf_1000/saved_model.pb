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
conv2d_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: * 
shared_nameconv2d_6/kernel
{
#conv2d_6/kernel/Read/ReadVariableOpReadVariableOpconv2d_6/kernel*&
_output_shapes
: *
dtype0
r
conv2d_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_6/bias
k
!conv2d_6/bias/Read/ReadVariableOpReadVariableOpconv2d_6/bias*
_output_shapes
: *
dtype0
?
conv2d_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @* 
shared_nameconv2d_7/kernel
{
#conv2d_7/kernel/Read/ReadVariableOpReadVariableOpconv2d_7/kernel*&
_output_shapes
: @*
dtype0
r
conv2d_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_7/bias
k
!conv2d_7/bias/Read/ReadVariableOpReadVariableOpconv2d_7/bias*
_output_shapes
:@*
dtype0
y
dense_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*
shared_namedense_6/kernel
r
"dense_6/kernel/Read/ReadVariableOpReadVariableOpdense_6/kernel*
_output_shapes
:	?*
dtype0
p
dense_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_6/bias
i
 dense_6/bias/Read/ReadVariableOpReadVariableOpdense_6/bias*
_output_shapes
:*
dtype0
y
dense_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*
shared_namedense_7/kernel
r
"dense_7/kernel/Read/ReadVariableOpReadVariableOpdense_7/kernel*
_output_shapes
:	?*
dtype0
q
dense_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_7/bias
j
 dense_7/bias/Read/ReadVariableOpReadVariableOpdense_7/bias*
_output_shapes	
:?*
dtype0
?
conv2d_transpose_9/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@ **
shared_nameconv2d_transpose_9/kernel
?
-conv2d_transpose_9/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_9/kernel*&
_output_shapes
:@ *
dtype0
?
conv2d_transpose_9/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*(
shared_nameconv2d_transpose_9/bias

+conv2d_transpose_9/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose_9/bias*
_output_shapes
:@*
dtype0
?
conv2d_transpose_10/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*+
shared_nameconv2d_transpose_10/kernel
?
.conv2d_transpose_10/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_10/kernel*&
_output_shapes
: @*
dtype0
?
conv2d_transpose_10/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *)
shared_nameconv2d_transpose_10/bias
?
,conv2d_transpose_10/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose_10/bias*
_output_shapes
: *
dtype0
?
conv2d_transpose_11/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *+
shared_nameconv2d_transpose_11/kernel
?
.conv2d_transpose_11/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_11/kernel*&
_output_shapes
: *
dtype0
?
conv2d_transpose_11/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameconv2d_transpose_11/bias
?
,conv2d_transpose_11/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose_11/bias*
_output_shapes
:*
dtype0
?
Adam/conv2d_6/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_nameAdam/conv2d_6/kernel/m
?
*Adam/conv2d_6/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_6/kernel/m*&
_output_shapes
: *
dtype0
?
Adam/conv2d_6/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/conv2d_6/bias/m
y
(Adam/conv2d_6/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_6/bias/m*
_output_shapes
: *
dtype0
?
Adam/conv2d_7/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*'
shared_nameAdam/conv2d_7/kernel/m
?
*Adam/conv2d_7/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_7/kernel/m*&
_output_shapes
: @*
dtype0
?
Adam/conv2d_7/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/conv2d_7/bias/m
y
(Adam/conv2d_7/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_7/bias/m*
_output_shapes
:@*
dtype0
?
Adam/dense_6/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*&
shared_nameAdam/dense_6/kernel/m
?
)Adam/dense_6/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_6/kernel/m*
_output_shapes
:	?*
dtype0
~
Adam/dense_6/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_6/bias/m
w
'Adam/dense_6/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_6/bias/m*
_output_shapes
:*
dtype0
?
Adam/dense_7/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*&
shared_nameAdam/dense_7/kernel/m
?
)Adam/dense_7/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_7/kernel/m*
_output_shapes
:	?*
dtype0

Adam/dense_7/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*$
shared_nameAdam/dense_7/bias/m
x
'Adam/dense_7/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_7/bias/m*
_output_shapes	
:?*
dtype0
?
 Adam/conv2d_transpose_9/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@ *1
shared_name" Adam/conv2d_transpose_9/kernel/m
?
4Adam/conv2d_transpose_9/kernel/m/Read/ReadVariableOpReadVariableOp Adam/conv2d_transpose_9/kernel/m*&
_output_shapes
:@ *
dtype0
?
Adam/conv2d_transpose_9/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*/
shared_name Adam/conv2d_transpose_9/bias/m
?
2Adam/conv2d_transpose_9/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_transpose_9/bias/m*
_output_shapes
:@*
dtype0
?
!Adam/conv2d_transpose_10/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*2
shared_name#!Adam/conv2d_transpose_10/kernel/m
?
5Adam/conv2d_transpose_10/kernel/m/Read/ReadVariableOpReadVariableOp!Adam/conv2d_transpose_10/kernel/m*&
_output_shapes
: @*
dtype0
?
Adam/conv2d_transpose_10/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *0
shared_name!Adam/conv2d_transpose_10/bias/m
?
3Adam/conv2d_transpose_10/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_transpose_10/bias/m*
_output_shapes
: *
dtype0
?
!Adam/conv2d_transpose_11/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!Adam/conv2d_transpose_11/kernel/m
?
5Adam/conv2d_transpose_11/kernel/m/Read/ReadVariableOpReadVariableOp!Adam/conv2d_transpose_11/kernel/m*&
_output_shapes
: *
dtype0
?
Adam/conv2d_transpose_11/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!Adam/conv2d_transpose_11/bias/m
?
3Adam/conv2d_transpose_11/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_transpose_11/bias/m*
_output_shapes
:*
dtype0
?
Adam/conv2d_6/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_nameAdam/conv2d_6/kernel/v
?
*Adam/conv2d_6/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_6/kernel/v*&
_output_shapes
: *
dtype0
?
Adam/conv2d_6/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/conv2d_6/bias/v
y
(Adam/conv2d_6/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_6/bias/v*
_output_shapes
: *
dtype0
?
Adam/conv2d_7/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*'
shared_nameAdam/conv2d_7/kernel/v
?
*Adam/conv2d_7/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_7/kernel/v*&
_output_shapes
: @*
dtype0
?
Adam/conv2d_7/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/conv2d_7/bias/v
y
(Adam/conv2d_7/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_7/bias/v*
_output_shapes
:@*
dtype0
?
Adam/dense_6/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*&
shared_nameAdam/dense_6/kernel/v
?
)Adam/dense_6/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_6/kernel/v*
_output_shapes
:	?*
dtype0
~
Adam/dense_6/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_6/bias/v
w
'Adam/dense_6/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_6/bias/v*
_output_shapes
:*
dtype0
?
Adam/dense_7/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*&
shared_nameAdam/dense_7/kernel/v
?
)Adam/dense_7/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_7/kernel/v*
_output_shapes
:	?*
dtype0

Adam/dense_7/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*$
shared_nameAdam/dense_7/bias/v
x
'Adam/dense_7/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_7/bias/v*
_output_shapes	
:?*
dtype0
?
 Adam/conv2d_transpose_9/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@ *1
shared_name" Adam/conv2d_transpose_9/kernel/v
?
4Adam/conv2d_transpose_9/kernel/v/Read/ReadVariableOpReadVariableOp Adam/conv2d_transpose_9/kernel/v*&
_output_shapes
:@ *
dtype0
?
Adam/conv2d_transpose_9/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*/
shared_name Adam/conv2d_transpose_9/bias/v
?
2Adam/conv2d_transpose_9/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_transpose_9/bias/v*
_output_shapes
:@*
dtype0
?
!Adam/conv2d_transpose_10/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*2
shared_name#!Adam/conv2d_transpose_10/kernel/v
?
5Adam/conv2d_transpose_10/kernel/v/Read/ReadVariableOpReadVariableOp!Adam/conv2d_transpose_10/kernel/v*&
_output_shapes
: @*
dtype0
?
Adam/conv2d_transpose_10/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *0
shared_name!Adam/conv2d_transpose_10/bias/v
?
3Adam/conv2d_transpose_10/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_transpose_10/bias/v*
_output_shapes
: *
dtype0
?
!Adam/conv2d_transpose_11/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!Adam/conv2d_transpose_11/kernel/v
?
5Adam/conv2d_transpose_11/kernel/v/Read/ReadVariableOpReadVariableOp!Adam/conv2d_transpose_11/kernel/v*&
_output_shapes
: *
dtype0
?
Adam/conv2d_transpose_11/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!Adam/conv2d_transpose_11/bias/v
?
3Adam/conv2d_transpose_11/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_transpose_11/bias/v*
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
VARIABLE_VALUEconv2d_6/kernel0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEconv2d_6/bias0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEconv2d_7/kernel0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEconv2d_7/bias0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEdense_6/kernel0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUEdense_6/bias0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEdense_7/kernel0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUEdense_7/bias0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUEconv2d_transpose_9/kernel0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUEconv2d_transpose_9/bias0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUEconv2d_transpose_10/kernel1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUEconv2d_transpose_10/bias1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUEconv2d_transpose_11/kernel1trainable_variables/12/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUEconv2d_transpose_11/bias1trainable_variables/13/.ATTRIBUTES/VARIABLE_VALUE
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
VARIABLE_VALUEAdam/conv2d_6/kernel/mLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/conv2d_6/bias/mLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/conv2d_7/kernel/mLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/conv2d_7/bias/mLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense_6/kernel/mLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUEAdam/dense_6/bias/mLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense_7/kernel/mLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUEAdam/dense_7/bias/mLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE Adam/conv2d_transpose_9/kernel/mLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/conv2d_transpose_9/bias/mLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE!Adam/conv2d_transpose_10/kernel/mMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/conv2d_transpose_10/bias/mMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE!Adam/conv2d_transpose_11/kernel/mMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/conv2d_transpose_11/bias/mMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/conv2d_6/kernel/vLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/conv2d_6/bias/vLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/conv2d_7/kernel/vLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/conv2d_7/bias/vLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense_6/kernel/vLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUEAdam/dense_6/bias/vLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense_7/kernel/vLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUEAdam/dense_7/bias/vLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE Adam/conv2d_transpose_9/kernel/vLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/conv2d_transpose_9/bias/vLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE!Adam/conv2d_transpose_10/kernel/vMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/conv2d_transpose_10/bias/vMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE!Adam/conv2d_transpose_11/kernel/vMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/conv2d_transpose_11/bias/vMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
serving_default_input_1Placeholder*/
_output_shapes
:?????????*
dtype0*$
shape:?????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1conv2d_6/kernelconv2d_6/biasconv2d_7/kernelconv2d_7/biasdense_6/kerneldense_6/biasdense_7/kerneldense_7/biasconv2d_transpose_9/kernelconv2d_transpose_9/biasconv2d_transpose_10/kernelconv2d_transpose_10/biasconv2d_transpose_11/kernelconv2d_transpose_11/bias*
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
#__inference_signature_wrapper_62190
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenametotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal_2/Read/ReadVariableOpcount_2/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp#conv2d_6/kernel/Read/ReadVariableOp!conv2d_6/bias/Read/ReadVariableOp#conv2d_7/kernel/Read/ReadVariableOp!conv2d_7/bias/Read/ReadVariableOp"dense_6/kernel/Read/ReadVariableOp dense_6/bias/Read/ReadVariableOp"dense_7/kernel/Read/ReadVariableOp dense_7/bias/Read/ReadVariableOp-conv2d_transpose_9/kernel/Read/ReadVariableOp+conv2d_transpose_9/bias/Read/ReadVariableOp.conv2d_transpose_10/kernel/Read/ReadVariableOp,conv2d_transpose_10/bias/Read/ReadVariableOp.conv2d_transpose_11/kernel/Read/ReadVariableOp,conv2d_transpose_11/bias/Read/ReadVariableOp*Adam/conv2d_6/kernel/m/Read/ReadVariableOp(Adam/conv2d_6/bias/m/Read/ReadVariableOp*Adam/conv2d_7/kernel/m/Read/ReadVariableOp(Adam/conv2d_7/bias/m/Read/ReadVariableOp)Adam/dense_6/kernel/m/Read/ReadVariableOp'Adam/dense_6/bias/m/Read/ReadVariableOp)Adam/dense_7/kernel/m/Read/ReadVariableOp'Adam/dense_7/bias/m/Read/ReadVariableOp4Adam/conv2d_transpose_9/kernel/m/Read/ReadVariableOp2Adam/conv2d_transpose_9/bias/m/Read/ReadVariableOp5Adam/conv2d_transpose_10/kernel/m/Read/ReadVariableOp3Adam/conv2d_transpose_10/bias/m/Read/ReadVariableOp5Adam/conv2d_transpose_11/kernel/m/Read/ReadVariableOp3Adam/conv2d_transpose_11/bias/m/Read/ReadVariableOp*Adam/conv2d_6/kernel/v/Read/ReadVariableOp(Adam/conv2d_6/bias/v/Read/ReadVariableOp*Adam/conv2d_7/kernel/v/Read/ReadVariableOp(Adam/conv2d_7/bias/v/Read/ReadVariableOp)Adam/dense_6/kernel/v/Read/ReadVariableOp'Adam/dense_6/bias/v/Read/ReadVariableOp)Adam/dense_7/kernel/v/Read/ReadVariableOp'Adam/dense_7/bias/v/Read/ReadVariableOp4Adam/conv2d_transpose_9/kernel/v/Read/ReadVariableOp2Adam/conv2d_transpose_9/bias/v/Read/ReadVariableOp5Adam/conv2d_transpose_10/kernel/v/Read/ReadVariableOp3Adam/conv2d_transpose_10/bias/v/Read/ReadVariableOp5Adam/conv2d_transpose_11/kernel/v/Read/ReadVariableOp3Adam/conv2d_transpose_11/bias/v/Read/ReadVariableOpConst*B
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
__inference__traced_save_63511
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenametotalcounttotal_1count_1total_2count_2	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_rateconv2d_6/kernelconv2d_6/biasconv2d_7/kernelconv2d_7/biasdense_6/kerneldense_6/biasdense_7/kerneldense_7/biasconv2d_transpose_9/kernelconv2d_transpose_9/biasconv2d_transpose_10/kernelconv2d_transpose_10/biasconv2d_transpose_11/kernelconv2d_transpose_11/biasAdam/conv2d_6/kernel/mAdam/conv2d_6/bias/mAdam/conv2d_7/kernel/mAdam/conv2d_7/bias/mAdam/dense_6/kernel/mAdam/dense_6/bias/mAdam/dense_7/kernel/mAdam/dense_7/bias/m Adam/conv2d_transpose_9/kernel/mAdam/conv2d_transpose_9/bias/m!Adam/conv2d_transpose_10/kernel/mAdam/conv2d_transpose_10/bias/m!Adam/conv2d_transpose_11/kernel/mAdam/conv2d_transpose_11/bias/mAdam/conv2d_6/kernel/vAdam/conv2d_6/bias/vAdam/conv2d_7/kernel/vAdam/conv2d_7/bias/vAdam/dense_6/kernel/vAdam/dense_6/bias/vAdam/dense_7/kernel/vAdam/dense_7/bias/v Adam/conv2d_transpose_9/kernel/vAdam/conv2d_transpose_9/bias/v!Adam/conv2d_transpose_10/kernel/vAdam/conv2d_transpose_10/bias/v!Adam/conv2d_transpose_11/kernel/vAdam/conv2d_transpose_11/bias/v*A
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
!__inference__traced_restore_63680܆
?&
?
N__inference_conv2d_transpose_11_layer_call_and_return_conditional_losses_63268

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
S
7__inference_normal_sampling_layer_3_layer_call_fn_62931

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
R__inference_normal_sampling_layer_3_layer_call_and_return_conditional_losses_618562
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
??
?
G__inference_sequential_7_layer_call_and_return_conditional_losses_62760

inputs9
&dense_7_matmul_readvariableop_resource:	?6
'dense_7_biasadd_readvariableop_resource:	?U
;conv2d_transpose_9_conv2d_transpose_readvariableop_resource:@ @
2conv2d_transpose_9_biasadd_readvariableop_resource:@V
<conv2d_transpose_10_conv2d_transpose_readvariableop_resource: @A
3conv2d_transpose_10_biasadd_readvariableop_resource: V
<conv2d_transpose_11_conv2d_transpose_readvariableop_resource: A
3conv2d_transpose_11_biasadd_readvariableop_resource:
identity??*conv2d_transpose_10/BiasAdd/ReadVariableOp?3conv2d_transpose_10/conv2d_transpose/ReadVariableOp?*conv2d_transpose_11/BiasAdd/ReadVariableOp?3conv2d_transpose_11/conv2d_transpose/ReadVariableOp?)conv2d_transpose_9/BiasAdd/ReadVariableOp?2conv2d_transpose_9/conv2d_transpose/ReadVariableOp?dense_7/BiasAdd/ReadVariableOp?dense_7/MatMul/ReadVariableOp?
dense_7/MatMul/ReadVariableOpReadVariableOp&dense_7_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
dense_7/MatMul/ReadVariableOp?
dense_7/MatMulMatMulinputs%dense_7/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_7/MatMul?
dense_7/BiasAdd/ReadVariableOpReadVariableOp'dense_7_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
dense_7/BiasAdd/ReadVariableOp?
dense_7/BiasAddBiasAdddense_7/MatMul:product:0&dense_7/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_7/BiasAddq
dense_7/ReluReludense_7/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_7/Relun
reshape_14/ShapeShapedense_7/Relu:activations:0*
T0*
_output_shapes
:2
reshape_14/Shape?
reshape_14/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
reshape_14/strided_slice/stack?
 reshape_14/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_14/strided_slice/stack_1?
 reshape_14/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_14/strided_slice/stack_2?
reshape_14/strided_sliceStridedSlicereshape_14/Shape:output:0'reshape_14/strided_slice/stack:output:0)reshape_14/strided_slice/stack_1:output:0)reshape_14/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_14/strided_slicez
reshape_14/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_14/Reshape/shape/1z
reshape_14/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_14/Reshape/shape/2z
reshape_14/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B : 2
reshape_14/Reshape/shape/3?
reshape_14/Reshape/shapePack!reshape_14/strided_slice:output:0#reshape_14/Reshape/shape/1:output:0#reshape_14/Reshape/shape/2:output:0#reshape_14/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
reshape_14/Reshape/shape?
reshape_14/ReshapeReshapedense_7/Relu:activations:0!reshape_14/Reshape/shape:output:0*
T0*/
_output_shapes
:????????? 2
reshape_14/Reshape
conv2d_transpose_9/ShapeShapereshape_14/Reshape:output:0*
T0*
_output_shapes
:2
conv2d_transpose_9/Shape?
&conv2d_transpose_9/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv2d_transpose_9/strided_slice/stack?
(conv2d_transpose_9/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_9/strided_slice/stack_1?
(conv2d_transpose_9/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_9/strided_slice/stack_2?
 conv2d_transpose_9/strided_sliceStridedSlice!conv2d_transpose_9/Shape:output:0/conv2d_transpose_9/strided_slice/stack:output:01conv2d_transpose_9/strided_slice/stack_1:output:01conv2d_transpose_9/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv2d_transpose_9/strided_slicez
conv2d_transpose_9/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_9/stack/1z
conv2d_transpose_9/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_9/stack/2z
conv2d_transpose_9/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@2
conv2d_transpose_9/stack/3?
conv2d_transpose_9/stackPack)conv2d_transpose_9/strided_slice:output:0#conv2d_transpose_9/stack/1:output:0#conv2d_transpose_9/stack/2:output:0#conv2d_transpose_9/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_9/stack?
(conv2d_transpose_9/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(conv2d_transpose_9/strided_slice_1/stack?
*conv2d_transpose_9/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_9/strided_slice_1/stack_1?
*conv2d_transpose_9/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_9/strided_slice_1/stack_2?
"conv2d_transpose_9/strided_slice_1StridedSlice!conv2d_transpose_9/stack:output:01conv2d_transpose_9/strided_slice_1/stack:output:03conv2d_transpose_9/strided_slice_1/stack_1:output:03conv2d_transpose_9/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv2d_transpose_9/strided_slice_1?
2conv2d_transpose_9/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_9_conv2d_transpose_readvariableop_resource*&
_output_shapes
:@ *
dtype024
2conv2d_transpose_9/conv2d_transpose/ReadVariableOp?
#conv2d_transpose_9/conv2d_transposeConv2DBackpropInput!conv2d_transpose_9/stack:output:0:conv2d_transpose_9/conv2d_transpose/ReadVariableOp:value:0reshape_14/Reshape:output:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2%
#conv2d_transpose_9/conv2d_transpose?
)conv2d_transpose_9/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_9_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02+
)conv2d_transpose_9/BiasAdd/ReadVariableOp?
conv2d_transpose_9/BiasAddBiasAdd,conv2d_transpose_9/conv2d_transpose:output:01conv2d_transpose_9/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
conv2d_transpose_9/BiasAdd?
conv2d_transpose_9/ReluRelu#conv2d_transpose_9/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
conv2d_transpose_9/Relu?
conv2d_transpose_10/ShapeShape%conv2d_transpose_9/Relu:activations:0*
T0*
_output_shapes
:2
conv2d_transpose_10/Shape?
'conv2d_transpose_10/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'conv2d_transpose_10/strided_slice/stack?
)conv2d_transpose_10/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_10/strided_slice/stack_1?
)conv2d_transpose_10/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_10/strided_slice/stack_2?
!conv2d_transpose_10/strided_sliceStridedSlice"conv2d_transpose_10/Shape:output:00conv2d_transpose_10/strided_slice/stack:output:02conv2d_transpose_10/strided_slice/stack_1:output:02conv2d_transpose_10/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!conv2d_transpose_10/strided_slice|
conv2d_transpose_10/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_10/stack/1|
conv2d_transpose_10/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_10/stack/2|
conv2d_transpose_10/stack/3Const*
_output_shapes
: *
dtype0*
value	B : 2
conv2d_transpose_10/stack/3?
conv2d_transpose_10/stackPack*conv2d_transpose_10/strided_slice:output:0$conv2d_transpose_10/stack/1:output:0$conv2d_transpose_10/stack/2:output:0$conv2d_transpose_10/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_10/stack?
)conv2d_transpose_10/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)conv2d_transpose_10/strided_slice_1/stack?
+conv2d_transpose_10/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_10/strided_slice_1/stack_1?
+conv2d_transpose_10/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_10/strided_slice_1/stack_2?
#conv2d_transpose_10/strided_slice_1StridedSlice"conv2d_transpose_10/stack:output:02conv2d_transpose_10/strided_slice_1/stack:output:04conv2d_transpose_10/strided_slice_1/stack_1:output:04conv2d_transpose_10/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#conv2d_transpose_10/strided_slice_1?
3conv2d_transpose_10/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_10_conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype025
3conv2d_transpose_10/conv2d_transpose/ReadVariableOp?
$conv2d_transpose_10/conv2d_transposeConv2DBackpropInput"conv2d_transpose_10/stack:output:0;conv2d_transpose_10/conv2d_transpose/ReadVariableOp:value:0%conv2d_transpose_9/Relu:activations:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
2&
$conv2d_transpose_10/conv2d_transpose?
*conv2d_transpose_10/BiasAdd/ReadVariableOpReadVariableOp3conv2d_transpose_10_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02,
*conv2d_transpose_10/BiasAdd/ReadVariableOp?
conv2d_transpose_10/BiasAddBiasAdd-conv2d_transpose_10/conv2d_transpose:output:02conv2d_transpose_10/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2
conv2d_transpose_10/BiasAdd?
conv2d_transpose_10/ReluRelu$conv2d_transpose_10/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
conv2d_transpose_10/Relu?
conv2d_transpose_11/ShapeShape&conv2d_transpose_10/Relu:activations:0*
T0*
_output_shapes
:2
conv2d_transpose_11/Shape?
'conv2d_transpose_11/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'conv2d_transpose_11/strided_slice/stack?
)conv2d_transpose_11/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_11/strided_slice/stack_1?
)conv2d_transpose_11/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_11/strided_slice/stack_2?
!conv2d_transpose_11/strided_sliceStridedSlice"conv2d_transpose_11/Shape:output:00conv2d_transpose_11/strided_slice/stack:output:02conv2d_transpose_11/strided_slice/stack_1:output:02conv2d_transpose_11/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!conv2d_transpose_11/strided_slice|
conv2d_transpose_11/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_11/stack/1|
conv2d_transpose_11/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_11/stack/2|
conv2d_transpose_11/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_11/stack/3?
conv2d_transpose_11/stackPack*conv2d_transpose_11/strided_slice:output:0$conv2d_transpose_11/stack/1:output:0$conv2d_transpose_11/stack/2:output:0$conv2d_transpose_11/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_11/stack?
)conv2d_transpose_11/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)conv2d_transpose_11/strided_slice_1/stack?
+conv2d_transpose_11/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_11/strided_slice_1/stack_1?
+conv2d_transpose_11/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_11/strided_slice_1/stack_2?
#conv2d_transpose_11/strided_slice_1StridedSlice"conv2d_transpose_11/stack:output:02conv2d_transpose_11/strided_slice_1/stack:output:04conv2d_transpose_11/strided_slice_1/stack_1:output:04conv2d_transpose_11/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#conv2d_transpose_11/strided_slice_1?
3conv2d_transpose_11/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_11_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype025
3conv2d_transpose_11/conv2d_transpose/ReadVariableOp?
$conv2d_transpose_11/conv2d_transposeConv2DBackpropInput"conv2d_transpose_11/stack:output:0;conv2d_transpose_11/conv2d_transpose/ReadVariableOp:value:0&conv2d_transpose_10/Relu:activations:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2&
$conv2d_transpose_11/conv2d_transpose?
*conv2d_transpose_11/BiasAdd/ReadVariableOpReadVariableOp3conv2d_transpose_11_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02,
*conv2d_transpose_11/BiasAdd/ReadVariableOp?
conv2d_transpose_11/BiasAddBiasAdd-conv2d_transpose_11/conv2d_transpose:output:02conv2d_transpose_11/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
conv2d_transpose_11/BiasAdd?
conv2d_transpose_11/SigmoidSigmoid$conv2d_transpose_11/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
conv2d_transpose_11/Sigmoids
reshape_15/ShapeShapeconv2d_transpose_11/Sigmoid:y:0*
T0*
_output_shapes
:2
reshape_15/Shape?
reshape_15/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
reshape_15/strided_slice/stack?
 reshape_15/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_15/strided_slice/stack_1?
 reshape_15/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_15/strided_slice/stack_2?
reshape_15/strided_sliceStridedSlicereshape_15/Shape:output:0'reshape_15/strided_slice/stack:output:0)reshape_15/strided_slice/stack_1:output:0)reshape_15/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_15/strided_slicez
reshape_15/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_15/Reshape/shape/1z
reshape_15/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_15/Reshape/shape/2z
reshape_15/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_15/Reshape/shape/3?
reshape_15/Reshape/shapePack!reshape_15/strided_slice:output:0#reshape_15/Reshape/shape/1:output:0#reshape_15/Reshape/shape/2:output:0#reshape_15/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
reshape_15/Reshape/shape?
reshape_15/ReshapeReshapeconv2d_transpose_11/Sigmoid:y:0!reshape_15/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????2
reshape_15/Reshape~
IdentityIdentityreshape_15/Reshape:output:0^NoOp*
T0*/
_output_shapes
:?????????2

Identity?
NoOpNoOp+^conv2d_transpose_10/BiasAdd/ReadVariableOp4^conv2d_transpose_10/conv2d_transpose/ReadVariableOp+^conv2d_transpose_11/BiasAdd/ReadVariableOp4^conv2d_transpose_11/conv2d_transpose/ReadVariableOp*^conv2d_transpose_9/BiasAdd/ReadVariableOp3^conv2d_transpose_9/conv2d_transpose/ReadVariableOp^dense_7/BiasAdd/ReadVariableOp^dense_7/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : : : : : 2X
*conv2d_transpose_10/BiasAdd/ReadVariableOp*conv2d_transpose_10/BiasAdd/ReadVariableOp2j
3conv2d_transpose_10/conv2d_transpose/ReadVariableOp3conv2d_transpose_10/conv2d_transpose/ReadVariableOp2X
*conv2d_transpose_11/BiasAdd/ReadVariableOp*conv2d_transpose_11/BiasAdd/ReadVariableOp2j
3conv2d_transpose_11/conv2d_transpose/ReadVariableOp3conv2d_transpose_11/conv2d_transpose/ReadVariableOp2V
)conv2d_transpose_9/BiasAdd/ReadVariableOp)conv2d_transpose_9/BiasAdd/ReadVariableOp2h
2conv2d_transpose_9/conv2d_transpose/ReadVariableOp2conv2d_transpose_9/conv2d_transpose/ReadVariableOp2@
dense_7/BiasAdd/ReadVariableOpdense_7/BiasAdd/ReadVariableOp2>
dense_7/MatMul/ReadVariableOpdense_7/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
??
?
 __inference__wrapped_model_60936
input_1h
Nvariational_autoencoder_3_sequential_6_conv2d_6_conv2d_readvariableop_resource: ]
Ovariational_autoencoder_3_sequential_6_conv2d_6_biasadd_readvariableop_resource: h
Nvariational_autoencoder_3_sequential_6_conv2d_7_conv2d_readvariableop_resource: @]
Ovariational_autoencoder_3_sequential_6_conv2d_7_biasadd_readvariableop_resource:@`
Mvariational_autoencoder_3_sequential_6_dense_6_matmul_readvariableop_resource:	?\
Nvariational_autoencoder_3_sequential_6_dense_6_biasadd_readvariableop_resource:`
Mvariational_autoencoder_3_sequential_7_dense_7_matmul_readvariableop_resource:	?]
Nvariational_autoencoder_3_sequential_7_dense_7_biasadd_readvariableop_resource:	?|
bvariational_autoencoder_3_sequential_7_conv2d_transpose_9_conv2d_transpose_readvariableop_resource:@ g
Yvariational_autoencoder_3_sequential_7_conv2d_transpose_9_biasadd_readvariableop_resource:@}
cvariational_autoencoder_3_sequential_7_conv2d_transpose_10_conv2d_transpose_readvariableop_resource: @h
Zvariational_autoencoder_3_sequential_7_conv2d_transpose_10_biasadd_readvariableop_resource: }
cvariational_autoencoder_3_sequential_7_conv2d_transpose_11_conv2d_transpose_readvariableop_resource: h
Zvariational_autoencoder_3_sequential_7_conv2d_transpose_11_biasadd_readvariableop_resource:
identity??Fvariational_autoencoder_3/sequential_6/conv2d_6/BiasAdd/ReadVariableOp?Evariational_autoencoder_3/sequential_6/conv2d_6/Conv2D/ReadVariableOp?Fvariational_autoencoder_3/sequential_6/conv2d_7/BiasAdd/ReadVariableOp?Evariational_autoencoder_3/sequential_6/conv2d_7/Conv2D/ReadVariableOp?Evariational_autoencoder_3/sequential_6/dense_6/BiasAdd/ReadVariableOp?Dvariational_autoencoder_3/sequential_6/dense_6/MatMul/ReadVariableOp?Qvariational_autoencoder_3/sequential_7/conv2d_transpose_10/BiasAdd/ReadVariableOp?Zvariational_autoencoder_3/sequential_7/conv2d_transpose_10/conv2d_transpose/ReadVariableOp?Qvariational_autoencoder_3/sequential_7/conv2d_transpose_11/BiasAdd/ReadVariableOp?Zvariational_autoencoder_3/sequential_7/conv2d_transpose_11/conv2d_transpose/ReadVariableOp?Pvariational_autoencoder_3/sequential_7/conv2d_transpose_9/BiasAdd/ReadVariableOp?Yvariational_autoencoder_3/sequential_7/conv2d_transpose_9/conv2d_transpose/ReadVariableOp?Evariational_autoencoder_3/sequential_7/dense_7/BiasAdd/ReadVariableOp?Dvariational_autoencoder_3/sequential_7/dense_7/MatMul/ReadVariableOp?
7variational_autoencoder_3/sequential_6/reshape_12/ShapeShapeinput_1*
T0*
_output_shapes
:29
7variational_autoencoder_3/sequential_6/reshape_12/Shape?
Evariational_autoencoder_3/sequential_6/reshape_12/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2G
Evariational_autoencoder_3/sequential_6/reshape_12/strided_slice/stack?
Gvariational_autoencoder_3/sequential_6/reshape_12/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2I
Gvariational_autoencoder_3/sequential_6/reshape_12/strided_slice/stack_1?
Gvariational_autoencoder_3/sequential_6/reshape_12/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2I
Gvariational_autoencoder_3/sequential_6/reshape_12/strided_slice/stack_2?
?variational_autoencoder_3/sequential_6/reshape_12/strided_sliceStridedSlice@variational_autoencoder_3/sequential_6/reshape_12/Shape:output:0Nvariational_autoencoder_3/sequential_6/reshape_12/strided_slice/stack:output:0Pvariational_autoencoder_3/sequential_6/reshape_12/strided_slice/stack_1:output:0Pvariational_autoencoder_3/sequential_6/reshape_12/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2A
?variational_autoencoder_3/sequential_6/reshape_12/strided_slice?
Avariational_autoencoder_3/sequential_6/reshape_12/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2C
Avariational_autoencoder_3/sequential_6/reshape_12/Reshape/shape/1?
Avariational_autoencoder_3/sequential_6/reshape_12/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2C
Avariational_autoencoder_3/sequential_6/reshape_12/Reshape/shape/2?
Avariational_autoencoder_3/sequential_6/reshape_12/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2C
Avariational_autoencoder_3/sequential_6/reshape_12/Reshape/shape/3?
?variational_autoencoder_3/sequential_6/reshape_12/Reshape/shapePackHvariational_autoencoder_3/sequential_6/reshape_12/strided_slice:output:0Jvariational_autoencoder_3/sequential_6/reshape_12/Reshape/shape/1:output:0Jvariational_autoencoder_3/sequential_6/reshape_12/Reshape/shape/2:output:0Jvariational_autoencoder_3/sequential_6/reshape_12/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2A
?variational_autoencoder_3/sequential_6/reshape_12/Reshape/shape?
9variational_autoencoder_3/sequential_6/reshape_12/ReshapeReshapeinput_1Hvariational_autoencoder_3/sequential_6/reshape_12/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????2;
9variational_autoencoder_3/sequential_6/reshape_12/Reshape?
Evariational_autoencoder_3/sequential_6/conv2d_6/Conv2D/ReadVariableOpReadVariableOpNvariational_autoencoder_3_sequential_6_conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02G
Evariational_autoencoder_3/sequential_6/conv2d_6/Conv2D/ReadVariableOp?
6variational_autoencoder_3/sequential_6/conv2d_6/Conv2DConv2DBvariational_autoencoder_3/sequential_6/reshape_12/Reshape:output:0Mvariational_autoencoder_3/sequential_6/conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
28
6variational_autoencoder_3/sequential_6/conv2d_6/Conv2D?
Fvariational_autoencoder_3/sequential_6/conv2d_6/BiasAdd/ReadVariableOpReadVariableOpOvariational_autoencoder_3_sequential_6_conv2d_6_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02H
Fvariational_autoencoder_3/sequential_6/conv2d_6/BiasAdd/ReadVariableOp?
7variational_autoencoder_3/sequential_6/conv2d_6/BiasAddBiasAdd?variational_autoencoder_3/sequential_6/conv2d_6/Conv2D:output:0Nvariational_autoencoder_3/sequential_6/conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 29
7variational_autoencoder_3/sequential_6/conv2d_6/BiasAdd?
4variational_autoencoder_3/sequential_6/conv2d_6/ReluRelu@variational_autoencoder_3/sequential_6/conv2d_6/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 26
4variational_autoencoder_3/sequential_6/conv2d_6/Relu?
Evariational_autoencoder_3/sequential_6/conv2d_7/Conv2D/ReadVariableOpReadVariableOpNvariational_autoencoder_3_sequential_6_conv2d_7_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02G
Evariational_autoencoder_3/sequential_6/conv2d_7/Conv2D/ReadVariableOp?
6variational_autoencoder_3/sequential_6/conv2d_7/Conv2DConv2DBvariational_autoencoder_3/sequential_6/conv2d_6/Relu:activations:0Mvariational_autoencoder_3/sequential_6/conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
28
6variational_autoencoder_3/sequential_6/conv2d_7/Conv2D?
Fvariational_autoencoder_3/sequential_6/conv2d_7/BiasAdd/ReadVariableOpReadVariableOpOvariational_autoencoder_3_sequential_6_conv2d_7_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02H
Fvariational_autoencoder_3/sequential_6/conv2d_7/BiasAdd/ReadVariableOp?
7variational_autoencoder_3/sequential_6/conv2d_7/BiasAddBiasAdd?variational_autoencoder_3/sequential_6/conv2d_7/Conv2D:output:0Nvariational_autoencoder_3/sequential_6/conv2d_7/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@29
7variational_autoencoder_3/sequential_6/conv2d_7/BiasAdd?
4variational_autoencoder_3/sequential_6/conv2d_7/ReluRelu@variational_autoencoder_3/sequential_6/conv2d_7/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@26
4variational_autoencoder_3/sequential_6/conv2d_7/Relu?
6variational_autoencoder_3/sequential_6/flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"???? 	  28
6variational_autoencoder_3/sequential_6/flatten_3/Const?
8variational_autoencoder_3/sequential_6/flatten_3/ReshapeReshapeBvariational_autoencoder_3/sequential_6/conv2d_7/Relu:activations:0?variational_autoencoder_3/sequential_6/flatten_3/Const:output:0*
T0*(
_output_shapes
:??????????2:
8variational_autoencoder_3/sequential_6/flatten_3/Reshape?
Dvariational_autoencoder_3/sequential_6/dense_6/MatMul/ReadVariableOpReadVariableOpMvariational_autoencoder_3_sequential_6_dense_6_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02F
Dvariational_autoencoder_3/sequential_6/dense_6/MatMul/ReadVariableOp?
5variational_autoencoder_3/sequential_6/dense_6/MatMulMatMulAvariational_autoencoder_3/sequential_6/flatten_3/Reshape:output:0Lvariational_autoencoder_3/sequential_6/dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????27
5variational_autoencoder_3/sequential_6/dense_6/MatMul?
Evariational_autoencoder_3/sequential_6/dense_6/BiasAdd/ReadVariableOpReadVariableOpNvariational_autoencoder_3_sequential_6_dense_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02G
Evariational_autoencoder_3/sequential_6/dense_6/BiasAdd/ReadVariableOp?
6variational_autoencoder_3/sequential_6/dense_6/BiasAddBiasAdd?variational_autoencoder_3/sequential_6/dense_6/MatMul:product:0Mvariational_autoencoder_3/sequential_6/dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????28
6variational_autoencoder_3/sequential_6/dense_6/BiasAdd?
7variational_autoencoder_3/sequential_6/reshape_13/ShapeShape?variational_autoencoder_3/sequential_6/dense_6/BiasAdd:output:0*
T0*
_output_shapes
:29
7variational_autoencoder_3/sequential_6/reshape_13/Shape?
Evariational_autoencoder_3/sequential_6/reshape_13/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2G
Evariational_autoencoder_3/sequential_6/reshape_13/strided_slice/stack?
Gvariational_autoencoder_3/sequential_6/reshape_13/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2I
Gvariational_autoencoder_3/sequential_6/reshape_13/strided_slice/stack_1?
Gvariational_autoencoder_3/sequential_6/reshape_13/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2I
Gvariational_autoencoder_3/sequential_6/reshape_13/strided_slice/stack_2?
?variational_autoencoder_3/sequential_6/reshape_13/strided_sliceStridedSlice@variational_autoencoder_3/sequential_6/reshape_13/Shape:output:0Nvariational_autoencoder_3/sequential_6/reshape_13/strided_slice/stack:output:0Pvariational_autoencoder_3/sequential_6/reshape_13/strided_slice/stack_1:output:0Pvariational_autoencoder_3/sequential_6/reshape_13/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2A
?variational_autoencoder_3/sequential_6/reshape_13/strided_slice?
Avariational_autoencoder_3/sequential_6/reshape_13/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2C
Avariational_autoencoder_3/sequential_6/reshape_13/Reshape/shape/1?
Avariational_autoencoder_3/sequential_6/reshape_13/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2C
Avariational_autoencoder_3/sequential_6/reshape_13/Reshape/shape/2?
?variational_autoencoder_3/sequential_6/reshape_13/Reshape/shapePackHvariational_autoencoder_3/sequential_6/reshape_13/strided_slice:output:0Jvariational_autoencoder_3/sequential_6/reshape_13/Reshape/shape/1:output:0Jvariational_autoencoder_3/sequential_6/reshape_13/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2A
?variational_autoencoder_3/sequential_6/reshape_13/Reshape/shape?
9variational_autoencoder_3/sequential_6/reshape_13/ReshapeReshape?variational_autoencoder_3/sequential_6/dense_6/BiasAdd:output:0Hvariational_autoencoder_3/sequential_6/reshape_13/Reshape/shape:output:0*
T0*+
_output_shapes
:?????????2;
9variational_autoencoder_3/sequential_6/reshape_13/Reshape?
9variational_autoencoder_3/normal_sampling_layer_3/unstackUnpackBvariational_autoencoder_3/sequential_6/reshape_13/Reshape:output:0*
T0*:
_output_shapes(
&:?????????:?????????*

axis*	
num2;
9variational_autoencoder_3/normal_sampling_layer_3/unstack?
Dvariational_autoencoder_3/sequential_7/dense_7/MatMul/ReadVariableOpReadVariableOpMvariational_autoencoder_3_sequential_7_dense_7_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02F
Dvariational_autoencoder_3/sequential_7/dense_7/MatMul/ReadVariableOp?
5variational_autoencoder_3/sequential_7/dense_7/MatMulMatMulBvariational_autoencoder_3/normal_sampling_layer_3/unstack:output:0Lvariational_autoencoder_3/sequential_7/dense_7/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????27
5variational_autoencoder_3/sequential_7/dense_7/MatMul?
Evariational_autoencoder_3/sequential_7/dense_7/BiasAdd/ReadVariableOpReadVariableOpNvariational_autoencoder_3_sequential_7_dense_7_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02G
Evariational_autoencoder_3/sequential_7/dense_7/BiasAdd/ReadVariableOp?
6variational_autoencoder_3/sequential_7/dense_7/BiasAddBiasAdd?variational_autoencoder_3/sequential_7/dense_7/MatMul:product:0Mvariational_autoencoder_3/sequential_7/dense_7/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????28
6variational_autoencoder_3/sequential_7/dense_7/BiasAdd?
3variational_autoencoder_3/sequential_7/dense_7/ReluRelu?variational_autoencoder_3/sequential_7/dense_7/BiasAdd:output:0*
T0*(
_output_shapes
:??????????25
3variational_autoencoder_3/sequential_7/dense_7/Relu?
7variational_autoencoder_3/sequential_7/reshape_14/ShapeShapeAvariational_autoencoder_3/sequential_7/dense_7/Relu:activations:0*
T0*
_output_shapes
:29
7variational_autoencoder_3/sequential_7/reshape_14/Shape?
Evariational_autoencoder_3/sequential_7/reshape_14/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2G
Evariational_autoencoder_3/sequential_7/reshape_14/strided_slice/stack?
Gvariational_autoencoder_3/sequential_7/reshape_14/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2I
Gvariational_autoencoder_3/sequential_7/reshape_14/strided_slice/stack_1?
Gvariational_autoencoder_3/sequential_7/reshape_14/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2I
Gvariational_autoencoder_3/sequential_7/reshape_14/strided_slice/stack_2?
?variational_autoencoder_3/sequential_7/reshape_14/strided_sliceStridedSlice@variational_autoencoder_3/sequential_7/reshape_14/Shape:output:0Nvariational_autoencoder_3/sequential_7/reshape_14/strided_slice/stack:output:0Pvariational_autoencoder_3/sequential_7/reshape_14/strided_slice/stack_1:output:0Pvariational_autoencoder_3/sequential_7/reshape_14/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2A
?variational_autoencoder_3/sequential_7/reshape_14/strided_slice?
Avariational_autoencoder_3/sequential_7/reshape_14/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2C
Avariational_autoencoder_3/sequential_7/reshape_14/Reshape/shape/1?
Avariational_autoencoder_3/sequential_7/reshape_14/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2C
Avariational_autoencoder_3/sequential_7/reshape_14/Reshape/shape/2?
Avariational_autoencoder_3/sequential_7/reshape_14/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B : 2C
Avariational_autoencoder_3/sequential_7/reshape_14/Reshape/shape/3?
?variational_autoencoder_3/sequential_7/reshape_14/Reshape/shapePackHvariational_autoencoder_3/sequential_7/reshape_14/strided_slice:output:0Jvariational_autoencoder_3/sequential_7/reshape_14/Reshape/shape/1:output:0Jvariational_autoencoder_3/sequential_7/reshape_14/Reshape/shape/2:output:0Jvariational_autoencoder_3/sequential_7/reshape_14/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2A
?variational_autoencoder_3/sequential_7/reshape_14/Reshape/shape?
9variational_autoencoder_3/sequential_7/reshape_14/ReshapeReshapeAvariational_autoencoder_3/sequential_7/dense_7/Relu:activations:0Hvariational_autoencoder_3/sequential_7/reshape_14/Reshape/shape:output:0*
T0*/
_output_shapes
:????????? 2;
9variational_autoencoder_3/sequential_7/reshape_14/Reshape?
?variational_autoencoder_3/sequential_7/conv2d_transpose_9/ShapeShapeBvariational_autoencoder_3/sequential_7/reshape_14/Reshape:output:0*
T0*
_output_shapes
:2A
?variational_autoencoder_3/sequential_7/conv2d_transpose_9/Shape?
Mvariational_autoencoder_3/sequential_7/conv2d_transpose_9/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2O
Mvariational_autoencoder_3/sequential_7/conv2d_transpose_9/strided_slice/stack?
Ovariational_autoencoder_3/sequential_7/conv2d_transpose_9/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2Q
Ovariational_autoencoder_3/sequential_7/conv2d_transpose_9/strided_slice/stack_1?
Ovariational_autoencoder_3/sequential_7/conv2d_transpose_9/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2Q
Ovariational_autoencoder_3/sequential_7/conv2d_transpose_9/strided_slice/stack_2?
Gvariational_autoencoder_3/sequential_7/conv2d_transpose_9/strided_sliceStridedSliceHvariational_autoencoder_3/sequential_7/conv2d_transpose_9/Shape:output:0Vvariational_autoencoder_3/sequential_7/conv2d_transpose_9/strided_slice/stack:output:0Xvariational_autoencoder_3/sequential_7/conv2d_transpose_9/strided_slice/stack_1:output:0Xvariational_autoencoder_3/sequential_7/conv2d_transpose_9/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2I
Gvariational_autoencoder_3/sequential_7/conv2d_transpose_9/strided_slice?
Avariational_autoencoder_3/sequential_7/conv2d_transpose_9/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2C
Avariational_autoencoder_3/sequential_7/conv2d_transpose_9/stack/1?
Avariational_autoencoder_3/sequential_7/conv2d_transpose_9/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2C
Avariational_autoencoder_3/sequential_7/conv2d_transpose_9/stack/2?
Avariational_autoencoder_3/sequential_7/conv2d_transpose_9/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@2C
Avariational_autoencoder_3/sequential_7/conv2d_transpose_9/stack/3?
?variational_autoencoder_3/sequential_7/conv2d_transpose_9/stackPackPvariational_autoencoder_3/sequential_7/conv2d_transpose_9/strided_slice:output:0Jvariational_autoencoder_3/sequential_7/conv2d_transpose_9/stack/1:output:0Jvariational_autoencoder_3/sequential_7/conv2d_transpose_9/stack/2:output:0Jvariational_autoencoder_3/sequential_7/conv2d_transpose_9/stack/3:output:0*
N*
T0*
_output_shapes
:2A
?variational_autoencoder_3/sequential_7/conv2d_transpose_9/stack?
Ovariational_autoencoder_3/sequential_7/conv2d_transpose_9/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2Q
Ovariational_autoencoder_3/sequential_7/conv2d_transpose_9/strided_slice_1/stack?
Qvariational_autoencoder_3/sequential_7/conv2d_transpose_9/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2S
Qvariational_autoencoder_3/sequential_7/conv2d_transpose_9/strided_slice_1/stack_1?
Qvariational_autoencoder_3/sequential_7/conv2d_transpose_9/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2S
Qvariational_autoencoder_3/sequential_7/conv2d_transpose_9/strided_slice_1/stack_2?
Ivariational_autoencoder_3/sequential_7/conv2d_transpose_9/strided_slice_1StridedSliceHvariational_autoencoder_3/sequential_7/conv2d_transpose_9/stack:output:0Xvariational_autoencoder_3/sequential_7/conv2d_transpose_9/strided_slice_1/stack:output:0Zvariational_autoencoder_3/sequential_7/conv2d_transpose_9/strided_slice_1/stack_1:output:0Zvariational_autoencoder_3/sequential_7/conv2d_transpose_9/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2K
Ivariational_autoencoder_3/sequential_7/conv2d_transpose_9/strided_slice_1?
Yvariational_autoencoder_3/sequential_7/conv2d_transpose_9/conv2d_transpose/ReadVariableOpReadVariableOpbvariational_autoencoder_3_sequential_7_conv2d_transpose_9_conv2d_transpose_readvariableop_resource*&
_output_shapes
:@ *
dtype02[
Yvariational_autoencoder_3/sequential_7/conv2d_transpose_9/conv2d_transpose/ReadVariableOp?
Jvariational_autoencoder_3/sequential_7/conv2d_transpose_9/conv2d_transposeConv2DBackpropInputHvariational_autoencoder_3/sequential_7/conv2d_transpose_9/stack:output:0avariational_autoencoder_3/sequential_7/conv2d_transpose_9/conv2d_transpose/ReadVariableOp:value:0Bvariational_autoencoder_3/sequential_7/reshape_14/Reshape:output:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2L
Jvariational_autoencoder_3/sequential_7/conv2d_transpose_9/conv2d_transpose?
Pvariational_autoencoder_3/sequential_7/conv2d_transpose_9/BiasAdd/ReadVariableOpReadVariableOpYvariational_autoencoder_3_sequential_7_conv2d_transpose_9_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02R
Pvariational_autoencoder_3/sequential_7/conv2d_transpose_9/BiasAdd/ReadVariableOp?
Avariational_autoencoder_3/sequential_7/conv2d_transpose_9/BiasAddBiasAddSvariational_autoencoder_3/sequential_7/conv2d_transpose_9/conv2d_transpose:output:0Xvariational_autoencoder_3/sequential_7/conv2d_transpose_9/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2C
Avariational_autoencoder_3/sequential_7/conv2d_transpose_9/BiasAdd?
>variational_autoencoder_3/sequential_7/conv2d_transpose_9/ReluReluJvariational_autoencoder_3/sequential_7/conv2d_transpose_9/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2@
>variational_autoencoder_3/sequential_7/conv2d_transpose_9/Relu?
@variational_autoencoder_3/sequential_7/conv2d_transpose_10/ShapeShapeLvariational_autoencoder_3/sequential_7/conv2d_transpose_9/Relu:activations:0*
T0*
_output_shapes
:2B
@variational_autoencoder_3/sequential_7/conv2d_transpose_10/Shape?
Nvariational_autoencoder_3/sequential_7/conv2d_transpose_10/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2P
Nvariational_autoencoder_3/sequential_7/conv2d_transpose_10/strided_slice/stack?
Pvariational_autoencoder_3/sequential_7/conv2d_transpose_10/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2R
Pvariational_autoencoder_3/sequential_7/conv2d_transpose_10/strided_slice/stack_1?
Pvariational_autoencoder_3/sequential_7/conv2d_transpose_10/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2R
Pvariational_autoencoder_3/sequential_7/conv2d_transpose_10/strided_slice/stack_2?
Hvariational_autoencoder_3/sequential_7/conv2d_transpose_10/strided_sliceStridedSliceIvariational_autoencoder_3/sequential_7/conv2d_transpose_10/Shape:output:0Wvariational_autoencoder_3/sequential_7/conv2d_transpose_10/strided_slice/stack:output:0Yvariational_autoencoder_3/sequential_7/conv2d_transpose_10/strided_slice/stack_1:output:0Yvariational_autoencoder_3/sequential_7/conv2d_transpose_10/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2J
Hvariational_autoencoder_3/sequential_7/conv2d_transpose_10/strided_slice?
Bvariational_autoencoder_3/sequential_7/conv2d_transpose_10/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2D
Bvariational_autoencoder_3/sequential_7/conv2d_transpose_10/stack/1?
Bvariational_autoencoder_3/sequential_7/conv2d_transpose_10/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2D
Bvariational_autoencoder_3/sequential_7/conv2d_transpose_10/stack/2?
Bvariational_autoencoder_3/sequential_7/conv2d_transpose_10/stack/3Const*
_output_shapes
: *
dtype0*
value	B : 2D
Bvariational_autoencoder_3/sequential_7/conv2d_transpose_10/stack/3?
@variational_autoencoder_3/sequential_7/conv2d_transpose_10/stackPackQvariational_autoencoder_3/sequential_7/conv2d_transpose_10/strided_slice:output:0Kvariational_autoencoder_3/sequential_7/conv2d_transpose_10/stack/1:output:0Kvariational_autoencoder_3/sequential_7/conv2d_transpose_10/stack/2:output:0Kvariational_autoencoder_3/sequential_7/conv2d_transpose_10/stack/3:output:0*
N*
T0*
_output_shapes
:2B
@variational_autoencoder_3/sequential_7/conv2d_transpose_10/stack?
Pvariational_autoencoder_3/sequential_7/conv2d_transpose_10/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2R
Pvariational_autoencoder_3/sequential_7/conv2d_transpose_10/strided_slice_1/stack?
Rvariational_autoencoder_3/sequential_7/conv2d_transpose_10/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2T
Rvariational_autoencoder_3/sequential_7/conv2d_transpose_10/strided_slice_1/stack_1?
Rvariational_autoencoder_3/sequential_7/conv2d_transpose_10/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2T
Rvariational_autoencoder_3/sequential_7/conv2d_transpose_10/strided_slice_1/stack_2?
Jvariational_autoencoder_3/sequential_7/conv2d_transpose_10/strided_slice_1StridedSliceIvariational_autoencoder_3/sequential_7/conv2d_transpose_10/stack:output:0Yvariational_autoencoder_3/sequential_7/conv2d_transpose_10/strided_slice_1/stack:output:0[variational_autoencoder_3/sequential_7/conv2d_transpose_10/strided_slice_1/stack_1:output:0[variational_autoencoder_3/sequential_7/conv2d_transpose_10/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2L
Jvariational_autoencoder_3/sequential_7/conv2d_transpose_10/strided_slice_1?
Zvariational_autoencoder_3/sequential_7/conv2d_transpose_10/conv2d_transpose/ReadVariableOpReadVariableOpcvariational_autoencoder_3_sequential_7_conv2d_transpose_10_conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype02\
Zvariational_autoencoder_3/sequential_7/conv2d_transpose_10/conv2d_transpose/ReadVariableOp?
Kvariational_autoencoder_3/sequential_7/conv2d_transpose_10/conv2d_transposeConv2DBackpropInputIvariational_autoencoder_3/sequential_7/conv2d_transpose_10/stack:output:0bvariational_autoencoder_3/sequential_7/conv2d_transpose_10/conv2d_transpose/ReadVariableOp:value:0Lvariational_autoencoder_3/sequential_7/conv2d_transpose_9/Relu:activations:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
2M
Kvariational_autoencoder_3/sequential_7/conv2d_transpose_10/conv2d_transpose?
Qvariational_autoencoder_3/sequential_7/conv2d_transpose_10/BiasAdd/ReadVariableOpReadVariableOpZvariational_autoencoder_3_sequential_7_conv2d_transpose_10_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02S
Qvariational_autoencoder_3/sequential_7/conv2d_transpose_10/BiasAdd/ReadVariableOp?
Bvariational_autoencoder_3/sequential_7/conv2d_transpose_10/BiasAddBiasAddTvariational_autoencoder_3/sequential_7/conv2d_transpose_10/conv2d_transpose:output:0Yvariational_autoencoder_3/sequential_7/conv2d_transpose_10/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2D
Bvariational_autoencoder_3/sequential_7/conv2d_transpose_10/BiasAdd?
?variational_autoencoder_3/sequential_7/conv2d_transpose_10/ReluReluKvariational_autoencoder_3/sequential_7/conv2d_transpose_10/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2A
?variational_autoencoder_3/sequential_7/conv2d_transpose_10/Relu?
@variational_autoencoder_3/sequential_7/conv2d_transpose_11/ShapeShapeMvariational_autoencoder_3/sequential_7/conv2d_transpose_10/Relu:activations:0*
T0*
_output_shapes
:2B
@variational_autoencoder_3/sequential_7/conv2d_transpose_11/Shape?
Nvariational_autoencoder_3/sequential_7/conv2d_transpose_11/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2P
Nvariational_autoencoder_3/sequential_7/conv2d_transpose_11/strided_slice/stack?
Pvariational_autoencoder_3/sequential_7/conv2d_transpose_11/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2R
Pvariational_autoencoder_3/sequential_7/conv2d_transpose_11/strided_slice/stack_1?
Pvariational_autoencoder_3/sequential_7/conv2d_transpose_11/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2R
Pvariational_autoencoder_3/sequential_7/conv2d_transpose_11/strided_slice/stack_2?
Hvariational_autoencoder_3/sequential_7/conv2d_transpose_11/strided_sliceStridedSliceIvariational_autoencoder_3/sequential_7/conv2d_transpose_11/Shape:output:0Wvariational_autoencoder_3/sequential_7/conv2d_transpose_11/strided_slice/stack:output:0Yvariational_autoencoder_3/sequential_7/conv2d_transpose_11/strided_slice/stack_1:output:0Yvariational_autoencoder_3/sequential_7/conv2d_transpose_11/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2J
Hvariational_autoencoder_3/sequential_7/conv2d_transpose_11/strided_slice?
Bvariational_autoencoder_3/sequential_7/conv2d_transpose_11/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2D
Bvariational_autoencoder_3/sequential_7/conv2d_transpose_11/stack/1?
Bvariational_autoencoder_3/sequential_7/conv2d_transpose_11/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2D
Bvariational_autoencoder_3/sequential_7/conv2d_transpose_11/stack/2?
Bvariational_autoencoder_3/sequential_7/conv2d_transpose_11/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2D
Bvariational_autoencoder_3/sequential_7/conv2d_transpose_11/stack/3?
@variational_autoencoder_3/sequential_7/conv2d_transpose_11/stackPackQvariational_autoencoder_3/sequential_7/conv2d_transpose_11/strided_slice:output:0Kvariational_autoencoder_3/sequential_7/conv2d_transpose_11/stack/1:output:0Kvariational_autoencoder_3/sequential_7/conv2d_transpose_11/stack/2:output:0Kvariational_autoencoder_3/sequential_7/conv2d_transpose_11/stack/3:output:0*
N*
T0*
_output_shapes
:2B
@variational_autoencoder_3/sequential_7/conv2d_transpose_11/stack?
Pvariational_autoencoder_3/sequential_7/conv2d_transpose_11/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2R
Pvariational_autoencoder_3/sequential_7/conv2d_transpose_11/strided_slice_1/stack?
Rvariational_autoencoder_3/sequential_7/conv2d_transpose_11/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2T
Rvariational_autoencoder_3/sequential_7/conv2d_transpose_11/strided_slice_1/stack_1?
Rvariational_autoencoder_3/sequential_7/conv2d_transpose_11/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2T
Rvariational_autoencoder_3/sequential_7/conv2d_transpose_11/strided_slice_1/stack_2?
Jvariational_autoencoder_3/sequential_7/conv2d_transpose_11/strided_slice_1StridedSliceIvariational_autoencoder_3/sequential_7/conv2d_transpose_11/stack:output:0Yvariational_autoencoder_3/sequential_7/conv2d_transpose_11/strided_slice_1/stack:output:0[variational_autoencoder_3/sequential_7/conv2d_transpose_11/strided_slice_1/stack_1:output:0[variational_autoencoder_3/sequential_7/conv2d_transpose_11/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2L
Jvariational_autoencoder_3/sequential_7/conv2d_transpose_11/strided_slice_1?
Zvariational_autoencoder_3/sequential_7/conv2d_transpose_11/conv2d_transpose/ReadVariableOpReadVariableOpcvariational_autoencoder_3_sequential_7_conv2d_transpose_11_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype02\
Zvariational_autoencoder_3/sequential_7/conv2d_transpose_11/conv2d_transpose/ReadVariableOp?
Kvariational_autoencoder_3/sequential_7/conv2d_transpose_11/conv2d_transposeConv2DBackpropInputIvariational_autoencoder_3/sequential_7/conv2d_transpose_11/stack:output:0bvariational_autoencoder_3/sequential_7/conv2d_transpose_11/conv2d_transpose/ReadVariableOp:value:0Mvariational_autoencoder_3/sequential_7/conv2d_transpose_10/Relu:activations:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2M
Kvariational_autoencoder_3/sequential_7/conv2d_transpose_11/conv2d_transpose?
Qvariational_autoencoder_3/sequential_7/conv2d_transpose_11/BiasAdd/ReadVariableOpReadVariableOpZvariational_autoencoder_3_sequential_7_conv2d_transpose_11_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02S
Qvariational_autoencoder_3/sequential_7/conv2d_transpose_11/BiasAdd/ReadVariableOp?
Bvariational_autoencoder_3/sequential_7/conv2d_transpose_11/BiasAddBiasAddTvariational_autoencoder_3/sequential_7/conv2d_transpose_11/conv2d_transpose:output:0Yvariational_autoencoder_3/sequential_7/conv2d_transpose_11/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2D
Bvariational_autoencoder_3/sequential_7/conv2d_transpose_11/BiasAdd?
Bvariational_autoencoder_3/sequential_7/conv2d_transpose_11/SigmoidSigmoidKvariational_autoencoder_3/sequential_7/conv2d_transpose_11/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2D
Bvariational_autoencoder_3/sequential_7/conv2d_transpose_11/Sigmoid?
7variational_autoencoder_3/sequential_7/reshape_15/ShapeShapeFvariational_autoencoder_3/sequential_7/conv2d_transpose_11/Sigmoid:y:0*
T0*
_output_shapes
:29
7variational_autoencoder_3/sequential_7/reshape_15/Shape?
Evariational_autoencoder_3/sequential_7/reshape_15/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2G
Evariational_autoencoder_3/sequential_7/reshape_15/strided_slice/stack?
Gvariational_autoencoder_3/sequential_7/reshape_15/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2I
Gvariational_autoencoder_3/sequential_7/reshape_15/strided_slice/stack_1?
Gvariational_autoencoder_3/sequential_7/reshape_15/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2I
Gvariational_autoencoder_3/sequential_7/reshape_15/strided_slice/stack_2?
?variational_autoencoder_3/sequential_7/reshape_15/strided_sliceStridedSlice@variational_autoencoder_3/sequential_7/reshape_15/Shape:output:0Nvariational_autoencoder_3/sequential_7/reshape_15/strided_slice/stack:output:0Pvariational_autoencoder_3/sequential_7/reshape_15/strided_slice/stack_1:output:0Pvariational_autoencoder_3/sequential_7/reshape_15/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2A
?variational_autoencoder_3/sequential_7/reshape_15/strided_slice?
Avariational_autoencoder_3/sequential_7/reshape_15/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2C
Avariational_autoencoder_3/sequential_7/reshape_15/Reshape/shape/1?
Avariational_autoencoder_3/sequential_7/reshape_15/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2C
Avariational_autoencoder_3/sequential_7/reshape_15/Reshape/shape/2?
Avariational_autoencoder_3/sequential_7/reshape_15/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2C
Avariational_autoencoder_3/sequential_7/reshape_15/Reshape/shape/3?
?variational_autoencoder_3/sequential_7/reshape_15/Reshape/shapePackHvariational_autoencoder_3/sequential_7/reshape_15/strided_slice:output:0Jvariational_autoencoder_3/sequential_7/reshape_15/Reshape/shape/1:output:0Jvariational_autoencoder_3/sequential_7/reshape_15/Reshape/shape/2:output:0Jvariational_autoencoder_3/sequential_7/reshape_15/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2A
?variational_autoencoder_3/sequential_7/reshape_15/Reshape/shape?
9variational_autoencoder_3/sequential_7/reshape_15/ReshapeReshapeFvariational_autoencoder_3/sequential_7/conv2d_transpose_11/Sigmoid:y:0Hvariational_autoencoder_3/sequential_7/reshape_15/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????2;
9variational_autoencoder_3/sequential_7/reshape_15/Reshape?
IdentityIdentityBvariational_autoencoder_3/sequential_7/reshape_15/Reshape:output:0^NoOp*
T0*/
_output_shapes
:?????????2

Identity?	
NoOpNoOpG^variational_autoencoder_3/sequential_6/conv2d_6/BiasAdd/ReadVariableOpF^variational_autoencoder_3/sequential_6/conv2d_6/Conv2D/ReadVariableOpG^variational_autoencoder_3/sequential_6/conv2d_7/BiasAdd/ReadVariableOpF^variational_autoencoder_3/sequential_6/conv2d_7/Conv2D/ReadVariableOpF^variational_autoencoder_3/sequential_6/dense_6/BiasAdd/ReadVariableOpE^variational_autoencoder_3/sequential_6/dense_6/MatMul/ReadVariableOpR^variational_autoencoder_3/sequential_7/conv2d_transpose_10/BiasAdd/ReadVariableOp[^variational_autoencoder_3/sequential_7/conv2d_transpose_10/conv2d_transpose/ReadVariableOpR^variational_autoencoder_3/sequential_7/conv2d_transpose_11/BiasAdd/ReadVariableOp[^variational_autoencoder_3/sequential_7/conv2d_transpose_11/conv2d_transpose/ReadVariableOpQ^variational_autoencoder_3/sequential_7/conv2d_transpose_9/BiasAdd/ReadVariableOpZ^variational_autoencoder_3/sequential_7/conv2d_transpose_9/conv2d_transpose/ReadVariableOpF^variational_autoencoder_3/sequential_7/dense_7/BiasAdd/ReadVariableOpE^variational_autoencoder_3/sequential_7/dense_7/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:?????????: : : : : : : : : : : : : : 2?
Fvariational_autoencoder_3/sequential_6/conv2d_6/BiasAdd/ReadVariableOpFvariational_autoencoder_3/sequential_6/conv2d_6/BiasAdd/ReadVariableOp2?
Evariational_autoencoder_3/sequential_6/conv2d_6/Conv2D/ReadVariableOpEvariational_autoencoder_3/sequential_6/conv2d_6/Conv2D/ReadVariableOp2?
Fvariational_autoencoder_3/sequential_6/conv2d_7/BiasAdd/ReadVariableOpFvariational_autoencoder_3/sequential_6/conv2d_7/BiasAdd/ReadVariableOp2?
Evariational_autoencoder_3/sequential_6/conv2d_7/Conv2D/ReadVariableOpEvariational_autoencoder_3/sequential_6/conv2d_7/Conv2D/ReadVariableOp2?
Evariational_autoencoder_3/sequential_6/dense_6/BiasAdd/ReadVariableOpEvariational_autoencoder_3/sequential_6/dense_6/BiasAdd/ReadVariableOp2?
Dvariational_autoencoder_3/sequential_6/dense_6/MatMul/ReadVariableOpDvariational_autoencoder_3/sequential_6/dense_6/MatMul/ReadVariableOp2?
Qvariational_autoencoder_3/sequential_7/conv2d_transpose_10/BiasAdd/ReadVariableOpQvariational_autoencoder_3/sequential_7/conv2d_transpose_10/BiasAdd/ReadVariableOp2?
Zvariational_autoencoder_3/sequential_7/conv2d_transpose_10/conv2d_transpose/ReadVariableOpZvariational_autoencoder_3/sequential_7/conv2d_transpose_10/conv2d_transpose/ReadVariableOp2?
Qvariational_autoencoder_3/sequential_7/conv2d_transpose_11/BiasAdd/ReadVariableOpQvariational_autoencoder_3/sequential_7/conv2d_transpose_11/BiasAdd/ReadVariableOp2?
Zvariational_autoencoder_3/sequential_7/conv2d_transpose_11/conv2d_transpose/ReadVariableOpZvariational_autoencoder_3/sequential_7/conv2d_transpose_11/conv2d_transpose/ReadVariableOp2?
Pvariational_autoencoder_3/sequential_7/conv2d_transpose_9/BiasAdd/ReadVariableOpPvariational_autoencoder_3/sequential_7/conv2d_transpose_9/BiasAdd/ReadVariableOp2?
Yvariational_autoencoder_3/sequential_7/conv2d_transpose_9/conv2d_transpose/ReadVariableOpYvariational_autoencoder_3/sequential_7/conv2d_transpose_9/conv2d_transpose/ReadVariableOp2?
Evariational_autoencoder_3/sequential_7/dense_7/BiasAdd/ReadVariableOpEvariational_autoencoder_3/sequential_7/dense_7/BiasAdd/ReadVariableOp2?
Dvariational_autoencoder_3/sequential_7/dense_7/MatMul/ReadVariableOpDvariational_autoencoder_3/sequential_7/dense_7/MatMul/ReadVariableOp:X T
/
_output_shapes
:?????????
!
_user_specified_name	input_1
?
?
9__inference_variational_autoencoder_3_layer_call_fn_62079
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
T__inference_variational_autoencoder_3_layer_call_and_return_conditional_losses_620152
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
?
E
)__inference_flatten_3_layer_call_fn_63006

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
D__inference_flatten_3_layer_call_and_return_conditional_losses_609992
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
,__inference_sequential_6_layer_call_fn_62652

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
G__inference_sequential_6_layer_call_and_return_conditional_losses_610332
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
?m
?
__inference__traced_save_63511
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
*savev2_conv2d_6_kernel_read_readvariableop,
(savev2_conv2d_6_bias_read_readvariableop.
*savev2_conv2d_7_kernel_read_readvariableop,
(savev2_conv2d_7_bias_read_readvariableop-
)savev2_dense_6_kernel_read_readvariableop+
'savev2_dense_6_bias_read_readvariableop-
)savev2_dense_7_kernel_read_readvariableop+
'savev2_dense_7_bias_read_readvariableop8
4savev2_conv2d_transpose_9_kernel_read_readvariableop6
2savev2_conv2d_transpose_9_bias_read_readvariableop9
5savev2_conv2d_transpose_10_kernel_read_readvariableop7
3savev2_conv2d_transpose_10_bias_read_readvariableop9
5savev2_conv2d_transpose_11_kernel_read_readvariableop7
3savev2_conv2d_transpose_11_bias_read_readvariableop5
1savev2_adam_conv2d_6_kernel_m_read_readvariableop3
/savev2_adam_conv2d_6_bias_m_read_readvariableop5
1savev2_adam_conv2d_7_kernel_m_read_readvariableop3
/savev2_adam_conv2d_7_bias_m_read_readvariableop4
0savev2_adam_dense_6_kernel_m_read_readvariableop2
.savev2_adam_dense_6_bias_m_read_readvariableop4
0savev2_adam_dense_7_kernel_m_read_readvariableop2
.savev2_adam_dense_7_bias_m_read_readvariableop?
;savev2_adam_conv2d_transpose_9_kernel_m_read_readvariableop=
9savev2_adam_conv2d_transpose_9_bias_m_read_readvariableop@
<savev2_adam_conv2d_transpose_10_kernel_m_read_readvariableop>
:savev2_adam_conv2d_transpose_10_bias_m_read_readvariableop@
<savev2_adam_conv2d_transpose_11_kernel_m_read_readvariableop>
:savev2_adam_conv2d_transpose_11_bias_m_read_readvariableop5
1savev2_adam_conv2d_6_kernel_v_read_readvariableop3
/savev2_adam_conv2d_6_bias_v_read_readvariableop5
1savev2_adam_conv2d_7_kernel_v_read_readvariableop3
/savev2_adam_conv2d_7_bias_v_read_readvariableop4
0savev2_adam_dense_6_kernel_v_read_readvariableop2
.savev2_adam_dense_6_bias_v_read_readvariableop4
0savev2_adam_dense_7_kernel_v_read_readvariableop2
.savev2_adam_dense_7_bias_v_read_readvariableop?
;savev2_adam_conv2d_transpose_9_kernel_v_read_readvariableop=
9savev2_adam_conv2d_transpose_9_bias_v_read_readvariableop@
<savev2_adam_conv2d_transpose_10_kernel_v_read_readvariableop>
:savev2_adam_conv2d_transpose_10_bias_v_read_readvariableop@
<savev2_adam_conv2d_transpose_11_kernel_v_read_readvariableop>
:savev2_adam_conv2d_transpose_11_bias_v_read_readvariableop
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
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0 savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop"savev2_total_2_read_readvariableop"savev2_count_2_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop*savev2_conv2d_6_kernel_read_readvariableop(savev2_conv2d_6_bias_read_readvariableop*savev2_conv2d_7_kernel_read_readvariableop(savev2_conv2d_7_bias_read_readvariableop)savev2_dense_6_kernel_read_readvariableop'savev2_dense_6_bias_read_readvariableop)savev2_dense_7_kernel_read_readvariableop'savev2_dense_7_bias_read_readvariableop4savev2_conv2d_transpose_9_kernel_read_readvariableop2savev2_conv2d_transpose_9_bias_read_readvariableop5savev2_conv2d_transpose_10_kernel_read_readvariableop3savev2_conv2d_transpose_10_bias_read_readvariableop5savev2_conv2d_transpose_11_kernel_read_readvariableop3savev2_conv2d_transpose_11_bias_read_readvariableop1savev2_adam_conv2d_6_kernel_m_read_readvariableop/savev2_adam_conv2d_6_bias_m_read_readvariableop1savev2_adam_conv2d_7_kernel_m_read_readvariableop/savev2_adam_conv2d_7_bias_m_read_readvariableop0savev2_adam_dense_6_kernel_m_read_readvariableop.savev2_adam_dense_6_bias_m_read_readvariableop0savev2_adam_dense_7_kernel_m_read_readvariableop.savev2_adam_dense_7_bias_m_read_readvariableop;savev2_adam_conv2d_transpose_9_kernel_m_read_readvariableop9savev2_adam_conv2d_transpose_9_bias_m_read_readvariableop<savev2_adam_conv2d_transpose_10_kernel_m_read_readvariableop:savev2_adam_conv2d_transpose_10_bias_m_read_readvariableop<savev2_adam_conv2d_transpose_11_kernel_m_read_readvariableop:savev2_adam_conv2d_transpose_11_bias_m_read_readvariableop1savev2_adam_conv2d_6_kernel_v_read_readvariableop/savev2_adam_conv2d_6_bias_v_read_readvariableop1savev2_adam_conv2d_7_kernel_v_read_readvariableop/savev2_adam_conv2d_7_bias_v_read_readvariableop0savev2_adam_dense_6_kernel_v_read_readvariableop.savev2_adam_dense_6_bias_v_read_readvariableop0savev2_adam_dense_7_kernel_v_read_readvariableop.savev2_adam_dense_7_bias_v_read_readvariableop;savev2_adam_conv2d_transpose_9_kernel_v_read_readvariableop9savev2_adam_conv2d_transpose_9_bias_v_read_readvariableop<savev2_adam_conv2d_transpose_10_kernel_v_read_readvariableop:savev2_adam_conv2d_transpose_10_bias_v_read_readvariableop<savev2_adam_conv2d_transpose_11_kernel_v_read_readvariableop:savev2_adam_conv2d_transpose_11_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
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
?
?
M__inference_conv2d_transpose_9_layer_call_and_return_conditional_losses_61540

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
N__inference_conv2d_transpose_10_layer_call_and_return_conditional_losses_61569

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
?
?
(__inference_conv2d_6_layer_call_fn_62975

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
C__inference_conv2d_6_layer_call_and_return_conditional_losses_609702
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
(__inference_conv2d_7_layer_call_fn_62995

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
C__inference_conv2d_7_layer_call_and_return_conditional_losses_609872
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
?
C__inference_conv2d_7_layer_call_and_return_conditional_losses_62986

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
a
E__inference_reshape_14_layer_call_and_return_conditional_losses_63077

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
?:
?
G__inference_sequential_6_layer_call_and_return_conditional_losses_62590

inputsA
'conv2d_6_conv2d_readvariableop_resource: 6
(conv2d_6_biasadd_readvariableop_resource: A
'conv2d_7_conv2d_readvariableop_resource: @6
(conv2d_7_biasadd_readvariableop_resource:@9
&dense_6_matmul_readvariableop_resource:	?5
'dense_6_biasadd_readvariableop_resource:
identity??conv2d_6/BiasAdd/ReadVariableOp?conv2d_6/Conv2D/ReadVariableOp?conv2d_7/BiasAdd/ReadVariableOp?conv2d_7/Conv2D/ReadVariableOp?dense_6/BiasAdd/ReadVariableOp?dense_6/MatMul/ReadVariableOpZ
reshape_12/ShapeShapeinputs*
T0*
_output_shapes
:2
reshape_12/Shape?
reshape_12/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
reshape_12/strided_slice/stack?
 reshape_12/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_12/strided_slice/stack_1?
 reshape_12/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_12/strided_slice/stack_2?
reshape_12/strided_sliceStridedSlicereshape_12/Shape:output:0'reshape_12/strided_slice/stack:output:0)reshape_12/strided_slice/stack_1:output:0)reshape_12/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_12/strided_slicez
reshape_12/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_12/Reshape/shape/1z
reshape_12/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_12/Reshape/shape/2z
reshape_12/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_12/Reshape/shape/3?
reshape_12/Reshape/shapePack!reshape_12/strided_slice:output:0#reshape_12/Reshape/shape/1:output:0#reshape_12/Reshape/shape/2:output:0#reshape_12/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
reshape_12/Reshape/shape?
reshape_12/ReshapeReshapeinputs!reshape_12/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????2
reshape_12/Reshape?
conv2d_6/Conv2D/ReadVariableOpReadVariableOp'conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02 
conv2d_6/Conv2D/ReadVariableOp?
conv2d_6/Conv2DConv2Dreshape_12/Reshape:output:0&conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
2
conv2d_6/Conv2D?
conv2d_6/BiasAdd/ReadVariableOpReadVariableOp(conv2d_6_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv2d_6/BiasAdd/ReadVariableOp?
conv2d_6/BiasAddBiasAddconv2d_6/Conv2D:output:0'conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2
conv2d_6/BiasAdd{
conv2d_6/ReluReluconv2d_6/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
conv2d_6/Relu?
conv2d_7/Conv2D/ReadVariableOpReadVariableOp'conv2d_7_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02 
conv2d_7/Conv2D/ReadVariableOp?
conv2d_7/Conv2DConv2Dconv2d_6/Relu:activations:0&conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
2
conv2d_7/Conv2D?
conv2d_7/BiasAdd/ReadVariableOpReadVariableOp(conv2d_7_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_7/BiasAdd/ReadVariableOp?
conv2d_7/BiasAddBiasAddconv2d_7/Conv2D:output:0'conv2d_7/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
conv2d_7/BiasAdd{
conv2d_7/ReluReluconv2d_7/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
conv2d_7/Relus
flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"???? 	  2
flatten_3/Const?
flatten_3/ReshapeReshapeconv2d_7/Relu:activations:0flatten_3/Const:output:0*
T0*(
_output_shapes
:??????????2
flatten_3/Reshape?
dense_6/MatMul/ReadVariableOpReadVariableOp&dense_6_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
dense_6/MatMul/ReadVariableOp?
dense_6/MatMulMatMulflatten_3/Reshape:output:0%dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_6/MatMul?
dense_6/BiasAdd/ReadVariableOpReadVariableOp'dense_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_6/BiasAdd/ReadVariableOp?
dense_6/BiasAddBiasAdddense_6/MatMul:product:0&dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_6/BiasAddl
reshape_13/ShapeShapedense_6/BiasAdd:output:0*
T0*
_output_shapes
:2
reshape_13/Shape?
reshape_13/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
reshape_13/strided_slice/stack?
 reshape_13/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_13/strided_slice/stack_1?
 reshape_13/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_13/strided_slice/stack_2?
reshape_13/strided_sliceStridedSlicereshape_13/Shape:output:0'reshape_13/strided_slice/stack:output:0)reshape_13/strided_slice/stack_1:output:0)reshape_13/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_13/strided_slicez
reshape_13/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_13/Reshape/shape/1z
reshape_13/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_13/Reshape/shape/2?
reshape_13/Reshape/shapePack!reshape_13/strided_slice:output:0#reshape_13/Reshape/shape/1:output:0#reshape_13/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
reshape_13/Reshape/shape?
reshape_13/ReshapeReshapedense_6/BiasAdd:output:0!reshape_13/Reshape/shape:output:0*
T0*+
_output_shapes
:?????????2
reshape_13/Reshapez
IdentityIdentityreshape_13/Reshape:output:0^NoOp*
T0*+
_output_shapes
:?????????2

Identity?
NoOpNoOp ^conv2d_6/BiasAdd/ReadVariableOp^conv2d_6/Conv2D/ReadVariableOp ^conv2d_7/BiasAdd/ReadVariableOp^conv2d_7/Conv2D/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : 2B
conv2d_6/BiasAdd/ReadVariableOpconv2d_6/BiasAdd/ReadVariableOp2@
conv2d_6/Conv2D/ReadVariableOpconv2d_6/Conv2D/ReadVariableOp2B
conv2d_7/BiasAdd/ReadVariableOpconv2d_7/BiasAdd/ReadVariableOp2@
conv2d_7/Conv2D/ReadVariableOpconv2d_7/Conv2D/ReadVariableOp2@
dense_6/BiasAdd/ReadVariableOpdense_6/BiasAdd/ReadVariableOp2>
dense_6/MatMul/ReadVariableOpdense_6/MatMul/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
9__inference_variational_autoencoder_3_layer_call_fn_61907
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
T__inference_variational_autoencoder_3_layer_call_and_return_conditional_losses_618762
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
?
?
M__inference_conv2d_transpose_9_layer_call_and_return_conditional_losses_63140

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
?
?
3__inference_conv2d_transpose_11_layer_call_fn_63301

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
N__inference_conv2d_transpose_11_layer_call_and_return_conditional_losses_614272
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
?
?
C__inference_conv2d_6_layer_call_and_return_conditional_losses_60970

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
G__inference_sequential_7_layer_call_and_return_conditional_losses_61738

inputs 
dense_7_61715:	?
dense_7_61717:	?2
conv2d_transpose_9_61721:@ &
conv2d_transpose_9_61723:@3
conv2d_transpose_10_61726: @'
conv2d_transpose_10_61728: 3
conv2d_transpose_11_61731: '
conv2d_transpose_11_61733:
identity??+conv2d_transpose_10/StatefulPartitionedCall?+conv2d_transpose_11/StatefulPartitionedCall?*conv2d_transpose_9/StatefulPartitionedCall?dense_7/StatefulPartitionedCall?
dense_7/StatefulPartitionedCallStatefulPartitionedCallinputsdense_7_61715dense_7_61717*
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
B__inference_dense_7_layer_call_and_return_conditional_losses_614952!
dense_7/StatefulPartitionedCall?
reshape_14/PartitionedCallPartitionedCall(dense_7/StatefulPartitionedCall:output:0*
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
E__inference_reshape_14_layer_call_and_return_conditional_losses_615152
reshape_14/PartitionedCall?
*conv2d_transpose_9/StatefulPartitionedCallStatefulPartitionedCall#reshape_14/PartitionedCall:output:0conv2d_transpose_9_61721conv2d_transpose_9_61723*
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
M__inference_conv2d_transpose_9_layer_call_and_return_conditional_losses_615402,
*conv2d_transpose_9/StatefulPartitionedCall?
+conv2d_transpose_10/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_9/StatefulPartitionedCall:output:0conv2d_transpose_10_61726conv2d_transpose_10_61728*
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
N__inference_conv2d_transpose_10_layer_call_and_return_conditional_losses_615692-
+conv2d_transpose_10/StatefulPartitionedCall?
+conv2d_transpose_11/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_10/StatefulPartitionedCall:output:0conv2d_transpose_11_61731conv2d_transpose_11_61733*
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
N__inference_conv2d_transpose_11_layer_call_and_return_conditional_losses_615982-
+conv2d_transpose_11/StatefulPartitionedCall?
reshape_15/PartitionedCallPartitionedCall4conv2d_transpose_11/StatefulPartitionedCall:output:0*
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
E__inference_reshape_15_layer_call_and_return_conditional_losses_616182
reshape_15/PartitionedCall?
IdentityIdentity#reshape_15/PartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????2

Identity?
NoOpNoOp,^conv2d_transpose_10/StatefulPartitionedCall,^conv2d_transpose_11/StatefulPartitionedCall+^conv2d_transpose_9/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : : : : : 2Z
+conv2d_transpose_10/StatefulPartitionedCall+conv2d_transpose_10/StatefulPartitionedCall2Z
+conv2d_transpose_11/StatefulPartitionedCall+conv2d_transpose_11/StatefulPartitionedCall2X
*conv2d_transpose_9/StatefulPartitionedCall*conv2d_transpose_9/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
p
7__inference_normal_sampling_layer_3_layer_call_fn_62936

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
R__inference_normal_sampling_layer_3_layer_call_and_return_conditional_losses_619422
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
?
G__inference_sequential_6_layer_call_and_return_conditional_losses_61137

inputs(
conv2d_6_61119: 
conv2d_6_61121: (
conv2d_7_61124: @
conv2d_7_61126:@ 
dense_6_61130:	?
dense_6_61132:
identity?? conv2d_6/StatefulPartitionedCall? conv2d_7/StatefulPartitionedCall?dense_6/StatefulPartitionedCall?
reshape_12/PartitionedCallPartitionedCallinputs*
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
E__inference_reshape_12_layer_call_and_return_conditional_losses_609572
reshape_12/PartitionedCall?
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCall#reshape_12/PartitionedCall:output:0conv2d_6_61119conv2d_6_61121*
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
C__inference_conv2d_6_layer_call_and_return_conditional_losses_609702"
 conv2d_6/StatefulPartitionedCall?
 conv2d_7/StatefulPartitionedCallStatefulPartitionedCall)conv2d_6/StatefulPartitionedCall:output:0conv2d_7_61124conv2d_7_61126*
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
C__inference_conv2d_7_layer_call_and_return_conditional_losses_609872"
 conv2d_7/StatefulPartitionedCall?
flatten_3/PartitionedCallPartitionedCall)conv2d_7/StatefulPartitionedCall:output:0*
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
D__inference_flatten_3_layer_call_and_return_conditional_losses_609992
flatten_3/PartitionedCall?
dense_6/StatefulPartitionedCallStatefulPartitionedCall"flatten_3/PartitionedCall:output:0dense_6_61130dense_6_61132*
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
B__inference_dense_6_layer_call_and_return_conditional_losses_610112!
dense_6/StatefulPartitionedCall?
reshape_13/PartitionedCallPartitionedCall(dense_6/StatefulPartitionedCall:output:0*
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
E__inference_reshape_13_layer_call_and_return_conditional_losses_610302
reshape_13/PartitionedCall?
IdentityIdentity#reshape_13/PartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????2

Identity?
NoOpNoOp!^conv2d_6/StatefulPartitionedCall!^conv2d_7/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : 2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall2D
 conv2d_7/StatefulPartitionedCall conv2d_7/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
T__inference_variational_autoencoder_3_layer_call_and_return_conditional_losses_62149
input_1,
sequential_6_62117:  
sequential_6_62119: ,
sequential_6_62121: @ 
sequential_6_62123:@%
sequential_6_62125:	? 
sequential_6_62127:%
sequential_7_62131:	?!
sequential_7_62133:	?,
sequential_7_62135:@  
sequential_7_62137:@,
sequential_7_62139: @ 
sequential_7_62141: ,
sequential_7_62143:  
sequential_7_62145:
identity??/normal_sampling_layer_3/StatefulPartitionedCall?$sequential_6/StatefulPartitionedCall?$sequential_7/StatefulPartitionedCall?
$sequential_6/StatefulPartitionedCallStatefulPartitionedCallinput_1sequential_6_62117sequential_6_62119sequential_6_62121sequential_6_62123sequential_6_62125sequential_6_62127*
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
G__inference_sequential_6_layer_call_and_return_conditional_losses_611372&
$sequential_6/StatefulPartitionedCall?
/normal_sampling_layer_3/StatefulPartitionedCallStatefulPartitionedCall-sequential_6/StatefulPartitionedCall:output:0*
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
R__inference_normal_sampling_layer_3_layer_call_and_return_conditional_losses_6194221
/normal_sampling_layer_3/StatefulPartitionedCall?
$sequential_7/StatefulPartitionedCallStatefulPartitionedCall8normal_sampling_layer_3/StatefulPartitionedCall:output:0sequential_7_62131sequential_7_62133sequential_7_62135sequential_7_62137sequential_7_62139sequential_7_62141sequential_7_62143sequential_7_62145*
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
G__inference_sequential_7_layer_call_and_return_conditional_losses_617382&
$sequential_7/StatefulPartitionedCall?
IdentityIdentity-sequential_7/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????2

Identity?
NoOpNoOp0^normal_sampling_layer_3/StatefulPartitionedCall%^sequential_6/StatefulPartitionedCall%^sequential_7/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:?????????: : : : : : : : : : : : : : 2b
/normal_sampling_layer_3/StatefulPartitionedCall/normal_sampling_layer_3/StatefulPartitionedCall2L
$sequential_6/StatefulPartitionedCall$sequential_6/StatefulPartitionedCall2L
$sequential_7/StatefulPartitionedCall$sequential_7/StatefulPartitionedCall:X T
/
_output_shapes
:?????????
!
_user_specified_name	input_1
?
`
D__inference_flatten_3_layer_call_and_return_conditional_losses_63001

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
?
C__inference_conv2d_7_layer_call_and_return_conditional_losses_60987

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
?
F
*__inference_reshape_12_layer_call_fn_62955

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
E__inference_reshape_12_layer_call_and_return_conditional_losses_609572
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
?
?
T__inference_variational_autoencoder_3_layer_call_and_return_conditional_losses_62015

inputs,
sequential_6_61983:  
sequential_6_61985: ,
sequential_6_61987: @ 
sequential_6_61989:@%
sequential_6_61991:	? 
sequential_6_61993:%
sequential_7_61997:	?!
sequential_7_61999:	?,
sequential_7_62001:@  
sequential_7_62003:@,
sequential_7_62005: @ 
sequential_7_62007: ,
sequential_7_62009:  
sequential_7_62011:
identity??/normal_sampling_layer_3/StatefulPartitionedCall?$sequential_6/StatefulPartitionedCall?$sequential_7/StatefulPartitionedCall?
$sequential_6/StatefulPartitionedCallStatefulPartitionedCallinputssequential_6_61983sequential_6_61985sequential_6_61987sequential_6_61989sequential_6_61991sequential_6_61993*
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
G__inference_sequential_6_layer_call_and_return_conditional_losses_611372&
$sequential_6/StatefulPartitionedCall?
/normal_sampling_layer_3/StatefulPartitionedCallStatefulPartitionedCall-sequential_6/StatefulPartitionedCall:output:0*
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
R__inference_normal_sampling_layer_3_layer_call_and_return_conditional_losses_6194221
/normal_sampling_layer_3/StatefulPartitionedCall?
$sequential_7/StatefulPartitionedCallStatefulPartitionedCall8normal_sampling_layer_3/StatefulPartitionedCall:output:0sequential_7_61997sequential_7_61999sequential_7_62001sequential_7_62003sequential_7_62005sequential_7_62007sequential_7_62009sequential_7_62011*
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
G__inference_sequential_7_layer_call_and_return_conditional_losses_617382&
$sequential_7/StatefulPartitionedCall?
IdentityIdentity-sequential_7/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????2

Identity?
NoOpNoOp0^normal_sampling_layer_3/StatefulPartitionedCall%^sequential_6/StatefulPartitionedCall%^sequential_7/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:?????????: : : : : : : : : : : : : : 2b
/normal_sampling_layer_3/StatefulPartitionedCall/normal_sampling_layer_3/StatefulPartitionedCall2L
$sequential_6/StatefulPartitionedCall$sequential_6/StatefulPartitionedCall2L
$sequential_7/StatefulPartitionedCall$sequential_7/StatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
G__inference_sequential_6_layer_call_and_return_conditional_losses_61213
input_13(
conv2d_6_61195: 
conv2d_6_61197: (
conv2d_7_61200: @
conv2d_7_61202:@ 
dense_6_61206:	?
dense_6_61208:
identity?? conv2d_6/StatefulPartitionedCall? conv2d_7/StatefulPartitionedCall?dense_6/StatefulPartitionedCall?
reshape_12/PartitionedCallPartitionedCallinput_13*
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
E__inference_reshape_12_layer_call_and_return_conditional_losses_609572
reshape_12/PartitionedCall?
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCall#reshape_12/PartitionedCall:output:0conv2d_6_61195conv2d_6_61197*
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
C__inference_conv2d_6_layer_call_and_return_conditional_losses_609702"
 conv2d_6/StatefulPartitionedCall?
 conv2d_7/StatefulPartitionedCallStatefulPartitionedCall)conv2d_6/StatefulPartitionedCall:output:0conv2d_7_61200conv2d_7_61202*
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
C__inference_conv2d_7_layer_call_and_return_conditional_losses_609872"
 conv2d_7/StatefulPartitionedCall?
flatten_3/PartitionedCallPartitionedCall)conv2d_7/StatefulPartitionedCall:output:0*
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
D__inference_flatten_3_layer_call_and_return_conditional_losses_609992
flatten_3/PartitionedCall?
dense_6/StatefulPartitionedCallStatefulPartitionedCall"flatten_3/PartitionedCall:output:0dense_6_61206dense_6_61208*
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
B__inference_dense_6_layer_call_and_return_conditional_losses_610112!
dense_6/StatefulPartitionedCall?
reshape_13/PartitionedCallPartitionedCall(dense_6/StatefulPartitionedCall:output:0*
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
E__inference_reshape_13_layer_call_and_return_conditional_losses_610302
reshape_13/PartitionedCall?
IdentityIdentity#reshape_13/PartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????2

Identity?
NoOpNoOp!^conv2d_6/StatefulPartitionedCall!^conv2d_7/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : 2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall2D
 conv2d_7/StatefulPartitionedCall conv2d_7/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall:Y U
/
_output_shapes
:?????????
"
_user_specified_name
input_13
?
n
R__inference_normal_sampling_layer_3_layer_call_and_return_conditional_losses_61856

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
a
E__inference_reshape_15_layer_call_and_return_conditional_losses_61618

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
?
G__inference_sequential_6_layer_call_and_return_conditional_losses_61033

inputs(
conv2d_6_60971: 
conv2d_6_60973: (
conv2d_7_60988: @
conv2d_7_60990:@ 
dense_6_61012:	?
dense_6_61014:
identity?? conv2d_6/StatefulPartitionedCall? conv2d_7/StatefulPartitionedCall?dense_6/StatefulPartitionedCall?
reshape_12/PartitionedCallPartitionedCallinputs*
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
E__inference_reshape_12_layer_call_and_return_conditional_losses_609572
reshape_12/PartitionedCall?
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCall#reshape_12/PartitionedCall:output:0conv2d_6_60971conv2d_6_60973*
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
C__inference_conv2d_6_layer_call_and_return_conditional_losses_609702"
 conv2d_6/StatefulPartitionedCall?
 conv2d_7/StatefulPartitionedCallStatefulPartitionedCall)conv2d_6/StatefulPartitionedCall:output:0conv2d_7_60988conv2d_7_60990*
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
C__inference_conv2d_7_layer_call_and_return_conditional_losses_609872"
 conv2d_7/StatefulPartitionedCall?
flatten_3/PartitionedCallPartitionedCall)conv2d_7/StatefulPartitionedCall:output:0*
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
D__inference_flatten_3_layer_call_and_return_conditional_losses_609992
flatten_3/PartitionedCall?
dense_6/StatefulPartitionedCallStatefulPartitionedCall"flatten_3/PartitionedCall:output:0dense_6_61012dense_6_61014*
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
B__inference_dense_6_layer_call_and_return_conditional_losses_610112!
dense_6/StatefulPartitionedCall?
reshape_13/PartitionedCallPartitionedCall(dense_6/StatefulPartitionedCall:output:0*
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
E__inference_reshape_13_layer_call_and_return_conditional_losses_610302
reshape_13/PartitionedCall?
IdentityIdentity#reshape_13/PartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????2

Identity?
NoOpNoOp!^conv2d_6/StatefulPartitionedCall!^conv2d_7/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : 2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall2D
 conv2d_7/StatefulPartitionedCall conv2d_7/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
,__inference_sequential_7_layer_call_fn_62893

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
G__inference_sequential_7_layer_call_and_return_conditional_losses_617382
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

?
,__inference_sequential_7_layer_call_fn_62872

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
G__inference_sequential_7_layer_call_and_return_conditional_losses_616212
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
?
,__inference_sequential_6_layer_call_fn_62669

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
G__inference_sequential_6_layer_call_and_return_conditional_losses_611372
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
?&
?
N__inference_conv2d_transpose_10_layer_call_and_return_conditional_losses_61339

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
?%
?
M__inference_conv2d_transpose_9_layer_call_and_return_conditional_losses_61251

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
?
?
T__inference_variational_autoencoder_3_layer_call_and_return_conditional_losses_61876

inputs,
sequential_6_61837:  
sequential_6_61839: ,
sequential_6_61841: @ 
sequential_6_61843:@%
sequential_6_61845:	? 
sequential_6_61847:%
sequential_7_61858:	?!
sequential_7_61860:	?,
sequential_7_61862:@  
sequential_7_61864:@,
sequential_7_61866: @ 
sequential_7_61868: ,
sequential_7_61870:  
sequential_7_61872:
identity??$sequential_6/StatefulPartitionedCall?$sequential_7/StatefulPartitionedCall?
$sequential_6/StatefulPartitionedCallStatefulPartitionedCallinputssequential_6_61837sequential_6_61839sequential_6_61841sequential_6_61843sequential_6_61845sequential_6_61847*
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
G__inference_sequential_6_layer_call_and_return_conditional_losses_610332&
$sequential_6/StatefulPartitionedCall?
'normal_sampling_layer_3/PartitionedCallPartitionedCall-sequential_6/StatefulPartitionedCall:output:0*
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
R__inference_normal_sampling_layer_3_layer_call_and_return_conditional_losses_618562)
'normal_sampling_layer_3/PartitionedCall?
$sequential_7/StatefulPartitionedCallStatefulPartitionedCall0normal_sampling_layer_3/PartitionedCall:output:0sequential_7_61858sequential_7_61860sequential_7_61862sequential_7_61864sequential_7_61866sequential_7_61868sequential_7_61870sequential_7_61872*
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
G__inference_sequential_7_layer_call_and_return_conditional_losses_616212&
$sequential_7/StatefulPartitionedCall?
IdentityIdentity-sequential_7/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????2

Identity?
NoOpNoOp%^sequential_6/StatefulPartitionedCall%^sequential_7/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:?????????: : : : : : : : : : : : : : 2L
$sequential_6/StatefulPartitionedCall$sequential_6/StatefulPartitionedCall2L
$sequential_7/StatefulPartitionedCall$sequential_7/StatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
2__inference_conv2d_transpose_9_layer_call_fn_63149

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
M__inference_conv2d_transpose_9_layer_call_and_return_conditional_losses_612512
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
?
B__inference_dense_7_layer_call_and_return_conditional_losses_61495

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
E__inference_reshape_15_layer_call_and_return_conditional_losses_63324

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
?&
?
N__inference_conv2d_transpose_11_layer_call_and_return_conditional_losses_61427

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

?
B__inference_dense_6_layer_call_and_return_conditional_losses_61011

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
?
?
'__inference_dense_6_layer_call_fn_63025

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
B__inference_dense_6_layer_call_and_return_conditional_losses_610112
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
?
q
R__inference_normal_sampling_layer_3_layer_call_and_return_conditional_losses_62926

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
seed2?ҹ2$
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
a
E__inference_reshape_12_layer_call_and_return_conditional_losses_60957

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
a
E__inference_reshape_12_layer_call_and_return_conditional_losses_62950

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
M__inference_conv2d_transpose_9_layer_call_and_return_conditional_losses_63116

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
?
?
T__inference_variational_autoencoder_3_layer_call_and_return_conditional_losses_62114
input_1,
sequential_6_62082:  
sequential_6_62084: ,
sequential_6_62086: @ 
sequential_6_62088:@%
sequential_6_62090:	? 
sequential_6_62092:%
sequential_7_62096:	?!
sequential_7_62098:	?,
sequential_7_62100:@  
sequential_7_62102:@,
sequential_7_62104: @ 
sequential_7_62106: ,
sequential_7_62108:  
sequential_7_62110:
identity??$sequential_6/StatefulPartitionedCall?$sequential_7/StatefulPartitionedCall?
$sequential_6/StatefulPartitionedCallStatefulPartitionedCallinput_1sequential_6_62082sequential_6_62084sequential_6_62086sequential_6_62088sequential_6_62090sequential_6_62092*
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
G__inference_sequential_6_layer_call_and_return_conditional_losses_610332&
$sequential_6/StatefulPartitionedCall?
'normal_sampling_layer_3/PartitionedCallPartitionedCall-sequential_6/StatefulPartitionedCall:output:0*
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
R__inference_normal_sampling_layer_3_layer_call_and_return_conditional_losses_618562)
'normal_sampling_layer_3/PartitionedCall?
$sequential_7/StatefulPartitionedCallStatefulPartitionedCall0normal_sampling_layer_3/PartitionedCall:output:0sequential_7_62096sequential_7_62098sequential_7_62100sequential_7_62102sequential_7_62104sequential_7_62106sequential_7_62108sequential_7_62110*
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
G__inference_sequential_7_layer_call_and_return_conditional_losses_616212&
$sequential_7/StatefulPartitionedCall?
IdentityIdentity-sequential_7/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????2

Identity?
NoOpNoOp%^sequential_6/StatefulPartitionedCall%^sequential_7/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:?????????: : : : : : : : : : : : : : 2L
$sequential_6/StatefulPartitionedCall$sequential_6/StatefulPartitionedCall2L
$sequential_7/StatefulPartitionedCall$sequential_7/StatefulPartitionedCall:X T
/
_output_shapes
:?????????
!
_user_specified_name	input_1
?
q
R__inference_normal_sampling_layer_3_layer_call_and_return_conditional_losses_61942

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
seed2???2$
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
?!
?
G__inference_sequential_7_layer_call_and_return_conditional_losses_61830
input_14 
dense_7_61807:	?
dense_7_61809:	?2
conv2d_transpose_9_61813:@ &
conv2d_transpose_9_61815:@3
conv2d_transpose_10_61818: @'
conv2d_transpose_10_61820: 3
conv2d_transpose_11_61823: '
conv2d_transpose_11_61825:
identity??+conv2d_transpose_10/StatefulPartitionedCall?+conv2d_transpose_11/StatefulPartitionedCall?*conv2d_transpose_9/StatefulPartitionedCall?dense_7/StatefulPartitionedCall?
dense_7/StatefulPartitionedCallStatefulPartitionedCallinput_14dense_7_61807dense_7_61809*
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
B__inference_dense_7_layer_call_and_return_conditional_losses_614952!
dense_7/StatefulPartitionedCall?
reshape_14/PartitionedCallPartitionedCall(dense_7/StatefulPartitionedCall:output:0*
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
E__inference_reshape_14_layer_call_and_return_conditional_losses_615152
reshape_14/PartitionedCall?
*conv2d_transpose_9/StatefulPartitionedCallStatefulPartitionedCall#reshape_14/PartitionedCall:output:0conv2d_transpose_9_61813conv2d_transpose_9_61815*
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
M__inference_conv2d_transpose_9_layer_call_and_return_conditional_losses_615402,
*conv2d_transpose_9/StatefulPartitionedCall?
+conv2d_transpose_10/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_9/StatefulPartitionedCall:output:0conv2d_transpose_10_61818conv2d_transpose_10_61820*
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
N__inference_conv2d_transpose_10_layer_call_and_return_conditional_losses_615692-
+conv2d_transpose_10/StatefulPartitionedCall?
+conv2d_transpose_11/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_10/StatefulPartitionedCall:output:0conv2d_transpose_11_61823conv2d_transpose_11_61825*
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
N__inference_conv2d_transpose_11_layer_call_and_return_conditional_losses_615982-
+conv2d_transpose_11/StatefulPartitionedCall?
reshape_15/PartitionedCallPartitionedCall4conv2d_transpose_11/StatefulPartitionedCall:output:0*
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
E__inference_reshape_15_layer_call_and_return_conditional_losses_616182
reshape_15/PartitionedCall?
IdentityIdentity#reshape_15/PartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????2

Identity?
NoOpNoOp,^conv2d_transpose_10/StatefulPartitionedCall,^conv2d_transpose_11/StatefulPartitionedCall+^conv2d_transpose_9/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : : : : : 2Z
+conv2d_transpose_10/StatefulPartitionedCall+conv2d_transpose_10/StatefulPartitionedCall2Z
+conv2d_transpose_11/StatefulPartitionedCall+conv2d_transpose_11/StatefulPartitionedCall2X
*conv2d_transpose_9/StatefulPartitionedCall*conv2d_transpose_9/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall:Q M
'
_output_shapes
:?????????
"
_user_specified_name
input_14
??
?"
!__inference__traced_restore_63680
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
#assignvariableop_11_conv2d_6_kernel: /
!assignvariableop_12_conv2d_6_bias: =
#assignvariableop_13_conv2d_7_kernel: @/
!assignvariableop_14_conv2d_7_bias:@5
"assignvariableop_15_dense_6_kernel:	?.
 assignvariableop_16_dense_6_bias:5
"assignvariableop_17_dense_7_kernel:	?/
 assignvariableop_18_dense_7_bias:	?G
-assignvariableop_19_conv2d_transpose_9_kernel:@ 9
+assignvariableop_20_conv2d_transpose_9_bias:@H
.assignvariableop_21_conv2d_transpose_10_kernel: @:
,assignvariableop_22_conv2d_transpose_10_bias: H
.assignvariableop_23_conv2d_transpose_11_kernel: :
,assignvariableop_24_conv2d_transpose_11_bias:D
*assignvariableop_25_adam_conv2d_6_kernel_m: 6
(assignvariableop_26_adam_conv2d_6_bias_m: D
*assignvariableop_27_adam_conv2d_7_kernel_m: @6
(assignvariableop_28_adam_conv2d_7_bias_m:@<
)assignvariableop_29_adam_dense_6_kernel_m:	?5
'assignvariableop_30_adam_dense_6_bias_m:<
)assignvariableop_31_adam_dense_7_kernel_m:	?6
'assignvariableop_32_adam_dense_7_bias_m:	?N
4assignvariableop_33_adam_conv2d_transpose_9_kernel_m:@ @
2assignvariableop_34_adam_conv2d_transpose_9_bias_m:@O
5assignvariableop_35_adam_conv2d_transpose_10_kernel_m: @A
3assignvariableop_36_adam_conv2d_transpose_10_bias_m: O
5assignvariableop_37_adam_conv2d_transpose_11_kernel_m: A
3assignvariableop_38_adam_conv2d_transpose_11_bias_m:D
*assignvariableop_39_adam_conv2d_6_kernel_v: 6
(assignvariableop_40_adam_conv2d_6_bias_v: D
*assignvariableop_41_adam_conv2d_7_kernel_v: @6
(assignvariableop_42_adam_conv2d_7_bias_v:@<
)assignvariableop_43_adam_dense_6_kernel_v:	?5
'assignvariableop_44_adam_dense_6_bias_v:<
)assignvariableop_45_adam_dense_7_kernel_v:	?6
'assignvariableop_46_adam_dense_7_bias_v:	?N
4assignvariableop_47_adam_conv2d_transpose_9_kernel_v:@ @
2assignvariableop_48_adam_conv2d_transpose_9_bias_v:@O
5assignvariableop_49_adam_conv2d_transpose_10_kernel_v: @A
3assignvariableop_50_adam_conv2d_transpose_10_bias_v: O
5assignvariableop_51_adam_conv2d_transpose_11_kernel_v: A
3assignvariableop_52_adam_conv2d_transpose_11_bias_v:
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
AssignVariableOp_11AssignVariableOp#assignvariableop_11_conv2d_6_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp!assignvariableop_12_conv2d_6_biasIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOp#assignvariableop_13_conv2d_7_kernelIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOp!assignvariableop_14_conv2d_7_biasIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOp"assignvariableop_15_dense_6_kernelIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOp assignvariableop_16_dense_6_biasIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOp"assignvariableop_17_dense_7_kernelIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOp assignvariableop_18_dense_7_biasIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOp-assignvariableop_19_conv2d_transpose_9_kernelIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp+assignvariableop_20_conv2d_transpose_9_biasIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp.assignvariableop_21_conv2d_transpose_10_kernelIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp,assignvariableop_22_conv2d_transpose_10_biasIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp.assignvariableop_23_conv2d_transpose_11_kernelIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp,assignvariableop_24_conv2d_transpose_11_biasIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOp*assignvariableop_25_adam_conv2d_6_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOp(assignvariableop_26_adam_conv2d_6_bias_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOp*assignvariableop_27_adam_conv2d_7_kernel_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOp(assignvariableop_28_adam_conv2d_7_bias_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOp)assignvariableop_29_adam_dense_6_kernel_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOp'assignvariableop_30_adam_dense_6_bias_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOp)assignvariableop_31_adam_dense_7_kernel_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOp'assignvariableop_32_adam_dense_7_bias_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33?
AssignVariableOp_33AssignVariableOp4assignvariableop_33_adam_conv2d_transpose_9_kernel_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34?
AssignVariableOp_34AssignVariableOp2assignvariableop_34_adam_conv2d_transpose_9_bias_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35?
AssignVariableOp_35AssignVariableOp5assignvariableop_35_adam_conv2d_transpose_10_kernel_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36?
AssignVariableOp_36AssignVariableOp3assignvariableop_36_adam_conv2d_transpose_10_bias_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37?
AssignVariableOp_37AssignVariableOp5assignvariableop_37_adam_conv2d_transpose_11_kernel_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38?
AssignVariableOp_38AssignVariableOp3assignvariableop_38_adam_conv2d_transpose_11_bias_mIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39?
AssignVariableOp_39AssignVariableOp*assignvariableop_39_adam_conv2d_6_kernel_vIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40?
AssignVariableOp_40AssignVariableOp(assignvariableop_40_adam_conv2d_6_bias_vIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41?
AssignVariableOp_41AssignVariableOp*assignvariableop_41_adam_conv2d_7_kernel_vIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42?
AssignVariableOp_42AssignVariableOp(assignvariableop_42_adam_conv2d_7_bias_vIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43?
AssignVariableOp_43AssignVariableOp)assignvariableop_43_adam_dense_6_kernel_vIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44?
AssignVariableOp_44AssignVariableOp'assignvariableop_44_adam_dense_6_bias_vIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45?
AssignVariableOp_45AssignVariableOp)assignvariableop_45_adam_dense_7_kernel_vIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46?
AssignVariableOp_46AssignVariableOp'assignvariableop_46_adam_dense_7_bias_vIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47?
AssignVariableOp_47AssignVariableOp4assignvariableop_47_adam_conv2d_transpose_9_kernel_vIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48?
AssignVariableOp_48AssignVariableOp2assignvariableop_48_adam_conv2d_transpose_9_bias_vIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49?
AssignVariableOp_49AssignVariableOp5assignvariableop_49_adam_conv2d_transpose_10_kernel_vIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50?
AssignVariableOp_50AssignVariableOp3assignvariableop_50_adam_conv2d_transpose_10_bias_vIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51?
AssignVariableOp_51AssignVariableOp5assignvariableop_51_adam_conv2d_transpose_11_kernel_vIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52?
AssignVariableOp_52AssignVariableOp3assignvariableop_52_adam_conv2d_transpose_11_bias_vIdentity_52:output:0"/device:CPU:0*
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
?
?
B__inference_dense_7_layer_call_and_return_conditional_losses_63054

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
?
C__inference_conv2d_6_layer_call_and_return_conditional_losses_62966

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
?
?
9__inference_variational_autoencoder_3_layer_call_fn_62545

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
T__inference_variational_autoencoder_3_layer_call_and_return_conditional_losses_620152
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
G__inference_sequential_7_layer_call_and_return_conditional_losses_62851

inputs9
&dense_7_matmul_readvariableop_resource:	?6
'dense_7_biasadd_readvariableop_resource:	?U
;conv2d_transpose_9_conv2d_transpose_readvariableop_resource:@ @
2conv2d_transpose_9_biasadd_readvariableop_resource:@V
<conv2d_transpose_10_conv2d_transpose_readvariableop_resource: @A
3conv2d_transpose_10_biasadd_readvariableop_resource: V
<conv2d_transpose_11_conv2d_transpose_readvariableop_resource: A
3conv2d_transpose_11_biasadd_readvariableop_resource:
identity??*conv2d_transpose_10/BiasAdd/ReadVariableOp?3conv2d_transpose_10/conv2d_transpose/ReadVariableOp?*conv2d_transpose_11/BiasAdd/ReadVariableOp?3conv2d_transpose_11/conv2d_transpose/ReadVariableOp?)conv2d_transpose_9/BiasAdd/ReadVariableOp?2conv2d_transpose_9/conv2d_transpose/ReadVariableOp?dense_7/BiasAdd/ReadVariableOp?dense_7/MatMul/ReadVariableOp?
dense_7/MatMul/ReadVariableOpReadVariableOp&dense_7_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
dense_7/MatMul/ReadVariableOp?
dense_7/MatMulMatMulinputs%dense_7/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_7/MatMul?
dense_7/BiasAdd/ReadVariableOpReadVariableOp'dense_7_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
dense_7/BiasAdd/ReadVariableOp?
dense_7/BiasAddBiasAdddense_7/MatMul:product:0&dense_7/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_7/BiasAddq
dense_7/ReluReludense_7/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_7/Relun
reshape_14/ShapeShapedense_7/Relu:activations:0*
T0*
_output_shapes
:2
reshape_14/Shape?
reshape_14/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
reshape_14/strided_slice/stack?
 reshape_14/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_14/strided_slice/stack_1?
 reshape_14/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_14/strided_slice/stack_2?
reshape_14/strided_sliceStridedSlicereshape_14/Shape:output:0'reshape_14/strided_slice/stack:output:0)reshape_14/strided_slice/stack_1:output:0)reshape_14/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_14/strided_slicez
reshape_14/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_14/Reshape/shape/1z
reshape_14/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_14/Reshape/shape/2z
reshape_14/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B : 2
reshape_14/Reshape/shape/3?
reshape_14/Reshape/shapePack!reshape_14/strided_slice:output:0#reshape_14/Reshape/shape/1:output:0#reshape_14/Reshape/shape/2:output:0#reshape_14/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
reshape_14/Reshape/shape?
reshape_14/ReshapeReshapedense_7/Relu:activations:0!reshape_14/Reshape/shape:output:0*
T0*/
_output_shapes
:????????? 2
reshape_14/Reshape
conv2d_transpose_9/ShapeShapereshape_14/Reshape:output:0*
T0*
_output_shapes
:2
conv2d_transpose_9/Shape?
&conv2d_transpose_9/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv2d_transpose_9/strided_slice/stack?
(conv2d_transpose_9/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_9/strided_slice/stack_1?
(conv2d_transpose_9/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_9/strided_slice/stack_2?
 conv2d_transpose_9/strided_sliceStridedSlice!conv2d_transpose_9/Shape:output:0/conv2d_transpose_9/strided_slice/stack:output:01conv2d_transpose_9/strided_slice/stack_1:output:01conv2d_transpose_9/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv2d_transpose_9/strided_slicez
conv2d_transpose_9/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_9/stack/1z
conv2d_transpose_9/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_9/stack/2z
conv2d_transpose_9/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@2
conv2d_transpose_9/stack/3?
conv2d_transpose_9/stackPack)conv2d_transpose_9/strided_slice:output:0#conv2d_transpose_9/stack/1:output:0#conv2d_transpose_9/stack/2:output:0#conv2d_transpose_9/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_9/stack?
(conv2d_transpose_9/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(conv2d_transpose_9/strided_slice_1/stack?
*conv2d_transpose_9/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_9/strided_slice_1/stack_1?
*conv2d_transpose_9/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_9/strided_slice_1/stack_2?
"conv2d_transpose_9/strided_slice_1StridedSlice!conv2d_transpose_9/stack:output:01conv2d_transpose_9/strided_slice_1/stack:output:03conv2d_transpose_9/strided_slice_1/stack_1:output:03conv2d_transpose_9/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv2d_transpose_9/strided_slice_1?
2conv2d_transpose_9/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_9_conv2d_transpose_readvariableop_resource*&
_output_shapes
:@ *
dtype024
2conv2d_transpose_9/conv2d_transpose/ReadVariableOp?
#conv2d_transpose_9/conv2d_transposeConv2DBackpropInput!conv2d_transpose_9/stack:output:0:conv2d_transpose_9/conv2d_transpose/ReadVariableOp:value:0reshape_14/Reshape:output:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2%
#conv2d_transpose_9/conv2d_transpose?
)conv2d_transpose_9/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_9_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02+
)conv2d_transpose_9/BiasAdd/ReadVariableOp?
conv2d_transpose_9/BiasAddBiasAdd,conv2d_transpose_9/conv2d_transpose:output:01conv2d_transpose_9/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
conv2d_transpose_9/BiasAdd?
conv2d_transpose_9/ReluRelu#conv2d_transpose_9/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
conv2d_transpose_9/Relu?
conv2d_transpose_10/ShapeShape%conv2d_transpose_9/Relu:activations:0*
T0*
_output_shapes
:2
conv2d_transpose_10/Shape?
'conv2d_transpose_10/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'conv2d_transpose_10/strided_slice/stack?
)conv2d_transpose_10/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_10/strided_slice/stack_1?
)conv2d_transpose_10/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_10/strided_slice/stack_2?
!conv2d_transpose_10/strided_sliceStridedSlice"conv2d_transpose_10/Shape:output:00conv2d_transpose_10/strided_slice/stack:output:02conv2d_transpose_10/strided_slice/stack_1:output:02conv2d_transpose_10/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!conv2d_transpose_10/strided_slice|
conv2d_transpose_10/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_10/stack/1|
conv2d_transpose_10/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_10/stack/2|
conv2d_transpose_10/stack/3Const*
_output_shapes
: *
dtype0*
value	B : 2
conv2d_transpose_10/stack/3?
conv2d_transpose_10/stackPack*conv2d_transpose_10/strided_slice:output:0$conv2d_transpose_10/stack/1:output:0$conv2d_transpose_10/stack/2:output:0$conv2d_transpose_10/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_10/stack?
)conv2d_transpose_10/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)conv2d_transpose_10/strided_slice_1/stack?
+conv2d_transpose_10/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_10/strided_slice_1/stack_1?
+conv2d_transpose_10/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_10/strided_slice_1/stack_2?
#conv2d_transpose_10/strided_slice_1StridedSlice"conv2d_transpose_10/stack:output:02conv2d_transpose_10/strided_slice_1/stack:output:04conv2d_transpose_10/strided_slice_1/stack_1:output:04conv2d_transpose_10/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#conv2d_transpose_10/strided_slice_1?
3conv2d_transpose_10/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_10_conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype025
3conv2d_transpose_10/conv2d_transpose/ReadVariableOp?
$conv2d_transpose_10/conv2d_transposeConv2DBackpropInput"conv2d_transpose_10/stack:output:0;conv2d_transpose_10/conv2d_transpose/ReadVariableOp:value:0%conv2d_transpose_9/Relu:activations:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
2&
$conv2d_transpose_10/conv2d_transpose?
*conv2d_transpose_10/BiasAdd/ReadVariableOpReadVariableOp3conv2d_transpose_10_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02,
*conv2d_transpose_10/BiasAdd/ReadVariableOp?
conv2d_transpose_10/BiasAddBiasAdd-conv2d_transpose_10/conv2d_transpose:output:02conv2d_transpose_10/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2
conv2d_transpose_10/BiasAdd?
conv2d_transpose_10/ReluRelu$conv2d_transpose_10/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
conv2d_transpose_10/Relu?
conv2d_transpose_11/ShapeShape&conv2d_transpose_10/Relu:activations:0*
T0*
_output_shapes
:2
conv2d_transpose_11/Shape?
'conv2d_transpose_11/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'conv2d_transpose_11/strided_slice/stack?
)conv2d_transpose_11/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_11/strided_slice/stack_1?
)conv2d_transpose_11/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_11/strided_slice/stack_2?
!conv2d_transpose_11/strided_sliceStridedSlice"conv2d_transpose_11/Shape:output:00conv2d_transpose_11/strided_slice/stack:output:02conv2d_transpose_11/strided_slice/stack_1:output:02conv2d_transpose_11/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!conv2d_transpose_11/strided_slice|
conv2d_transpose_11/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_11/stack/1|
conv2d_transpose_11/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_11/stack/2|
conv2d_transpose_11/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_11/stack/3?
conv2d_transpose_11/stackPack*conv2d_transpose_11/strided_slice:output:0$conv2d_transpose_11/stack/1:output:0$conv2d_transpose_11/stack/2:output:0$conv2d_transpose_11/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_11/stack?
)conv2d_transpose_11/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)conv2d_transpose_11/strided_slice_1/stack?
+conv2d_transpose_11/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_11/strided_slice_1/stack_1?
+conv2d_transpose_11/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_11/strided_slice_1/stack_2?
#conv2d_transpose_11/strided_slice_1StridedSlice"conv2d_transpose_11/stack:output:02conv2d_transpose_11/strided_slice_1/stack:output:04conv2d_transpose_11/strided_slice_1/stack_1:output:04conv2d_transpose_11/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#conv2d_transpose_11/strided_slice_1?
3conv2d_transpose_11/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_11_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype025
3conv2d_transpose_11/conv2d_transpose/ReadVariableOp?
$conv2d_transpose_11/conv2d_transposeConv2DBackpropInput"conv2d_transpose_11/stack:output:0;conv2d_transpose_11/conv2d_transpose/ReadVariableOp:value:0&conv2d_transpose_10/Relu:activations:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2&
$conv2d_transpose_11/conv2d_transpose?
*conv2d_transpose_11/BiasAdd/ReadVariableOpReadVariableOp3conv2d_transpose_11_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02,
*conv2d_transpose_11/BiasAdd/ReadVariableOp?
conv2d_transpose_11/BiasAddBiasAdd-conv2d_transpose_11/conv2d_transpose:output:02conv2d_transpose_11/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
conv2d_transpose_11/BiasAdd?
conv2d_transpose_11/SigmoidSigmoid$conv2d_transpose_11/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
conv2d_transpose_11/Sigmoids
reshape_15/ShapeShapeconv2d_transpose_11/Sigmoid:y:0*
T0*
_output_shapes
:2
reshape_15/Shape?
reshape_15/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
reshape_15/strided_slice/stack?
 reshape_15/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_15/strided_slice/stack_1?
 reshape_15/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_15/strided_slice/stack_2?
reshape_15/strided_sliceStridedSlicereshape_15/Shape:output:0'reshape_15/strided_slice/stack:output:0)reshape_15/strided_slice/stack_1:output:0)reshape_15/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_15/strided_slicez
reshape_15/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_15/Reshape/shape/1z
reshape_15/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_15/Reshape/shape/2z
reshape_15/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_15/Reshape/shape/3?
reshape_15/Reshape/shapePack!reshape_15/strided_slice:output:0#reshape_15/Reshape/shape/1:output:0#reshape_15/Reshape/shape/2:output:0#reshape_15/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
reshape_15/Reshape/shape?
reshape_15/ReshapeReshapeconv2d_transpose_11/Sigmoid:y:0!reshape_15/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????2
reshape_15/Reshape~
IdentityIdentityreshape_15/Reshape:output:0^NoOp*
T0*/
_output_shapes
:?????????2

Identity?
NoOpNoOp+^conv2d_transpose_10/BiasAdd/ReadVariableOp4^conv2d_transpose_10/conv2d_transpose/ReadVariableOp+^conv2d_transpose_11/BiasAdd/ReadVariableOp4^conv2d_transpose_11/conv2d_transpose/ReadVariableOp*^conv2d_transpose_9/BiasAdd/ReadVariableOp3^conv2d_transpose_9/conv2d_transpose/ReadVariableOp^dense_7/BiasAdd/ReadVariableOp^dense_7/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : : : : : 2X
*conv2d_transpose_10/BiasAdd/ReadVariableOp*conv2d_transpose_10/BiasAdd/ReadVariableOp2j
3conv2d_transpose_10/conv2d_transpose/ReadVariableOp3conv2d_transpose_10/conv2d_transpose/ReadVariableOp2X
*conv2d_transpose_11/BiasAdd/ReadVariableOp*conv2d_transpose_11/BiasAdd/ReadVariableOp2j
3conv2d_transpose_11/conv2d_transpose/ReadVariableOp3conv2d_transpose_11/conv2d_transpose/ReadVariableOp2V
)conv2d_transpose_9/BiasAdd/ReadVariableOp)conv2d_transpose_9/BiasAdd/ReadVariableOp2h
2conv2d_transpose_9/conv2d_transpose/ReadVariableOp2conv2d_transpose_9/conv2d_transpose/ReadVariableOp2@
dense_7/BiasAdd/ReadVariableOpdense_7/BiasAdd/ReadVariableOp2>
dense_7/MatMul/ReadVariableOpdense_7/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
B__inference_dense_6_layer_call_and_return_conditional_losses_63016

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
?
?
3__inference_conv2d_transpose_10_layer_call_fn_63234

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
N__inference_conv2d_transpose_10_layer_call_and_return_conditional_losses_615692
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
?
F
*__inference_reshape_15_layer_call_fn_63329

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
E__inference_reshape_15_layer_call_and_return_conditional_losses_616182
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
?&
?
N__inference_conv2d_transpose_10_layer_call_and_return_conditional_losses_63192

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
?
,__inference_sequential_6_layer_call_fn_61169
input_13!
unknown: 
	unknown_0: #
	unknown_1: @
	unknown_2:@
	unknown_3:	?
	unknown_4:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_13unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
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
G__inference_sequential_6_layer_call_and_return_conditional_losses_611372
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
input_13
?
a
E__inference_reshape_13_layer_call_and_return_conditional_losses_61030

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
?
F
*__inference_reshape_14_layer_call_fn_63082

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
E__inference_reshape_14_layer_call_and_return_conditional_losses_615152
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

?
,__inference_sequential_7_layer_call_fn_61640
input_14
unknown:	?
	unknown_0:	?#
	unknown_1:@ 
	unknown_2:@#
	unknown_3: @
	unknown_4: #
	unknown_5: 
	unknown_6:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_14unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
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
G__inference_sequential_7_layer_call_and_return_conditional_losses_616212
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
input_14
?
?
N__inference_conv2d_transpose_11_layer_call_and_return_conditional_losses_63292

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
a
E__inference_reshape_14_layer_call_and_return_conditional_losses_61515

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
9__inference_variational_autoencoder_3_layer_call_fn_62512

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
T__inference_variational_autoencoder_3_layer_call_and_return_conditional_losses_618762
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
?
T__inference_variational_autoencoder_3_layer_call_and_return_conditional_losses_62324

inputsN
4sequential_6_conv2d_6_conv2d_readvariableop_resource: C
5sequential_6_conv2d_6_biasadd_readvariableop_resource: N
4sequential_6_conv2d_7_conv2d_readvariableop_resource: @C
5sequential_6_conv2d_7_biasadd_readvariableop_resource:@F
3sequential_6_dense_6_matmul_readvariableop_resource:	?B
4sequential_6_dense_6_biasadd_readvariableop_resource:F
3sequential_7_dense_7_matmul_readvariableop_resource:	?C
4sequential_7_dense_7_biasadd_readvariableop_resource:	?b
Hsequential_7_conv2d_transpose_9_conv2d_transpose_readvariableop_resource:@ M
?sequential_7_conv2d_transpose_9_biasadd_readvariableop_resource:@c
Isequential_7_conv2d_transpose_10_conv2d_transpose_readvariableop_resource: @N
@sequential_7_conv2d_transpose_10_biasadd_readvariableop_resource: c
Isequential_7_conv2d_transpose_11_conv2d_transpose_readvariableop_resource: N
@sequential_7_conv2d_transpose_11_biasadd_readvariableop_resource:
identity??,sequential_6/conv2d_6/BiasAdd/ReadVariableOp?+sequential_6/conv2d_6/Conv2D/ReadVariableOp?,sequential_6/conv2d_7/BiasAdd/ReadVariableOp?+sequential_6/conv2d_7/Conv2D/ReadVariableOp?+sequential_6/dense_6/BiasAdd/ReadVariableOp?*sequential_6/dense_6/MatMul/ReadVariableOp?7sequential_7/conv2d_transpose_10/BiasAdd/ReadVariableOp?@sequential_7/conv2d_transpose_10/conv2d_transpose/ReadVariableOp?7sequential_7/conv2d_transpose_11/BiasAdd/ReadVariableOp?@sequential_7/conv2d_transpose_11/conv2d_transpose/ReadVariableOp?6sequential_7/conv2d_transpose_9/BiasAdd/ReadVariableOp??sequential_7/conv2d_transpose_9/conv2d_transpose/ReadVariableOp?+sequential_7/dense_7/BiasAdd/ReadVariableOp?*sequential_7/dense_7/MatMul/ReadVariableOpt
sequential_6/reshape_12/ShapeShapeinputs*
T0*
_output_shapes
:2
sequential_6/reshape_12/Shape?
+sequential_6/reshape_12/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2-
+sequential_6/reshape_12/strided_slice/stack?
-sequential_6/reshape_12/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_6/reshape_12/strided_slice/stack_1?
-sequential_6/reshape_12/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_6/reshape_12/strided_slice/stack_2?
%sequential_6/reshape_12/strided_sliceStridedSlice&sequential_6/reshape_12/Shape:output:04sequential_6/reshape_12/strided_slice/stack:output:06sequential_6/reshape_12/strided_slice/stack_1:output:06sequential_6/reshape_12/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%sequential_6/reshape_12/strided_slice?
'sequential_6/reshape_12/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2)
'sequential_6/reshape_12/Reshape/shape/1?
'sequential_6/reshape_12/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2)
'sequential_6/reshape_12/Reshape/shape/2?
'sequential_6/reshape_12/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2)
'sequential_6/reshape_12/Reshape/shape/3?
%sequential_6/reshape_12/Reshape/shapePack.sequential_6/reshape_12/strided_slice:output:00sequential_6/reshape_12/Reshape/shape/1:output:00sequential_6/reshape_12/Reshape/shape/2:output:00sequential_6/reshape_12/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2'
%sequential_6/reshape_12/Reshape/shape?
sequential_6/reshape_12/ReshapeReshapeinputs.sequential_6/reshape_12/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????2!
sequential_6/reshape_12/Reshape?
+sequential_6/conv2d_6/Conv2D/ReadVariableOpReadVariableOp4sequential_6_conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02-
+sequential_6/conv2d_6/Conv2D/ReadVariableOp?
sequential_6/conv2d_6/Conv2DConv2D(sequential_6/reshape_12/Reshape:output:03sequential_6/conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
2
sequential_6/conv2d_6/Conv2D?
,sequential_6/conv2d_6/BiasAdd/ReadVariableOpReadVariableOp5sequential_6_conv2d_6_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02.
,sequential_6/conv2d_6/BiasAdd/ReadVariableOp?
sequential_6/conv2d_6/BiasAddBiasAdd%sequential_6/conv2d_6/Conv2D:output:04sequential_6/conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2
sequential_6/conv2d_6/BiasAdd?
sequential_6/conv2d_6/ReluRelu&sequential_6/conv2d_6/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
sequential_6/conv2d_6/Relu?
+sequential_6/conv2d_7/Conv2D/ReadVariableOpReadVariableOp4sequential_6_conv2d_7_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02-
+sequential_6/conv2d_7/Conv2D/ReadVariableOp?
sequential_6/conv2d_7/Conv2DConv2D(sequential_6/conv2d_6/Relu:activations:03sequential_6/conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
2
sequential_6/conv2d_7/Conv2D?
,sequential_6/conv2d_7/BiasAdd/ReadVariableOpReadVariableOp5sequential_6_conv2d_7_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02.
,sequential_6/conv2d_7/BiasAdd/ReadVariableOp?
sequential_6/conv2d_7/BiasAddBiasAdd%sequential_6/conv2d_7/Conv2D:output:04sequential_6/conv2d_7/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
sequential_6/conv2d_7/BiasAdd?
sequential_6/conv2d_7/ReluRelu&sequential_6/conv2d_7/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
sequential_6/conv2d_7/Relu?
sequential_6/flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"???? 	  2
sequential_6/flatten_3/Const?
sequential_6/flatten_3/ReshapeReshape(sequential_6/conv2d_7/Relu:activations:0%sequential_6/flatten_3/Const:output:0*
T0*(
_output_shapes
:??????????2 
sequential_6/flatten_3/Reshape?
*sequential_6/dense_6/MatMul/ReadVariableOpReadVariableOp3sequential_6_dense_6_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02,
*sequential_6/dense_6/MatMul/ReadVariableOp?
sequential_6/dense_6/MatMulMatMul'sequential_6/flatten_3/Reshape:output:02sequential_6/dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential_6/dense_6/MatMul?
+sequential_6/dense_6/BiasAdd/ReadVariableOpReadVariableOp4sequential_6_dense_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02-
+sequential_6/dense_6/BiasAdd/ReadVariableOp?
sequential_6/dense_6/BiasAddBiasAdd%sequential_6/dense_6/MatMul:product:03sequential_6/dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential_6/dense_6/BiasAdd?
sequential_6/reshape_13/ShapeShape%sequential_6/dense_6/BiasAdd:output:0*
T0*
_output_shapes
:2
sequential_6/reshape_13/Shape?
+sequential_6/reshape_13/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2-
+sequential_6/reshape_13/strided_slice/stack?
-sequential_6/reshape_13/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_6/reshape_13/strided_slice/stack_1?
-sequential_6/reshape_13/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_6/reshape_13/strided_slice/stack_2?
%sequential_6/reshape_13/strided_sliceStridedSlice&sequential_6/reshape_13/Shape:output:04sequential_6/reshape_13/strided_slice/stack:output:06sequential_6/reshape_13/strided_slice/stack_1:output:06sequential_6/reshape_13/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%sequential_6/reshape_13/strided_slice?
'sequential_6/reshape_13/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2)
'sequential_6/reshape_13/Reshape/shape/1?
'sequential_6/reshape_13/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2)
'sequential_6/reshape_13/Reshape/shape/2?
%sequential_6/reshape_13/Reshape/shapePack.sequential_6/reshape_13/strided_slice:output:00sequential_6/reshape_13/Reshape/shape/1:output:00sequential_6/reshape_13/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2'
%sequential_6/reshape_13/Reshape/shape?
sequential_6/reshape_13/ReshapeReshape%sequential_6/dense_6/BiasAdd:output:0.sequential_6/reshape_13/Reshape/shape:output:0*
T0*+
_output_shapes
:?????????2!
sequential_6/reshape_13/Reshape?
normal_sampling_layer_3/unstackUnpack(sequential_6/reshape_13/Reshape:output:0*
T0*:
_output_shapes(
&:?????????:?????????*

axis*	
num2!
normal_sampling_layer_3/unstack?
*sequential_7/dense_7/MatMul/ReadVariableOpReadVariableOp3sequential_7_dense_7_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02,
*sequential_7/dense_7/MatMul/ReadVariableOp?
sequential_7/dense_7/MatMulMatMul(normal_sampling_layer_3/unstack:output:02sequential_7/dense_7/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential_7/dense_7/MatMul?
+sequential_7/dense_7/BiasAdd/ReadVariableOpReadVariableOp4sequential_7_dense_7_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02-
+sequential_7/dense_7/BiasAdd/ReadVariableOp?
sequential_7/dense_7/BiasAddBiasAdd%sequential_7/dense_7/MatMul:product:03sequential_7/dense_7/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential_7/dense_7/BiasAdd?
sequential_7/dense_7/ReluRelu%sequential_7/dense_7/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
sequential_7/dense_7/Relu?
sequential_7/reshape_14/ShapeShape'sequential_7/dense_7/Relu:activations:0*
T0*
_output_shapes
:2
sequential_7/reshape_14/Shape?
+sequential_7/reshape_14/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2-
+sequential_7/reshape_14/strided_slice/stack?
-sequential_7/reshape_14/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_7/reshape_14/strided_slice/stack_1?
-sequential_7/reshape_14/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_7/reshape_14/strided_slice/stack_2?
%sequential_7/reshape_14/strided_sliceStridedSlice&sequential_7/reshape_14/Shape:output:04sequential_7/reshape_14/strided_slice/stack:output:06sequential_7/reshape_14/strided_slice/stack_1:output:06sequential_7/reshape_14/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%sequential_7/reshape_14/strided_slice?
'sequential_7/reshape_14/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2)
'sequential_7/reshape_14/Reshape/shape/1?
'sequential_7/reshape_14/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2)
'sequential_7/reshape_14/Reshape/shape/2?
'sequential_7/reshape_14/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B : 2)
'sequential_7/reshape_14/Reshape/shape/3?
%sequential_7/reshape_14/Reshape/shapePack.sequential_7/reshape_14/strided_slice:output:00sequential_7/reshape_14/Reshape/shape/1:output:00sequential_7/reshape_14/Reshape/shape/2:output:00sequential_7/reshape_14/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2'
%sequential_7/reshape_14/Reshape/shape?
sequential_7/reshape_14/ReshapeReshape'sequential_7/dense_7/Relu:activations:0.sequential_7/reshape_14/Reshape/shape:output:0*
T0*/
_output_shapes
:????????? 2!
sequential_7/reshape_14/Reshape?
%sequential_7/conv2d_transpose_9/ShapeShape(sequential_7/reshape_14/Reshape:output:0*
T0*
_output_shapes
:2'
%sequential_7/conv2d_transpose_9/Shape?
3sequential_7/conv2d_transpose_9/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 25
3sequential_7/conv2d_transpose_9/strided_slice/stack?
5sequential_7/conv2d_transpose_9/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential_7/conv2d_transpose_9/strided_slice/stack_1?
5sequential_7/conv2d_transpose_9/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential_7/conv2d_transpose_9/strided_slice/stack_2?
-sequential_7/conv2d_transpose_9/strided_sliceStridedSlice.sequential_7/conv2d_transpose_9/Shape:output:0<sequential_7/conv2d_transpose_9/strided_slice/stack:output:0>sequential_7/conv2d_transpose_9/strided_slice/stack_1:output:0>sequential_7/conv2d_transpose_9/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2/
-sequential_7/conv2d_transpose_9/strided_slice?
'sequential_7/conv2d_transpose_9/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2)
'sequential_7/conv2d_transpose_9/stack/1?
'sequential_7/conv2d_transpose_9/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2)
'sequential_7/conv2d_transpose_9/stack/2?
'sequential_7/conv2d_transpose_9/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@2)
'sequential_7/conv2d_transpose_9/stack/3?
%sequential_7/conv2d_transpose_9/stackPack6sequential_7/conv2d_transpose_9/strided_slice:output:00sequential_7/conv2d_transpose_9/stack/1:output:00sequential_7/conv2d_transpose_9/stack/2:output:00sequential_7/conv2d_transpose_9/stack/3:output:0*
N*
T0*
_output_shapes
:2'
%sequential_7/conv2d_transpose_9/stack?
5sequential_7/conv2d_transpose_9/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 27
5sequential_7/conv2d_transpose_9/strided_slice_1/stack?
7sequential_7/conv2d_transpose_9/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:29
7sequential_7/conv2d_transpose_9/strided_slice_1/stack_1?
7sequential_7/conv2d_transpose_9/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:29
7sequential_7/conv2d_transpose_9/strided_slice_1/stack_2?
/sequential_7/conv2d_transpose_9/strided_slice_1StridedSlice.sequential_7/conv2d_transpose_9/stack:output:0>sequential_7/conv2d_transpose_9/strided_slice_1/stack:output:0@sequential_7/conv2d_transpose_9/strided_slice_1/stack_1:output:0@sequential_7/conv2d_transpose_9/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask21
/sequential_7/conv2d_transpose_9/strided_slice_1?
?sequential_7/conv2d_transpose_9/conv2d_transpose/ReadVariableOpReadVariableOpHsequential_7_conv2d_transpose_9_conv2d_transpose_readvariableop_resource*&
_output_shapes
:@ *
dtype02A
?sequential_7/conv2d_transpose_9/conv2d_transpose/ReadVariableOp?
0sequential_7/conv2d_transpose_9/conv2d_transposeConv2DBackpropInput.sequential_7/conv2d_transpose_9/stack:output:0Gsequential_7/conv2d_transpose_9/conv2d_transpose/ReadVariableOp:value:0(sequential_7/reshape_14/Reshape:output:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
22
0sequential_7/conv2d_transpose_9/conv2d_transpose?
6sequential_7/conv2d_transpose_9/BiasAdd/ReadVariableOpReadVariableOp?sequential_7_conv2d_transpose_9_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype028
6sequential_7/conv2d_transpose_9/BiasAdd/ReadVariableOp?
'sequential_7/conv2d_transpose_9/BiasAddBiasAdd9sequential_7/conv2d_transpose_9/conv2d_transpose:output:0>sequential_7/conv2d_transpose_9/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2)
'sequential_7/conv2d_transpose_9/BiasAdd?
$sequential_7/conv2d_transpose_9/ReluRelu0sequential_7/conv2d_transpose_9/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2&
$sequential_7/conv2d_transpose_9/Relu?
&sequential_7/conv2d_transpose_10/ShapeShape2sequential_7/conv2d_transpose_9/Relu:activations:0*
T0*
_output_shapes
:2(
&sequential_7/conv2d_transpose_10/Shape?
4sequential_7/conv2d_transpose_10/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 26
4sequential_7/conv2d_transpose_10/strided_slice/stack?
6sequential_7/conv2d_transpose_10/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:28
6sequential_7/conv2d_transpose_10/strided_slice/stack_1?
6sequential_7/conv2d_transpose_10/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:28
6sequential_7/conv2d_transpose_10/strided_slice/stack_2?
.sequential_7/conv2d_transpose_10/strided_sliceStridedSlice/sequential_7/conv2d_transpose_10/Shape:output:0=sequential_7/conv2d_transpose_10/strided_slice/stack:output:0?sequential_7/conv2d_transpose_10/strided_slice/stack_1:output:0?sequential_7/conv2d_transpose_10/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask20
.sequential_7/conv2d_transpose_10/strided_slice?
(sequential_7/conv2d_transpose_10/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2*
(sequential_7/conv2d_transpose_10/stack/1?
(sequential_7/conv2d_transpose_10/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2*
(sequential_7/conv2d_transpose_10/stack/2?
(sequential_7/conv2d_transpose_10/stack/3Const*
_output_shapes
: *
dtype0*
value	B : 2*
(sequential_7/conv2d_transpose_10/stack/3?
&sequential_7/conv2d_transpose_10/stackPack7sequential_7/conv2d_transpose_10/strided_slice:output:01sequential_7/conv2d_transpose_10/stack/1:output:01sequential_7/conv2d_transpose_10/stack/2:output:01sequential_7/conv2d_transpose_10/stack/3:output:0*
N*
T0*
_output_shapes
:2(
&sequential_7/conv2d_transpose_10/stack?
6sequential_7/conv2d_transpose_10/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 28
6sequential_7/conv2d_transpose_10/strided_slice_1/stack?
8sequential_7/conv2d_transpose_10/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2:
8sequential_7/conv2d_transpose_10/strided_slice_1/stack_1?
8sequential_7/conv2d_transpose_10/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2:
8sequential_7/conv2d_transpose_10/strided_slice_1/stack_2?
0sequential_7/conv2d_transpose_10/strided_slice_1StridedSlice/sequential_7/conv2d_transpose_10/stack:output:0?sequential_7/conv2d_transpose_10/strided_slice_1/stack:output:0Asequential_7/conv2d_transpose_10/strided_slice_1/stack_1:output:0Asequential_7/conv2d_transpose_10/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask22
0sequential_7/conv2d_transpose_10/strided_slice_1?
@sequential_7/conv2d_transpose_10/conv2d_transpose/ReadVariableOpReadVariableOpIsequential_7_conv2d_transpose_10_conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype02B
@sequential_7/conv2d_transpose_10/conv2d_transpose/ReadVariableOp?
1sequential_7/conv2d_transpose_10/conv2d_transposeConv2DBackpropInput/sequential_7/conv2d_transpose_10/stack:output:0Hsequential_7/conv2d_transpose_10/conv2d_transpose/ReadVariableOp:value:02sequential_7/conv2d_transpose_9/Relu:activations:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
23
1sequential_7/conv2d_transpose_10/conv2d_transpose?
7sequential_7/conv2d_transpose_10/BiasAdd/ReadVariableOpReadVariableOp@sequential_7_conv2d_transpose_10_biasadd_readvariableop_resource*
_output_shapes
: *
dtype029
7sequential_7/conv2d_transpose_10/BiasAdd/ReadVariableOp?
(sequential_7/conv2d_transpose_10/BiasAddBiasAdd:sequential_7/conv2d_transpose_10/conv2d_transpose:output:0?sequential_7/conv2d_transpose_10/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2*
(sequential_7/conv2d_transpose_10/BiasAdd?
%sequential_7/conv2d_transpose_10/ReluRelu1sequential_7/conv2d_transpose_10/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2'
%sequential_7/conv2d_transpose_10/Relu?
&sequential_7/conv2d_transpose_11/ShapeShape3sequential_7/conv2d_transpose_10/Relu:activations:0*
T0*
_output_shapes
:2(
&sequential_7/conv2d_transpose_11/Shape?
4sequential_7/conv2d_transpose_11/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 26
4sequential_7/conv2d_transpose_11/strided_slice/stack?
6sequential_7/conv2d_transpose_11/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:28
6sequential_7/conv2d_transpose_11/strided_slice/stack_1?
6sequential_7/conv2d_transpose_11/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:28
6sequential_7/conv2d_transpose_11/strided_slice/stack_2?
.sequential_7/conv2d_transpose_11/strided_sliceStridedSlice/sequential_7/conv2d_transpose_11/Shape:output:0=sequential_7/conv2d_transpose_11/strided_slice/stack:output:0?sequential_7/conv2d_transpose_11/strided_slice/stack_1:output:0?sequential_7/conv2d_transpose_11/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask20
.sequential_7/conv2d_transpose_11/strided_slice?
(sequential_7/conv2d_transpose_11/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2*
(sequential_7/conv2d_transpose_11/stack/1?
(sequential_7/conv2d_transpose_11/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2*
(sequential_7/conv2d_transpose_11/stack/2?
(sequential_7/conv2d_transpose_11/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2*
(sequential_7/conv2d_transpose_11/stack/3?
&sequential_7/conv2d_transpose_11/stackPack7sequential_7/conv2d_transpose_11/strided_slice:output:01sequential_7/conv2d_transpose_11/stack/1:output:01sequential_7/conv2d_transpose_11/stack/2:output:01sequential_7/conv2d_transpose_11/stack/3:output:0*
N*
T0*
_output_shapes
:2(
&sequential_7/conv2d_transpose_11/stack?
6sequential_7/conv2d_transpose_11/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 28
6sequential_7/conv2d_transpose_11/strided_slice_1/stack?
8sequential_7/conv2d_transpose_11/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2:
8sequential_7/conv2d_transpose_11/strided_slice_1/stack_1?
8sequential_7/conv2d_transpose_11/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2:
8sequential_7/conv2d_transpose_11/strided_slice_1/stack_2?
0sequential_7/conv2d_transpose_11/strided_slice_1StridedSlice/sequential_7/conv2d_transpose_11/stack:output:0?sequential_7/conv2d_transpose_11/strided_slice_1/stack:output:0Asequential_7/conv2d_transpose_11/strided_slice_1/stack_1:output:0Asequential_7/conv2d_transpose_11/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask22
0sequential_7/conv2d_transpose_11/strided_slice_1?
@sequential_7/conv2d_transpose_11/conv2d_transpose/ReadVariableOpReadVariableOpIsequential_7_conv2d_transpose_11_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype02B
@sequential_7/conv2d_transpose_11/conv2d_transpose/ReadVariableOp?
1sequential_7/conv2d_transpose_11/conv2d_transposeConv2DBackpropInput/sequential_7/conv2d_transpose_11/stack:output:0Hsequential_7/conv2d_transpose_11/conv2d_transpose/ReadVariableOp:value:03sequential_7/conv2d_transpose_10/Relu:activations:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
23
1sequential_7/conv2d_transpose_11/conv2d_transpose?
7sequential_7/conv2d_transpose_11/BiasAdd/ReadVariableOpReadVariableOp@sequential_7_conv2d_transpose_11_biasadd_readvariableop_resource*
_output_shapes
:*
dtype029
7sequential_7/conv2d_transpose_11/BiasAdd/ReadVariableOp?
(sequential_7/conv2d_transpose_11/BiasAddBiasAdd:sequential_7/conv2d_transpose_11/conv2d_transpose:output:0?sequential_7/conv2d_transpose_11/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2*
(sequential_7/conv2d_transpose_11/BiasAdd?
(sequential_7/conv2d_transpose_11/SigmoidSigmoid1sequential_7/conv2d_transpose_11/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2*
(sequential_7/conv2d_transpose_11/Sigmoid?
sequential_7/reshape_15/ShapeShape,sequential_7/conv2d_transpose_11/Sigmoid:y:0*
T0*
_output_shapes
:2
sequential_7/reshape_15/Shape?
+sequential_7/reshape_15/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2-
+sequential_7/reshape_15/strided_slice/stack?
-sequential_7/reshape_15/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_7/reshape_15/strided_slice/stack_1?
-sequential_7/reshape_15/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_7/reshape_15/strided_slice/stack_2?
%sequential_7/reshape_15/strided_sliceStridedSlice&sequential_7/reshape_15/Shape:output:04sequential_7/reshape_15/strided_slice/stack:output:06sequential_7/reshape_15/strided_slice/stack_1:output:06sequential_7/reshape_15/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%sequential_7/reshape_15/strided_slice?
'sequential_7/reshape_15/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2)
'sequential_7/reshape_15/Reshape/shape/1?
'sequential_7/reshape_15/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2)
'sequential_7/reshape_15/Reshape/shape/2?
'sequential_7/reshape_15/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2)
'sequential_7/reshape_15/Reshape/shape/3?
%sequential_7/reshape_15/Reshape/shapePack.sequential_7/reshape_15/strided_slice:output:00sequential_7/reshape_15/Reshape/shape/1:output:00sequential_7/reshape_15/Reshape/shape/2:output:00sequential_7/reshape_15/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2'
%sequential_7/reshape_15/Reshape/shape?
sequential_7/reshape_15/ReshapeReshape,sequential_7/conv2d_transpose_11/Sigmoid:y:0.sequential_7/reshape_15/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????2!
sequential_7/reshape_15/Reshape?
IdentityIdentity(sequential_7/reshape_15/Reshape:output:0^NoOp*
T0*/
_output_shapes
:?????????2

Identity?
NoOpNoOp-^sequential_6/conv2d_6/BiasAdd/ReadVariableOp,^sequential_6/conv2d_6/Conv2D/ReadVariableOp-^sequential_6/conv2d_7/BiasAdd/ReadVariableOp,^sequential_6/conv2d_7/Conv2D/ReadVariableOp,^sequential_6/dense_6/BiasAdd/ReadVariableOp+^sequential_6/dense_6/MatMul/ReadVariableOp8^sequential_7/conv2d_transpose_10/BiasAdd/ReadVariableOpA^sequential_7/conv2d_transpose_10/conv2d_transpose/ReadVariableOp8^sequential_7/conv2d_transpose_11/BiasAdd/ReadVariableOpA^sequential_7/conv2d_transpose_11/conv2d_transpose/ReadVariableOp7^sequential_7/conv2d_transpose_9/BiasAdd/ReadVariableOp@^sequential_7/conv2d_transpose_9/conv2d_transpose/ReadVariableOp,^sequential_7/dense_7/BiasAdd/ReadVariableOp+^sequential_7/dense_7/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:?????????: : : : : : : : : : : : : : 2\
,sequential_6/conv2d_6/BiasAdd/ReadVariableOp,sequential_6/conv2d_6/BiasAdd/ReadVariableOp2Z
+sequential_6/conv2d_6/Conv2D/ReadVariableOp+sequential_6/conv2d_6/Conv2D/ReadVariableOp2\
,sequential_6/conv2d_7/BiasAdd/ReadVariableOp,sequential_6/conv2d_7/BiasAdd/ReadVariableOp2Z
+sequential_6/conv2d_7/Conv2D/ReadVariableOp+sequential_6/conv2d_7/Conv2D/ReadVariableOp2Z
+sequential_6/dense_6/BiasAdd/ReadVariableOp+sequential_6/dense_6/BiasAdd/ReadVariableOp2X
*sequential_6/dense_6/MatMul/ReadVariableOp*sequential_6/dense_6/MatMul/ReadVariableOp2r
7sequential_7/conv2d_transpose_10/BiasAdd/ReadVariableOp7sequential_7/conv2d_transpose_10/BiasAdd/ReadVariableOp2?
@sequential_7/conv2d_transpose_10/conv2d_transpose/ReadVariableOp@sequential_7/conv2d_transpose_10/conv2d_transpose/ReadVariableOp2r
7sequential_7/conv2d_transpose_11/BiasAdd/ReadVariableOp7sequential_7/conv2d_transpose_11/BiasAdd/ReadVariableOp2?
@sequential_7/conv2d_transpose_11/conv2d_transpose/ReadVariableOp@sequential_7/conv2d_transpose_11/conv2d_transpose/ReadVariableOp2p
6sequential_7/conv2d_transpose_9/BiasAdd/ReadVariableOp6sequential_7/conv2d_transpose_9/BiasAdd/ReadVariableOp2?
?sequential_7/conv2d_transpose_9/conv2d_transpose/ReadVariableOp?sequential_7/conv2d_transpose_9/conv2d_transpose/ReadVariableOp2Z
+sequential_7/dense_7/BiasAdd/ReadVariableOp+sequential_7/dense_7/BiasAdd/ReadVariableOp2X
*sequential_7/dense_7/MatMul/ReadVariableOp*sequential_7/dense_7/MatMul/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
n
R__inference_normal_sampling_layer_3_layer_call_and_return_conditional_losses_62899

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
?
?
N__inference_conv2d_transpose_11_layer_call_and_return_conditional_losses_61598

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
?
?
2__inference_conv2d_transpose_9_layer_call_fn_63158

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
M__inference_conv2d_transpose_9_layer_call_and_return_conditional_losses_615402
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
?
?
3__inference_conv2d_transpose_11_layer_call_fn_63310

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
N__inference_conv2d_transpose_11_layer_call_and_return_conditional_losses_615982
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
??
?
T__inference_variational_autoencoder_3_layer_call_and_return_conditional_losses_62479

inputsN
4sequential_6_conv2d_6_conv2d_readvariableop_resource: C
5sequential_6_conv2d_6_biasadd_readvariableop_resource: N
4sequential_6_conv2d_7_conv2d_readvariableop_resource: @C
5sequential_6_conv2d_7_biasadd_readvariableop_resource:@F
3sequential_6_dense_6_matmul_readvariableop_resource:	?B
4sequential_6_dense_6_biasadd_readvariableop_resource:F
3sequential_7_dense_7_matmul_readvariableop_resource:	?C
4sequential_7_dense_7_biasadd_readvariableop_resource:	?b
Hsequential_7_conv2d_transpose_9_conv2d_transpose_readvariableop_resource:@ M
?sequential_7_conv2d_transpose_9_biasadd_readvariableop_resource:@c
Isequential_7_conv2d_transpose_10_conv2d_transpose_readvariableop_resource: @N
@sequential_7_conv2d_transpose_10_biasadd_readvariableop_resource: c
Isequential_7_conv2d_transpose_11_conv2d_transpose_readvariableop_resource: N
@sequential_7_conv2d_transpose_11_biasadd_readvariableop_resource:
identity??,sequential_6/conv2d_6/BiasAdd/ReadVariableOp?+sequential_6/conv2d_6/Conv2D/ReadVariableOp?,sequential_6/conv2d_7/BiasAdd/ReadVariableOp?+sequential_6/conv2d_7/Conv2D/ReadVariableOp?+sequential_6/dense_6/BiasAdd/ReadVariableOp?*sequential_6/dense_6/MatMul/ReadVariableOp?7sequential_7/conv2d_transpose_10/BiasAdd/ReadVariableOp?@sequential_7/conv2d_transpose_10/conv2d_transpose/ReadVariableOp?7sequential_7/conv2d_transpose_11/BiasAdd/ReadVariableOp?@sequential_7/conv2d_transpose_11/conv2d_transpose/ReadVariableOp?6sequential_7/conv2d_transpose_9/BiasAdd/ReadVariableOp??sequential_7/conv2d_transpose_9/conv2d_transpose/ReadVariableOp?+sequential_7/dense_7/BiasAdd/ReadVariableOp?*sequential_7/dense_7/MatMul/ReadVariableOpt
sequential_6/reshape_12/ShapeShapeinputs*
T0*
_output_shapes
:2
sequential_6/reshape_12/Shape?
+sequential_6/reshape_12/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2-
+sequential_6/reshape_12/strided_slice/stack?
-sequential_6/reshape_12/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_6/reshape_12/strided_slice/stack_1?
-sequential_6/reshape_12/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_6/reshape_12/strided_slice/stack_2?
%sequential_6/reshape_12/strided_sliceStridedSlice&sequential_6/reshape_12/Shape:output:04sequential_6/reshape_12/strided_slice/stack:output:06sequential_6/reshape_12/strided_slice/stack_1:output:06sequential_6/reshape_12/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%sequential_6/reshape_12/strided_slice?
'sequential_6/reshape_12/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2)
'sequential_6/reshape_12/Reshape/shape/1?
'sequential_6/reshape_12/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2)
'sequential_6/reshape_12/Reshape/shape/2?
'sequential_6/reshape_12/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2)
'sequential_6/reshape_12/Reshape/shape/3?
%sequential_6/reshape_12/Reshape/shapePack.sequential_6/reshape_12/strided_slice:output:00sequential_6/reshape_12/Reshape/shape/1:output:00sequential_6/reshape_12/Reshape/shape/2:output:00sequential_6/reshape_12/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2'
%sequential_6/reshape_12/Reshape/shape?
sequential_6/reshape_12/ReshapeReshapeinputs.sequential_6/reshape_12/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????2!
sequential_6/reshape_12/Reshape?
+sequential_6/conv2d_6/Conv2D/ReadVariableOpReadVariableOp4sequential_6_conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02-
+sequential_6/conv2d_6/Conv2D/ReadVariableOp?
sequential_6/conv2d_6/Conv2DConv2D(sequential_6/reshape_12/Reshape:output:03sequential_6/conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
2
sequential_6/conv2d_6/Conv2D?
,sequential_6/conv2d_6/BiasAdd/ReadVariableOpReadVariableOp5sequential_6_conv2d_6_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02.
,sequential_6/conv2d_6/BiasAdd/ReadVariableOp?
sequential_6/conv2d_6/BiasAddBiasAdd%sequential_6/conv2d_6/Conv2D:output:04sequential_6/conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2
sequential_6/conv2d_6/BiasAdd?
sequential_6/conv2d_6/ReluRelu&sequential_6/conv2d_6/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
sequential_6/conv2d_6/Relu?
+sequential_6/conv2d_7/Conv2D/ReadVariableOpReadVariableOp4sequential_6_conv2d_7_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02-
+sequential_6/conv2d_7/Conv2D/ReadVariableOp?
sequential_6/conv2d_7/Conv2DConv2D(sequential_6/conv2d_6/Relu:activations:03sequential_6/conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
2
sequential_6/conv2d_7/Conv2D?
,sequential_6/conv2d_7/BiasAdd/ReadVariableOpReadVariableOp5sequential_6_conv2d_7_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02.
,sequential_6/conv2d_7/BiasAdd/ReadVariableOp?
sequential_6/conv2d_7/BiasAddBiasAdd%sequential_6/conv2d_7/Conv2D:output:04sequential_6/conv2d_7/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
sequential_6/conv2d_7/BiasAdd?
sequential_6/conv2d_7/ReluRelu&sequential_6/conv2d_7/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
sequential_6/conv2d_7/Relu?
sequential_6/flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"???? 	  2
sequential_6/flatten_3/Const?
sequential_6/flatten_3/ReshapeReshape(sequential_6/conv2d_7/Relu:activations:0%sequential_6/flatten_3/Const:output:0*
T0*(
_output_shapes
:??????????2 
sequential_6/flatten_3/Reshape?
*sequential_6/dense_6/MatMul/ReadVariableOpReadVariableOp3sequential_6_dense_6_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02,
*sequential_6/dense_6/MatMul/ReadVariableOp?
sequential_6/dense_6/MatMulMatMul'sequential_6/flatten_3/Reshape:output:02sequential_6/dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential_6/dense_6/MatMul?
+sequential_6/dense_6/BiasAdd/ReadVariableOpReadVariableOp4sequential_6_dense_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02-
+sequential_6/dense_6/BiasAdd/ReadVariableOp?
sequential_6/dense_6/BiasAddBiasAdd%sequential_6/dense_6/MatMul:product:03sequential_6/dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential_6/dense_6/BiasAdd?
sequential_6/reshape_13/ShapeShape%sequential_6/dense_6/BiasAdd:output:0*
T0*
_output_shapes
:2
sequential_6/reshape_13/Shape?
+sequential_6/reshape_13/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2-
+sequential_6/reshape_13/strided_slice/stack?
-sequential_6/reshape_13/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_6/reshape_13/strided_slice/stack_1?
-sequential_6/reshape_13/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_6/reshape_13/strided_slice/stack_2?
%sequential_6/reshape_13/strided_sliceStridedSlice&sequential_6/reshape_13/Shape:output:04sequential_6/reshape_13/strided_slice/stack:output:06sequential_6/reshape_13/strided_slice/stack_1:output:06sequential_6/reshape_13/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%sequential_6/reshape_13/strided_slice?
'sequential_6/reshape_13/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2)
'sequential_6/reshape_13/Reshape/shape/1?
'sequential_6/reshape_13/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2)
'sequential_6/reshape_13/Reshape/shape/2?
%sequential_6/reshape_13/Reshape/shapePack.sequential_6/reshape_13/strided_slice:output:00sequential_6/reshape_13/Reshape/shape/1:output:00sequential_6/reshape_13/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2'
%sequential_6/reshape_13/Reshape/shape?
sequential_6/reshape_13/ReshapeReshape%sequential_6/dense_6/BiasAdd:output:0.sequential_6/reshape_13/Reshape/shape:output:0*
T0*+
_output_shapes
:?????????2!
sequential_6/reshape_13/Reshape?
normal_sampling_layer_3/unstackUnpack(sequential_6/reshape_13/Reshape:output:0*
T0*:
_output_shapes(
&:?????????:?????????*

axis*	
num2!
normal_sampling_layer_3/unstack?
normal_sampling_layer_3/ShapeShape(normal_sampling_layer_3/unstack:output:0*
T0*
_output_shapes
:2
normal_sampling_layer_3/Shape?
+normal_sampling_layer_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2-
+normal_sampling_layer_3/strided_slice/stack?
-normal_sampling_layer_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-normal_sampling_layer_3/strided_slice/stack_1?
-normal_sampling_layer_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-normal_sampling_layer_3/strided_slice/stack_2?
%normal_sampling_layer_3/strided_sliceStridedSlice&normal_sampling_layer_3/Shape:output:04normal_sampling_layer_3/strided_slice/stack:output:06normal_sampling_layer_3/strided_slice/stack_1:output:06normal_sampling_layer_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%normal_sampling_layer_3/strided_slice?
normal_sampling_layer_3/Shape_1Shape(normal_sampling_layer_3/unstack:output:0*
T0*
_output_shapes
:2!
normal_sampling_layer_3/Shape_1?
-normal_sampling_layer_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2/
-normal_sampling_layer_3/strided_slice_1/stack?
/normal_sampling_layer_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:21
/normal_sampling_layer_3/strided_slice_1/stack_1?
/normal_sampling_layer_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:21
/normal_sampling_layer_3/strided_slice_1/stack_2?
'normal_sampling_layer_3/strided_slice_1StridedSlice(normal_sampling_layer_3/Shape_1:output:06normal_sampling_layer_3/strided_slice_1/stack:output:08normal_sampling_layer_3/strided_slice_1/stack_1:output:08normal_sampling_layer_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2)
'normal_sampling_layer_3/strided_slice_1?
+normal_sampling_layer_3/random_normal/shapePack.normal_sampling_layer_3/strided_slice:output:00normal_sampling_layer_3/strided_slice_1:output:0*
N*
T0*
_output_shapes
:2-
+normal_sampling_layer_3/random_normal/shape?
*normal_sampling_layer_3/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2,
*normal_sampling_layer_3/random_normal/mean?
,normal_sampling_layer_3/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2.
,normal_sampling_layer_3/random_normal/stddev?
:normal_sampling_layer_3/random_normal/RandomStandardNormalRandomStandardNormal4normal_sampling_layer_3/random_normal/shape:output:0*
T0*0
_output_shapes
:??????????????????*
dtype0*
seed???)*
seed2???2<
:normal_sampling_layer_3/random_normal/RandomStandardNormal?
)normal_sampling_layer_3/random_normal/mulMulCnormal_sampling_layer_3/random_normal/RandomStandardNormal:output:05normal_sampling_layer_3/random_normal/stddev:output:0*
T0*0
_output_shapes
:??????????????????2+
)normal_sampling_layer_3/random_normal/mul?
%normal_sampling_layer_3/random_normalAddV2-normal_sampling_layer_3/random_normal/mul:z:03normal_sampling_layer_3/random_normal/mean:output:0*
T0*0
_output_shapes
:??????????????????2'
%normal_sampling_layer_3/random_normal?
normal_sampling_layer_3/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
normal_sampling_layer_3/mul/x?
normal_sampling_layer_3/mulMul&normal_sampling_layer_3/mul/x:output:0(normal_sampling_layer_3/unstack:output:1*
T0*'
_output_shapes
:?????????2
normal_sampling_layer_3/mul?
normal_sampling_layer_3/ExpExpnormal_sampling_layer_3/mul:z:0*
T0*'
_output_shapes
:?????????2
normal_sampling_layer_3/Exp?
normal_sampling_layer_3/mul_1Mulnormal_sampling_layer_3/Exp:y:0)normal_sampling_layer_3/random_normal:z:0*
T0*'
_output_shapes
:?????????2
normal_sampling_layer_3/mul_1?
normal_sampling_layer_3/addAddV2(normal_sampling_layer_3/unstack:output:0!normal_sampling_layer_3/mul_1:z:0*
T0*'
_output_shapes
:?????????2
normal_sampling_layer_3/add?
*sequential_7/dense_7/MatMul/ReadVariableOpReadVariableOp3sequential_7_dense_7_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02,
*sequential_7/dense_7/MatMul/ReadVariableOp?
sequential_7/dense_7/MatMulMatMulnormal_sampling_layer_3/add:z:02sequential_7/dense_7/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential_7/dense_7/MatMul?
+sequential_7/dense_7/BiasAdd/ReadVariableOpReadVariableOp4sequential_7_dense_7_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02-
+sequential_7/dense_7/BiasAdd/ReadVariableOp?
sequential_7/dense_7/BiasAddBiasAdd%sequential_7/dense_7/MatMul:product:03sequential_7/dense_7/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential_7/dense_7/BiasAdd?
sequential_7/dense_7/ReluRelu%sequential_7/dense_7/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
sequential_7/dense_7/Relu?
sequential_7/reshape_14/ShapeShape'sequential_7/dense_7/Relu:activations:0*
T0*
_output_shapes
:2
sequential_7/reshape_14/Shape?
+sequential_7/reshape_14/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2-
+sequential_7/reshape_14/strided_slice/stack?
-sequential_7/reshape_14/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_7/reshape_14/strided_slice/stack_1?
-sequential_7/reshape_14/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_7/reshape_14/strided_slice/stack_2?
%sequential_7/reshape_14/strided_sliceStridedSlice&sequential_7/reshape_14/Shape:output:04sequential_7/reshape_14/strided_slice/stack:output:06sequential_7/reshape_14/strided_slice/stack_1:output:06sequential_7/reshape_14/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%sequential_7/reshape_14/strided_slice?
'sequential_7/reshape_14/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2)
'sequential_7/reshape_14/Reshape/shape/1?
'sequential_7/reshape_14/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2)
'sequential_7/reshape_14/Reshape/shape/2?
'sequential_7/reshape_14/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B : 2)
'sequential_7/reshape_14/Reshape/shape/3?
%sequential_7/reshape_14/Reshape/shapePack.sequential_7/reshape_14/strided_slice:output:00sequential_7/reshape_14/Reshape/shape/1:output:00sequential_7/reshape_14/Reshape/shape/2:output:00sequential_7/reshape_14/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2'
%sequential_7/reshape_14/Reshape/shape?
sequential_7/reshape_14/ReshapeReshape'sequential_7/dense_7/Relu:activations:0.sequential_7/reshape_14/Reshape/shape:output:0*
T0*/
_output_shapes
:????????? 2!
sequential_7/reshape_14/Reshape?
%sequential_7/conv2d_transpose_9/ShapeShape(sequential_7/reshape_14/Reshape:output:0*
T0*
_output_shapes
:2'
%sequential_7/conv2d_transpose_9/Shape?
3sequential_7/conv2d_transpose_9/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 25
3sequential_7/conv2d_transpose_9/strided_slice/stack?
5sequential_7/conv2d_transpose_9/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential_7/conv2d_transpose_9/strided_slice/stack_1?
5sequential_7/conv2d_transpose_9/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential_7/conv2d_transpose_9/strided_slice/stack_2?
-sequential_7/conv2d_transpose_9/strided_sliceStridedSlice.sequential_7/conv2d_transpose_9/Shape:output:0<sequential_7/conv2d_transpose_9/strided_slice/stack:output:0>sequential_7/conv2d_transpose_9/strided_slice/stack_1:output:0>sequential_7/conv2d_transpose_9/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2/
-sequential_7/conv2d_transpose_9/strided_slice?
'sequential_7/conv2d_transpose_9/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2)
'sequential_7/conv2d_transpose_9/stack/1?
'sequential_7/conv2d_transpose_9/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2)
'sequential_7/conv2d_transpose_9/stack/2?
'sequential_7/conv2d_transpose_9/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@2)
'sequential_7/conv2d_transpose_9/stack/3?
%sequential_7/conv2d_transpose_9/stackPack6sequential_7/conv2d_transpose_9/strided_slice:output:00sequential_7/conv2d_transpose_9/stack/1:output:00sequential_7/conv2d_transpose_9/stack/2:output:00sequential_7/conv2d_transpose_9/stack/3:output:0*
N*
T0*
_output_shapes
:2'
%sequential_7/conv2d_transpose_9/stack?
5sequential_7/conv2d_transpose_9/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 27
5sequential_7/conv2d_transpose_9/strided_slice_1/stack?
7sequential_7/conv2d_transpose_9/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:29
7sequential_7/conv2d_transpose_9/strided_slice_1/stack_1?
7sequential_7/conv2d_transpose_9/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:29
7sequential_7/conv2d_transpose_9/strided_slice_1/stack_2?
/sequential_7/conv2d_transpose_9/strided_slice_1StridedSlice.sequential_7/conv2d_transpose_9/stack:output:0>sequential_7/conv2d_transpose_9/strided_slice_1/stack:output:0@sequential_7/conv2d_transpose_9/strided_slice_1/stack_1:output:0@sequential_7/conv2d_transpose_9/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask21
/sequential_7/conv2d_transpose_9/strided_slice_1?
?sequential_7/conv2d_transpose_9/conv2d_transpose/ReadVariableOpReadVariableOpHsequential_7_conv2d_transpose_9_conv2d_transpose_readvariableop_resource*&
_output_shapes
:@ *
dtype02A
?sequential_7/conv2d_transpose_9/conv2d_transpose/ReadVariableOp?
0sequential_7/conv2d_transpose_9/conv2d_transposeConv2DBackpropInput.sequential_7/conv2d_transpose_9/stack:output:0Gsequential_7/conv2d_transpose_9/conv2d_transpose/ReadVariableOp:value:0(sequential_7/reshape_14/Reshape:output:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
22
0sequential_7/conv2d_transpose_9/conv2d_transpose?
6sequential_7/conv2d_transpose_9/BiasAdd/ReadVariableOpReadVariableOp?sequential_7_conv2d_transpose_9_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype028
6sequential_7/conv2d_transpose_9/BiasAdd/ReadVariableOp?
'sequential_7/conv2d_transpose_9/BiasAddBiasAdd9sequential_7/conv2d_transpose_9/conv2d_transpose:output:0>sequential_7/conv2d_transpose_9/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2)
'sequential_7/conv2d_transpose_9/BiasAdd?
$sequential_7/conv2d_transpose_9/ReluRelu0sequential_7/conv2d_transpose_9/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2&
$sequential_7/conv2d_transpose_9/Relu?
&sequential_7/conv2d_transpose_10/ShapeShape2sequential_7/conv2d_transpose_9/Relu:activations:0*
T0*
_output_shapes
:2(
&sequential_7/conv2d_transpose_10/Shape?
4sequential_7/conv2d_transpose_10/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 26
4sequential_7/conv2d_transpose_10/strided_slice/stack?
6sequential_7/conv2d_transpose_10/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:28
6sequential_7/conv2d_transpose_10/strided_slice/stack_1?
6sequential_7/conv2d_transpose_10/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:28
6sequential_7/conv2d_transpose_10/strided_slice/stack_2?
.sequential_7/conv2d_transpose_10/strided_sliceStridedSlice/sequential_7/conv2d_transpose_10/Shape:output:0=sequential_7/conv2d_transpose_10/strided_slice/stack:output:0?sequential_7/conv2d_transpose_10/strided_slice/stack_1:output:0?sequential_7/conv2d_transpose_10/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask20
.sequential_7/conv2d_transpose_10/strided_slice?
(sequential_7/conv2d_transpose_10/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2*
(sequential_7/conv2d_transpose_10/stack/1?
(sequential_7/conv2d_transpose_10/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2*
(sequential_7/conv2d_transpose_10/stack/2?
(sequential_7/conv2d_transpose_10/stack/3Const*
_output_shapes
: *
dtype0*
value	B : 2*
(sequential_7/conv2d_transpose_10/stack/3?
&sequential_7/conv2d_transpose_10/stackPack7sequential_7/conv2d_transpose_10/strided_slice:output:01sequential_7/conv2d_transpose_10/stack/1:output:01sequential_7/conv2d_transpose_10/stack/2:output:01sequential_7/conv2d_transpose_10/stack/3:output:0*
N*
T0*
_output_shapes
:2(
&sequential_7/conv2d_transpose_10/stack?
6sequential_7/conv2d_transpose_10/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 28
6sequential_7/conv2d_transpose_10/strided_slice_1/stack?
8sequential_7/conv2d_transpose_10/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2:
8sequential_7/conv2d_transpose_10/strided_slice_1/stack_1?
8sequential_7/conv2d_transpose_10/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2:
8sequential_7/conv2d_transpose_10/strided_slice_1/stack_2?
0sequential_7/conv2d_transpose_10/strided_slice_1StridedSlice/sequential_7/conv2d_transpose_10/stack:output:0?sequential_7/conv2d_transpose_10/strided_slice_1/stack:output:0Asequential_7/conv2d_transpose_10/strided_slice_1/stack_1:output:0Asequential_7/conv2d_transpose_10/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask22
0sequential_7/conv2d_transpose_10/strided_slice_1?
@sequential_7/conv2d_transpose_10/conv2d_transpose/ReadVariableOpReadVariableOpIsequential_7_conv2d_transpose_10_conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype02B
@sequential_7/conv2d_transpose_10/conv2d_transpose/ReadVariableOp?
1sequential_7/conv2d_transpose_10/conv2d_transposeConv2DBackpropInput/sequential_7/conv2d_transpose_10/stack:output:0Hsequential_7/conv2d_transpose_10/conv2d_transpose/ReadVariableOp:value:02sequential_7/conv2d_transpose_9/Relu:activations:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
23
1sequential_7/conv2d_transpose_10/conv2d_transpose?
7sequential_7/conv2d_transpose_10/BiasAdd/ReadVariableOpReadVariableOp@sequential_7_conv2d_transpose_10_biasadd_readvariableop_resource*
_output_shapes
: *
dtype029
7sequential_7/conv2d_transpose_10/BiasAdd/ReadVariableOp?
(sequential_7/conv2d_transpose_10/BiasAddBiasAdd:sequential_7/conv2d_transpose_10/conv2d_transpose:output:0?sequential_7/conv2d_transpose_10/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2*
(sequential_7/conv2d_transpose_10/BiasAdd?
%sequential_7/conv2d_transpose_10/ReluRelu1sequential_7/conv2d_transpose_10/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2'
%sequential_7/conv2d_transpose_10/Relu?
&sequential_7/conv2d_transpose_11/ShapeShape3sequential_7/conv2d_transpose_10/Relu:activations:0*
T0*
_output_shapes
:2(
&sequential_7/conv2d_transpose_11/Shape?
4sequential_7/conv2d_transpose_11/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 26
4sequential_7/conv2d_transpose_11/strided_slice/stack?
6sequential_7/conv2d_transpose_11/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:28
6sequential_7/conv2d_transpose_11/strided_slice/stack_1?
6sequential_7/conv2d_transpose_11/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:28
6sequential_7/conv2d_transpose_11/strided_slice/stack_2?
.sequential_7/conv2d_transpose_11/strided_sliceStridedSlice/sequential_7/conv2d_transpose_11/Shape:output:0=sequential_7/conv2d_transpose_11/strided_slice/stack:output:0?sequential_7/conv2d_transpose_11/strided_slice/stack_1:output:0?sequential_7/conv2d_transpose_11/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask20
.sequential_7/conv2d_transpose_11/strided_slice?
(sequential_7/conv2d_transpose_11/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2*
(sequential_7/conv2d_transpose_11/stack/1?
(sequential_7/conv2d_transpose_11/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2*
(sequential_7/conv2d_transpose_11/stack/2?
(sequential_7/conv2d_transpose_11/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2*
(sequential_7/conv2d_transpose_11/stack/3?
&sequential_7/conv2d_transpose_11/stackPack7sequential_7/conv2d_transpose_11/strided_slice:output:01sequential_7/conv2d_transpose_11/stack/1:output:01sequential_7/conv2d_transpose_11/stack/2:output:01sequential_7/conv2d_transpose_11/stack/3:output:0*
N*
T0*
_output_shapes
:2(
&sequential_7/conv2d_transpose_11/stack?
6sequential_7/conv2d_transpose_11/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 28
6sequential_7/conv2d_transpose_11/strided_slice_1/stack?
8sequential_7/conv2d_transpose_11/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2:
8sequential_7/conv2d_transpose_11/strided_slice_1/stack_1?
8sequential_7/conv2d_transpose_11/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2:
8sequential_7/conv2d_transpose_11/strided_slice_1/stack_2?
0sequential_7/conv2d_transpose_11/strided_slice_1StridedSlice/sequential_7/conv2d_transpose_11/stack:output:0?sequential_7/conv2d_transpose_11/strided_slice_1/stack:output:0Asequential_7/conv2d_transpose_11/strided_slice_1/stack_1:output:0Asequential_7/conv2d_transpose_11/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask22
0sequential_7/conv2d_transpose_11/strided_slice_1?
@sequential_7/conv2d_transpose_11/conv2d_transpose/ReadVariableOpReadVariableOpIsequential_7_conv2d_transpose_11_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype02B
@sequential_7/conv2d_transpose_11/conv2d_transpose/ReadVariableOp?
1sequential_7/conv2d_transpose_11/conv2d_transposeConv2DBackpropInput/sequential_7/conv2d_transpose_11/stack:output:0Hsequential_7/conv2d_transpose_11/conv2d_transpose/ReadVariableOp:value:03sequential_7/conv2d_transpose_10/Relu:activations:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
23
1sequential_7/conv2d_transpose_11/conv2d_transpose?
7sequential_7/conv2d_transpose_11/BiasAdd/ReadVariableOpReadVariableOp@sequential_7_conv2d_transpose_11_biasadd_readvariableop_resource*
_output_shapes
:*
dtype029
7sequential_7/conv2d_transpose_11/BiasAdd/ReadVariableOp?
(sequential_7/conv2d_transpose_11/BiasAddBiasAdd:sequential_7/conv2d_transpose_11/conv2d_transpose:output:0?sequential_7/conv2d_transpose_11/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2*
(sequential_7/conv2d_transpose_11/BiasAdd?
(sequential_7/conv2d_transpose_11/SigmoidSigmoid1sequential_7/conv2d_transpose_11/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2*
(sequential_7/conv2d_transpose_11/Sigmoid?
sequential_7/reshape_15/ShapeShape,sequential_7/conv2d_transpose_11/Sigmoid:y:0*
T0*
_output_shapes
:2
sequential_7/reshape_15/Shape?
+sequential_7/reshape_15/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2-
+sequential_7/reshape_15/strided_slice/stack?
-sequential_7/reshape_15/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_7/reshape_15/strided_slice/stack_1?
-sequential_7/reshape_15/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_7/reshape_15/strided_slice/stack_2?
%sequential_7/reshape_15/strided_sliceStridedSlice&sequential_7/reshape_15/Shape:output:04sequential_7/reshape_15/strided_slice/stack:output:06sequential_7/reshape_15/strided_slice/stack_1:output:06sequential_7/reshape_15/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%sequential_7/reshape_15/strided_slice?
'sequential_7/reshape_15/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2)
'sequential_7/reshape_15/Reshape/shape/1?
'sequential_7/reshape_15/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2)
'sequential_7/reshape_15/Reshape/shape/2?
'sequential_7/reshape_15/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2)
'sequential_7/reshape_15/Reshape/shape/3?
%sequential_7/reshape_15/Reshape/shapePack.sequential_7/reshape_15/strided_slice:output:00sequential_7/reshape_15/Reshape/shape/1:output:00sequential_7/reshape_15/Reshape/shape/2:output:00sequential_7/reshape_15/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2'
%sequential_7/reshape_15/Reshape/shape?
sequential_7/reshape_15/ReshapeReshape,sequential_7/conv2d_transpose_11/Sigmoid:y:0.sequential_7/reshape_15/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????2!
sequential_7/reshape_15/Reshape?
IdentityIdentity(sequential_7/reshape_15/Reshape:output:0^NoOp*
T0*/
_output_shapes
:?????????2

Identity?
NoOpNoOp-^sequential_6/conv2d_6/BiasAdd/ReadVariableOp,^sequential_6/conv2d_6/Conv2D/ReadVariableOp-^sequential_6/conv2d_7/BiasAdd/ReadVariableOp,^sequential_6/conv2d_7/Conv2D/ReadVariableOp,^sequential_6/dense_6/BiasAdd/ReadVariableOp+^sequential_6/dense_6/MatMul/ReadVariableOp8^sequential_7/conv2d_transpose_10/BiasAdd/ReadVariableOpA^sequential_7/conv2d_transpose_10/conv2d_transpose/ReadVariableOp8^sequential_7/conv2d_transpose_11/BiasAdd/ReadVariableOpA^sequential_7/conv2d_transpose_11/conv2d_transpose/ReadVariableOp7^sequential_7/conv2d_transpose_9/BiasAdd/ReadVariableOp@^sequential_7/conv2d_transpose_9/conv2d_transpose/ReadVariableOp,^sequential_7/dense_7/BiasAdd/ReadVariableOp+^sequential_7/dense_7/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:?????????: : : : : : : : : : : : : : 2\
,sequential_6/conv2d_6/BiasAdd/ReadVariableOp,sequential_6/conv2d_6/BiasAdd/ReadVariableOp2Z
+sequential_6/conv2d_6/Conv2D/ReadVariableOp+sequential_6/conv2d_6/Conv2D/ReadVariableOp2\
,sequential_6/conv2d_7/BiasAdd/ReadVariableOp,sequential_6/conv2d_7/BiasAdd/ReadVariableOp2Z
+sequential_6/conv2d_7/Conv2D/ReadVariableOp+sequential_6/conv2d_7/Conv2D/ReadVariableOp2Z
+sequential_6/dense_6/BiasAdd/ReadVariableOp+sequential_6/dense_6/BiasAdd/ReadVariableOp2X
*sequential_6/dense_6/MatMul/ReadVariableOp*sequential_6/dense_6/MatMul/ReadVariableOp2r
7sequential_7/conv2d_transpose_10/BiasAdd/ReadVariableOp7sequential_7/conv2d_transpose_10/BiasAdd/ReadVariableOp2?
@sequential_7/conv2d_transpose_10/conv2d_transpose/ReadVariableOp@sequential_7/conv2d_transpose_10/conv2d_transpose/ReadVariableOp2r
7sequential_7/conv2d_transpose_11/BiasAdd/ReadVariableOp7sequential_7/conv2d_transpose_11/BiasAdd/ReadVariableOp2?
@sequential_7/conv2d_transpose_11/conv2d_transpose/ReadVariableOp@sequential_7/conv2d_transpose_11/conv2d_transpose/ReadVariableOp2p
6sequential_7/conv2d_transpose_9/BiasAdd/ReadVariableOp6sequential_7/conv2d_transpose_9/BiasAdd/ReadVariableOp2?
?sequential_7/conv2d_transpose_9/conv2d_transpose/ReadVariableOp?sequential_7/conv2d_transpose_9/conv2d_transpose/ReadVariableOp2Z
+sequential_7/dense_7/BiasAdd/ReadVariableOp+sequential_7/dense_7/BiasAdd/ReadVariableOp2X
*sequential_7/dense_7/MatMul/ReadVariableOp*sequential_7/dense_7/MatMul/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?!
?
G__inference_sequential_7_layer_call_and_return_conditional_losses_61804
input_14 
dense_7_61781:	?
dense_7_61783:	?2
conv2d_transpose_9_61787:@ &
conv2d_transpose_9_61789:@3
conv2d_transpose_10_61792: @'
conv2d_transpose_10_61794: 3
conv2d_transpose_11_61797: '
conv2d_transpose_11_61799:
identity??+conv2d_transpose_10/StatefulPartitionedCall?+conv2d_transpose_11/StatefulPartitionedCall?*conv2d_transpose_9/StatefulPartitionedCall?dense_7/StatefulPartitionedCall?
dense_7/StatefulPartitionedCallStatefulPartitionedCallinput_14dense_7_61781dense_7_61783*
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
B__inference_dense_7_layer_call_and_return_conditional_losses_614952!
dense_7/StatefulPartitionedCall?
reshape_14/PartitionedCallPartitionedCall(dense_7/StatefulPartitionedCall:output:0*
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
E__inference_reshape_14_layer_call_and_return_conditional_losses_615152
reshape_14/PartitionedCall?
*conv2d_transpose_9/StatefulPartitionedCallStatefulPartitionedCall#reshape_14/PartitionedCall:output:0conv2d_transpose_9_61787conv2d_transpose_9_61789*
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
M__inference_conv2d_transpose_9_layer_call_and_return_conditional_losses_615402,
*conv2d_transpose_9/StatefulPartitionedCall?
+conv2d_transpose_10/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_9/StatefulPartitionedCall:output:0conv2d_transpose_10_61792conv2d_transpose_10_61794*
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
N__inference_conv2d_transpose_10_layer_call_and_return_conditional_losses_615692-
+conv2d_transpose_10/StatefulPartitionedCall?
+conv2d_transpose_11/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_10/StatefulPartitionedCall:output:0conv2d_transpose_11_61797conv2d_transpose_11_61799*
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
N__inference_conv2d_transpose_11_layer_call_and_return_conditional_losses_615982-
+conv2d_transpose_11/StatefulPartitionedCall?
reshape_15/PartitionedCallPartitionedCall4conv2d_transpose_11/StatefulPartitionedCall:output:0*
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
E__inference_reshape_15_layer_call_and_return_conditional_losses_616182
reshape_15/PartitionedCall?
IdentityIdentity#reshape_15/PartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????2

Identity?
NoOpNoOp,^conv2d_transpose_10/StatefulPartitionedCall,^conv2d_transpose_11/StatefulPartitionedCall+^conv2d_transpose_9/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : : : : : 2Z
+conv2d_transpose_10/StatefulPartitionedCall+conv2d_transpose_10/StatefulPartitionedCall2Z
+conv2d_transpose_11/StatefulPartitionedCall+conv2d_transpose_11/StatefulPartitionedCall2X
*conv2d_transpose_9/StatefulPartitionedCall*conv2d_transpose_9/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall:Q M
'
_output_shapes
:?????????
"
_user_specified_name
input_14
?!
?
G__inference_sequential_7_layer_call_and_return_conditional_losses_61621

inputs 
dense_7_61496:	?
dense_7_61498:	?2
conv2d_transpose_9_61541:@ &
conv2d_transpose_9_61543:@3
conv2d_transpose_10_61570: @'
conv2d_transpose_10_61572: 3
conv2d_transpose_11_61599: '
conv2d_transpose_11_61601:
identity??+conv2d_transpose_10/StatefulPartitionedCall?+conv2d_transpose_11/StatefulPartitionedCall?*conv2d_transpose_9/StatefulPartitionedCall?dense_7/StatefulPartitionedCall?
dense_7/StatefulPartitionedCallStatefulPartitionedCallinputsdense_7_61496dense_7_61498*
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
B__inference_dense_7_layer_call_and_return_conditional_losses_614952!
dense_7/StatefulPartitionedCall?
reshape_14/PartitionedCallPartitionedCall(dense_7/StatefulPartitionedCall:output:0*
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
E__inference_reshape_14_layer_call_and_return_conditional_losses_615152
reshape_14/PartitionedCall?
*conv2d_transpose_9/StatefulPartitionedCallStatefulPartitionedCall#reshape_14/PartitionedCall:output:0conv2d_transpose_9_61541conv2d_transpose_9_61543*
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
M__inference_conv2d_transpose_9_layer_call_and_return_conditional_losses_615402,
*conv2d_transpose_9/StatefulPartitionedCall?
+conv2d_transpose_10/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_9/StatefulPartitionedCall:output:0conv2d_transpose_10_61570conv2d_transpose_10_61572*
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
N__inference_conv2d_transpose_10_layer_call_and_return_conditional_losses_615692-
+conv2d_transpose_10/StatefulPartitionedCall?
+conv2d_transpose_11/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_10/StatefulPartitionedCall:output:0conv2d_transpose_11_61599conv2d_transpose_11_61601*
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
N__inference_conv2d_transpose_11_layer_call_and_return_conditional_losses_615982-
+conv2d_transpose_11/StatefulPartitionedCall?
reshape_15/PartitionedCallPartitionedCall4conv2d_transpose_11/StatefulPartitionedCall:output:0*
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
E__inference_reshape_15_layer_call_and_return_conditional_losses_616182
reshape_15/PartitionedCall?
IdentityIdentity#reshape_15/PartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????2

Identity?
NoOpNoOp,^conv2d_transpose_10/StatefulPartitionedCall,^conv2d_transpose_11/StatefulPartitionedCall+^conv2d_transpose_9/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : : : : : 2Z
+conv2d_transpose_10/StatefulPartitionedCall+conv2d_transpose_10/StatefulPartitionedCall2Z
+conv2d_transpose_11/StatefulPartitionedCall+conv2d_transpose_11/StatefulPartitionedCall2X
*conv2d_transpose_9/StatefulPartitionedCall*conv2d_transpose_9/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
N__inference_conv2d_transpose_10_layer_call_and_return_conditional_losses_63216

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
?:
?
G__inference_sequential_6_layer_call_and_return_conditional_losses_62635

inputsA
'conv2d_6_conv2d_readvariableop_resource: 6
(conv2d_6_biasadd_readvariableop_resource: A
'conv2d_7_conv2d_readvariableop_resource: @6
(conv2d_7_biasadd_readvariableop_resource:@9
&dense_6_matmul_readvariableop_resource:	?5
'dense_6_biasadd_readvariableop_resource:
identity??conv2d_6/BiasAdd/ReadVariableOp?conv2d_6/Conv2D/ReadVariableOp?conv2d_7/BiasAdd/ReadVariableOp?conv2d_7/Conv2D/ReadVariableOp?dense_6/BiasAdd/ReadVariableOp?dense_6/MatMul/ReadVariableOpZ
reshape_12/ShapeShapeinputs*
T0*
_output_shapes
:2
reshape_12/Shape?
reshape_12/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
reshape_12/strided_slice/stack?
 reshape_12/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_12/strided_slice/stack_1?
 reshape_12/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_12/strided_slice/stack_2?
reshape_12/strided_sliceStridedSlicereshape_12/Shape:output:0'reshape_12/strided_slice/stack:output:0)reshape_12/strided_slice/stack_1:output:0)reshape_12/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_12/strided_slicez
reshape_12/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_12/Reshape/shape/1z
reshape_12/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_12/Reshape/shape/2z
reshape_12/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_12/Reshape/shape/3?
reshape_12/Reshape/shapePack!reshape_12/strided_slice:output:0#reshape_12/Reshape/shape/1:output:0#reshape_12/Reshape/shape/2:output:0#reshape_12/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
reshape_12/Reshape/shape?
reshape_12/ReshapeReshapeinputs!reshape_12/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????2
reshape_12/Reshape?
conv2d_6/Conv2D/ReadVariableOpReadVariableOp'conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02 
conv2d_6/Conv2D/ReadVariableOp?
conv2d_6/Conv2DConv2Dreshape_12/Reshape:output:0&conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
2
conv2d_6/Conv2D?
conv2d_6/BiasAdd/ReadVariableOpReadVariableOp(conv2d_6_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv2d_6/BiasAdd/ReadVariableOp?
conv2d_6/BiasAddBiasAddconv2d_6/Conv2D:output:0'conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2
conv2d_6/BiasAdd{
conv2d_6/ReluReluconv2d_6/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
conv2d_6/Relu?
conv2d_7/Conv2D/ReadVariableOpReadVariableOp'conv2d_7_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02 
conv2d_7/Conv2D/ReadVariableOp?
conv2d_7/Conv2DConv2Dconv2d_6/Relu:activations:0&conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
2
conv2d_7/Conv2D?
conv2d_7/BiasAdd/ReadVariableOpReadVariableOp(conv2d_7_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_7/BiasAdd/ReadVariableOp?
conv2d_7/BiasAddBiasAddconv2d_7/Conv2D:output:0'conv2d_7/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
conv2d_7/BiasAdd{
conv2d_7/ReluReluconv2d_7/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
conv2d_7/Relus
flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"???? 	  2
flatten_3/Const?
flatten_3/ReshapeReshapeconv2d_7/Relu:activations:0flatten_3/Const:output:0*
T0*(
_output_shapes
:??????????2
flatten_3/Reshape?
dense_6/MatMul/ReadVariableOpReadVariableOp&dense_6_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
dense_6/MatMul/ReadVariableOp?
dense_6/MatMulMatMulflatten_3/Reshape:output:0%dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_6/MatMul?
dense_6/BiasAdd/ReadVariableOpReadVariableOp'dense_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_6/BiasAdd/ReadVariableOp?
dense_6/BiasAddBiasAdddense_6/MatMul:product:0&dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_6/BiasAddl
reshape_13/ShapeShapedense_6/BiasAdd:output:0*
T0*
_output_shapes
:2
reshape_13/Shape?
reshape_13/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
reshape_13/strided_slice/stack?
 reshape_13/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_13/strided_slice/stack_1?
 reshape_13/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_13/strided_slice/stack_2?
reshape_13/strided_sliceStridedSlicereshape_13/Shape:output:0'reshape_13/strided_slice/stack:output:0)reshape_13/strided_slice/stack_1:output:0)reshape_13/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_13/strided_slicez
reshape_13/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_13/Reshape/shape/1z
reshape_13/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_13/Reshape/shape/2?
reshape_13/Reshape/shapePack!reshape_13/strided_slice:output:0#reshape_13/Reshape/shape/1:output:0#reshape_13/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
reshape_13/Reshape/shape?
reshape_13/ReshapeReshapedense_6/BiasAdd:output:0!reshape_13/Reshape/shape:output:0*
T0*+
_output_shapes
:?????????2
reshape_13/Reshapez
IdentityIdentityreshape_13/Reshape:output:0^NoOp*
T0*+
_output_shapes
:?????????2

Identity?
NoOpNoOp ^conv2d_6/BiasAdd/ReadVariableOp^conv2d_6/Conv2D/ReadVariableOp ^conv2d_7/BiasAdd/ReadVariableOp^conv2d_7/Conv2D/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : 2B
conv2d_6/BiasAdd/ReadVariableOpconv2d_6/BiasAdd/ReadVariableOp2@
conv2d_6/Conv2D/ReadVariableOpconv2d_6/Conv2D/ReadVariableOp2B
conv2d_7/BiasAdd/ReadVariableOpconv2d_7/BiasAdd/ReadVariableOp2@
conv2d_7/Conv2D/ReadVariableOpconv2d_7/Conv2D/ReadVariableOp2@
dense_6/BiasAdd/ReadVariableOpdense_6/BiasAdd/ReadVariableOp2>
dense_6/MatMul/ReadVariableOpdense_6/MatMul/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
G__inference_sequential_6_layer_call_and_return_conditional_losses_61191
input_13(
conv2d_6_61173: 
conv2d_6_61175: (
conv2d_7_61178: @
conv2d_7_61180:@ 
dense_6_61184:	?
dense_6_61186:
identity?? conv2d_6/StatefulPartitionedCall? conv2d_7/StatefulPartitionedCall?dense_6/StatefulPartitionedCall?
reshape_12/PartitionedCallPartitionedCallinput_13*
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
E__inference_reshape_12_layer_call_and_return_conditional_losses_609572
reshape_12/PartitionedCall?
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCall#reshape_12/PartitionedCall:output:0conv2d_6_61173conv2d_6_61175*
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
C__inference_conv2d_6_layer_call_and_return_conditional_losses_609702"
 conv2d_6/StatefulPartitionedCall?
 conv2d_7/StatefulPartitionedCallStatefulPartitionedCall)conv2d_6/StatefulPartitionedCall:output:0conv2d_7_61178conv2d_7_61180*
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
C__inference_conv2d_7_layer_call_and_return_conditional_losses_609872"
 conv2d_7/StatefulPartitionedCall?
flatten_3/PartitionedCallPartitionedCall)conv2d_7/StatefulPartitionedCall:output:0*
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
D__inference_flatten_3_layer_call_and_return_conditional_losses_609992
flatten_3/PartitionedCall?
dense_6/StatefulPartitionedCallStatefulPartitionedCall"flatten_3/PartitionedCall:output:0dense_6_61184dense_6_61186*
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
B__inference_dense_6_layer_call_and_return_conditional_losses_610112!
dense_6/StatefulPartitionedCall?
reshape_13/PartitionedCallPartitionedCall(dense_6/StatefulPartitionedCall:output:0*
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
E__inference_reshape_13_layer_call_and_return_conditional_losses_610302
reshape_13/PartitionedCall?
IdentityIdentity#reshape_13/PartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????2

Identity?
NoOpNoOp!^conv2d_6/StatefulPartitionedCall!^conv2d_7/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : 2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall2D
 conv2d_7/StatefulPartitionedCall conv2d_7/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall:Y U
/
_output_shapes
:?????????
"
_user_specified_name
input_13
?
?
#__inference_signature_wrapper_62190
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
 __inference__wrapped_model_609362
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
a
E__inference_reshape_13_layer_call_and_return_conditional_losses_63038

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
?
'__inference_dense_7_layer_call_fn_63063

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
B__inference_dense_7_layer_call_and_return_conditional_losses_614952
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
?
,__inference_sequential_6_layer_call_fn_61048
input_13!
unknown: 
	unknown_0: #
	unknown_1: @
	unknown_2:@
	unknown_3:	?
	unknown_4:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_13unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
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
G__inference_sequential_6_layer_call_and_return_conditional_losses_610332
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
input_13
?
`
D__inference_flatten_3_layer_call_and_return_conditional_losses_60999

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
?
?
3__inference_conv2d_transpose_10_layer_call_fn_63225

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
N__inference_conv2d_transpose_10_layer_call_and_return_conditional_losses_613392
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
F
*__inference_reshape_13_layer_call_fn_63043

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
E__inference_reshape_13_layer_call_and_return_conditional_losses_610302
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

?
,__inference_sequential_7_layer_call_fn_61778
input_14
unknown:	?
	unknown_0:	?#
	unknown_1:@ 
	unknown_2:@#
	unknown_3: @
	unknown_4: #
	unknown_5: 
	unknown_6:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_14unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
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
G__inference_sequential_7_layer_call_and_return_conditional_losses_617382
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
input_14"?L
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
StatefulPartitionedCall:0?????????tensorflow/serving/predict:ȟ
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
):' 2conv2d_6/kernel
: 2conv2d_6/bias
):' @2conv2d_7/kernel
:@2conv2d_7/bias
!:	?2dense_6/kernel
:2dense_6/bias
!:	?2dense_7/kernel
:?2dense_7/bias
3:1@ 2conv2d_transpose_9/kernel
%:#@2conv2d_transpose_9/bias
4:2 @2conv2d_transpose_10/kernel
&:$ 2conv2d_transpose_10/bias
4:2 2conv2d_transpose_11/kernel
&:$2conv2d_transpose_11/bias
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
.:, 2Adam/conv2d_6/kernel/m
 : 2Adam/conv2d_6/bias/m
.:, @2Adam/conv2d_7/kernel/m
 :@2Adam/conv2d_7/bias/m
&:$	?2Adam/dense_6/kernel/m
:2Adam/dense_6/bias/m
&:$	?2Adam/dense_7/kernel/m
 :?2Adam/dense_7/bias/m
8:6@ 2 Adam/conv2d_transpose_9/kernel/m
*:(@2Adam/conv2d_transpose_9/bias/m
9:7 @2!Adam/conv2d_transpose_10/kernel/m
+:) 2Adam/conv2d_transpose_10/bias/m
9:7 2!Adam/conv2d_transpose_11/kernel/m
+:)2Adam/conv2d_transpose_11/bias/m
.:, 2Adam/conv2d_6/kernel/v
 : 2Adam/conv2d_6/bias/v
.:, @2Adam/conv2d_7/kernel/v
 :@2Adam/conv2d_7/bias/v
&:$	?2Adam/dense_6/kernel/v
:2Adam/dense_6/bias/v
&:$	?2Adam/dense_7/kernel/v
 :?2Adam/dense_7/bias/v
8:6@ 2 Adam/conv2d_transpose_9/kernel/v
*:(@2Adam/conv2d_transpose_9/bias/v
9:7 @2!Adam/conv2d_transpose_10/kernel/v
+:) 2Adam/conv2d_transpose_10/bias/v
9:7 2!Adam/conv2d_transpose_11/kernel/v
+:)2Adam/conv2d_transpose_11/bias/v
?B?
 __inference__wrapped_model_60936input_1"?
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
T__inference_variational_autoencoder_3_layer_call_and_return_conditional_losses_62324
T__inference_variational_autoencoder_3_layer_call_and_return_conditional_losses_62479
T__inference_variational_autoencoder_3_layer_call_and_return_conditional_losses_62114
T__inference_variational_autoencoder_3_layer_call_and_return_conditional_losses_62149?
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
9__inference_variational_autoencoder_3_layer_call_fn_61907
9__inference_variational_autoencoder_3_layer_call_fn_62512
9__inference_variational_autoencoder_3_layer_call_fn_62545
9__inference_variational_autoencoder_3_layer_call_fn_62079?
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
G__inference_sequential_6_layer_call_and_return_conditional_losses_62590
G__inference_sequential_6_layer_call_and_return_conditional_losses_62635
G__inference_sequential_6_layer_call_and_return_conditional_losses_61191
G__inference_sequential_6_layer_call_and_return_conditional_losses_61213?
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
,__inference_sequential_6_layer_call_fn_61048
,__inference_sequential_6_layer_call_fn_62652
,__inference_sequential_6_layer_call_fn_62669
,__inference_sequential_6_layer_call_fn_61169?
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
G__inference_sequential_7_layer_call_and_return_conditional_losses_62760
G__inference_sequential_7_layer_call_and_return_conditional_losses_62851
G__inference_sequential_7_layer_call_and_return_conditional_losses_61804
G__inference_sequential_7_layer_call_and_return_conditional_losses_61830?
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
,__inference_sequential_7_layer_call_fn_61640
,__inference_sequential_7_layer_call_fn_62872
,__inference_sequential_7_layer_call_fn_62893
,__inference_sequential_7_layer_call_fn_61778?
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
R__inference_normal_sampling_layer_3_layer_call_and_return_conditional_losses_62899
R__inference_normal_sampling_layer_3_layer_call_and_return_conditional_losses_62926?
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
7__inference_normal_sampling_layer_3_layer_call_fn_62931
7__inference_normal_sampling_layer_3_layer_call_fn_62936?
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
#__inference_signature_wrapper_62190input_1"?
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
E__inference_reshape_12_layer_call_and_return_conditional_losses_62950?
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
*__inference_reshape_12_layer_call_fn_62955?
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
C__inference_conv2d_6_layer_call_and_return_conditional_losses_62966?
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
(__inference_conv2d_6_layer_call_fn_62975?
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
C__inference_conv2d_7_layer_call_and_return_conditional_losses_62986?
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
(__inference_conv2d_7_layer_call_fn_62995?
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
D__inference_flatten_3_layer_call_and_return_conditional_losses_63001?
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
)__inference_flatten_3_layer_call_fn_63006?
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
B__inference_dense_6_layer_call_and_return_conditional_losses_63016?
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
'__inference_dense_6_layer_call_fn_63025?
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
E__inference_reshape_13_layer_call_and_return_conditional_losses_63038?
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
*__inference_reshape_13_layer_call_fn_63043?
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
B__inference_dense_7_layer_call_and_return_conditional_losses_63054?
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
'__inference_dense_7_layer_call_fn_63063?
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
E__inference_reshape_14_layer_call_and_return_conditional_losses_63077?
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
*__inference_reshape_14_layer_call_fn_63082?
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
M__inference_conv2d_transpose_9_layer_call_and_return_conditional_losses_63116
M__inference_conv2d_transpose_9_layer_call_and_return_conditional_losses_63140?
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
2__inference_conv2d_transpose_9_layer_call_fn_63149
2__inference_conv2d_transpose_9_layer_call_fn_63158?
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
N__inference_conv2d_transpose_10_layer_call_and_return_conditional_losses_63192
N__inference_conv2d_transpose_10_layer_call_and_return_conditional_losses_63216?
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
3__inference_conv2d_transpose_10_layer_call_fn_63225
3__inference_conv2d_transpose_10_layer_call_fn_63234?
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
N__inference_conv2d_transpose_11_layer_call_and_return_conditional_losses_63268
N__inference_conv2d_transpose_11_layer_call_and_return_conditional_losses_63292?
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
3__inference_conv2d_transpose_11_layer_call_fn_63301
3__inference_conv2d_transpose_11_layer_call_fn_63310?
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
E__inference_reshape_15_layer_call_and_return_conditional_losses_63324?
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
*__inference_reshape_15_layer_call_fn_63329?
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
 __inference__wrapped_model_60936?789:;<=>?@ABCD8?5
.?+
)?&
input_1?????????
? ";?8
6
output_1*?'
output_1??????????
C__inference_conv2d_6_layer_call_and_return_conditional_losses_62966l787?4
-?*
(?%
inputs?????????
? "-?*
#? 
0????????? 
? ?
(__inference_conv2d_6_layer_call_fn_62975_787?4
-?*
(?%
inputs?????????
? " ?????????? ?
C__inference_conv2d_7_layer_call_and_return_conditional_losses_62986l9:7?4
-?*
(?%
inputs????????? 
? "-?*
#? 
0?????????@
? ?
(__inference_conv2d_7_layer_call_fn_62995_9:7?4
-?*
(?%
inputs????????? 
? " ??????????@?
N__inference_conv2d_transpose_10_layer_call_and_return_conditional_losses_63192?ABI?F
??<
:?7
inputs+???????????????????????????@
? "??<
5?2
0+??????????????????????????? 
? ?
N__inference_conv2d_transpose_10_layer_call_and_return_conditional_losses_63216lAB7?4
-?*
(?%
inputs?????????@
? "-?*
#? 
0????????? 
? ?
3__inference_conv2d_transpose_10_layer_call_fn_63225?ABI?F
??<
:?7
inputs+???????????????????????????@
? "2?/+??????????????????????????? ?
3__inference_conv2d_transpose_10_layer_call_fn_63234_AB7?4
-?*
(?%
inputs?????????@
? " ?????????? ?
N__inference_conv2d_transpose_11_layer_call_and_return_conditional_losses_63268?CDI?F
??<
:?7
inputs+??????????????????????????? 
? "??<
5?2
0+???????????????????????????
? ?
N__inference_conv2d_transpose_11_layer_call_and_return_conditional_losses_63292lCD7?4
-?*
(?%
inputs????????? 
? "-?*
#? 
0?????????
? ?
3__inference_conv2d_transpose_11_layer_call_fn_63301?CDI?F
??<
:?7
inputs+??????????????????????????? 
? "2?/+????????????????????????????
3__inference_conv2d_transpose_11_layer_call_fn_63310_CD7?4
-?*
(?%
inputs????????? 
? " ???????????
M__inference_conv2d_transpose_9_layer_call_and_return_conditional_losses_63116??@I?F
??<
:?7
inputs+??????????????????????????? 
? "??<
5?2
0+???????????????????????????@
? ?
M__inference_conv2d_transpose_9_layer_call_and_return_conditional_losses_63140l?@7?4
-?*
(?%
inputs????????? 
? "-?*
#? 
0?????????@
? ?
2__inference_conv2d_transpose_9_layer_call_fn_63149??@I?F
??<
:?7
inputs+??????????????????????????? 
? "2?/+???????????????????????????@?
2__inference_conv2d_transpose_9_layer_call_fn_63158_?@7?4
-?*
(?%
inputs????????? 
? " ??????????@?
B__inference_dense_6_layer_call_and_return_conditional_losses_63016];<0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????
? {
'__inference_dense_6_layer_call_fn_63025P;<0?-
&?#
!?
inputs??????????
? "???????????
B__inference_dense_7_layer_call_and_return_conditional_losses_63054]=>/?,
%?"
 ?
inputs?????????
? "&?#
?
0??????????
? {
'__inference_dense_7_layer_call_fn_63063P=>/?,
%?"
 ?
inputs?????????
? "????????????
D__inference_flatten_3_layer_call_and_return_conditional_losses_63001a7?4
-?*
(?%
inputs?????????@
? "&?#
?
0??????????
? ?
)__inference_flatten_3_layer_call_fn_63006T7?4
-?*
(?%
inputs?????????@
? "????????????
R__inference_normal_sampling_layer_3_layer_call_and_return_conditional_losses_62899`7?4
-?*
$?!
inputs?????????
p 
? "%?"
?
0?????????
? ?
R__inference_normal_sampling_layer_3_layer_call_and_return_conditional_losses_62926`7?4
-?*
$?!
inputs?????????
p
? "%?"
?
0?????????
? ?
7__inference_normal_sampling_layer_3_layer_call_fn_62931S7?4
-?*
$?!
inputs?????????
p 
? "???????????
7__inference_normal_sampling_layer_3_layer_call_fn_62936S7?4
-?*
$?!
inputs?????????
p
? "???????????
E__inference_reshape_12_layer_call_and_return_conditional_losses_62950h7?4
-?*
(?%
inputs?????????
? "-?*
#? 
0?????????
? ?
*__inference_reshape_12_layer_call_fn_62955[7?4
-?*
(?%
inputs?????????
? " ???????????
E__inference_reshape_13_layer_call_and_return_conditional_losses_63038\/?,
%?"
 ?
inputs?????????
? ")?&
?
0?????????
? }
*__inference_reshape_13_layer_call_fn_63043O/?,
%?"
 ?
inputs?????????
? "???????????
E__inference_reshape_14_layer_call_and_return_conditional_losses_63077a0?-
&?#
!?
inputs??????????
? "-?*
#? 
0????????? 
? ?
*__inference_reshape_14_layer_call_fn_63082T0?-
&?#
!?
inputs??????????
? " ?????????? ?
E__inference_reshape_15_layer_call_and_return_conditional_losses_63324h7?4
-?*
(?%
inputs?????????
? "-?*
#? 
0?????????
? ?
*__inference_reshape_15_layer_call_fn_63329[7?4
-?*
(?%
inputs?????????
? " ???????????
G__inference_sequential_6_layer_call_and_return_conditional_losses_61191v789:;<A?>
7?4
*?'
input_13?????????
p 

 
? ")?&
?
0?????????
? ?
G__inference_sequential_6_layer_call_and_return_conditional_losses_61213v789:;<A?>
7?4
*?'
input_13?????????
p

 
? ")?&
?
0?????????
? ?
G__inference_sequential_6_layer_call_and_return_conditional_losses_62590t789:;<??<
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
G__inference_sequential_6_layer_call_and_return_conditional_losses_62635t789:;<??<
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
,__inference_sequential_6_layer_call_fn_61048i789:;<A?>
7?4
*?'
input_13?????????
p 

 
? "???????????
,__inference_sequential_6_layer_call_fn_61169i789:;<A?>
7?4
*?'
input_13?????????
p

 
? "???????????
,__inference_sequential_6_layer_call_fn_62652g789:;<??<
5?2
(?%
inputs?????????
p 

 
? "???????????
,__inference_sequential_6_layer_call_fn_62669g789:;<??<
5?2
(?%
inputs?????????
p

 
? "???????????
G__inference_sequential_7_layer_call_and_return_conditional_losses_61804t=>?@ABCD9?6
/?,
"?
input_14?????????
p 

 
? "-?*
#? 
0?????????
? ?
G__inference_sequential_7_layer_call_and_return_conditional_losses_61830t=>?@ABCD9?6
/?,
"?
input_14?????????
p

 
? "-?*
#? 
0?????????
? ?
G__inference_sequential_7_layer_call_and_return_conditional_losses_62760r=>?@ABCD7?4
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
G__inference_sequential_7_layer_call_and_return_conditional_losses_62851r=>?@ABCD7?4
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
,__inference_sequential_7_layer_call_fn_61640g=>?@ABCD9?6
/?,
"?
input_14?????????
p 

 
? " ???????????
,__inference_sequential_7_layer_call_fn_61778g=>?@ABCD9?6
/?,
"?
input_14?????????
p

 
? " ???????????
,__inference_sequential_7_layer_call_fn_62872e=>?@ABCD7?4
-?*
 ?
inputs?????????
p 

 
? " ???????????
,__inference_sequential_7_layer_call_fn_62893e=>?@ABCD7?4
-?*
 ?
inputs?????????
p

 
? " ???????????
#__inference_signature_wrapper_62190?789:;<=>?@ABCDC?@
? 
9?6
4
input_1)?&
input_1?????????";?8
6
output_1*?'
output_1??????????
T__inference_variational_autoencoder_3_layer_call_and_return_conditional_losses_62114}789:;<=>?@ABCD<?9
2?/
)?&
input_1?????????
p 
? "-?*
#? 
0?????????
? ?
T__inference_variational_autoencoder_3_layer_call_and_return_conditional_losses_62149}789:;<=>?@ABCD<?9
2?/
)?&
input_1?????????
p
? "-?*
#? 
0?????????
? ?
T__inference_variational_autoencoder_3_layer_call_and_return_conditional_losses_62324|789:;<=>?@ABCD;?8
1?.
(?%
inputs?????????
p 
? "-?*
#? 
0?????????
? ?
T__inference_variational_autoencoder_3_layer_call_and_return_conditional_losses_62479|789:;<=>?@ABCD;?8
1?.
(?%
inputs?????????
p
? "-?*
#? 
0?????????
? ?
9__inference_variational_autoencoder_3_layer_call_fn_61907p789:;<=>?@ABCD<?9
2?/
)?&
input_1?????????
p 
? " ???????????
9__inference_variational_autoencoder_3_layer_call_fn_62079p789:;<=>?@ABCD<?9
2?/
)?&
input_1?????????
p
? " ???????????
9__inference_variational_autoencoder_3_layer_call_fn_62512o789:;<=>?@ABCD;?8
1?.
(?%
inputs?????????
p 
? " ???????????
9__inference_variational_autoencoder_3_layer_call_fn_62545o789:;<=>?@ABCD;?8
1?.
(?%
inputs?????????
p
? " ??????????