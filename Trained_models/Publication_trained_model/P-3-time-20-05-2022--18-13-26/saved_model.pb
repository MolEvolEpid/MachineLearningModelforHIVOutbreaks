Ǯ
??
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( ?
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
FusedBatchNormV3
x"T

scale"U
offset"U	
mean"U
variance"U
y"T

batch_mean"U
batch_variance"U
reserve_space_1"U
reserve_space_2"U
reserve_space_3"U"
Ttype:
2"
Utype:
2"
epsilonfloat%??8"&
exponential_avg_factorfloat%  ??";
data_formatstringNHWC:
NHWCNCHWNDHWCNCDHW"
is_trainingbool(
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
?
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
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
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
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
executor_typestring ??
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.8.02v2.8.0-rc1-32-g3f878cff5b68??
~
Conv_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameConv_1/kernel
w
!Conv_1/kernel/Read/ReadVariableOpReadVariableOpConv_1/kernel*&
_output_shapes
: *
dtype0
n
Conv_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameConv_1/bias
g
Conv_1/bias/Read/ReadVariableOpReadVariableOpConv_1/bias*
_output_shapes
: *
dtype0
z
BatchNorm_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *"
shared_nameBatchNorm_1/gamma
s
%BatchNorm_1/gamma/Read/ReadVariableOpReadVariableOpBatchNorm_1/gamma*
_output_shapes
: *
dtype0
x
BatchNorm_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_nameBatchNorm_1/beta
q
$BatchNorm_1/beta/Read/ReadVariableOpReadVariableOpBatchNorm_1/beta*
_output_shapes
: *
dtype0
?
BatchNorm_1/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameBatchNorm_1/moving_mean

+BatchNorm_1/moving_mean/Read/ReadVariableOpReadVariableOpBatchNorm_1/moving_mean*
_output_shapes
: *
dtype0
?
BatchNorm_1/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_nameBatchNorm_1/moving_variance
?
/BatchNorm_1/moving_variance/Read/ReadVariableOpReadVariableOpBatchNorm_1/moving_variance*
_output_shapes
: *
dtype0
~
Conv_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *
shared_nameConv_2/kernel
w
!Conv_2/kernel/Read/ReadVariableOpReadVariableOpConv_2/kernel*&
_output_shapes
:  *
dtype0
n
Conv_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameConv_2/bias
g
Conv_2/bias/Read/ReadVariableOpReadVariableOpConv_2/bias*
_output_shapes
: *
dtype0
~
Conv_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *
shared_nameConv_3/kernel
w
!Conv_3/kernel/Read/ReadVariableOpReadVariableOpConv_3/kernel*&
_output_shapes
:  *
dtype0
n
Conv_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameConv_3/bias
g
Conv_3/bias/Read/ReadVariableOpReadVariableOpConv_3/bias*
_output_shapes
: *
dtype0
z
BatchNorm_2/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *"
shared_nameBatchNorm_2/gamma
s
%BatchNorm_2/gamma/Read/ReadVariableOpReadVariableOpBatchNorm_2/gamma*
_output_shapes
: *
dtype0
x
BatchNorm_2/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_nameBatchNorm_2/beta
q
$BatchNorm_2/beta/Read/ReadVariableOpReadVariableOpBatchNorm_2/beta*
_output_shapes
: *
dtype0
?
BatchNorm_2/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameBatchNorm_2/moving_mean

+BatchNorm_2/moving_mean/Read/ReadVariableOpReadVariableOpBatchNorm_2/moving_mean*
_output_shapes
: *
dtype0
?
BatchNorm_2/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_nameBatchNorm_2/moving_variance
?
/BatchNorm_2/moving_variance/Read/ReadVariableOpReadVariableOpBatchNorm_2/moving_variance*
_output_shapes
: *
dtype0
~
Conv_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *
shared_nameConv_4/kernel
w
!Conv_4/kernel/Read/ReadVariableOpReadVariableOpConv_4/kernel*&
_output_shapes
:  *
dtype0
n
Conv_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameConv_4/bias
g
Conv_4/bias/Read/ReadVariableOpReadVariableOpConv_4/bias*
_output_shapes
: *
dtype0
x
Dense_0/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: @*
shared_nameDense_0/kernel
q
"Dense_0/kernel/Read/ReadVariableOpReadVariableOpDense_0/kernel*
_output_shapes

: @*
dtype0
p
Dense_0/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameDense_0/bias
i
 Dense_0/bias/Read/ReadVariableOpReadVariableOpDense_0/bias*
_output_shapes
:@*
dtype0
x
Dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*
shared_nameDense_1/kernel
q
"Dense_1/kernel/Read/ReadVariableOpReadVariableOpDense_1/kernel*
_output_shapes

:@@*
dtype0
p
Dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameDense_1/bias
i
 Dense_1/bias/Read/ReadVariableOpReadVariableOpDense_1/bias*
_output_shapes
:@*
dtype0
?
Dense_output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*$
shared_nameDense_output/kernel
{
'Dense_output/kernel/Read/ReadVariableOpReadVariableOpDense_output/kernel*
_output_shapes

:@*
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
?
Adam/Conv_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/Conv_1/kernel/m
?
(Adam/Conv_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/Conv_1/kernel/m*&
_output_shapes
: *
dtype0
|
Adam/Conv_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/Conv_1/bias/m
u
&Adam/Conv_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/Conv_1/bias/m*
_output_shapes
: *
dtype0
?
Adam/BatchNorm_1/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *)
shared_nameAdam/BatchNorm_1/gamma/m
?
,Adam/BatchNorm_1/gamma/m/Read/ReadVariableOpReadVariableOpAdam/BatchNorm_1/gamma/m*
_output_shapes
: *
dtype0
?
Adam/BatchNorm_1/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameAdam/BatchNorm_1/beta/m

+Adam/BatchNorm_1/beta/m/Read/ReadVariableOpReadVariableOpAdam/BatchNorm_1/beta/m*
_output_shapes
: *
dtype0
?
Adam/Conv_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *%
shared_nameAdam/Conv_2/kernel/m
?
(Adam/Conv_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/Conv_2/kernel/m*&
_output_shapes
:  *
dtype0
|
Adam/Conv_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/Conv_2/bias/m
u
&Adam/Conv_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/Conv_2/bias/m*
_output_shapes
: *
dtype0
?
Adam/Conv_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *%
shared_nameAdam/Conv_3/kernel/m
?
(Adam/Conv_3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/Conv_3/kernel/m*&
_output_shapes
:  *
dtype0
|
Adam/Conv_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/Conv_3/bias/m
u
&Adam/Conv_3/bias/m/Read/ReadVariableOpReadVariableOpAdam/Conv_3/bias/m*
_output_shapes
: *
dtype0
?
Adam/BatchNorm_2/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *)
shared_nameAdam/BatchNorm_2/gamma/m
?
,Adam/BatchNorm_2/gamma/m/Read/ReadVariableOpReadVariableOpAdam/BatchNorm_2/gamma/m*
_output_shapes
: *
dtype0
?
Adam/BatchNorm_2/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameAdam/BatchNorm_2/beta/m

+Adam/BatchNorm_2/beta/m/Read/ReadVariableOpReadVariableOpAdam/BatchNorm_2/beta/m*
_output_shapes
: *
dtype0
?
Adam/Conv_4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *%
shared_nameAdam/Conv_4/kernel/m
?
(Adam/Conv_4/kernel/m/Read/ReadVariableOpReadVariableOpAdam/Conv_4/kernel/m*&
_output_shapes
:  *
dtype0
|
Adam/Conv_4/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/Conv_4/bias/m
u
&Adam/Conv_4/bias/m/Read/ReadVariableOpReadVariableOpAdam/Conv_4/bias/m*
_output_shapes
: *
dtype0
?
Adam/Dense_0/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: @*&
shared_nameAdam/Dense_0/kernel/m

)Adam/Dense_0/kernel/m/Read/ReadVariableOpReadVariableOpAdam/Dense_0/kernel/m*
_output_shapes

: @*
dtype0
~
Adam/Dense_0/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*$
shared_nameAdam/Dense_0/bias/m
w
'Adam/Dense_0/bias/m/Read/ReadVariableOpReadVariableOpAdam/Dense_0/bias/m*
_output_shapes
:@*
dtype0
?
Adam/Dense_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*&
shared_nameAdam/Dense_1/kernel/m

)Adam/Dense_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/Dense_1/kernel/m*
_output_shapes

:@@*
dtype0
~
Adam/Dense_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*$
shared_nameAdam/Dense_1/bias/m
w
'Adam/Dense_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/Dense_1/bias/m*
_output_shapes
:@*
dtype0
?
Adam/Dense_output/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*+
shared_nameAdam/Dense_output/kernel/m
?
.Adam/Dense_output/kernel/m/Read/ReadVariableOpReadVariableOpAdam/Dense_output/kernel/m*
_output_shapes

:@*
dtype0
?
Adam/Conv_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/Conv_1/kernel/v
?
(Adam/Conv_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/Conv_1/kernel/v*&
_output_shapes
: *
dtype0
|
Adam/Conv_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/Conv_1/bias/v
u
&Adam/Conv_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/Conv_1/bias/v*
_output_shapes
: *
dtype0
?
Adam/BatchNorm_1/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *)
shared_nameAdam/BatchNorm_1/gamma/v
?
,Adam/BatchNorm_1/gamma/v/Read/ReadVariableOpReadVariableOpAdam/BatchNorm_1/gamma/v*
_output_shapes
: *
dtype0
?
Adam/BatchNorm_1/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameAdam/BatchNorm_1/beta/v

+Adam/BatchNorm_1/beta/v/Read/ReadVariableOpReadVariableOpAdam/BatchNorm_1/beta/v*
_output_shapes
: *
dtype0
?
Adam/Conv_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *%
shared_nameAdam/Conv_2/kernel/v
?
(Adam/Conv_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/Conv_2/kernel/v*&
_output_shapes
:  *
dtype0
|
Adam/Conv_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/Conv_2/bias/v
u
&Adam/Conv_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/Conv_2/bias/v*
_output_shapes
: *
dtype0
?
Adam/Conv_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *%
shared_nameAdam/Conv_3/kernel/v
?
(Adam/Conv_3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/Conv_3/kernel/v*&
_output_shapes
:  *
dtype0
|
Adam/Conv_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/Conv_3/bias/v
u
&Adam/Conv_3/bias/v/Read/ReadVariableOpReadVariableOpAdam/Conv_3/bias/v*
_output_shapes
: *
dtype0
?
Adam/BatchNorm_2/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *)
shared_nameAdam/BatchNorm_2/gamma/v
?
,Adam/BatchNorm_2/gamma/v/Read/ReadVariableOpReadVariableOpAdam/BatchNorm_2/gamma/v*
_output_shapes
: *
dtype0
?
Adam/BatchNorm_2/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameAdam/BatchNorm_2/beta/v

+Adam/BatchNorm_2/beta/v/Read/ReadVariableOpReadVariableOpAdam/BatchNorm_2/beta/v*
_output_shapes
: *
dtype0
?
Adam/Conv_4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *%
shared_nameAdam/Conv_4/kernel/v
?
(Adam/Conv_4/kernel/v/Read/ReadVariableOpReadVariableOpAdam/Conv_4/kernel/v*&
_output_shapes
:  *
dtype0
|
Adam/Conv_4/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/Conv_4/bias/v
u
&Adam/Conv_4/bias/v/Read/ReadVariableOpReadVariableOpAdam/Conv_4/bias/v*
_output_shapes
: *
dtype0
?
Adam/Dense_0/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: @*&
shared_nameAdam/Dense_0/kernel/v

)Adam/Dense_0/kernel/v/Read/ReadVariableOpReadVariableOpAdam/Dense_0/kernel/v*
_output_shapes

: @*
dtype0
~
Adam/Dense_0/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*$
shared_nameAdam/Dense_0/bias/v
w
'Adam/Dense_0/bias/v/Read/ReadVariableOpReadVariableOpAdam/Dense_0/bias/v*
_output_shapes
:@*
dtype0
?
Adam/Dense_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*&
shared_nameAdam/Dense_1/kernel/v

)Adam/Dense_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/Dense_1/kernel/v*
_output_shapes

:@@*
dtype0
~
Adam/Dense_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*$
shared_nameAdam/Dense_1/bias/v
w
'Adam/Dense_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/Dense_1/bias/v*
_output_shapes
:@*
dtype0
?
Adam/Dense_output/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*+
shared_nameAdam/Dense_output/kernel/v
?
.Adam/Dense_output/kernel/v/Read/ReadVariableOpReadVariableOpAdam/Dense_output/kernel/v*
_output_shapes

:@*
dtype0
?
Adam/Conv_1/kernel/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameAdam/Conv_1/kernel/vhat
?
+Adam/Conv_1/kernel/vhat/Read/ReadVariableOpReadVariableOpAdam/Conv_1/kernel/vhat*&
_output_shapes
: *
dtype0
?
Adam/Conv_1/bias/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/Conv_1/bias/vhat
{
)Adam/Conv_1/bias/vhat/Read/ReadVariableOpReadVariableOpAdam/Conv_1/bias/vhat*
_output_shapes
: *
dtype0
?
Adam/BatchNorm_1/gamma/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_nameAdam/BatchNorm_1/gamma/vhat
?
/Adam/BatchNorm_1/gamma/vhat/Read/ReadVariableOpReadVariableOpAdam/BatchNorm_1/gamma/vhat*
_output_shapes
: *
dtype0
?
Adam/BatchNorm_1/beta/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape: *+
shared_nameAdam/BatchNorm_1/beta/vhat
?
.Adam/BatchNorm_1/beta/vhat/Read/ReadVariableOpReadVariableOpAdam/BatchNorm_1/beta/vhat*
_output_shapes
: *
dtype0
?
Adam/Conv_2/kernel/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *(
shared_nameAdam/Conv_2/kernel/vhat
?
+Adam/Conv_2/kernel/vhat/Read/ReadVariableOpReadVariableOpAdam/Conv_2/kernel/vhat*&
_output_shapes
:  *
dtype0
?
Adam/Conv_2/bias/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/Conv_2/bias/vhat
{
)Adam/Conv_2/bias/vhat/Read/ReadVariableOpReadVariableOpAdam/Conv_2/bias/vhat*
_output_shapes
: *
dtype0
?
Adam/Conv_3/kernel/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *(
shared_nameAdam/Conv_3/kernel/vhat
?
+Adam/Conv_3/kernel/vhat/Read/ReadVariableOpReadVariableOpAdam/Conv_3/kernel/vhat*&
_output_shapes
:  *
dtype0
?
Adam/Conv_3/bias/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/Conv_3/bias/vhat
{
)Adam/Conv_3/bias/vhat/Read/ReadVariableOpReadVariableOpAdam/Conv_3/bias/vhat*
_output_shapes
: *
dtype0
?
Adam/BatchNorm_2/gamma/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_nameAdam/BatchNorm_2/gamma/vhat
?
/Adam/BatchNorm_2/gamma/vhat/Read/ReadVariableOpReadVariableOpAdam/BatchNorm_2/gamma/vhat*
_output_shapes
: *
dtype0
?
Adam/BatchNorm_2/beta/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape: *+
shared_nameAdam/BatchNorm_2/beta/vhat
?
.Adam/BatchNorm_2/beta/vhat/Read/ReadVariableOpReadVariableOpAdam/BatchNorm_2/beta/vhat*
_output_shapes
: *
dtype0
?
Adam/Conv_4/kernel/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *(
shared_nameAdam/Conv_4/kernel/vhat
?
+Adam/Conv_4/kernel/vhat/Read/ReadVariableOpReadVariableOpAdam/Conv_4/kernel/vhat*&
_output_shapes
:  *
dtype0
?
Adam/Conv_4/bias/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/Conv_4/bias/vhat
{
)Adam/Conv_4/bias/vhat/Read/ReadVariableOpReadVariableOpAdam/Conv_4/bias/vhat*
_output_shapes
: *
dtype0
?
Adam/Dense_0/kernel/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape
: @*)
shared_nameAdam/Dense_0/kernel/vhat
?
,Adam/Dense_0/kernel/vhat/Read/ReadVariableOpReadVariableOpAdam/Dense_0/kernel/vhat*
_output_shapes

: @*
dtype0
?
Adam/Dense_0/bias/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*'
shared_nameAdam/Dense_0/bias/vhat
}
*Adam/Dense_0/bias/vhat/Read/ReadVariableOpReadVariableOpAdam/Dense_0/bias/vhat*
_output_shapes
:@*
dtype0
?
Adam/Dense_1/kernel/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*)
shared_nameAdam/Dense_1/kernel/vhat
?
,Adam/Dense_1/kernel/vhat/Read/ReadVariableOpReadVariableOpAdam/Dense_1/kernel/vhat*
_output_shapes

:@@*
dtype0
?
Adam/Dense_1/bias/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*'
shared_nameAdam/Dense_1/bias/vhat
}
*Adam/Dense_1/bias/vhat/Read/ReadVariableOpReadVariableOpAdam/Dense_1/bias/vhat*
_output_shapes
:@*
dtype0
?
Adam/Dense_output/kernel/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*.
shared_nameAdam/Dense_output/kernel/vhat
?
1Adam/Dense_output/kernel/vhat/Read/ReadVariableOpReadVariableOpAdam/Dense_output/kernel/vhat*
_output_shapes

:@*
dtype0

NoOpNoOp
??
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*??
value??B?? B??
?
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer-4
layer_with_weights-3
layer-5
layer-6
layer-7
	layer_with_weights-4
	layer-8

layer_with_weights-5

layer-9
layer-10
layer-11
layer_with_weights-6
layer-12
layer_with_weights-7
layer-13
layer-14
layer_with_weights-8
layer-15
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures*
* 
?

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
 __call__
*!&call_and_return_all_conditional_losses*
?
"axis
	#gamma
$beta
%moving_mean
&moving_variance
'	variables
(trainable_variables
)regularization_losses
*	keras_api
+__call__
*,&call_and_return_all_conditional_losses*
?

-kernel
.bias
/	variables
0trainable_variables
1regularization_losses
2	keras_api
3__call__
*4&call_and_return_all_conditional_losses*
?
5	variables
6trainable_variables
7regularization_losses
8	keras_api
9__call__
*:&call_and_return_all_conditional_losses* 
?

;kernel
<bias
=	variables
>trainable_variables
?regularization_losses
@	keras_api
A__call__
*B&call_and_return_all_conditional_losses*
?
C	variables
Dtrainable_variables
Eregularization_losses
F	keras_api
G__call__
*H&call_and_return_all_conditional_losses* 
?
I	variables
Jtrainable_variables
Kregularization_losses
L	keras_api
M__call__
*N&call_and_return_all_conditional_losses* 
?
Oaxis
	Pgamma
Qbeta
Rmoving_mean
Smoving_variance
T	variables
Utrainable_variables
Vregularization_losses
W	keras_api
X__call__
*Y&call_and_return_all_conditional_losses*
?

Zkernel
[bias
\	variables
]trainable_variables
^regularization_losses
_	keras_api
`__call__
*a&call_and_return_all_conditional_losses*
?
b	variables
ctrainable_variables
dregularization_losses
e	keras_api
f__call__
*g&call_and_return_all_conditional_losses* 
?
h	variables
itrainable_variables
jregularization_losses
k	keras_api
l_random_generator
m__call__
*n&call_and_return_all_conditional_losses* 
?

okernel
pbias
q	variables
rtrainable_variables
sregularization_losses
t	keras_api
u__call__
*v&call_and_return_all_conditional_losses*
?

wkernel
xbias
y	variables
ztrainable_variables
{regularization_losses
|	keras_api
}__call__
*~&call_and_return_all_conditional_losses*
?
	variables
?trainable_variables
?regularization_losses
?	keras_api
?_random_generator
?__call__
+?&call_and_return_all_conditional_losses* 
?
?kernel
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses*
?
	?iter
?beta_1
?beta_2

?decaym?m?#m?$m?-m?.m?;m?<m?Pm?Qm?Zm?[m?om?pm?wm?xm?	?m?v?v?#v?$v?-v?.v?;v?<v?Pv?Qv?Zv?[v?ov?pv?wv?xv?	?v?vhat?vhat?#vhat?$vhat?-vhat?.vhat?;vhat?<vhat?Pvhat?Qvhat?Zvhat?[vhat?ovhat?pvhat?wvhat?xvhat??vhat?*
?
0
1
#2
$3
%4
&5
-6
.7
;8
<9
P10
Q11
R12
S13
Z14
[15
o16
p17
w18
x19
?20*
?
0
1
#2
$3
-4
.5
;6
<7
P8
Q9
Z10
[11
o12
p13
w14
x15
?16*
"
?0
?1
?2
?3* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
* 

?serving_default* 
]W
VARIABLE_VALUEConv_1/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEConv_1/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

0
1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
	variables
trainable_variables
regularization_losses
 __call__
*!&call_and_return_all_conditional_losses
&!"call_and_return_conditional_losses*
* 
* 
* 
`Z
VARIABLE_VALUEBatchNorm_1/gamma5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEBatchNorm_1/beta4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUE*
lf
VARIABLE_VALUEBatchNorm_1/moving_mean;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
tn
VARIABLE_VALUEBatchNorm_1/moving_variance?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
 
#0
$1
%2
&3*

#0
$1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
'	variables
(trainable_variables
)regularization_losses
+__call__
*,&call_and_return_all_conditional_losses
&,"call_and_return_conditional_losses*
* 
* 
]W
VARIABLE_VALUEConv_2/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEConv_2/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*

-0
.1*

-0
.1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
/	variables
0trainable_variables
1regularization_losses
3__call__
*4&call_and_return_all_conditional_losses
&4"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
5	variables
6trainable_variables
7regularization_losses
9__call__
*:&call_and_return_all_conditional_losses
&:"call_and_return_conditional_losses* 
* 
* 
]W
VARIABLE_VALUEConv_3/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEConv_3/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*

;0
<1*

;0
<1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
=	variables
>trainable_variables
?regularization_losses
A__call__
*B&call_and_return_all_conditional_losses
&B"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
C	variables
Dtrainable_variables
Eregularization_losses
G__call__
*H&call_and_return_all_conditional_losses
&H"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
I	variables
Jtrainable_variables
Kregularization_losses
M__call__
*N&call_and_return_all_conditional_losses
&N"call_and_return_conditional_losses* 
* 
* 
* 
`Z
VARIABLE_VALUEBatchNorm_2/gamma5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEBatchNorm_2/beta4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUE*
lf
VARIABLE_VALUEBatchNorm_2/moving_mean;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
tn
VARIABLE_VALUEBatchNorm_2/moving_variance?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
 
P0
Q1
R2
S3*

P0
Q1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
T	variables
Utrainable_variables
Vregularization_losses
X__call__
*Y&call_and_return_all_conditional_losses
&Y"call_and_return_conditional_losses*
* 
* 
]W
VARIABLE_VALUEConv_4/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEConv_4/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE*

Z0
[1*

Z0
[1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
\	variables
]trainable_variables
^regularization_losses
`__call__
*a&call_and_return_all_conditional_losses
&a"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
b	variables
ctrainable_variables
dregularization_losses
f__call__
*g&call_and_return_all_conditional_losses
&g"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
h	variables
itrainable_variables
jregularization_losses
m__call__
*n&call_and_return_all_conditional_losses
&n"call_and_return_conditional_losses* 
* 
* 
* 
^X
VARIABLE_VALUEDense_0/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEDense_0/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE*

o0
p1*

o0
p1*

?0
?1* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
q	variables
rtrainable_variables
sregularization_losses
u__call__
*v&call_and_return_all_conditional_losses
&v"call_and_return_conditional_losses*
* 
* 
^X
VARIABLE_VALUEDense_1/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEDense_1/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE*

w0
x1*

w0
x1*

?0
?1* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
y	variables
ztrainable_variables
{regularization_losses
}__call__
*~&call_and_return_all_conditional_losses
&~"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses* 
* 
* 
* 
c]
VARIABLE_VALUEDense_output/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE*

?0*

?0*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*
* 
* 
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
 
%0
&1
R2
S3*
z
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15*

?0
?1
?2*
* 
* 
* 
* 
* 
* 
* 
* 

%0
&1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

R0
S1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

?0
?1* 
* 
* 
* 
* 

?0
?1* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
<

?total

?count
?	variables
?	keras_api*
M

?total

?count
?
_fn_kwargs
?	variables
?	keras_api*
M

?total

?count
?
_fn_kwargs
?	variables
?	keras_api*
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

?0
?1*

?	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 

?0
?1*

?	variables*
UO
VARIABLE_VALUEtotal_24keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_24keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUE*
* 

?0
?1*

?	variables*
?z
VARIABLE_VALUEAdam/Conv_1/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUEAdam/Conv_1/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?}
VARIABLE_VALUEAdam/BatchNorm_1/gamma/mQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?{
VARIABLE_VALUEAdam/BatchNorm_1/beta/mPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?z
VARIABLE_VALUEAdam/Conv_2/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUEAdam/Conv_2/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?z
VARIABLE_VALUEAdam/Conv_3/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUEAdam/Conv_3/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?}
VARIABLE_VALUEAdam/BatchNorm_2/gamma/mQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?{
VARIABLE_VALUEAdam/BatchNorm_2/beta/mPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?z
VARIABLE_VALUEAdam/Conv_4/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUEAdam/Conv_4/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?{
VARIABLE_VALUEAdam/Dense_0/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/Dense_0/bias/mPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?{
VARIABLE_VALUEAdam/Dense_1/kernel/mRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/Dense_1/bias/mPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEAdam/Dense_output/kernel/mRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?z
VARIABLE_VALUEAdam/Conv_1/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUEAdam/Conv_1/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?}
VARIABLE_VALUEAdam/BatchNorm_1/gamma/vQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?{
VARIABLE_VALUEAdam/BatchNorm_1/beta/vPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?z
VARIABLE_VALUEAdam/Conv_2/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUEAdam/Conv_2/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?z
VARIABLE_VALUEAdam/Conv_3/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUEAdam/Conv_3/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?}
VARIABLE_VALUEAdam/BatchNorm_2/gamma/vQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?{
VARIABLE_VALUEAdam/BatchNorm_2/beta/vPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?z
VARIABLE_VALUEAdam/Conv_4/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUEAdam/Conv_4/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?{
VARIABLE_VALUEAdam/Dense_0/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/Dense_0/bias/vPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?{
VARIABLE_VALUEAdam/Dense_1/kernel/vRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/Dense_1/bias/vPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEAdam/Dense_output/kernel/vRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEAdam/Conv_1/kernel/vhatUlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE*
?|
VARIABLE_VALUEAdam/Conv_1/bias/vhatSlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEAdam/BatchNorm_1/gamma/vhatTlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEAdam/BatchNorm_1/beta/vhatSlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEAdam/Conv_2/kernel/vhatUlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE*
?|
VARIABLE_VALUEAdam/Conv_2/bias/vhatSlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEAdam/Conv_3/kernel/vhatUlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE*
?|
VARIABLE_VALUEAdam/Conv_3/bias/vhatSlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEAdam/BatchNorm_2/gamma/vhatTlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEAdam/BatchNorm_2/beta/vhatSlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEAdam/Conv_4/kernel/vhatUlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE*
?|
VARIABLE_VALUEAdam/Conv_4/bias/vhatSlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEAdam/Dense_0/kernel/vhatUlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE*
?}
VARIABLE_VALUEAdam/Dense_0/bias/vhatSlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEAdam/Dense_1/kernel/vhatUlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE*
?}
VARIABLE_VALUEAdam/Dense_1/bias/vhatSlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEAdam/Dense_output/kernel/vhatUlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE*
?
serving_default_input_8Placeholder*/
_output_shapes
:?????????*
dtype0*$
shape:?????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_8Conv_1/kernelConv_1/biasBatchNorm_1/gammaBatchNorm_1/betaBatchNorm_1/moving_meanBatchNorm_1/moving_varianceConv_2/kernelConv_2/biasConv_3/kernelConv_3/biasBatchNorm_2/gammaBatchNorm_2/betaBatchNorm_2/moving_meanBatchNorm_2/moving_varianceConv_4/kernelConv_4/biasDense_0/kernelDense_0/biasDense_1/kernelDense_1/biasDense_output/kernel*!
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*7
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *.
f)R'
%__inference_signature_wrapper_6707035
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename!Conv_1/kernel/Read/ReadVariableOpConv_1/bias/Read/ReadVariableOp%BatchNorm_1/gamma/Read/ReadVariableOp$BatchNorm_1/beta/Read/ReadVariableOp+BatchNorm_1/moving_mean/Read/ReadVariableOp/BatchNorm_1/moving_variance/Read/ReadVariableOp!Conv_2/kernel/Read/ReadVariableOpConv_2/bias/Read/ReadVariableOp!Conv_3/kernel/Read/ReadVariableOpConv_3/bias/Read/ReadVariableOp%BatchNorm_2/gamma/Read/ReadVariableOp$BatchNorm_2/beta/Read/ReadVariableOp+BatchNorm_2/moving_mean/Read/ReadVariableOp/BatchNorm_2/moving_variance/Read/ReadVariableOp!Conv_4/kernel/Read/ReadVariableOpConv_4/bias/Read/ReadVariableOp"Dense_0/kernel/Read/ReadVariableOp Dense_0/bias/Read/ReadVariableOp"Dense_1/kernel/Read/ReadVariableOp Dense_1/bias/Read/ReadVariableOp'Dense_output/kernel/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal_2/Read/ReadVariableOpcount_2/Read/ReadVariableOp(Adam/Conv_1/kernel/m/Read/ReadVariableOp&Adam/Conv_1/bias/m/Read/ReadVariableOp,Adam/BatchNorm_1/gamma/m/Read/ReadVariableOp+Adam/BatchNorm_1/beta/m/Read/ReadVariableOp(Adam/Conv_2/kernel/m/Read/ReadVariableOp&Adam/Conv_2/bias/m/Read/ReadVariableOp(Adam/Conv_3/kernel/m/Read/ReadVariableOp&Adam/Conv_3/bias/m/Read/ReadVariableOp,Adam/BatchNorm_2/gamma/m/Read/ReadVariableOp+Adam/BatchNorm_2/beta/m/Read/ReadVariableOp(Adam/Conv_4/kernel/m/Read/ReadVariableOp&Adam/Conv_4/bias/m/Read/ReadVariableOp)Adam/Dense_0/kernel/m/Read/ReadVariableOp'Adam/Dense_0/bias/m/Read/ReadVariableOp)Adam/Dense_1/kernel/m/Read/ReadVariableOp'Adam/Dense_1/bias/m/Read/ReadVariableOp.Adam/Dense_output/kernel/m/Read/ReadVariableOp(Adam/Conv_1/kernel/v/Read/ReadVariableOp&Adam/Conv_1/bias/v/Read/ReadVariableOp,Adam/BatchNorm_1/gamma/v/Read/ReadVariableOp+Adam/BatchNorm_1/beta/v/Read/ReadVariableOp(Adam/Conv_2/kernel/v/Read/ReadVariableOp&Adam/Conv_2/bias/v/Read/ReadVariableOp(Adam/Conv_3/kernel/v/Read/ReadVariableOp&Adam/Conv_3/bias/v/Read/ReadVariableOp,Adam/BatchNorm_2/gamma/v/Read/ReadVariableOp+Adam/BatchNorm_2/beta/v/Read/ReadVariableOp(Adam/Conv_4/kernel/v/Read/ReadVariableOp&Adam/Conv_4/bias/v/Read/ReadVariableOp)Adam/Dense_0/kernel/v/Read/ReadVariableOp'Adam/Dense_0/bias/v/Read/ReadVariableOp)Adam/Dense_1/kernel/v/Read/ReadVariableOp'Adam/Dense_1/bias/v/Read/ReadVariableOp.Adam/Dense_output/kernel/v/Read/ReadVariableOp+Adam/Conv_1/kernel/vhat/Read/ReadVariableOp)Adam/Conv_1/bias/vhat/Read/ReadVariableOp/Adam/BatchNorm_1/gamma/vhat/Read/ReadVariableOp.Adam/BatchNorm_1/beta/vhat/Read/ReadVariableOp+Adam/Conv_2/kernel/vhat/Read/ReadVariableOp)Adam/Conv_2/bias/vhat/Read/ReadVariableOp+Adam/Conv_3/kernel/vhat/Read/ReadVariableOp)Adam/Conv_3/bias/vhat/Read/ReadVariableOp/Adam/BatchNorm_2/gamma/vhat/Read/ReadVariableOp.Adam/BatchNorm_2/beta/vhat/Read/ReadVariableOp+Adam/Conv_4/kernel/vhat/Read/ReadVariableOp)Adam/Conv_4/bias/vhat/Read/ReadVariableOp,Adam/Dense_0/kernel/vhat/Read/ReadVariableOp*Adam/Dense_0/bias/vhat/Read/ReadVariableOp,Adam/Dense_1/kernel/vhat/Read/ReadVariableOp*Adam/Dense_1/bias/vhat/Read/ReadVariableOp1Adam/Dense_output/kernel/vhat/Read/ReadVariableOpConst*_
TinX
V2T	*
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
GPU2*0J 8? *)
f$R"
 __inference__traced_save_6707744
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameConv_1/kernelConv_1/biasBatchNorm_1/gammaBatchNorm_1/betaBatchNorm_1/moving_meanBatchNorm_1/moving_varianceConv_2/kernelConv_2/biasConv_3/kernelConv_3/biasBatchNorm_2/gammaBatchNorm_2/betaBatchNorm_2/moving_meanBatchNorm_2/moving_varianceConv_4/kernelConv_4/biasDense_0/kernelDense_0/biasDense_1/kernelDense_1/biasDense_output/kernel	Adam/iterAdam/beta_1Adam/beta_2
Adam/decaytotalcounttotal_1count_1total_2count_2Adam/Conv_1/kernel/mAdam/Conv_1/bias/mAdam/BatchNorm_1/gamma/mAdam/BatchNorm_1/beta/mAdam/Conv_2/kernel/mAdam/Conv_2/bias/mAdam/Conv_3/kernel/mAdam/Conv_3/bias/mAdam/BatchNorm_2/gamma/mAdam/BatchNorm_2/beta/mAdam/Conv_4/kernel/mAdam/Conv_4/bias/mAdam/Dense_0/kernel/mAdam/Dense_0/bias/mAdam/Dense_1/kernel/mAdam/Dense_1/bias/mAdam/Dense_output/kernel/mAdam/Conv_1/kernel/vAdam/Conv_1/bias/vAdam/BatchNorm_1/gamma/vAdam/BatchNorm_1/beta/vAdam/Conv_2/kernel/vAdam/Conv_2/bias/vAdam/Conv_3/kernel/vAdam/Conv_3/bias/vAdam/BatchNorm_2/gamma/vAdam/BatchNorm_2/beta/vAdam/Conv_4/kernel/vAdam/Conv_4/bias/vAdam/Dense_0/kernel/vAdam/Dense_0/bias/vAdam/Dense_1/kernel/vAdam/Dense_1/bias/vAdam/Dense_output/kernel/vAdam/Conv_1/kernel/vhatAdam/Conv_1/bias/vhatAdam/BatchNorm_1/gamma/vhatAdam/BatchNorm_1/beta/vhatAdam/Conv_2/kernel/vhatAdam/Conv_2/bias/vhatAdam/Conv_3/kernel/vhatAdam/Conv_3/bias/vhatAdam/BatchNorm_2/gamma/vhatAdam/BatchNorm_2/beta/vhatAdam/Conv_4/kernel/vhatAdam/Conv_4/bias/vhatAdam/Dense_0/kernel/vhatAdam/Dense_0/bias/vhatAdam/Dense_1/kernel/vhatAdam/Dense_1/bias/vhatAdam/Dense_output/kernel/vhat*^
TinW
U2S*
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
GPU2*0J 8? *,
f'R%
#__inference__traced_restore_6708000??
??
?!
 __inference__traced_save_6707744
file_prefix,
(savev2_conv_1_kernel_read_readvariableop*
&savev2_conv_1_bias_read_readvariableop0
,savev2_batchnorm_1_gamma_read_readvariableop/
+savev2_batchnorm_1_beta_read_readvariableop6
2savev2_batchnorm_1_moving_mean_read_readvariableop:
6savev2_batchnorm_1_moving_variance_read_readvariableop,
(savev2_conv_2_kernel_read_readvariableop*
&savev2_conv_2_bias_read_readvariableop,
(savev2_conv_3_kernel_read_readvariableop*
&savev2_conv_3_bias_read_readvariableop0
,savev2_batchnorm_2_gamma_read_readvariableop/
+savev2_batchnorm_2_beta_read_readvariableop6
2savev2_batchnorm_2_moving_mean_read_readvariableop:
6savev2_batchnorm_2_moving_variance_read_readvariableop,
(savev2_conv_4_kernel_read_readvariableop*
&savev2_conv_4_bias_read_readvariableop-
)savev2_dense_0_kernel_read_readvariableop+
'savev2_dense_0_bias_read_readvariableop-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop2
.savev2_dense_output_kernel_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop&
"savev2_total_2_read_readvariableop&
"savev2_count_2_read_readvariableop3
/savev2_adam_conv_1_kernel_m_read_readvariableop1
-savev2_adam_conv_1_bias_m_read_readvariableop7
3savev2_adam_batchnorm_1_gamma_m_read_readvariableop6
2savev2_adam_batchnorm_1_beta_m_read_readvariableop3
/savev2_adam_conv_2_kernel_m_read_readvariableop1
-savev2_adam_conv_2_bias_m_read_readvariableop3
/savev2_adam_conv_3_kernel_m_read_readvariableop1
-savev2_adam_conv_3_bias_m_read_readvariableop7
3savev2_adam_batchnorm_2_gamma_m_read_readvariableop6
2savev2_adam_batchnorm_2_beta_m_read_readvariableop3
/savev2_adam_conv_4_kernel_m_read_readvariableop1
-savev2_adam_conv_4_bias_m_read_readvariableop4
0savev2_adam_dense_0_kernel_m_read_readvariableop2
.savev2_adam_dense_0_bias_m_read_readvariableop4
0savev2_adam_dense_1_kernel_m_read_readvariableop2
.savev2_adam_dense_1_bias_m_read_readvariableop9
5savev2_adam_dense_output_kernel_m_read_readvariableop3
/savev2_adam_conv_1_kernel_v_read_readvariableop1
-savev2_adam_conv_1_bias_v_read_readvariableop7
3savev2_adam_batchnorm_1_gamma_v_read_readvariableop6
2savev2_adam_batchnorm_1_beta_v_read_readvariableop3
/savev2_adam_conv_2_kernel_v_read_readvariableop1
-savev2_adam_conv_2_bias_v_read_readvariableop3
/savev2_adam_conv_3_kernel_v_read_readvariableop1
-savev2_adam_conv_3_bias_v_read_readvariableop7
3savev2_adam_batchnorm_2_gamma_v_read_readvariableop6
2savev2_adam_batchnorm_2_beta_v_read_readvariableop3
/savev2_adam_conv_4_kernel_v_read_readvariableop1
-savev2_adam_conv_4_bias_v_read_readvariableop4
0savev2_adam_dense_0_kernel_v_read_readvariableop2
.savev2_adam_dense_0_bias_v_read_readvariableop4
0savev2_adam_dense_1_kernel_v_read_readvariableop2
.savev2_adam_dense_1_bias_v_read_readvariableop9
5savev2_adam_dense_output_kernel_v_read_readvariableop6
2savev2_adam_conv_1_kernel_vhat_read_readvariableop4
0savev2_adam_conv_1_bias_vhat_read_readvariableop:
6savev2_adam_batchnorm_1_gamma_vhat_read_readvariableop9
5savev2_adam_batchnorm_1_beta_vhat_read_readvariableop6
2savev2_adam_conv_2_kernel_vhat_read_readvariableop4
0savev2_adam_conv_2_bias_vhat_read_readvariableop6
2savev2_adam_conv_3_kernel_vhat_read_readvariableop4
0savev2_adam_conv_3_bias_vhat_read_readvariableop:
6savev2_adam_batchnorm_2_gamma_vhat_read_readvariableop9
5savev2_adam_batchnorm_2_beta_vhat_read_readvariableop6
2savev2_adam_conv_4_kernel_vhat_read_readvariableop4
0savev2_adam_conv_4_bias_vhat_read_readvariableop7
3savev2_adam_dense_0_kernel_vhat_read_readvariableop5
1savev2_adam_dense_0_bias_vhat_read_readvariableop7
3savev2_adam_dense_1_kernel_vhat_read_readvariableop5
1savev2_adam_dense_1_bias_vhat_read_readvariableop<
8savev2_adam_dense_output_kernel_vhat_read_readvariableop
savev2_const

identity_1??MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : ?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ?/
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:S*
dtype0*?/
value?.B?.SB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:S*
dtype0*?
value?B?SB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ? 
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0(savev2_conv_1_kernel_read_readvariableop&savev2_conv_1_bias_read_readvariableop,savev2_batchnorm_1_gamma_read_readvariableop+savev2_batchnorm_1_beta_read_readvariableop2savev2_batchnorm_1_moving_mean_read_readvariableop6savev2_batchnorm_1_moving_variance_read_readvariableop(savev2_conv_2_kernel_read_readvariableop&savev2_conv_2_bias_read_readvariableop(savev2_conv_3_kernel_read_readvariableop&savev2_conv_3_bias_read_readvariableop,savev2_batchnorm_2_gamma_read_readvariableop+savev2_batchnorm_2_beta_read_readvariableop2savev2_batchnorm_2_moving_mean_read_readvariableop6savev2_batchnorm_2_moving_variance_read_readvariableop(savev2_conv_4_kernel_read_readvariableop&savev2_conv_4_bias_read_readvariableop)savev2_dense_0_kernel_read_readvariableop'savev2_dense_0_bias_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop.savev2_dense_output_kernel_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop"savev2_total_2_read_readvariableop"savev2_count_2_read_readvariableop/savev2_adam_conv_1_kernel_m_read_readvariableop-savev2_adam_conv_1_bias_m_read_readvariableop3savev2_adam_batchnorm_1_gamma_m_read_readvariableop2savev2_adam_batchnorm_1_beta_m_read_readvariableop/savev2_adam_conv_2_kernel_m_read_readvariableop-savev2_adam_conv_2_bias_m_read_readvariableop/savev2_adam_conv_3_kernel_m_read_readvariableop-savev2_adam_conv_3_bias_m_read_readvariableop3savev2_adam_batchnorm_2_gamma_m_read_readvariableop2savev2_adam_batchnorm_2_beta_m_read_readvariableop/savev2_adam_conv_4_kernel_m_read_readvariableop-savev2_adam_conv_4_bias_m_read_readvariableop0savev2_adam_dense_0_kernel_m_read_readvariableop.savev2_adam_dense_0_bias_m_read_readvariableop0savev2_adam_dense_1_kernel_m_read_readvariableop.savev2_adam_dense_1_bias_m_read_readvariableop5savev2_adam_dense_output_kernel_m_read_readvariableop/savev2_adam_conv_1_kernel_v_read_readvariableop-savev2_adam_conv_1_bias_v_read_readvariableop3savev2_adam_batchnorm_1_gamma_v_read_readvariableop2savev2_adam_batchnorm_1_beta_v_read_readvariableop/savev2_adam_conv_2_kernel_v_read_readvariableop-savev2_adam_conv_2_bias_v_read_readvariableop/savev2_adam_conv_3_kernel_v_read_readvariableop-savev2_adam_conv_3_bias_v_read_readvariableop3savev2_adam_batchnorm_2_gamma_v_read_readvariableop2savev2_adam_batchnorm_2_beta_v_read_readvariableop/savev2_adam_conv_4_kernel_v_read_readvariableop-savev2_adam_conv_4_bias_v_read_readvariableop0savev2_adam_dense_0_kernel_v_read_readvariableop.savev2_adam_dense_0_bias_v_read_readvariableop0savev2_adam_dense_1_kernel_v_read_readvariableop.savev2_adam_dense_1_bias_v_read_readvariableop5savev2_adam_dense_output_kernel_v_read_readvariableop2savev2_adam_conv_1_kernel_vhat_read_readvariableop0savev2_adam_conv_1_bias_vhat_read_readvariableop6savev2_adam_batchnorm_1_gamma_vhat_read_readvariableop5savev2_adam_batchnorm_1_beta_vhat_read_readvariableop2savev2_adam_conv_2_kernel_vhat_read_readvariableop0savev2_adam_conv_2_bias_vhat_read_readvariableop2savev2_adam_conv_3_kernel_vhat_read_readvariableop0savev2_adam_conv_3_bias_vhat_read_readvariableop6savev2_adam_batchnorm_2_gamma_vhat_read_readvariableop5savev2_adam_batchnorm_2_beta_vhat_read_readvariableop2savev2_adam_conv_4_kernel_vhat_read_readvariableop0savev2_adam_conv_4_bias_vhat_read_readvariableop3savev2_adam_dense_0_kernel_vhat_read_readvariableop1savev2_adam_dense_0_bias_vhat_read_readvariableop3savev2_adam_dense_1_kernel_vhat_read_readvariableop1savev2_adam_dense_1_bias_vhat_read_readvariableop8savev2_adam_dense_output_kernel_vhat_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *a
dtypesW
U2S	?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*?
_input_shapes?
?: : : : : : : :  : :  : : : : : :  : : @:@:@@:@:@: : : : : : : : : : : : : : :  : :  : : : :  : : @:@:@@:@:@: : : : :  : :  : : : :  : : @:@:@@:@:@: : : : :  : :  : : : :  : : @:@:@@:@:@: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
:  : 

_output_shapes
: :,	(
&
_output_shapes
:  : 


_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
:  : 

_output_shapes
: :$ 

_output_shapes

: @: 

_output_shapes
:@:$ 

_output_shapes

:@@: 

_output_shapes
:@:$ 

_output_shapes

:@:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :, (
&
_output_shapes
: : !

_output_shapes
: : "

_output_shapes
: : #

_output_shapes
: :,$(
&
_output_shapes
:  : %

_output_shapes
: :,&(
&
_output_shapes
:  : '

_output_shapes
: : (

_output_shapes
: : )

_output_shapes
: :,*(
&
_output_shapes
:  : +

_output_shapes
: :$, 

_output_shapes

: @: -

_output_shapes
:@:$. 

_output_shapes

:@@: /

_output_shapes
:@:$0 

_output_shapes

:@:,1(
&
_output_shapes
: : 2

_output_shapes
: : 3

_output_shapes
: : 4

_output_shapes
: :,5(
&
_output_shapes
:  : 6

_output_shapes
: :,7(
&
_output_shapes
:  : 8

_output_shapes
: : 9

_output_shapes
: : :

_output_shapes
: :,;(
&
_output_shapes
:  : <

_output_shapes
: :$= 

_output_shapes

: @: >

_output_shapes
:@:$? 

_output_shapes

:@@: @

_output_shapes
:@:$A 

_output_shapes

:@:,B(
&
_output_shapes
: : C

_output_shapes
: : D

_output_shapes
: : E

_output_shapes
: :,F(
&
_output_shapes
:  : G

_output_shapes
: :,H(
&
_output_shapes
:  : I

_output_shapes
: : J

_output_shapes
: : K

_output_shapes
: :,L(
&
_output_shapes
:  : M

_output_shapes
: :$N 

_output_shapes

: @: O

_output_shapes
:@:$P 

_output_shapes

:@@: Q

_output_shapes
:@:$R 

_output_shapes

:@:S

_output_shapes
: 
?
J
.__inference_activation_6_layer_call_fn_6707140

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
:?????????		 * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_activation_6_layer_call_and_return_conditional_losses_6705905h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????		 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????		 :W S
/
_output_shapes
:?????????		 
 
_user_specified_nameinputs
?
b
F__inference_flatten_3_layer_call_and_return_conditional_losses_6705962

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"????    \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:????????? X
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:????????? "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:????????? :W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
e
I__inference_activation_6_layer_call_and_return_conditional_losses_6705905

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:?????????		 b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????		 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????		 :W S
/
_output_shapes
:?????????		 
 
_user_specified_nameinputs
?	
?
-__inference_BatchNorm_2_layer_call_fn_6707197

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_BatchNorm_2_layer_call_and_return_conditional_losses_6705810?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+??????????????????????????? `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+??????????????????????????? : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?

?
C__inference_Conv_1_layer_call_and_return_conditional_losses_6705869

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:????????? w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
__inference_loss_fn_2_6707464K
9dense_1_kernel_regularizer_square_readvariableop_resource:@@
identity??0Dense_1/kernel/Regularizer/Square/ReadVariableOp?
0Dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp9dense_1_kernel_regularizer_square_readvariableop_resource*
_output_shapes

:@@*
dtype0?
!Dense_1/kernel/Regularizer/SquareSquare8Dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:@@q
 Dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
Dense_1/kernel/Regularizer/SumSum%Dense_1/kernel/Regularizer/Square:y:0)Dense_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: e
 Dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
Dense_1/kernel/Regularizer/mulMul)Dense_1/kernel/Regularizer/mul/x:output:0'Dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: `
IdentityIdentity"Dense_1/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: y
NoOpNoOp1^Dense_1/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2d
0Dense_1/kernel/Regularizer/Square/ReadVariableOp0Dense_1/kernel/Regularizer/Square/ReadVariableOp
?
?
H__inference_BatchNorm_1_layer_call_and_return_conditional_losses_6707098

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+??????????????????????????? ?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+??????????????????????????? : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
d
+__inference_Dropout_0_layer_call_fn_6707286

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
:????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_Dropout_0_layer_call_and_return_conditional_losses_6706187o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:????????? `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:????????? 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
D__inference_Dense_0_layer_call_and_return_conditional_losses_6707346

inputs0
matmul_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?+Dense_0/bias/Regularizer/Abs/ReadVariableOp?0Dense_0/kernel/Regularizer/Square/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: @*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
0Dense_0/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: @*
dtype0?
!Dense_0/kernel/Regularizer/SquareSquare8Dense_0/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: @q
 Dense_0/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
Dense_0/kernel/Regularizer/SumSum%Dense_0/kernel/Regularizer/Square:y:0)Dense_0/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: e
 Dense_0/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??L=?
Dense_0/kernel/Regularizer/mulMul)Dense_0/kernel/Regularizer/mul/x:output:0'Dense_0/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
+Dense_0/bias/Regularizer/Abs/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
Dense_0/bias/Regularizer/AbsAbs3Dense_0/bias/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:@h
Dense_0/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
Dense_0/bias/Regularizer/SumSum Dense_0/bias/Regularizer/Abs:y:0'Dense_0/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: c
Dense_0/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
Dense_0/bias/Regularizer/mulMul'Dense_0/bias/Regularizer/mul/x:output:0%Dense_0/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????@?
NoOpNoOp^BiasAdd/ReadVariableOp,^Dense_0/bias/Regularizer/Abs/ReadVariableOp1^Dense_0/kernel/Regularizer/Square/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2Z
+Dense_0/bias/Regularizer/Abs/ReadVariableOp+Dense_0/bias/Regularizer/Abs/ReadVariableOp2d
0Dense_0/kernel/Regularizer/Square/ReadVariableOp0Dense_0/kernel/Regularizer/Square/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
e
I__inference_activation_7_layer_call_and_return_conditional_losses_6707184

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:????????? b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:????????? "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:????????? :W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
%__inference_signature_wrapper_6707035
input_8!
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: #
	unknown_5:  
	unknown_6: #
	unknown_7:  
	unknown_8: 
	unknown_9: 

unknown_10: 

unknown_11: 

unknown_12: $

unknown_13:  

unknown_14: 

unknown_15: @

unknown_16:@

unknown_17:@@

unknown_18:@

unknown_19:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_8unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19*!
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*7
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *+
f&R$
"__inference__wrapped_model_6705712o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*X
_input_shapesG
E:?????????: : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:?????????
!
_user_specified_name	input_8
?

?
C__inference_Conv_2_layer_call_and_return_conditional_losses_6705894

inputs8
conv2d_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????		 *
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????		 g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:?????????		 w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
(__inference_Conv_2_layer_call_fn_6707125

inputs!
unknown:  
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
:?????????		 *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_Conv_2_layer_call_and_return_conditional_losses_6705894w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????		 `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:????????? : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
G
+__inference_Dropout_0_layer_call_fn_6707281

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
:????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_Dropout_0_layer_call_and_return_conditional_losses_6705969`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:????????? "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:????????? :O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?

?
C__inference_Conv_4_layer_call_and_return_conditional_losses_6705950

inputs8
conv2d_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:????????? w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
d
F__inference_Dropout_2_layer_call_and_return_conditional_losses_6707404

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:?????????@[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:?????????@"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????@:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
e
I__inference_activation_6_layer_call_and_return_conditional_losses_6707145

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:?????????		 b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????		 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????		 :W S
/
_output_shapes
:?????????		 
 
_user_specified_nameinputs
?	
e
F__inference_Dropout_0_layer_call_and_return_conditional_losses_6707303

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:????????? C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:????????? *
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?>?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:????????? o
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:????????? i
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:????????? Y
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:????????? "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:????????? :O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
e
I__inference_activation_7_layer_call_and_return_conditional_losses_6705929

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:????????? b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:????????? "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:????????? :W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
)__inference_model_7_layer_call_fn_6706719

inputs!
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: #
	unknown_5:  
	unknown_6: #
	unknown_7:  
	unknown_8: 
	unknown_9: 

unknown_10: 

unknown_11: 

unknown_12: $

unknown_13:  

unknown_14: 

unknown_15: @

unknown_16:@

unknown_17:@@

unknown_18:@

unknown_19:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19*!
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*7
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_model_7_layer_call_and_return_conditional_losses_6706071o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*X
_input_shapesG
E:?????????: : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
-__inference_BatchNorm_1_layer_call_fn_6707067

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_BatchNorm_1_layer_call_and_return_conditional_losses_6705734?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+??????????????????????????? `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+??????????????????????????? : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?x
?
D__inference_model_7_layer_call_and_return_conditional_losses_6706869

inputs?
%conv_1_conv2d_readvariableop_resource: 4
&conv_1_biasadd_readvariableop_resource: 1
#batchnorm_1_readvariableop_resource: 3
%batchnorm_1_readvariableop_1_resource: B
4batchnorm_1_fusedbatchnormv3_readvariableop_resource: D
6batchnorm_1_fusedbatchnormv3_readvariableop_1_resource: ?
%conv_2_conv2d_readvariableop_resource:  4
&conv_2_biasadd_readvariableop_resource: ?
%conv_3_conv2d_readvariableop_resource:  4
&conv_3_biasadd_readvariableop_resource: 1
#batchnorm_2_readvariableop_resource: 3
%batchnorm_2_readvariableop_1_resource: B
4batchnorm_2_fusedbatchnormv3_readvariableop_resource: D
6batchnorm_2_fusedbatchnormv3_readvariableop_1_resource: ?
%conv_4_conv2d_readvariableop_resource:  4
&conv_4_biasadd_readvariableop_resource: 8
&dense_0_matmul_readvariableop_resource: @5
'dense_0_biasadd_readvariableop_resource:@8
&dense_1_matmul_readvariableop_resource:@@5
'dense_1_biasadd_readvariableop_resource:@=
+dense_output_matmul_readvariableop_resource:@
identity??+BatchNorm_1/FusedBatchNormV3/ReadVariableOp?-BatchNorm_1/FusedBatchNormV3/ReadVariableOp_1?BatchNorm_1/ReadVariableOp?BatchNorm_1/ReadVariableOp_1?+BatchNorm_2/FusedBatchNormV3/ReadVariableOp?-BatchNorm_2/FusedBatchNormV3/ReadVariableOp_1?BatchNorm_2/ReadVariableOp?BatchNorm_2/ReadVariableOp_1?Conv_1/BiasAdd/ReadVariableOp?Conv_1/Conv2D/ReadVariableOp?Conv_2/BiasAdd/ReadVariableOp?Conv_2/Conv2D/ReadVariableOp?Conv_3/BiasAdd/ReadVariableOp?Conv_3/Conv2D/ReadVariableOp?Conv_4/BiasAdd/ReadVariableOp?Conv_4/Conv2D/ReadVariableOp?Dense_0/BiasAdd/ReadVariableOp?Dense_0/MatMul/ReadVariableOp?+Dense_0/bias/Regularizer/Abs/ReadVariableOp?0Dense_0/kernel/Regularizer/Square/ReadVariableOp?Dense_1/BiasAdd/ReadVariableOp?Dense_1/MatMul/ReadVariableOp?+Dense_1/bias/Regularizer/Abs/ReadVariableOp?0Dense_1/kernel/Regularizer/Square/ReadVariableOp?"Dense_output/MatMul/ReadVariableOp?
Conv_1/Conv2D/ReadVariableOpReadVariableOp%conv_1_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0?
Conv_1/Conv2DConv2Dinputs$Conv_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
?
Conv_1/BiasAdd/ReadVariableOpReadVariableOp&conv_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
Conv_1/BiasAddBiasAddConv_1/Conv2D:output:0%Conv_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? z
BatchNorm_1/ReadVariableOpReadVariableOp#batchnorm_1_readvariableop_resource*
_output_shapes
: *
dtype0~
BatchNorm_1/ReadVariableOp_1ReadVariableOp%batchnorm_1_readvariableop_1_resource*
_output_shapes
: *
dtype0?
+BatchNorm_1/FusedBatchNormV3/ReadVariableOpReadVariableOp4batchnorm_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
-BatchNorm_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp6batchnorm_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
BatchNorm_1/FusedBatchNormV3FusedBatchNormV3Conv_1/BiasAdd:output:0"BatchNorm_1/ReadVariableOp:value:0$BatchNorm_1/ReadVariableOp_1:value:03BatchNorm_1/FusedBatchNormV3/ReadVariableOp:value:05BatchNorm_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:????????? : : : : :*
epsilon%o?:*
is_training( ?
Conv_2/Conv2D/ReadVariableOpReadVariableOp%conv_2_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0?
Conv_2/Conv2DConv2D BatchNorm_1/FusedBatchNormV3:y:0$Conv_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????		 *
paddingVALID*
strides
?
Conv_2/BiasAdd/ReadVariableOpReadVariableOp&conv_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
Conv_2/BiasAddBiasAddConv_2/Conv2D:output:0%Conv_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????		 l
activation_6/ReluReluConv_2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????		 ?
Conv_3/Conv2D/ReadVariableOpReadVariableOp%conv_3_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0?
Conv_3/Conv2DConv2Dactivation_6/Relu:activations:0$Conv_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
?
Conv_3/BiasAdd/ReadVariableOpReadVariableOp&conv_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
Conv_3/BiasAddBiasAddConv_3/Conv2D:output:0%Conv_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? ?
MaxPool_1/MaxPoolMaxPoolConv_3/BiasAdd:output:0*/
_output_shapes
:????????? *
ksize
*
paddingVALID*
strides
o
activation_7/ReluReluMaxPool_1/MaxPool:output:0*
T0*/
_output_shapes
:????????? z
BatchNorm_2/ReadVariableOpReadVariableOp#batchnorm_2_readvariableop_resource*
_output_shapes
: *
dtype0~
BatchNorm_2/ReadVariableOp_1ReadVariableOp%batchnorm_2_readvariableop_1_resource*
_output_shapes
: *
dtype0?
+BatchNorm_2/FusedBatchNormV3/ReadVariableOpReadVariableOp4batchnorm_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
-BatchNorm_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp6batchnorm_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
BatchNorm_2/FusedBatchNormV3FusedBatchNormV3activation_7/Relu:activations:0"BatchNorm_2/ReadVariableOp:value:0$BatchNorm_2/ReadVariableOp_1:value:03BatchNorm_2/FusedBatchNormV3/ReadVariableOp:value:05BatchNorm_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:????????? : : : : :*
epsilon%o?:*
is_training( ?
Conv_4/Conv2D/ReadVariableOpReadVariableOp%conv_4_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0?
Conv_4/Conv2DConv2D BatchNorm_2/FusedBatchNormV3:y:0$Conv_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
?
Conv_4/BiasAdd/ReadVariableOpReadVariableOp&conv_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
Conv_4/BiasAddBiasAddConv_4/Conv2D:output:0%Conv_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? `
flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"????    ?
flatten_3/ReshapeReshapeConv_4/BiasAdd:output:0flatten_3/Const:output:0*
T0*'
_output_shapes
:????????? l
Dropout_0/IdentityIdentityflatten_3/Reshape:output:0*
T0*'
_output_shapes
:????????? ?
Dense_0/MatMul/ReadVariableOpReadVariableOp&dense_0_matmul_readvariableop_resource*
_output_shapes

: @*
dtype0?
Dense_0/MatMulMatMulDropout_0/Identity:output:0%Dense_0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
Dense_0/BiasAdd/ReadVariableOpReadVariableOp'dense_0_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
Dense_0/BiasAddBiasAddDense_0/MatMul:product:0&Dense_0/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
Dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0?
Dense_1/MatMulMatMulDense_0/BiasAdd:output:0%Dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
Dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
Dense_1/BiasAddBiasAddDense_1/MatMul:product:0&Dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@j
Dropout_2/IdentityIdentityDense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@?
"Dense_output/MatMul/ReadVariableOpReadVariableOp+dense_output_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0?
Dense_output/MatMulMatMulDropout_2/Identity:output:0*Dense_output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????p
Dense_output/SoftmaxSoftmaxDense_output/MatMul:product:0*
T0*'
_output_shapes
:??????????
0Dense_0/kernel/Regularizer/Square/ReadVariableOpReadVariableOp&dense_0_matmul_readvariableop_resource*
_output_shapes

: @*
dtype0?
!Dense_0/kernel/Regularizer/SquareSquare8Dense_0/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: @q
 Dense_0/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
Dense_0/kernel/Regularizer/SumSum%Dense_0/kernel/Regularizer/Square:y:0)Dense_0/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: e
 Dense_0/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??L=?
Dense_0/kernel/Regularizer/mulMul)Dense_0/kernel/Regularizer/mul/x:output:0'Dense_0/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
+Dense_0/bias/Regularizer/Abs/ReadVariableOpReadVariableOp'dense_0_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
Dense_0/bias/Regularizer/AbsAbs3Dense_0/bias/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:@h
Dense_0/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
Dense_0/bias/Regularizer/SumSum Dense_0/bias/Regularizer/Abs:y:0'Dense_0/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: c
Dense_0/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
Dense_0/bias/Regularizer/mulMul'Dense_0/bias/Regularizer/mul/x:output:0%Dense_0/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
0Dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0?
!Dense_1/kernel/Regularizer/SquareSquare8Dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:@@q
 Dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
Dense_1/kernel/Regularizer/SumSum%Dense_1/kernel/Regularizer/Square:y:0)Dense_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: e
 Dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
Dense_1/kernel/Regularizer/mulMul)Dense_1/kernel/Regularizer/mul/x:output:0'Dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
+Dense_1/bias/Regularizer/Abs/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
Dense_1/bias/Regularizer/AbsAbs3Dense_1/bias/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:@h
Dense_1/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
Dense_1/bias/Regularizer/SumSum Dense_1/bias/Regularizer/Abs:y:0'Dense_1/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: c
Dense_1/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
Dense_1/bias/Regularizer/mulMul'Dense_1/bias/Regularizer/mul/x:output:0%Dense_1/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: m
IdentityIdentityDense_output/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp,^BatchNorm_1/FusedBatchNormV3/ReadVariableOp.^BatchNorm_1/FusedBatchNormV3/ReadVariableOp_1^BatchNorm_1/ReadVariableOp^BatchNorm_1/ReadVariableOp_1,^BatchNorm_2/FusedBatchNormV3/ReadVariableOp.^BatchNorm_2/FusedBatchNormV3/ReadVariableOp_1^BatchNorm_2/ReadVariableOp^BatchNorm_2/ReadVariableOp_1^Conv_1/BiasAdd/ReadVariableOp^Conv_1/Conv2D/ReadVariableOp^Conv_2/BiasAdd/ReadVariableOp^Conv_2/Conv2D/ReadVariableOp^Conv_3/BiasAdd/ReadVariableOp^Conv_3/Conv2D/ReadVariableOp^Conv_4/BiasAdd/ReadVariableOp^Conv_4/Conv2D/ReadVariableOp^Dense_0/BiasAdd/ReadVariableOp^Dense_0/MatMul/ReadVariableOp,^Dense_0/bias/Regularizer/Abs/ReadVariableOp1^Dense_0/kernel/Regularizer/Square/ReadVariableOp^Dense_1/BiasAdd/ReadVariableOp^Dense_1/MatMul/ReadVariableOp,^Dense_1/bias/Regularizer/Abs/ReadVariableOp1^Dense_1/kernel/Regularizer/Square/ReadVariableOp#^Dense_output/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*X
_input_shapesG
E:?????????: : : : : : : : : : : : : : : : : : : : : 2Z
+BatchNorm_1/FusedBatchNormV3/ReadVariableOp+BatchNorm_1/FusedBatchNormV3/ReadVariableOp2^
-BatchNorm_1/FusedBatchNormV3/ReadVariableOp_1-BatchNorm_1/FusedBatchNormV3/ReadVariableOp_128
BatchNorm_1/ReadVariableOpBatchNorm_1/ReadVariableOp2<
BatchNorm_1/ReadVariableOp_1BatchNorm_1/ReadVariableOp_12Z
+BatchNorm_2/FusedBatchNormV3/ReadVariableOp+BatchNorm_2/FusedBatchNormV3/ReadVariableOp2^
-BatchNorm_2/FusedBatchNormV3/ReadVariableOp_1-BatchNorm_2/FusedBatchNormV3/ReadVariableOp_128
BatchNorm_2/ReadVariableOpBatchNorm_2/ReadVariableOp2<
BatchNorm_2/ReadVariableOp_1BatchNorm_2/ReadVariableOp_12>
Conv_1/BiasAdd/ReadVariableOpConv_1/BiasAdd/ReadVariableOp2<
Conv_1/Conv2D/ReadVariableOpConv_1/Conv2D/ReadVariableOp2>
Conv_2/BiasAdd/ReadVariableOpConv_2/BiasAdd/ReadVariableOp2<
Conv_2/Conv2D/ReadVariableOpConv_2/Conv2D/ReadVariableOp2>
Conv_3/BiasAdd/ReadVariableOpConv_3/BiasAdd/ReadVariableOp2<
Conv_3/Conv2D/ReadVariableOpConv_3/Conv2D/ReadVariableOp2>
Conv_4/BiasAdd/ReadVariableOpConv_4/BiasAdd/ReadVariableOp2<
Conv_4/Conv2D/ReadVariableOpConv_4/Conv2D/ReadVariableOp2@
Dense_0/BiasAdd/ReadVariableOpDense_0/BiasAdd/ReadVariableOp2>
Dense_0/MatMul/ReadVariableOpDense_0/MatMul/ReadVariableOp2Z
+Dense_0/bias/Regularizer/Abs/ReadVariableOp+Dense_0/bias/Regularizer/Abs/ReadVariableOp2d
0Dense_0/kernel/Regularizer/Square/ReadVariableOp0Dense_0/kernel/Regularizer/Square/ReadVariableOp2@
Dense_1/BiasAdd/ReadVariableOpDense_1/BiasAdd/ReadVariableOp2>
Dense_1/MatMul/ReadVariableOpDense_1/MatMul/ReadVariableOp2Z
+Dense_1/bias/Regularizer/Abs/ReadVariableOp+Dense_1/bias/Regularizer/Abs/ReadVariableOp2d
0Dense_1/kernel/Regularizer/Square/ReadVariableOp0Dense_1/kernel/Regularizer/Square/ReadVariableOp2H
"Dense_output/MatMul/ReadVariableOp"Dense_output/MatMul/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
d
+__inference_Dropout_2_layer_call_fn_6707399

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
:?????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_Dropout_2_layer_call_and_return_conditional_losses_6706144o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????@22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
)__inference_Dense_1_layer_call_fn_6707367

inputs
unknown:@@
	unknown_0:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_Dense_1_layer_call_and_return_conditional_losses_6706021o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
??
?
D__inference_model_7_layer_call_and_return_conditional_losses_6706986

inputs?
%conv_1_conv2d_readvariableop_resource: 4
&conv_1_biasadd_readvariableop_resource: 1
#batchnorm_1_readvariableop_resource: 3
%batchnorm_1_readvariableop_1_resource: B
4batchnorm_1_fusedbatchnormv3_readvariableop_resource: D
6batchnorm_1_fusedbatchnormv3_readvariableop_1_resource: ?
%conv_2_conv2d_readvariableop_resource:  4
&conv_2_biasadd_readvariableop_resource: ?
%conv_3_conv2d_readvariableop_resource:  4
&conv_3_biasadd_readvariableop_resource: 1
#batchnorm_2_readvariableop_resource: 3
%batchnorm_2_readvariableop_1_resource: B
4batchnorm_2_fusedbatchnormv3_readvariableop_resource: D
6batchnorm_2_fusedbatchnormv3_readvariableop_1_resource: ?
%conv_4_conv2d_readvariableop_resource:  4
&conv_4_biasadd_readvariableop_resource: 8
&dense_0_matmul_readvariableop_resource: @5
'dense_0_biasadd_readvariableop_resource:@8
&dense_1_matmul_readvariableop_resource:@@5
'dense_1_biasadd_readvariableop_resource:@=
+dense_output_matmul_readvariableop_resource:@
identity??BatchNorm_1/AssignNewValue?BatchNorm_1/AssignNewValue_1?+BatchNorm_1/FusedBatchNormV3/ReadVariableOp?-BatchNorm_1/FusedBatchNormV3/ReadVariableOp_1?BatchNorm_1/ReadVariableOp?BatchNorm_1/ReadVariableOp_1?BatchNorm_2/AssignNewValue?BatchNorm_2/AssignNewValue_1?+BatchNorm_2/FusedBatchNormV3/ReadVariableOp?-BatchNorm_2/FusedBatchNormV3/ReadVariableOp_1?BatchNorm_2/ReadVariableOp?BatchNorm_2/ReadVariableOp_1?Conv_1/BiasAdd/ReadVariableOp?Conv_1/Conv2D/ReadVariableOp?Conv_2/BiasAdd/ReadVariableOp?Conv_2/Conv2D/ReadVariableOp?Conv_3/BiasAdd/ReadVariableOp?Conv_3/Conv2D/ReadVariableOp?Conv_4/BiasAdd/ReadVariableOp?Conv_4/Conv2D/ReadVariableOp?Dense_0/BiasAdd/ReadVariableOp?Dense_0/MatMul/ReadVariableOp?+Dense_0/bias/Regularizer/Abs/ReadVariableOp?0Dense_0/kernel/Regularizer/Square/ReadVariableOp?Dense_1/BiasAdd/ReadVariableOp?Dense_1/MatMul/ReadVariableOp?+Dense_1/bias/Regularizer/Abs/ReadVariableOp?0Dense_1/kernel/Regularizer/Square/ReadVariableOp?"Dense_output/MatMul/ReadVariableOp?
Conv_1/Conv2D/ReadVariableOpReadVariableOp%conv_1_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0?
Conv_1/Conv2DConv2Dinputs$Conv_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
?
Conv_1/BiasAdd/ReadVariableOpReadVariableOp&conv_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
Conv_1/BiasAddBiasAddConv_1/Conv2D:output:0%Conv_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? z
BatchNorm_1/ReadVariableOpReadVariableOp#batchnorm_1_readvariableop_resource*
_output_shapes
: *
dtype0~
BatchNorm_1/ReadVariableOp_1ReadVariableOp%batchnorm_1_readvariableop_1_resource*
_output_shapes
: *
dtype0?
+BatchNorm_1/FusedBatchNormV3/ReadVariableOpReadVariableOp4batchnorm_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
-BatchNorm_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp6batchnorm_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
BatchNorm_1/FusedBatchNormV3FusedBatchNormV3Conv_1/BiasAdd:output:0"BatchNorm_1/ReadVariableOp:value:0$BatchNorm_1/ReadVariableOp_1:value:03BatchNorm_1/FusedBatchNormV3/ReadVariableOp:value:05BatchNorm_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:????????? : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<?
BatchNorm_1/AssignNewValueAssignVariableOp4batchnorm_1_fusedbatchnormv3_readvariableop_resource)BatchNorm_1/FusedBatchNormV3:batch_mean:0,^BatchNorm_1/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
BatchNorm_1/AssignNewValue_1AssignVariableOp6batchnorm_1_fusedbatchnormv3_readvariableop_1_resource-BatchNorm_1/FusedBatchNormV3:batch_variance:0.^BatchNorm_1/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0?
Conv_2/Conv2D/ReadVariableOpReadVariableOp%conv_2_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0?
Conv_2/Conv2DConv2D BatchNorm_1/FusedBatchNormV3:y:0$Conv_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????		 *
paddingVALID*
strides
?
Conv_2/BiasAdd/ReadVariableOpReadVariableOp&conv_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
Conv_2/BiasAddBiasAddConv_2/Conv2D:output:0%Conv_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????		 l
activation_6/ReluReluConv_2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????		 ?
Conv_3/Conv2D/ReadVariableOpReadVariableOp%conv_3_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0?
Conv_3/Conv2DConv2Dactivation_6/Relu:activations:0$Conv_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
?
Conv_3/BiasAdd/ReadVariableOpReadVariableOp&conv_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
Conv_3/BiasAddBiasAddConv_3/Conv2D:output:0%Conv_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? ?
MaxPool_1/MaxPoolMaxPoolConv_3/BiasAdd:output:0*/
_output_shapes
:????????? *
ksize
*
paddingVALID*
strides
o
activation_7/ReluReluMaxPool_1/MaxPool:output:0*
T0*/
_output_shapes
:????????? z
BatchNorm_2/ReadVariableOpReadVariableOp#batchnorm_2_readvariableop_resource*
_output_shapes
: *
dtype0~
BatchNorm_2/ReadVariableOp_1ReadVariableOp%batchnorm_2_readvariableop_1_resource*
_output_shapes
: *
dtype0?
+BatchNorm_2/FusedBatchNormV3/ReadVariableOpReadVariableOp4batchnorm_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
-BatchNorm_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp6batchnorm_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
BatchNorm_2/FusedBatchNormV3FusedBatchNormV3activation_7/Relu:activations:0"BatchNorm_2/ReadVariableOp:value:0$BatchNorm_2/ReadVariableOp_1:value:03BatchNorm_2/FusedBatchNormV3/ReadVariableOp:value:05BatchNorm_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:????????? : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<?
BatchNorm_2/AssignNewValueAssignVariableOp4batchnorm_2_fusedbatchnormv3_readvariableop_resource)BatchNorm_2/FusedBatchNormV3:batch_mean:0,^BatchNorm_2/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
BatchNorm_2/AssignNewValue_1AssignVariableOp6batchnorm_2_fusedbatchnormv3_readvariableop_1_resource-BatchNorm_2/FusedBatchNormV3:batch_variance:0.^BatchNorm_2/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0?
Conv_4/Conv2D/ReadVariableOpReadVariableOp%conv_4_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0?
Conv_4/Conv2DConv2D BatchNorm_2/FusedBatchNormV3:y:0$Conv_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
?
Conv_4/BiasAdd/ReadVariableOpReadVariableOp&conv_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
Conv_4/BiasAddBiasAddConv_4/Conv2D:output:0%Conv_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? `
flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"????    ?
flatten_3/ReshapeReshapeConv_4/BiasAdd:output:0flatten_3/Const:output:0*
T0*'
_output_shapes
:????????? \
Dropout_0/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?????
Dropout_0/dropout/MulMulflatten_3/Reshape:output:0 Dropout_0/dropout/Const:output:0*
T0*'
_output_shapes
:????????? a
Dropout_0/dropout/ShapeShapeflatten_3/Reshape:output:0*
T0*
_output_shapes
:?
.Dropout_0/dropout/random_uniform/RandomUniformRandomUniform Dropout_0/dropout/Shape:output:0*
T0*'
_output_shapes
:????????? *
dtype0e
 Dropout_0/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?>?
Dropout_0/dropout/GreaterEqualGreaterEqual7Dropout_0/dropout/random_uniform/RandomUniform:output:0)Dropout_0/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:????????? ?
Dropout_0/dropout/CastCast"Dropout_0/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:????????? ?
Dropout_0/dropout/Mul_1MulDropout_0/dropout/Mul:z:0Dropout_0/dropout/Cast:y:0*
T0*'
_output_shapes
:????????? ?
Dense_0/MatMul/ReadVariableOpReadVariableOp&dense_0_matmul_readvariableop_resource*
_output_shapes

: @*
dtype0?
Dense_0/MatMulMatMulDropout_0/dropout/Mul_1:z:0%Dense_0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
Dense_0/BiasAdd/ReadVariableOpReadVariableOp'dense_0_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
Dense_0/BiasAddBiasAddDense_0/MatMul:product:0&Dense_0/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
Dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0?
Dense_1/MatMulMatMulDense_0/BiasAdd:output:0%Dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
Dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
Dense_1/BiasAddBiasAddDense_1/MatMul:product:0&Dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@\
Dropout_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?????
Dropout_2/dropout/MulMulDense_1/BiasAdd:output:0 Dropout_2/dropout/Const:output:0*
T0*'
_output_shapes
:?????????@_
Dropout_2/dropout/ShapeShapeDense_1/BiasAdd:output:0*
T0*
_output_shapes
:?
.Dropout_2/dropout/random_uniform/RandomUniformRandomUniform Dropout_2/dropout/Shape:output:0*
T0*'
_output_shapes
:?????????@*
dtype0e
 Dropout_2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?>?
Dropout_2/dropout/GreaterEqualGreaterEqual7Dropout_2/dropout/random_uniform/RandomUniform:output:0)Dropout_2/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????@?
Dropout_2/dropout/CastCast"Dropout_2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????@?
Dropout_2/dropout/Mul_1MulDropout_2/dropout/Mul:z:0Dropout_2/dropout/Cast:y:0*
T0*'
_output_shapes
:?????????@?
"Dense_output/MatMul/ReadVariableOpReadVariableOp+dense_output_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0?
Dense_output/MatMulMatMulDropout_2/dropout/Mul_1:z:0*Dense_output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????p
Dense_output/SoftmaxSoftmaxDense_output/MatMul:product:0*
T0*'
_output_shapes
:??????????
0Dense_0/kernel/Regularizer/Square/ReadVariableOpReadVariableOp&dense_0_matmul_readvariableop_resource*
_output_shapes

: @*
dtype0?
!Dense_0/kernel/Regularizer/SquareSquare8Dense_0/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: @q
 Dense_0/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
Dense_0/kernel/Regularizer/SumSum%Dense_0/kernel/Regularizer/Square:y:0)Dense_0/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: e
 Dense_0/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??L=?
Dense_0/kernel/Regularizer/mulMul)Dense_0/kernel/Regularizer/mul/x:output:0'Dense_0/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
+Dense_0/bias/Regularizer/Abs/ReadVariableOpReadVariableOp'dense_0_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
Dense_0/bias/Regularizer/AbsAbs3Dense_0/bias/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:@h
Dense_0/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
Dense_0/bias/Regularizer/SumSum Dense_0/bias/Regularizer/Abs:y:0'Dense_0/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: c
Dense_0/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
Dense_0/bias/Regularizer/mulMul'Dense_0/bias/Regularizer/mul/x:output:0%Dense_0/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
0Dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0?
!Dense_1/kernel/Regularizer/SquareSquare8Dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:@@q
 Dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
Dense_1/kernel/Regularizer/SumSum%Dense_1/kernel/Regularizer/Square:y:0)Dense_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: e
 Dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
Dense_1/kernel/Regularizer/mulMul)Dense_1/kernel/Regularizer/mul/x:output:0'Dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
+Dense_1/bias/Regularizer/Abs/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
Dense_1/bias/Regularizer/AbsAbs3Dense_1/bias/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:@h
Dense_1/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
Dense_1/bias/Regularizer/SumSum Dense_1/bias/Regularizer/Abs:y:0'Dense_1/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: c
Dense_1/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
Dense_1/bias/Regularizer/mulMul'Dense_1/bias/Regularizer/mul/x:output:0%Dense_1/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: m
IdentityIdentityDense_output/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^BatchNorm_1/AssignNewValue^BatchNorm_1/AssignNewValue_1,^BatchNorm_1/FusedBatchNormV3/ReadVariableOp.^BatchNorm_1/FusedBatchNormV3/ReadVariableOp_1^BatchNorm_1/ReadVariableOp^BatchNorm_1/ReadVariableOp_1^BatchNorm_2/AssignNewValue^BatchNorm_2/AssignNewValue_1,^BatchNorm_2/FusedBatchNormV3/ReadVariableOp.^BatchNorm_2/FusedBatchNormV3/ReadVariableOp_1^BatchNorm_2/ReadVariableOp^BatchNorm_2/ReadVariableOp_1^Conv_1/BiasAdd/ReadVariableOp^Conv_1/Conv2D/ReadVariableOp^Conv_2/BiasAdd/ReadVariableOp^Conv_2/Conv2D/ReadVariableOp^Conv_3/BiasAdd/ReadVariableOp^Conv_3/Conv2D/ReadVariableOp^Conv_4/BiasAdd/ReadVariableOp^Conv_4/Conv2D/ReadVariableOp^Dense_0/BiasAdd/ReadVariableOp^Dense_0/MatMul/ReadVariableOp,^Dense_0/bias/Regularizer/Abs/ReadVariableOp1^Dense_0/kernel/Regularizer/Square/ReadVariableOp^Dense_1/BiasAdd/ReadVariableOp^Dense_1/MatMul/ReadVariableOp,^Dense_1/bias/Regularizer/Abs/ReadVariableOp1^Dense_1/kernel/Regularizer/Square/ReadVariableOp#^Dense_output/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*X
_input_shapesG
E:?????????: : : : : : : : : : : : : : : : : : : : : 28
BatchNorm_1/AssignNewValueBatchNorm_1/AssignNewValue2<
BatchNorm_1/AssignNewValue_1BatchNorm_1/AssignNewValue_12Z
+BatchNorm_1/FusedBatchNormV3/ReadVariableOp+BatchNorm_1/FusedBatchNormV3/ReadVariableOp2^
-BatchNorm_1/FusedBatchNormV3/ReadVariableOp_1-BatchNorm_1/FusedBatchNormV3/ReadVariableOp_128
BatchNorm_1/ReadVariableOpBatchNorm_1/ReadVariableOp2<
BatchNorm_1/ReadVariableOp_1BatchNorm_1/ReadVariableOp_128
BatchNorm_2/AssignNewValueBatchNorm_2/AssignNewValue2<
BatchNorm_2/AssignNewValue_1BatchNorm_2/AssignNewValue_12Z
+BatchNorm_2/FusedBatchNormV3/ReadVariableOp+BatchNorm_2/FusedBatchNormV3/ReadVariableOp2^
-BatchNorm_2/FusedBatchNormV3/ReadVariableOp_1-BatchNorm_2/FusedBatchNormV3/ReadVariableOp_128
BatchNorm_2/ReadVariableOpBatchNorm_2/ReadVariableOp2<
BatchNorm_2/ReadVariableOp_1BatchNorm_2/ReadVariableOp_12>
Conv_1/BiasAdd/ReadVariableOpConv_1/BiasAdd/ReadVariableOp2<
Conv_1/Conv2D/ReadVariableOpConv_1/Conv2D/ReadVariableOp2>
Conv_2/BiasAdd/ReadVariableOpConv_2/BiasAdd/ReadVariableOp2<
Conv_2/Conv2D/ReadVariableOpConv_2/Conv2D/ReadVariableOp2>
Conv_3/BiasAdd/ReadVariableOpConv_3/BiasAdd/ReadVariableOp2<
Conv_3/Conv2D/ReadVariableOpConv_3/Conv2D/ReadVariableOp2>
Conv_4/BiasAdd/ReadVariableOpConv_4/BiasAdd/ReadVariableOp2<
Conv_4/Conv2D/ReadVariableOpConv_4/Conv2D/ReadVariableOp2@
Dense_0/BiasAdd/ReadVariableOpDense_0/BiasAdd/ReadVariableOp2>
Dense_0/MatMul/ReadVariableOpDense_0/MatMul/ReadVariableOp2Z
+Dense_0/bias/Regularizer/Abs/ReadVariableOp+Dense_0/bias/Regularizer/Abs/ReadVariableOp2d
0Dense_0/kernel/Regularizer/Square/ReadVariableOp0Dense_0/kernel/Regularizer/Square/ReadVariableOp2@
Dense_1/BiasAdd/ReadVariableOpDense_1/BiasAdd/ReadVariableOp2>
Dense_1/MatMul/ReadVariableOpDense_1/MatMul/ReadVariableOp2Z
+Dense_1/bias/Regularizer/Abs/ReadVariableOp+Dense_1/bias/Regularizer/Abs/ReadVariableOp2d
0Dense_1/kernel/Regularizer/Square/ReadVariableOp0Dense_1/kernel/Regularizer/Square/ReadVariableOp2H
"Dense_output/MatMul/ReadVariableOp"Dense_output/MatMul/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
)__inference_model_7_layer_call_fn_6706474
input_8!
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: #
	unknown_5:  
	unknown_6: #
	unknown_7:  
	unknown_8: 
	unknown_9: 

unknown_10: 

unknown_11: 

unknown_12: $

unknown_13:  

unknown_14: 

unknown_15: @

unknown_16:@

unknown_17:@@

unknown_18:@

unknown_19:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_8unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19*!
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*3
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_model_7_layer_call_and_return_conditional_losses_6706382o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*X
_input_shapesG
E:?????????: : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:?????????
!
_user_specified_name	input_8
?
?
.__inference_Dense_output_layer_call_fn_6707423

inputs
unknown:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_Dense_output_layer_call_and_return_conditional_losses_6706042o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????@: 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
__inference_loss_fn_0_6707442K
9dense_0_kernel_regularizer_square_readvariableop_resource: @
identity??0Dense_0/kernel/Regularizer/Square/ReadVariableOp?
0Dense_0/kernel/Regularizer/Square/ReadVariableOpReadVariableOp9dense_0_kernel_regularizer_square_readvariableop_resource*
_output_shapes

: @*
dtype0?
!Dense_0/kernel/Regularizer/SquareSquare8Dense_0/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: @q
 Dense_0/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
Dense_0/kernel/Regularizer/SumSum%Dense_0/kernel/Regularizer/Square:y:0)Dense_0/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: e
 Dense_0/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??L=?
Dense_0/kernel/Regularizer/mulMul)Dense_0/kernel/Regularizer/mul/x:output:0'Dense_0/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: `
IdentityIdentity"Dense_0/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: y
NoOpNoOp1^Dense_0/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2d
0Dense_0/kernel/Regularizer/Square/ReadVariableOp0Dense_0/kernel/Regularizer/Square/ReadVariableOp
?
G
+__inference_Dropout_2_layer_call_fn_6707394

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
:?????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_Dropout_2_layer_call_and_return_conditional_losses_6706032`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????@:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
(__inference_Conv_1_layer_call_fn_6707044

inputs!
unknown: 
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
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_Conv_1_layer_call_and_return_conditional_losses_6705869w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:????????? `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
G
+__inference_MaxPool_1_layer_call_fn_6707169

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_MaxPool_1_layer_call_and_return_conditional_losses_6705785?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?	
?
-__inference_BatchNorm_1_layer_call_fn_6707080

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
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
GPU2*0J 8? *Q
fLRJ
H__inference_BatchNorm_1_layer_call_and_return_conditional_losses_6705765?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+??????????????????????????? `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+??????????????????????????? : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?

?
C__inference_Conv_2_layer_call_and_return_conditional_losses_6707135

inputs8
conv2d_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????		 *
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????		 g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:?????????		 w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?

?
C__inference_Conv_1_layer_call_and_return_conditional_losses_6707054

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:????????? w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
(__inference_Conv_4_layer_call_fn_6707255

inputs!
unknown:  
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
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_Conv_4_layer_call_and_return_conditional_losses_6705950w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:????????? `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:????????? : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
)__inference_model_7_layer_call_fn_6706766

inputs!
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: #
	unknown_5:  
	unknown_6: #
	unknown_7:  
	unknown_8: 
	unknown_9: 

unknown_10: 

unknown_11: 

unknown_12: $

unknown_13:  

unknown_14: 

unknown_15: @

unknown_16:@

unknown_17:@@

unknown_18:@

unknown_19:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19*!
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*3
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_model_7_layer_call_and_return_conditional_losses_6706382o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*X
_input_shapesG
E:?????????: : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
H__inference_BatchNorm_2_layer_call_and_return_conditional_losses_6705810

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+??????????????????????????? ?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+??????????????????????????? : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?

?
C__inference_Conv_4_layer_call_and_return_conditional_losses_6707265

inputs8
conv2d_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:????????? w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
H__inference_BatchNorm_1_layer_call_and_return_conditional_losses_6705734

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+??????????????????????????? ?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+??????????????????????????? : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
D__inference_Dense_0_layer_call_and_return_conditional_losses_6705993

inputs0
matmul_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?+Dense_0/bias/Regularizer/Abs/ReadVariableOp?0Dense_0/kernel/Regularizer/Square/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: @*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
0Dense_0/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: @*
dtype0?
!Dense_0/kernel/Regularizer/SquareSquare8Dense_0/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: @q
 Dense_0/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
Dense_0/kernel/Regularizer/SumSum%Dense_0/kernel/Regularizer/Square:y:0)Dense_0/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: e
 Dense_0/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??L=?
Dense_0/kernel/Regularizer/mulMul)Dense_0/kernel/Regularizer/mul/x:output:0'Dense_0/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
+Dense_0/bias/Regularizer/Abs/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
Dense_0/bias/Regularizer/AbsAbs3Dense_0/bias/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:@h
Dense_0/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
Dense_0/bias/Regularizer/SumSum Dense_0/bias/Regularizer/Abs:y:0'Dense_0/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: c
Dense_0/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
Dense_0/bias/Regularizer/mulMul'Dense_0/bias/Regularizer/mul/x:output:0%Dense_0/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????@?
NoOpNoOp^BiasAdd/ReadVariableOp,^Dense_0/bias/Regularizer/Abs/ReadVariableOp1^Dense_0/kernel/Regularizer/Square/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2Z
+Dense_0/bias/Regularizer/Abs/ReadVariableOp+Dense_0/bias/Regularizer/Abs/ReadVariableOp2d
0Dense_0/kernel/Regularizer/Square/ReadVariableOp0Dense_0/kernel/Regularizer/Square/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
H__inference_BatchNorm_2_layer_call_and_return_conditional_losses_6707246

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+??????????????????????????? ?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+??????????????????????????? : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
D__inference_Dense_1_layer_call_and_return_conditional_losses_6706021

inputs0
matmul_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?+Dense_1/bias/Regularizer/Abs/ReadVariableOp?0Dense_1/kernel/Regularizer/Square/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
0Dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
dtype0?
!Dense_1/kernel/Regularizer/SquareSquare8Dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:@@q
 Dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
Dense_1/kernel/Regularizer/SumSum%Dense_1/kernel/Regularizer/Square:y:0)Dense_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: e
 Dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
Dense_1/kernel/Regularizer/mulMul)Dense_1/kernel/Regularizer/mul/x:output:0'Dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
+Dense_1/bias/Regularizer/Abs/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
Dense_1/bias/Regularizer/AbsAbs3Dense_1/bias/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:@h
Dense_1/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
Dense_1/bias/Regularizer/SumSum Dense_1/bias/Regularizer/Abs:y:0'Dense_1/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: c
Dense_1/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
Dense_1/bias/Regularizer/mulMul'Dense_1/bias/Regularizer/mul/x:output:0%Dense_1/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????@?
NoOpNoOp^BiasAdd/ReadVariableOp,^Dense_1/bias/Regularizer/Abs/ReadVariableOp1^Dense_1/kernel/Regularizer/Square/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2Z
+Dense_1/bias/Regularizer/Abs/ReadVariableOp+Dense_1/bias/Regularizer/Abs/ReadVariableOp2d
0Dense_1/kernel/Regularizer/Square/ReadVariableOp0Dense_1/kernel/Regularizer/Square/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
b
F__inference_MaxPool_1_layer_call_and_return_conditional_losses_6705785

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?

?
C__inference_Conv_3_layer_call_and_return_conditional_losses_6705917

inputs8
conv2d_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:????????? w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????		 : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????		 
 
_user_specified_nameinputs
?
d
F__inference_Dropout_2_layer_call_and_return_conditional_losses_6706032

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:?????????@[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:?????????@"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????@:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
H__inference_BatchNorm_2_layer_call_and_return_conditional_losses_6707228

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+??????????????????????????? ?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+??????????????????????????? : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
I__inference_Dense_output_layer_call_and_return_conditional_losses_6707431

inputs0
matmul_readvariableop_resource:@
identity??MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????V
SoftmaxSoftmaxMatMul:product:0*
T0*'
_output_shapes
:?????????`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????^
NoOpNoOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????@: 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?c
?
D__inference_model_7_layer_call_and_return_conditional_losses_6706382

inputs(
conv_1_6706300: 
conv_1_6706302: !
batchnorm_1_6706305: !
batchnorm_1_6706307: !
batchnorm_1_6706309: !
batchnorm_1_6706311: (
conv_2_6706314:  
conv_2_6706316: (
conv_3_6706320:  
conv_3_6706322: !
batchnorm_2_6706327: !
batchnorm_2_6706329: !
batchnorm_2_6706331: !
batchnorm_2_6706333: (
conv_4_6706336:  
conv_4_6706338: !
dense_0_6706343: @
dense_0_6706345:@!
dense_1_6706348:@@
dense_1_6706350:@&
dense_output_6706354:@
identity??#BatchNorm_1/StatefulPartitionedCall?#BatchNorm_2/StatefulPartitionedCall?Conv_1/StatefulPartitionedCall?Conv_2/StatefulPartitionedCall?Conv_3/StatefulPartitionedCall?Conv_4/StatefulPartitionedCall?Dense_0/StatefulPartitionedCall?+Dense_0/bias/Regularizer/Abs/ReadVariableOp?0Dense_0/kernel/Regularizer/Square/ReadVariableOp?Dense_1/StatefulPartitionedCall?+Dense_1/bias/Regularizer/Abs/ReadVariableOp?0Dense_1/kernel/Regularizer/Square/ReadVariableOp?$Dense_output/StatefulPartitionedCall?!Dropout_0/StatefulPartitionedCall?!Dropout_2/StatefulPartitionedCall?
Conv_1/StatefulPartitionedCallStatefulPartitionedCallinputsconv_1_6706300conv_1_6706302*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_Conv_1_layer_call_and_return_conditional_losses_6705869?
#BatchNorm_1/StatefulPartitionedCallStatefulPartitionedCall'Conv_1/StatefulPartitionedCall:output:0batchnorm_1_6706305batchnorm_1_6706307batchnorm_1_6706309batchnorm_1_6706311*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_BatchNorm_1_layer_call_and_return_conditional_losses_6705765?
Conv_2/StatefulPartitionedCallStatefulPartitionedCall,BatchNorm_1/StatefulPartitionedCall:output:0conv_2_6706314conv_2_6706316*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????		 *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_Conv_2_layer_call_and_return_conditional_losses_6705894?
activation_6/PartitionedCallPartitionedCall'Conv_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????		 * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_activation_6_layer_call_and_return_conditional_losses_6705905?
Conv_3/StatefulPartitionedCallStatefulPartitionedCall%activation_6/PartitionedCall:output:0conv_3_6706320conv_3_6706322*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_Conv_3_layer_call_and_return_conditional_losses_6705917?
MaxPool_1/PartitionedCallPartitionedCall'Conv_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_MaxPool_1_layer_call_and_return_conditional_losses_6705785?
activation_7/PartitionedCallPartitionedCall"MaxPool_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_activation_7_layer_call_and_return_conditional_losses_6705929?
#BatchNorm_2/StatefulPartitionedCallStatefulPartitionedCall%activation_7/PartitionedCall:output:0batchnorm_2_6706327batchnorm_2_6706329batchnorm_2_6706331batchnorm_2_6706333*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_BatchNorm_2_layer_call_and_return_conditional_losses_6705841?
Conv_4/StatefulPartitionedCallStatefulPartitionedCall,BatchNorm_2/StatefulPartitionedCall:output:0conv_4_6706336conv_4_6706338*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_Conv_4_layer_call_and_return_conditional_losses_6705950?
flatten_3/PartitionedCallPartitionedCall'Conv_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_flatten_3_layer_call_and_return_conditional_losses_6705962?
!Dropout_0/StatefulPartitionedCallStatefulPartitionedCall"flatten_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_Dropout_0_layer_call_and_return_conditional_losses_6706187?
Dense_0/StatefulPartitionedCallStatefulPartitionedCall*Dropout_0/StatefulPartitionedCall:output:0dense_0_6706343dense_0_6706345*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_Dense_0_layer_call_and_return_conditional_losses_6705993?
Dense_1/StatefulPartitionedCallStatefulPartitionedCall(Dense_0/StatefulPartitionedCall:output:0dense_1_6706348dense_1_6706350*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_Dense_1_layer_call_and_return_conditional_losses_6706021?
!Dropout_2/StatefulPartitionedCallStatefulPartitionedCall(Dense_1/StatefulPartitionedCall:output:0"^Dropout_0/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_Dropout_2_layer_call_and_return_conditional_losses_6706144?
$Dense_output/StatefulPartitionedCallStatefulPartitionedCall*Dropout_2/StatefulPartitionedCall:output:0dense_output_6706354*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_Dense_output_layer_call_and_return_conditional_losses_6706042?
0Dense_0/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_0_6706343*
_output_shapes

: @*
dtype0?
!Dense_0/kernel/Regularizer/SquareSquare8Dense_0/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: @q
 Dense_0/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
Dense_0/kernel/Regularizer/SumSum%Dense_0/kernel/Regularizer/Square:y:0)Dense_0/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: e
 Dense_0/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??L=?
Dense_0/kernel/Regularizer/mulMul)Dense_0/kernel/Regularizer/mul/x:output:0'Dense_0/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: w
+Dense_0/bias/Regularizer/Abs/ReadVariableOpReadVariableOpdense_0_6706345*
_output_shapes
:@*
dtype0}
Dense_0/bias/Regularizer/AbsAbs3Dense_0/bias/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:@h
Dense_0/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
Dense_0/bias/Regularizer/SumSum Dense_0/bias/Regularizer/Abs:y:0'Dense_0/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: c
Dense_0/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
Dense_0/bias/Regularizer/mulMul'Dense_0/bias/Regularizer/mul/x:output:0%Dense_0/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
0Dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_1_6706348*
_output_shapes

:@@*
dtype0?
!Dense_1/kernel/Regularizer/SquareSquare8Dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:@@q
 Dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
Dense_1/kernel/Regularizer/SumSum%Dense_1/kernel/Regularizer/Square:y:0)Dense_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: e
 Dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
Dense_1/kernel/Regularizer/mulMul)Dense_1/kernel/Regularizer/mul/x:output:0'Dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: w
+Dense_1/bias/Regularizer/Abs/ReadVariableOpReadVariableOpdense_1_6706350*
_output_shapes
:@*
dtype0}
Dense_1/bias/Regularizer/AbsAbs3Dense_1/bias/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:@h
Dense_1/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
Dense_1/bias/Regularizer/SumSum Dense_1/bias/Regularizer/Abs:y:0'Dense_1/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: c
Dense_1/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
Dense_1/bias/Regularizer/mulMul'Dense_1/bias/Regularizer/mul/x:output:0%Dense_1/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: |
IdentityIdentity-Dense_output/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp$^BatchNorm_1/StatefulPartitionedCall$^BatchNorm_2/StatefulPartitionedCall^Conv_1/StatefulPartitionedCall^Conv_2/StatefulPartitionedCall^Conv_3/StatefulPartitionedCall^Conv_4/StatefulPartitionedCall ^Dense_0/StatefulPartitionedCall,^Dense_0/bias/Regularizer/Abs/ReadVariableOp1^Dense_0/kernel/Regularizer/Square/ReadVariableOp ^Dense_1/StatefulPartitionedCall,^Dense_1/bias/Regularizer/Abs/ReadVariableOp1^Dense_1/kernel/Regularizer/Square/ReadVariableOp%^Dense_output/StatefulPartitionedCall"^Dropout_0/StatefulPartitionedCall"^Dropout_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*X
_input_shapesG
E:?????????: : : : : : : : : : : : : : : : : : : : : 2J
#BatchNorm_1/StatefulPartitionedCall#BatchNorm_1/StatefulPartitionedCall2J
#BatchNorm_2/StatefulPartitionedCall#BatchNorm_2/StatefulPartitionedCall2@
Conv_1/StatefulPartitionedCallConv_1/StatefulPartitionedCall2@
Conv_2/StatefulPartitionedCallConv_2/StatefulPartitionedCall2@
Conv_3/StatefulPartitionedCallConv_3/StatefulPartitionedCall2@
Conv_4/StatefulPartitionedCallConv_4/StatefulPartitionedCall2B
Dense_0/StatefulPartitionedCallDense_0/StatefulPartitionedCall2Z
+Dense_0/bias/Regularizer/Abs/ReadVariableOp+Dense_0/bias/Regularizer/Abs/ReadVariableOp2d
0Dense_0/kernel/Regularizer/Square/ReadVariableOp0Dense_0/kernel/Regularizer/Square/ReadVariableOp2B
Dense_1/StatefulPartitionedCallDense_1/StatefulPartitionedCall2Z
+Dense_1/bias/Regularizer/Abs/ReadVariableOp+Dense_1/bias/Regularizer/Abs/ReadVariableOp2d
0Dense_1/kernel/Regularizer/Square/ReadVariableOp0Dense_1/kernel/Regularizer/Square/ReadVariableOp2L
$Dense_output/StatefulPartitionedCall$Dense_output/StatefulPartitionedCall2F
!Dropout_0/StatefulPartitionedCall!Dropout_0/StatefulPartitionedCall2F
!Dropout_2/StatefulPartitionedCall!Dropout_2/StatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
(__inference_Conv_3_layer_call_fn_6707154

inputs!
unknown:  
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
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_Conv_3_layer_call_and_return_conditional_losses_6705917w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:????????? `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????		 : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????		 
 
_user_specified_nameinputs
?
?
H__inference_BatchNorm_1_layer_call_and_return_conditional_losses_6707116

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+??????????????????????????? ?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+??????????????????????????? : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
J
.__inference_activation_7_layer_call_fn_6707179

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
:????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_activation_7_layer_call_and_return_conditional_losses_6705929h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:????????? "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:????????? :W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
I__inference_Dense_output_layer_call_and_return_conditional_losses_6706042

inputs0
matmul_readvariableop_resource:@
identity??MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????V
SoftmaxSoftmaxMatMul:product:0*
T0*'
_output_shapes
:?????????`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????^
NoOpNoOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????@: 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
)__inference_Dense_0_layer_call_fn_6707324

inputs
unknown: @
	unknown_0:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_Dense_0_layer_call_and_return_conditional_losses_6705993o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:????????? : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?	
e
F__inference_Dropout_0_layer_call_and_return_conditional_losses_6706187

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:????????? C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:????????? *
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?>?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:????????? o
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:????????? i
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:????????? Y
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:????????? "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:????????? :O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
)__inference_model_7_layer_call_fn_6706116
input_8!
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: #
	unknown_5:  
	unknown_6: #
	unknown_7:  
	unknown_8: 
	unknown_9: 

unknown_10: 

unknown_11: 

unknown_12: $

unknown_13:  

unknown_14: 

unknown_15: @

unknown_16:@

unknown_17:@@

unknown_18:@

unknown_19:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_8unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19*!
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*7
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_model_7_layer_call_and_return_conditional_losses_6706071o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*X
_input_shapesG
E:?????????: : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:?????????
!
_user_specified_name	input_8
?
?
H__inference_BatchNorm_1_layer_call_and_return_conditional_losses_6705765

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+??????????????????????????? ?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+??????????????????????????? : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?

?
C__inference_Conv_3_layer_call_and_return_conditional_losses_6707164

inputs8
conv2d_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:????????? w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????		 : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????		 
 
_user_specified_nameinputs
?

?
__inference_loss_fn_1_6707453B
4dense_0_bias_regularizer_abs_readvariableop_resource:@
identity??+Dense_0/bias/Regularizer/Abs/ReadVariableOp?
+Dense_0/bias/Regularizer/Abs/ReadVariableOpReadVariableOp4dense_0_bias_regularizer_abs_readvariableop_resource*
_output_shapes
:@*
dtype0}
Dense_0/bias/Regularizer/AbsAbs3Dense_0/bias/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:@h
Dense_0/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
Dense_0/bias/Regularizer/SumSum Dense_0/bias/Regularizer/Abs:y:0'Dense_0/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: c
Dense_0/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
Dense_0/bias/Regularizer/mulMul'Dense_0/bias/Regularizer/mul/x:output:0%Dense_0/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: ^
IdentityIdentity Dense_0/bias/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: t
NoOpNoOp,^Dense_0/bias/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2Z
+Dense_0/bias/Regularizer/Abs/ReadVariableOp+Dense_0/bias/Regularizer/Abs/ReadVariableOp
?
d
F__inference_Dropout_0_layer_call_and_return_conditional_losses_6707291

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:????????? [

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:????????? "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:????????? :O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
G
+__inference_flatten_3_layer_call_fn_6707270

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
:????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_flatten_3_layer_call_and_return_conditional_losses_6705962`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:????????? "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:????????? :W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?	
e
F__inference_Dropout_2_layer_call_and_return_conditional_losses_6707416

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:?????????@C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:?????????@*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?>?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????@o
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????@i
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:?????????@Y
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:?????????@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????@:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?	
e
F__inference_Dropout_2_layer_call_and_return_conditional_losses_6706144

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:?????????@C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:?????????@*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?>?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????@o
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????@i
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:?????????@Y
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:?????????@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????@:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?`
?

D__inference_model_7_layer_call_and_return_conditional_losses_6706559
input_8(
conv_1_6706477: 
conv_1_6706479: !
batchnorm_1_6706482: !
batchnorm_1_6706484: !
batchnorm_1_6706486: !
batchnorm_1_6706488: (
conv_2_6706491:  
conv_2_6706493: (
conv_3_6706497:  
conv_3_6706499: !
batchnorm_2_6706504: !
batchnorm_2_6706506: !
batchnorm_2_6706508: !
batchnorm_2_6706510: (
conv_4_6706513:  
conv_4_6706515: !
dense_0_6706520: @
dense_0_6706522:@!
dense_1_6706525:@@
dense_1_6706527:@&
dense_output_6706531:@
identity??#BatchNorm_1/StatefulPartitionedCall?#BatchNorm_2/StatefulPartitionedCall?Conv_1/StatefulPartitionedCall?Conv_2/StatefulPartitionedCall?Conv_3/StatefulPartitionedCall?Conv_4/StatefulPartitionedCall?Dense_0/StatefulPartitionedCall?+Dense_0/bias/Regularizer/Abs/ReadVariableOp?0Dense_0/kernel/Regularizer/Square/ReadVariableOp?Dense_1/StatefulPartitionedCall?+Dense_1/bias/Regularizer/Abs/ReadVariableOp?0Dense_1/kernel/Regularizer/Square/ReadVariableOp?$Dense_output/StatefulPartitionedCall?
Conv_1/StatefulPartitionedCallStatefulPartitionedCallinput_8conv_1_6706477conv_1_6706479*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_Conv_1_layer_call_and_return_conditional_losses_6705869?
#BatchNorm_1/StatefulPartitionedCallStatefulPartitionedCall'Conv_1/StatefulPartitionedCall:output:0batchnorm_1_6706482batchnorm_1_6706484batchnorm_1_6706486batchnorm_1_6706488*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_BatchNorm_1_layer_call_and_return_conditional_losses_6705734?
Conv_2/StatefulPartitionedCallStatefulPartitionedCall,BatchNorm_1/StatefulPartitionedCall:output:0conv_2_6706491conv_2_6706493*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????		 *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_Conv_2_layer_call_and_return_conditional_losses_6705894?
activation_6/PartitionedCallPartitionedCall'Conv_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????		 * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_activation_6_layer_call_and_return_conditional_losses_6705905?
Conv_3/StatefulPartitionedCallStatefulPartitionedCall%activation_6/PartitionedCall:output:0conv_3_6706497conv_3_6706499*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_Conv_3_layer_call_and_return_conditional_losses_6705917?
MaxPool_1/PartitionedCallPartitionedCall'Conv_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_MaxPool_1_layer_call_and_return_conditional_losses_6705785?
activation_7/PartitionedCallPartitionedCall"MaxPool_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_activation_7_layer_call_and_return_conditional_losses_6705929?
#BatchNorm_2/StatefulPartitionedCallStatefulPartitionedCall%activation_7/PartitionedCall:output:0batchnorm_2_6706504batchnorm_2_6706506batchnorm_2_6706508batchnorm_2_6706510*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_BatchNorm_2_layer_call_and_return_conditional_losses_6705810?
Conv_4/StatefulPartitionedCallStatefulPartitionedCall,BatchNorm_2/StatefulPartitionedCall:output:0conv_4_6706513conv_4_6706515*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_Conv_4_layer_call_and_return_conditional_losses_6705950?
flatten_3/PartitionedCallPartitionedCall'Conv_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_flatten_3_layer_call_and_return_conditional_losses_6705962?
Dropout_0/PartitionedCallPartitionedCall"flatten_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_Dropout_0_layer_call_and_return_conditional_losses_6705969?
Dense_0/StatefulPartitionedCallStatefulPartitionedCall"Dropout_0/PartitionedCall:output:0dense_0_6706520dense_0_6706522*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_Dense_0_layer_call_and_return_conditional_losses_6705993?
Dense_1/StatefulPartitionedCallStatefulPartitionedCall(Dense_0/StatefulPartitionedCall:output:0dense_1_6706525dense_1_6706527*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_Dense_1_layer_call_and_return_conditional_losses_6706021?
Dropout_2/PartitionedCallPartitionedCall(Dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_Dropout_2_layer_call_and_return_conditional_losses_6706032?
$Dense_output/StatefulPartitionedCallStatefulPartitionedCall"Dropout_2/PartitionedCall:output:0dense_output_6706531*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_Dense_output_layer_call_and_return_conditional_losses_6706042?
0Dense_0/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_0_6706520*
_output_shapes

: @*
dtype0?
!Dense_0/kernel/Regularizer/SquareSquare8Dense_0/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: @q
 Dense_0/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
Dense_0/kernel/Regularizer/SumSum%Dense_0/kernel/Regularizer/Square:y:0)Dense_0/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: e
 Dense_0/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??L=?
Dense_0/kernel/Regularizer/mulMul)Dense_0/kernel/Regularizer/mul/x:output:0'Dense_0/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: w
+Dense_0/bias/Regularizer/Abs/ReadVariableOpReadVariableOpdense_0_6706522*
_output_shapes
:@*
dtype0}
Dense_0/bias/Regularizer/AbsAbs3Dense_0/bias/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:@h
Dense_0/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
Dense_0/bias/Regularizer/SumSum Dense_0/bias/Regularizer/Abs:y:0'Dense_0/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: c
Dense_0/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
Dense_0/bias/Regularizer/mulMul'Dense_0/bias/Regularizer/mul/x:output:0%Dense_0/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
0Dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_1_6706525*
_output_shapes

:@@*
dtype0?
!Dense_1/kernel/Regularizer/SquareSquare8Dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:@@q
 Dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
Dense_1/kernel/Regularizer/SumSum%Dense_1/kernel/Regularizer/Square:y:0)Dense_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: e
 Dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
Dense_1/kernel/Regularizer/mulMul)Dense_1/kernel/Regularizer/mul/x:output:0'Dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: w
+Dense_1/bias/Regularizer/Abs/ReadVariableOpReadVariableOpdense_1_6706527*
_output_shapes
:@*
dtype0}
Dense_1/bias/Regularizer/AbsAbs3Dense_1/bias/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:@h
Dense_1/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
Dense_1/bias/Regularizer/SumSum Dense_1/bias/Regularizer/Abs:y:0'Dense_1/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: c
Dense_1/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
Dense_1/bias/Regularizer/mulMul'Dense_1/bias/Regularizer/mul/x:output:0%Dense_1/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: |
IdentityIdentity-Dense_output/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp$^BatchNorm_1/StatefulPartitionedCall$^BatchNorm_2/StatefulPartitionedCall^Conv_1/StatefulPartitionedCall^Conv_2/StatefulPartitionedCall^Conv_3/StatefulPartitionedCall^Conv_4/StatefulPartitionedCall ^Dense_0/StatefulPartitionedCall,^Dense_0/bias/Regularizer/Abs/ReadVariableOp1^Dense_0/kernel/Regularizer/Square/ReadVariableOp ^Dense_1/StatefulPartitionedCall,^Dense_1/bias/Regularizer/Abs/ReadVariableOp1^Dense_1/kernel/Regularizer/Square/ReadVariableOp%^Dense_output/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*X
_input_shapesG
E:?????????: : : : : : : : : : : : : : : : : : : : : 2J
#BatchNorm_1/StatefulPartitionedCall#BatchNorm_1/StatefulPartitionedCall2J
#BatchNorm_2/StatefulPartitionedCall#BatchNorm_2/StatefulPartitionedCall2@
Conv_1/StatefulPartitionedCallConv_1/StatefulPartitionedCall2@
Conv_2/StatefulPartitionedCallConv_2/StatefulPartitionedCall2@
Conv_3/StatefulPartitionedCallConv_3/StatefulPartitionedCall2@
Conv_4/StatefulPartitionedCallConv_4/StatefulPartitionedCall2B
Dense_0/StatefulPartitionedCallDense_0/StatefulPartitionedCall2Z
+Dense_0/bias/Regularizer/Abs/ReadVariableOp+Dense_0/bias/Regularizer/Abs/ReadVariableOp2d
0Dense_0/kernel/Regularizer/Square/ReadVariableOp0Dense_0/kernel/Regularizer/Square/ReadVariableOp2B
Dense_1/StatefulPartitionedCallDense_1/StatefulPartitionedCall2Z
+Dense_1/bias/Regularizer/Abs/ReadVariableOp+Dense_1/bias/Regularizer/Abs/ReadVariableOp2d
0Dense_1/kernel/Regularizer/Square/ReadVariableOp0Dense_1/kernel/Regularizer/Square/ReadVariableOp2L
$Dense_output/StatefulPartitionedCall$Dense_output/StatefulPartitionedCall:X T
/
_output_shapes
:?????????
!
_user_specified_name	input_8
?
?
H__inference_BatchNorm_2_layer_call_and_return_conditional_losses_6705841

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+??????????????????????????? ?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+??????????????????????????? : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?c
?
D__inference_model_7_layer_call_and_return_conditional_losses_6706644
input_8(
conv_1_6706562: 
conv_1_6706564: !
batchnorm_1_6706567: !
batchnorm_1_6706569: !
batchnorm_1_6706571: !
batchnorm_1_6706573: (
conv_2_6706576:  
conv_2_6706578: (
conv_3_6706582:  
conv_3_6706584: !
batchnorm_2_6706589: !
batchnorm_2_6706591: !
batchnorm_2_6706593: !
batchnorm_2_6706595: (
conv_4_6706598:  
conv_4_6706600: !
dense_0_6706605: @
dense_0_6706607:@!
dense_1_6706610:@@
dense_1_6706612:@&
dense_output_6706616:@
identity??#BatchNorm_1/StatefulPartitionedCall?#BatchNorm_2/StatefulPartitionedCall?Conv_1/StatefulPartitionedCall?Conv_2/StatefulPartitionedCall?Conv_3/StatefulPartitionedCall?Conv_4/StatefulPartitionedCall?Dense_0/StatefulPartitionedCall?+Dense_0/bias/Regularizer/Abs/ReadVariableOp?0Dense_0/kernel/Regularizer/Square/ReadVariableOp?Dense_1/StatefulPartitionedCall?+Dense_1/bias/Regularizer/Abs/ReadVariableOp?0Dense_1/kernel/Regularizer/Square/ReadVariableOp?$Dense_output/StatefulPartitionedCall?!Dropout_0/StatefulPartitionedCall?!Dropout_2/StatefulPartitionedCall?
Conv_1/StatefulPartitionedCallStatefulPartitionedCallinput_8conv_1_6706562conv_1_6706564*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_Conv_1_layer_call_and_return_conditional_losses_6705869?
#BatchNorm_1/StatefulPartitionedCallStatefulPartitionedCall'Conv_1/StatefulPartitionedCall:output:0batchnorm_1_6706567batchnorm_1_6706569batchnorm_1_6706571batchnorm_1_6706573*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_BatchNorm_1_layer_call_and_return_conditional_losses_6705765?
Conv_2/StatefulPartitionedCallStatefulPartitionedCall,BatchNorm_1/StatefulPartitionedCall:output:0conv_2_6706576conv_2_6706578*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????		 *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_Conv_2_layer_call_and_return_conditional_losses_6705894?
activation_6/PartitionedCallPartitionedCall'Conv_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????		 * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_activation_6_layer_call_and_return_conditional_losses_6705905?
Conv_3/StatefulPartitionedCallStatefulPartitionedCall%activation_6/PartitionedCall:output:0conv_3_6706582conv_3_6706584*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_Conv_3_layer_call_and_return_conditional_losses_6705917?
MaxPool_1/PartitionedCallPartitionedCall'Conv_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_MaxPool_1_layer_call_and_return_conditional_losses_6705785?
activation_7/PartitionedCallPartitionedCall"MaxPool_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_activation_7_layer_call_and_return_conditional_losses_6705929?
#BatchNorm_2/StatefulPartitionedCallStatefulPartitionedCall%activation_7/PartitionedCall:output:0batchnorm_2_6706589batchnorm_2_6706591batchnorm_2_6706593batchnorm_2_6706595*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_BatchNorm_2_layer_call_and_return_conditional_losses_6705841?
Conv_4/StatefulPartitionedCallStatefulPartitionedCall,BatchNorm_2/StatefulPartitionedCall:output:0conv_4_6706598conv_4_6706600*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_Conv_4_layer_call_and_return_conditional_losses_6705950?
flatten_3/PartitionedCallPartitionedCall'Conv_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_flatten_3_layer_call_and_return_conditional_losses_6705962?
!Dropout_0/StatefulPartitionedCallStatefulPartitionedCall"flatten_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_Dropout_0_layer_call_and_return_conditional_losses_6706187?
Dense_0/StatefulPartitionedCallStatefulPartitionedCall*Dropout_0/StatefulPartitionedCall:output:0dense_0_6706605dense_0_6706607*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_Dense_0_layer_call_and_return_conditional_losses_6705993?
Dense_1/StatefulPartitionedCallStatefulPartitionedCall(Dense_0/StatefulPartitionedCall:output:0dense_1_6706610dense_1_6706612*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_Dense_1_layer_call_and_return_conditional_losses_6706021?
!Dropout_2/StatefulPartitionedCallStatefulPartitionedCall(Dense_1/StatefulPartitionedCall:output:0"^Dropout_0/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_Dropout_2_layer_call_and_return_conditional_losses_6706144?
$Dense_output/StatefulPartitionedCallStatefulPartitionedCall*Dropout_2/StatefulPartitionedCall:output:0dense_output_6706616*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_Dense_output_layer_call_and_return_conditional_losses_6706042?
0Dense_0/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_0_6706605*
_output_shapes

: @*
dtype0?
!Dense_0/kernel/Regularizer/SquareSquare8Dense_0/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: @q
 Dense_0/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
Dense_0/kernel/Regularizer/SumSum%Dense_0/kernel/Regularizer/Square:y:0)Dense_0/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: e
 Dense_0/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??L=?
Dense_0/kernel/Regularizer/mulMul)Dense_0/kernel/Regularizer/mul/x:output:0'Dense_0/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: w
+Dense_0/bias/Regularizer/Abs/ReadVariableOpReadVariableOpdense_0_6706607*
_output_shapes
:@*
dtype0}
Dense_0/bias/Regularizer/AbsAbs3Dense_0/bias/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:@h
Dense_0/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
Dense_0/bias/Regularizer/SumSum Dense_0/bias/Regularizer/Abs:y:0'Dense_0/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: c
Dense_0/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
Dense_0/bias/Regularizer/mulMul'Dense_0/bias/Regularizer/mul/x:output:0%Dense_0/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
0Dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_1_6706610*
_output_shapes

:@@*
dtype0?
!Dense_1/kernel/Regularizer/SquareSquare8Dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:@@q
 Dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
Dense_1/kernel/Regularizer/SumSum%Dense_1/kernel/Regularizer/Square:y:0)Dense_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: e
 Dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
Dense_1/kernel/Regularizer/mulMul)Dense_1/kernel/Regularizer/mul/x:output:0'Dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: w
+Dense_1/bias/Regularizer/Abs/ReadVariableOpReadVariableOpdense_1_6706612*
_output_shapes
:@*
dtype0}
Dense_1/bias/Regularizer/AbsAbs3Dense_1/bias/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:@h
Dense_1/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
Dense_1/bias/Regularizer/SumSum Dense_1/bias/Regularizer/Abs:y:0'Dense_1/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: c
Dense_1/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
Dense_1/bias/Regularizer/mulMul'Dense_1/bias/Regularizer/mul/x:output:0%Dense_1/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: |
IdentityIdentity-Dense_output/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp$^BatchNorm_1/StatefulPartitionedCall$^BatchNorm_2/StatefulPartitionedCall^Conv_1/StatefulPartitionedCall^Conv_2/StatefulPartitionedCall^Conv_3/StatefulPartitionedCall^Conv_4/StatefulPartitionedCall ^Dense_0/StatefulPartitionedCall,^Dense_0/bias/Regularizer/Abs/ReadVariableOp1^Dense_0/kernel/Regularizer/Square/ReadVariableOp ^Dense_1/StatefulPartitionedCall,^Dense_1/bias/Regularizer/Abs/ReadVariableOp1^Dense_1/kernel/Regularizer/Square/ReadVariableOp%^Dense_output/StatefulPartitionedCall"^Dropout_0/StatefulPartitionedCall"^Dropout_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*X
_input_shapesG
E:?????????: : : : : : : : : : : : : : : : : : : : : 2J
#BatchNorm_1/StatefulPartitionedCall#BatchNorm_1/StatefulPartitionedCall2J
#BatchNorm_2/StatefulPartitionedCall#BatchNorm_2/StatefulPartitionedCall2@
Conv_1/StatefulPartitionedCallConv_1/StatefulPartitionedCall2@
Conv_2/StatefulPartitionedCallConv_2/StatefulPartitionedCall2@
Conv_3/StatefulPartitionedCallConv_3/StatefulPartitionedCall2@
Conv_4/StatefulPartitionedCallConv_4/StatefulPartitionedCall2B
Dense_0/StatefulPartitionedCallDense_0/StatefulPartitionedCall2Z
+Dense_0/bias/Regularizer/Abs/ReadVariableOp+Dense_0/bias/Regularizer/Abs/ReadVariableOp2d
0Dense_0/kernel/Regularizer/Square/ReadVariableOp0Dense_0/kernel/Regularizer/Square/ReadVariableOp2B
Dense_1/StatefulPartitionedCallDense_1/StatefulPartitionedCall2Z
+Dense_1/bias/Regularizer/Abs/ReadVariableOp+Dense_1/bias/Regularizer/Abs/ReadVariableOp2d
0Dense_1/kernel/Regularizer/Square/ReadVariableOp0Dense_1/kernel/Regularizer/Square/ReadVariableOp2L
$Dense_output/StatefulPartitionedCall$Dense_output/StatefulPartitionedCall2F
!Dropout_0/StatefulPartitionedCall!Dropout_0/StatefulPartitionedCall2F
!Dropout_2/StatefulPartitionedCall!Dropout_2/StatefulPartitionedCall:X T
/
_output_shapes
:?????????
!
_user_specified_name	input_8
?f
?
"__inference__wrapped_model_6705712
input_8G
-model_7_conv_1_conv2d_readvariableop_resource: <
.model_7_conv_1_biasadd_readvariableop_resource: 9
+model_7_batchnorm_1_readvariableop_resource: ;
-model_7_batchnorm_1_readvariableop_1_resource: J
<model_7_batchnorm_1_fusedbatchnormv3_readvariableop_resource: L
>model_7_batchnorm_1_fusedbatchnormv3_readvariableop_1_resource: G
-model_7_conv_2_conv2d_readvariableop_resource:  <
.model_7_conv_2_biasadd_readvariableop_resource: G
-model_7_conv_3_conv2d_readvariableop_resource:  <
.model_7_conv_3_biasadd_readvariableop_resource: 9
+model_7_batchnorm_2_readvariableop_resource: ;
-model_7_batchnorm_2_readvariableop_1_resource: J
<model_7_batchnorm_2_fusedbatchnormv3_readvariableop_resource: L
>model_7_batchnorm_2_fusedbatchnormv3_readvariableop_1_resource: G
-model_7_conv_4_conv2d_readvariableop_resource:  <
.model_7_conv_4_biasadd_readvariableop_resource: @
.model_7_dense_0_matmul_readvariableop_resource: @=
/model_7_dense_0_biasadd_readvariableop_resource:@@
.model_7_dense_1_matmul_readvariableop_resource:@@=
/model_7_dense_1_biasadd_readvariableop_resource:@E
3model_7_dense_output_matmul_readvariableop_resource:@
identity??3model_7/BatchNorm_1/FusedBatchNormV3/ReadVariableOp?5model_7/BatchNorm_1/FusedBatchNormV3/ReadVariableOp_1?"model_7/BatchNorm_1/ReadVariableOp?$model_7/BatchNorm_1/ReadVariableOp_1?3model_7/BatchNorm_2/FusedBatchNormV3/ReadVariableOp?5model_7/BatchNorm_2/FusedBatchNormV3/ReadVariableOp_1?"model_7/BatchNorm_2/ReadVariableOp?$model_7/BatchNorm_2/ReadVariableOp_1?%model_7/Conv_1/BiasAdd/ReadVariableOp?$model_7/Conv_1/Conv2D/ReadVariableOp?%model_7/Conv_2/BiasAdd/ReadVariableOp?$model_7/Conv_2/Conv2D/ReadVariableOp?%model_7/Conv_3/BiasAdd/ReadVariableOp?$model_7/Conv_3/Conv2D/ReadVariableOp?%model_7/Conv_4/BiasAdd/ReadVariableOp?$model_7/Conv_4/Conv2D/ReadVariableOp?&model_7/Dense_0/BiasAdd/ReadVariableOp?%model_7/Dense_0/MatMul/ReadVariableOp?&model_7/Dense_1/BiasAdd/ReadVariableOp?%model_7/Dense_1/MatMul/ReadVariableOp?*model_7/Dense_output/MatMul/ReadVariableOp?
$model_7/Conv_1/Conv2D/ReadVariableOpReadVariableOp-model_7_conv_1_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0?
model_7/Conv_1/Conv2DConv2Dinput_8,model_7/Conv_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
?
%model_7/Conv_1/BiasAdd/ReadVariableOpReadVariableOp.model_7_conv_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
model_7/Conv_1/BiasAddBiasAddmodel_7/Conv_1/Conv2D:output:0-model_7/Conv_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? ?
"model_7/BatchNorm_1/ReadVariableOpReadVariableOp+model_7_batchnorm_1_readvariableop_resource*
_output_shapes
: *
dtype0?
$model_7/BatchNorm_1/ReadVariableOp_1ReadVariableOp-model_7_batchnorm_1_readvariableop_1_resource*
_output_shapes
: *
dtype0?
3model_7/BatchNorm_1/FusedBatchNormV3/ReadVariableOpReadVariableOp<model_7_batchnorm_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
5model_7/BatchNorm_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp>model_7_batchnorm_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
$model_7/BatchNorm_1/FusedBatchNormV3FusedBatchNormV3model_7/Conv_1/BiasAdd:output:0*model_7/BatchNorm_1/ReadVariableOp:value:0,model_7/BatchNorm_1/ReadVariableOp_1:value:0;model_7/BatchNorm_1/FusedBatchNormV3/ReadVariableOp:value:0=model_7/BatchNorm_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:????????? : : : : :*
epsilon%o?:*
is_training( ?
$model_7/Conv_2/Conv2D/ReadVariableOpReadVariableOp-model_7_conv_2_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0?
model_7/Conv_2/Conv2DConv2D(model_7/BatchNorm_1/FusedBatchNormV3:y:0,model_7/Conv_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????		 *
paddingVALID*
strides
?
%model_7/Conv_2/BiasAdd/ReadVariableOpReadVariableOp.model_7_conv_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
model_7/Conv_2/BiasAddBiasAddmodel_7/Conv_2/Conv2D:output:0-model_7/Conv_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????		 |
model_7/activation_6/ReluRelumodel_7/Conv_2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????		 ?
$model_7/Conv_3/Conv2D/ReadVariableOpReadVariableOp-model_7_conv_3_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0?
model_7/Conv_3/Conv2DConv2D'model_7/activation_6/Relu:activations:0,model_7/Conv_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
?
%model_7/Conv_3/BiasAdd/ReadVariableOpReadVariableOp.model_7_conv_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
model_7/Conv_3/BiasAddBiasAddmodel_7/Conv_3/Conv2D:output:0-model_7/Conv_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? ?
model_7/MaxPool_1/MaxPoolMaxPoolmodel_7/Conv_3/BiasAdd:output:0*/
_output_shapes
:????????? *
ksize
*
paddingVALID*
strides

model_7/activation_7/ReluRelu"model_7/MaxPool_1/MaxPool:output:0*
T0*/
_output_shapes
:????????? ?
"model_7/BatchNorm_2/ReadVariableOpReadVariableOp+model_7_batchnorm_2_readvariableop_resource*
_output_shapes
: *
dtype0?
$model_7/BatchNorm_2/ReadVariableOp_1ReadVariableOp-model_7_batchnorm_2_readvariableop_1_resource*
_output_shapes
: *
dtype0?
3model_7/BatchNorm_2/FusedBatchNormV3/ReadVariableOpReadVariableOp<model_7_batchnorm_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
5model_7/BatchNorm_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp>model_7_batchnorm_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
$model_7/BatchNorm_2/FusedBatchNormV3FusedBatchNormV3'model_7/activation_7/Relu:activations:0*model_7/BatchNorm_2/ReadVariableOp:value:0,model_7/BatchNorm_2/ReadVariableOp_1:value:0;model_7/BatchNorm_2/FusedBatchNormV3/ReadVariableOp:value:0=model_7/BatchNorm_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:????????? : : : : :*
epsilon%o?:*
is_training( ?
$model_7/Conv_4/Conv2D/ReadVariableOpReadVariableOp-model_7_conv_4_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0?
model_7/Conv_4/Conv2DConv2D(model_7/BatchNorm_2/FusedBatchNormV3:y:0,model_7/Conv_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
?
%model_7/Conv_4/BiasAdd/ReadVariableOpReadVariableOp.model_7_conv_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
model_7/Conv_4/BiasAddBiasAddmodel_7/Conv_4/Conv2D:output:0-model_7/Conv_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? h
model_7/flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"????    ?
model_7/flatten_3/ReshapeReshapemodel_7/Conv_4/BiasAdd:output:0 model_7/flatten_3/Const:output:0*
T0*'
_output_shapes
:????????? |
model_7/Dropout_0/IdentityIdentity"model_7/flatten_3/Reshape:output:0*
T0*'
_output_shapes
:????????? ?
%model_7/Dense_0/MatMul/ReadVariableOpReadVariableOp.model_7_dense_0_matmul_readvariableop_resource*
_output_shapes

: @*
dtype0?
model_7/Dense_0/MatMulMatMul#model_7/Dropout_0/Identity:output:0-model_7/Dense_0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
&model_7/Dense_0/BiasAdd/ReadVariableOpReadVariableOp/model_7_dense_0_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
model_7/Dense_0/BiasAddBiasAdd model_7/Dense_0/MatMul:product:0.model_7/Dense_0/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
%model_7/Dense_1/MatMul/ReadVariableOpReadVariableOp.model_7_dense_1_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0?
model_7/Dense_1/MatMulMatMul model_7/Dense_0/BiasAdd:output:0-model_7/Dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
&model_7/Dense_1/BiasAdd/ReadVariableOpReadVariableOp/model_7_dense_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
model_7/Dense_1/BiasAddBiasAdd model_7/Dense_1/MatMul:product:0.model_7/Dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@z
model_7/Dropout_2/IdentityIdentity model_7/Dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@?
*model_7/Dense_output/MatMul/ReadVariableOpReadVariableOp3model_7_dense_output_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0?
model_7/Dense_output/MatMulMatMul#model_7/Dropout_2/Identity:output:02model_7/Dense_output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
model_7/Dense_output/SoftmaxSoftmax%model_7/Dense_output/MatMul:product:0*
T0*'
_output_shapes
:?????????u
IdentityIdentity&model_7/Dense_output/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp4^model_7/BatchNorm_1/FusedBatchNormV3/ReadVariableOp6^model_7/BatchNorm_1/FusedBatchNormV3/ReadVariableOp_1#^model_7/BatchNorm_1/ReadVariableOp%^model_7/BatchNorm_1/ReadVariableOp_14^model_7/BatchNorm_2/FusedBatchNormV3/ReadVariableOp6^model_7/BatchNorm_2/FusedBatchNormV3/ReadVariableOp_1#^model_7/BatchNorm_2/ReadVariableOp%^model_7/BatchNorm_2/ReadVariableOp_1&^model_7/Conv_1/BiasAdd/ReadVariableOp%^model_7/Conv_1/Conv2D/ReadVariableOp&^model_7/Conv_2/BiasAdd/ReadVariableOp%^model_7/Conv_2/Conv2D/ReadVariableOp&^model_7/Conv_3/BiasAdd/ReadVariableOp%^model_7/Conv_3/Conv2D/ReadVariableOp&^model_7/Conv_4/BiasAdd/ReadVariableOp%^model_7/Conv_4/Conv2D/ReadVariableOp'^model_7/Dense_0/BiasAdd/ReadVariableOp&^model_7/Dense_0/MatMul/ReadVariableOp'^model_7/Dense_1/BiasAdd/ReadVariableOp&^model_7/Dense_1/MatMul/ReadVariableOp+^model_7/Dense_output/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*X
_input_shapesG
E:?????????: : : : : : : : : : : : : : : : : : : : : 2j
3model_7/BatchNorm_1/FusedBatchNormV3/ReadVariableOp3model_7/BatchNorm_1/FusedBatchNormV3/ReadVariableOp2n
5model_7/BatchNorm_1/FusedBatchNormV3/ReadVariableOp_15model_7/BatchNorm_1/FusedBatchNormV3/ReadVariableOp_12H
"model_7/BatchNorm_1/ReadVariableOp"model_7/BatchNorm_1/ReadVariableOp2L
$model_7/BatchNorm_1/ReadVariableOp_1$model_7/BatchNorm_1/ReadVariableOp_12j
3model_7/BatchNorm_2/FusedBatchNormV3/ReadVariableOp3model_7/BatchNorm_2/FusedBatchNormV3/ReadVariableOp2n
5model_7/BatchNorm_2/FusedBatchNormV3/ReadVariableOp_15model_7/BatchNorm_2/FusedBatchNormV3/ReadVariableOp_12H
"model_7/BatchNorm_2/ReadVariableOp"model_7/BatchNorm_2/ReadVariableOp2L
$model_7/BatchNorm_2/ReadVariableOp_1$model_7/BatchNorm_2/ReadVariableOp_12N
%model_7/Conv_1/BiasAdd/ReadVariableOp%model_7/Conv_1/BiasAdd/ReadVariableOp2L
$model_7/Conv_1/Conv2D/ReadVariableOp$model_7/Conv_1/Conv2D/ReadVariableOp2N
%model_7/Conv_2/BiasAdd/ReadVariableOp%model_7/Conv_2/BiasAdd/ReadVariableOp2L
$model_7/Conv_2/Conv2D/ReadVariableOp$model_7/Conv_2/Conv2D/ReadVariableOp2N
%model_7/Conv_3/BiasAdd/ReadVariableOp%model_7/Conv_3/BiasAdd/ReadVariableOp2L
$model_7/Conv_3/Conv2D/ReadVariableOp$model_7/Conv_3/Conv2D/ReadVariableOp2N
%model_7/Conv_4/BiasAdd/ReadVariableOp%model_7/Conv_4/BiasAdd/ReadVariableOp2L
$model_7/Conv_4/Conv2D/ReadVariableOp$model_7/Conv_4/Conv2D/ReadVariableOp2P
&model_7/Dense_0/BiasAdd/ReadVariableOp&model_7/Dense_0/BiasAdd/ReadVariableOp2N
%model_7/Dense_0/MatMul/ReadVariableOp%model_7/Dense_0/MatMul/ReadVariableOp2P
&model_7/Dense_1/BiasAdd/ReadVariableOp&model_7/Dense_1/BiasAdd/ReadVariableOp2N
%model_7/Dense_1/MatMul/ReadVariableOp%model_7/Dense_1/MatMul/ReadVariableOp2X
*model_7/Dense_output/MatMul/ReadVariableOp*model_7/Dense_output/MatMul/ReadVariableOp:X T
/
_output_shapes
:?????????
!
_user_specified_name	input_8
?`
?

D__inference_model_7_layer_call_and_return_conditional_losses_6706071

inputs(
conv_1_6705870: 
conv_1_6705872: !
batchnorm_1_6705875: !
batchnorm_1_6705877: !
batchnorm_1_6705879: !
batchnorm_1_6705881: (
conv_2_6705895:  
conv_2_6705897: (
conv_3_6705918:  
conv_3_6705920: !
batchnorm_2_6705931: !
batchnorm_2_6705933: !
batchnorm_2_6705935: !
batchnorm_2_6705937: (
conv_4_6705951:  
conv_4_6705953: !
dense_0_6705994: @
dense_0_6705996:@!
dense_1_6706022:@@
dense_1_6706024:@&
dense_output_6706043:@
identity??#BatchNorm_1/StatefulPartitionedCall?#BatchNorm_2/StatefulPartitionedCall?Conv_1/StatefulPartitionedCall?Conv_2/StatefulPartitionedCall?Conv_3/StatefulPartitionedCall?Conv_4/StatefulPartitionedCall?Dense_0/StatefulPartitionedCall?+Dense_0/bias/Regularizer/Abs/ReadVariableOp?0Dense_0/kernel/Regularizer/Square/ReadVariableOp?Dense_1/StatefulPartitionedCall?+Dense_1/bias/Regularizer/Abs/ReadVariableOp?0Dense_1/kernel/Regularizer/Square/ReadVariableOp?$Dense_output/StatefulPartitionedCall?
Conv_1/StatefulPartitionedCallStatefulPartitionedCallinputsconv_1_6705870conv_1_6705872*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_Conv_1_layer_call_and_return_conditional_losses_6705869?
#BatchNorm_1/StatefulPartitionedCallStatefulPartitionedCall'Conv_1/StatefulPartitionedCall:output:0batchnorm_1_6705875batchnorm_1_6705877batchnorm_1_6705879batchnorm_1_6705881*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_BatchNorm_1_layer_call_and_return_conditional_losses_6705734?
Conv_2/StatefulPartitionedCallStatefulPartitionedCall,BatchNorm_1/StatefulPartitionedCall:output:0conv_2_6705895conv_2_6705897*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????		 *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_Conv_2_layer_call_and_return_conditional_losses_6705894?
activation_6/PartitionedCallPartitionedCall'Conv_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????		 * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_activation_6_layer_call_and_return_conditional_losses_6705905?
Conv_3/StatefulPartitionedCallStatefulPartitionedCall%activation_6/PartitionedCall:output:0conv_3_6705918conv_3_6705920*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_Conv_3_layer_call_and_return_conditional_losses_6705917?
MaxPool_1/PartitionedCallPartitionedCall'Conv_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_MaxPool_1_layer_call_and_return_conditional_losses_6705785?
activation_7/PartitionedCallPartitionedCall"MaxPool_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_activation_7_layer_call_and_return_conditional_losses_6705929?
#BatchNorm_2/StatefulPartitionedCallStatefulPartitionedCall%activation_7/PartitionedCall:output:0batchnorm_2_6705931batchnorm_2_6705933batchnorm_2_6705935batchnorm_2_6705937*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_BatchNorm_2_layer_call_and_return_conditional_losses_6705810?
Conv_4/StatefulPartitionedCallStatefulPartitionedCall,BatchNorm_2/StatefulPartitionedCall:output:0conv_4_6705951conv_4_6705953*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_Conv_4_layer_call_and_return_conditional_losses_6705950?
flatten_3/PartitionedCallPartitionedCall'Conv_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_flatten_3_layer_call_and_return_conditional_losses_6705962?
Dropout_0/PartitionedCallPartitionedCall"flatten_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_Dropout_0_layer_call_and_return_conditional_losses_6705969?
Dense_0/StatefulPartitionedCallStatefulPartitionedCall"Dropout_0/PartitionedCall:output:0dense_0_6705994dense_0_6705996*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_Dense_0_layer_call_and_return_conditional_losses_6705993?
Dense_1/StatefulPartitionedCallStatefulPartitionedCall(Dense_0/StatefulPartitionedCall:output:0dense_1_6706022dense_1_6706024*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_Dense_1_layer_call_and_return_conditional_losses_6706021?
Dropout_2/PartitionedCallPartitionedCall(Dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_Dropout_2_layer_call_and_return_conditional_losses_6706032?
$Dense_output/StatefulPartitionedCallStatefulPartitionedCall"Dropout_2/PartitionedCall:output:0dense_output_6706043*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_Dense_output_layer_call_and_return_conditional_losses_6706042?
0Dense_0/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_0_6705994*
_output_shapes

: @*
dtype0?
!Dense_0/kernel/Regularizer/SquareSquare8Dense_0/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: @q
 Dense_0/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
Dense_0/kernel/Regularizer/SumSum%Dense_0/kernel/Regularizer/Square:y:0)Dense_0/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: e
 Dense_0/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??L=?
Dense_0/kernel/Regularizer/mulMul)Dense_0/kernel/Regularizer/mul/x:output:0'Dense_0/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: w
+Dense_0/bias/Regularizer/Abs/ReadVariableOpReadVariableOpdense_0_6705996*
_output_shapes
:@*
dtype0}
Dense_0/bias/Regularizer/AbsAbs3Dense_0/bias/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:@h
Dense_0/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
Dense_0/bias/Regularizer/SumSum Dense_0/bias/Regularizer/Abs:y:0'Dense_0/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: c
Dense_0/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
Dense_0/bias/Regularizer/mulMul'Dense_0/bias/Regularizer/mul/x:output:0%Dense_0/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
0Dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_1_6706022*
_output_shapes

:@@*
dtype0?
!Dense_1/kernel/Regularizer/SquareSquare8Dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:@@q
 Dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
Dense_1/kernel/Regularizer/SumSum%Dense_1/kernel/Regularizer/Square:y:0)Dense_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: e
 Dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
Dense_1/kernel/Regularizer/mulMul)Dense_1/kernel/Regularizer/mul/x:output:0'Dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: w
+Dense_1/bias/Regularizer/Abs/ReadVariableOpReadVariableOpdense_1_6706024*
_output_shapes
:@*
dtype0}
Dense_1/bias/Regularizer/AbsAbs3Dense_1/bias/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:@h
Dense_1/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
Dense_1/bias/Regularizer/SumSum Dense_1/bias/Regularizer/Abs:y:0'Dense_1/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: c
Dense_1/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
Dense_1/bias/Regularizer/mulMul'Dense_1/bias/Regularizer/mul/x:output:0%Dense_1/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: |
IdentityIdentity-Dense_output/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp$^BatchNorm_1/StatefulPartitionedCall$^BatchNorm_2/StatefulPartitionedCall^Conv_1/StatefulPartitionedCall^Conv_2/StatefulPartitionedCall^Conv_3/StatefulPartitionedCall^Conv_4/StatefulPartitionedCall ^Dense_0/StatefulPartitionedCall,^Dense_0/bias/Regularizer/Abs/ReadVariableOp1^Dense_0/kernel/Regularizer/Square/ReadVariableOp ^Dense_1/StatefulPartitionedCall,^Dense_1/bias/Regularizer/Abs/ReadVariableOp1^Dense_1/kernel/Regularizer/Square/ReadVariableOp%^Dense_output/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*X
_input_shapesG
E:?????????: : : : : : : : : : : : : : : : : : : : : 2J
#BatchNorm_1/StatefulPartitionedCall#BatchNorm_1/StatefulPartitionedCall2J
#BatchNorm_2/StatefulPartitionedCall#BatchNorm_2/StatefulPartitionedCall2@
Conv_1/StatefulPartitionedCallConv_1/StatefulPartitionedCall2@
Conv_2/StatefulPartitionedCallConv_2/StatefulPartitionedCall2@
Conv_3/StatefulPartitionedCallConv_3/StatefulPartitionedCall2@
Conv_4/StatefulPartitionedCallConv_4/StatefulPartitionedCall2B
Dense_0/StatefulPartitionedCallDense_0/StatefulPartitionedCall2Z
+Dense_0/bias/Regularizer/Abs/ReadVariableOp+Dense_0/bias/Regularizer/Abs/ReadVariableOp2d
0Dense_0/kernel/Regularizer/Square/ReadVariableOp0Dense_0/kernel/Regularizer/Square/ReadVariableOp2B
Dense_1/StatefulPartitionedCallDense_1/StatefulPartitionedCall2Z
+Dense_1/bias/Regularizer/Abs/ReadVariableOp+Dense_1/bias/Regularizer/Abs/ReadVariableOp2d
0Dense_1/kernel/Regularizer/Square/ReadVariableOp0Dense_1/kernel/Regularizer/Square/ReadVariableOp2L
$Dense_output/StatefulPartitionedCall$Dense_output/StatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
b
F__inference_flatten_3_layer_call_and_return_conditional_losses_6707276

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"????    \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:????????? X
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:????????? "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:????????? :W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
d
F__inference_Dropout_0_layer_call_and_return_conditional_losses_6705969

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:????????? [

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:????????? "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:????????? :O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?	
?
-__inference_BatchNorm_2_layer_call_fn_6707210

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
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
GPU2*0J 8? *Q
fLRJ
H__inference_BatchNorm_2_layer_call_and_return_conditional_losses_6705841?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+??????????????????????????? `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+??????????????????????????? : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?

?
__inference_loss_fn_3_6707475B
4dense_1_bias_regularizer_abs_readvariableop_resource:@
identity??+Dense_1/bias/Regularizer/Abs/ReadVariableOp?
+Dense_1/bias/Regularizer/Abs/ReadVariableOpReadVariableOp4dense_1_bias_regularizer_abs_readvariableop_resource*
_output_shapes
:@*
dtype0}
Dense_1/bias/Regularizer/AbsAbs3Dense_1/bias/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:@h
Dense_1/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
Dense_1/bias/Regularizer/SumSum Dense_1/bias/Regularizer/Abs:y:0'Dense_1/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: c
Dense_1/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
Dense_1/bias/Regularizer/mulMul'Dense_1/bias/Regularizer/mul/x:output:0%Dense_1/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: ^
IdentityIdentity Dense_1/bias/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: t
NoOpNoOp,^Dense_1/bias/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2Z
+Dense_1/bias/Regularizer/Abs/ReadVariableOp+Dense_1/bias/Regularizer/Abs/ReadVariableOp
??
?2
#__inference__traced_restore_6708000
file_prefix8
assignvariableop_conv_1_kernel: ,
assignvariableop_1_conv_1_bias: 2
$assignvariableop_2_batchnorm_1_gamma: 1
#assignvariableop_3_batchnorm_1_beta: 8
*assignvariableop_4_batchnorm_1_moving_mean: <
.assignvariableop_5_batchnorm_1_moving_variance: :
 assignvariableop_6_conv_2_kernel:  ,
assignvariableop_7_conv_2_bias: :
 assignvariableop_8_conv_3_kernel:  ,
assignvariableop_9_conv_3_bias: 3
%assignvariableop_10_batchnorm_2_gamma: 2
$assignvariableop_11_batchnorm_2_beta: 9
+assignvariableop_12_batchnorm_2_moving_mean: =
/assignvariableop_13_batchnorm_2_moving_variance: ;
!assignvariableop_14_conv_4_kernel:  -
assignvariableop_15_conv_4_bias: 4
"assignvariableop_16_dense_0_kernel: @.
 assignvariableop_17_dense_0_bias:@4
"assignvariableop_18_dense_1_kernel:@@.
 assignvariableop_19_dense_1_bias:@9
'assignvariableop_20_dense_output_kernel:@'
assignvariableop_21_adam_iter:	 )
assignvariableop_22_adam_beta_1: )
assignvariableop_23_adam_beta_2: (
assignvariableop_24_adam_decay: #
assignvariableop_25_total: #
assignvariableop_26_count: %
assignvariableop_27_total_1: %
assignvariableop_28_count_1: %
assignvariableop_29_total_2: %
assignvariableop_30_count_2: B
(assignvariableop_31_adam_conv_1_kernel_m: 4
&assignvariableop_32_adam_conv_1_bias_m: :
,assignvariableop_33_adam_batchnorm_1_gamma_m: 9
+assignvariableop_34_adam_batchnorm_1_beta_m: B
(assignvariableop_35_adam_conv_2_kernel_m:  4
&assignvariableop_36_adam_conv_2_bias_m: B
(assignvariableop_37_adam_conv_3_kernel_m:  4
&assignvariableop_38_adam_conv_3_bias_m: :
,assignvariableop_39_adam_batchnorm_2_gamma_m: 9
+assignvariableop_40_adam_batchnorm_2_beta_m: B
(assignvariableop_41_adam_conv_4_kernel_m:  4
&assignvariableop_42_adam_conv_4_bias_m: ;
)assignvariableop_43_adam_dense_0_kernel_m: @5
'assignvariableop_44_adam_dense_0_bias_m:@;
)assignvariableop_45_adam_dense_1_kernel_m:@@5
'assignvariableop_46_adam_dense_1_bias_m:@@
.assignvariableop_47_adam_dense_output_kernel_m:@B
(assignvariableop_48_adam_conv_1_kernel_v: 4
&assignvariableop_49_adam_conv_1_bias_v: :
,assignvariableop_50_adam_batchnorm_1_gamma_v: 9
+assignvariableop_51_adam_batchnorm_1_beta_v: B
(assignvariableop_52_adam_conv_2_kernel_v:  4
&assignvariableop_53_adam_conv_2_bias_v: B
(assignvariableop_54_adam_conv_3_kernel_v:  4
&assignvariableop_55_adam_conv_3_bias_v: :
,assignvariableop_56_adam_batchnorm_2_gamma_v: 9
+assignvariableop_57_adam_batchnorm_2_beta_v: B
(assignvariableop_58_adam_conv_4_kernel_v:  4
&assignvariableop_59_adam_conv_4_bias_v: ;
)assignvariableop_60_adam_dense_0_kernel_v: @5
'assignvariableop_61_adam_dense_0_bias_v:@;
)assignvariableop_62_adam_dense_1_kernel_v:@@5
'assignvariableop_63_adam_dense_1_bias_v:@@
.assignvariableop_64_adam_dense_output_kernel_v:@E
+assignvariableop_65_adam_conv_1_kernel_vhat: 7
)assignvariableop_66_adam_conv_1_bias_vhat: =
/assignvariableop_67_adam_batchnorm_1_gamma_vhat: <
.assignvariableop_68_adam_batchnorm_1_beta_vhat: E
+assignvariableop_69_adam_conv_2_kernel_vhat:  7
)assignvariableop_70_adam_conv_2_bias_vhat: E
+assignvariableop_71_adam_conv_3_kernel_vhat:  7
)assignvariableop_72_adam_conv_3_bias_vhat: =
/assignvariableop_73_adam_batchnorm_2_gamma_vhat: <
.assignvariableop_74_adam_batchnorm_2_beta_vhat: E
+assignvariableop_75_adam_conv_4_kernel_vhat:  7
)assignvariableop_76_adam_conv_4_bias_vhat: >
,assignvariableop_77_adam_dense_0_kernel_vhat: @8
*assignvariableop_78_adam_dense_0_bias_vhat:@>
,assignvariableop_79_adam_dense_1_kernel_vhat:@@8
*assignvariableop_80_adam_dense_1_bias_vhat:@C
1assignvariableop_81_adam_dense_output_kernel_vhat:@
identity_83??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_43?AssignVariableOp_44?AssignVariableOp_45?AssignVariableOp_46?AssignVariableOp_47?AssignVariableOp_48?AssignVariableOp_49?AssignVariableOp_5?AssignVariableOp_50?AssignVariableOp_51?AssignVariableOp_52?AssignVariableOp_53?AssignVariableOp_54?AssignVariableOp_55?AssignVariableOp_56?AssignVariableOp_57?AssignVariableOp_58?AssignVariableOp_59?AssignVariableOp_6?AssignVariableOp_60?AssignVariableOp_61?AssignVariableOp_62?AssignVariableOp_63?AssignVariableOp_64?AssignVariableOp_65?AssignVariableOp_66?AssignVariableOp_67?AssignVariableOp_68?AssignVariableOp_69?AssignVariableOp_7?AssignVariableOp_70?AssignVariableOp_71?AssignVariableOp_72?AssignVariableOp_73?AssignVariableOp_74?AssignVariableOp_75?AssignVariableOp_76?AssignVariableOp_77?AssignVariableOp_78?AssignVariableOp_79?AssignVariableOp_8?AssignVariableOp_80?AssignVariableOp_81?AssignVariableOp_9?/
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:S*
dtype0*?/
value?.B?.SB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:S*
dtype0*?
value?B?SB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*a
dtypesW
U2S	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOpAssignVariableOpassignvariableop_conv_1_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_1AssignVariableOpassignvariableop_1_conv_1_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_2AssignVariableOp$assignvariableop_2_batchnorm_1_gammaIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_3AssignVariableOp#assignvariableop_3_batchnorm_1_betaIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_4AssignVariableOp*assignvariableop_4_batchnorm_1_moving_meanIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_5AssignVariableOp.assignvariableop_5_batchnorm_1_moving_varianceIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_6AssignVariableOp assignvariableop_6_conv_2_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_7AssignVariableOpassignvariableop_7_conv_2_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_8AssignVariableOp assignvariableop_8_conv_3_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_9AssignVariableOpassignvariableop_9_conv_3_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_10AssignVariableOp%assignvariableop_10_batchnorm_2_gammaIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_11AssignVariableOp$assignvariableop_11_batchnorm_2_betaIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_12AssignVariableOp+assignvariableop_12_batchnorm_2_moving_meanIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_13AssignVariableOp/assignvariableop_13_batchnorm_2_moving_varianceIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_14AssignVariableOp!assignvariableop_14_conv_4_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_15AssignVariableOpassignvariableop_15_conv_4_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_16AssignVariableOp"assignvariableop_16_dense_0_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_17AssignVariableOp assignvariableop_17_dense_0_biasIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_18AssignVariableOp"assignvariableop_18_dense_1_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_19AssignVariableOp assignvariableop_19_dense_1_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_20AssignVariableOp'assignvariableop_20_dense_output_kernelIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0	*
_output_shapes
:?
AssignVariableOp_21AssignVariableOpassignvariableop_21_adam_iterIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_22AssignVariableOpassignvariableop_22_adam_beta_1Identity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_23AssignVariableOpassignvariableop_23_adam_beta_2Identity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_24AssignVariableOpassignvariableop_24_adam_decayIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_25AssignVariableOpassignvariableop_25_totalIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_26AssignVariableOpassignvariableop_26_countIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_27AssignVariableOpassignvariableop_27_total_1Identity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_28AssignVariableOpassignvariableop_28_count_1Identity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_29AssignVariableOpassignvariableop_29_total_2Identity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_30AssignVariableOpassignvariableop_30_count_2Identity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_31AssignVariableOp(assignvariableop_31_adam_conv_1_kernel_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_32AssignVariableOp&assignvariableop_32_adam_conv_1_bias_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_33AssignVariableOp,assignvariableop_33_adam_batchnorm_1_gamma_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_34AssignVariableOp+assignvariableop_34_adam_batchnorm_1_beta_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_35AssignVariableOp(assignvariableop_35_adam_conv_2_kernel_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_36AssignVariableOp&assignvariableop_36_adam_conv_2_bias_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_37AssignVariableOp(assignvariableop_37_adam_conv_3_kernel_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_38AssignVariableOp&assignvariableop_38_adam_conv_3_bias_mIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_39AssignVariableOp,assignvariableop_39_adam_batchnorm_2_gamma_mIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_40AssignVariableOp+assignvariableop_40_adam_batchnorm_2_beta_mIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_41AssignVariableOp(assignvariableop_41_adam_conv_4_kernel_mIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_42AssignVariableOp&assignvariableop_42_adam_conv_4_bias_mIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_43AssignVariableOp)assignvariableop_43_adam_dense_0_kernel_mIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_44AssignVariableOp'assignvariableop_44_adam_dense_0_bias_mIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_45AssignVariableOp)assignvariableop_45_adam_dense_1_kernel_mIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_46AssignVariableOp'assignvariableop_46_adam_dense_1_bias_mIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_47AssignVariableOp.assignvariableop_47_adam_dense_output_kernel_mIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_48AssignVariableOp(assignvariableop_48_adam_conv_1_kernel_vIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_49AssignVariableOp&assignvariableop_49_adam_conv_1_bias_vIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_50AssignVariableOp,assignvariableop_50_adam_batchnorm_1_gamma_vIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_51AssignVariableOp+assignvariableop_51_adam_batchnorm_1_beta_vIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_52AssignVariableOp(assignvariableop_52_adam_conv_2_kernel_vIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_53AssignVariableOp&assignvariableop_53_adam_conv_2_bias_vIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_54AssignVariableOp(assignvariableop_54_adam_conv_3_kernel_vIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_55AssignVariableOp&assignvariableop_55_adam_conv_3_bias_vIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_56AssignVariableOp,assignvariableop_56_adam_batchnorm_2_gamma_vIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_57AssignVariableOp+assignvariableop_57_adam_batchnorm_2_beta_vIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_58AssignVariableOp(assignvariableop_58_adam_conv_4_kernel_vIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_59AssignVariableOp&assignvariableop_59_adam_conv_4_bias_vIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_60AssignVariableOp)assignvariableop_60_adam_dense_0_kernel_vIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_61AssignVariableOp'assignvariableop_61_adam_dense_0_bias_vIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_62AssignVariableOp)assignvariableop_62_adam_dense_1_kernel_vIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_63AssignVariableOp'assignvariableop_63_adam_dense_1_bias_vIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_64AssignVariableOp.assignvariableop_64_adam_dense_output_kernel_vIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_65AssignVariableOp+assignvariableop_65_adam_conv_1_kernel_vhatIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_66AssignVariableOp)assignvariableop_66_adam_conv_1_bias_vhatIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_67AssignVariableOp/assignvariableop_67_adam_batchnorm_1_gamma_vhatIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_68AssignVariableOp.assignvariableop_68_adam_batchnorm_1_beta_vhatIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_69AssignVariableOp+assignvariableop_69_adam_conv_2_kernel_vhatIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_70AssignVariableOp)assignvariableop_70_adam_conv_2_bias_vhatIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_71AssignVariableOp+assignvariableop_71_adam_conv_3_kernel_vhatIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_72AssignVariableOp)assignvariableop_72_adam_conv_3_bias_vhatIdentity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_73AssignVariableOp/assignvariableop_73_adam_batchnorm_2_gamma_vhatIdentity_73:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_74AssignVariableOp.assignvariableop_74_adam_batchnorm_2_beta_vhatIdentity_74:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_75AssignVariableOp+assignvariableop_75_adam_conv_4_kernel_vhatIdentity_75:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_76IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_76AssignVariableOp)assignvariableop_76_adam_conv_4_bias_vhatIdentity_76:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_77IdentityRestoreV2:tensors:77"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_77AssignVariableOp,assignvariableop_77_adam_dense_0_kernel_vhatIdentity_77:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_78IdentityRestoreV2:tensors:78"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_78AssignVariableOp*assignvariableop_78_adam_dense_0_bias_vhatIdentity_78:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_79IdentityRestoreV2:tensors:79"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_79AssignVariableOp,assignvariableop_79_adam_dense_1_kernel_vhatIdentity_79:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_80IdentityRestoreV2:tensors:80"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_80AssignVariableOp*assignvariableop_80_adam_dense_1_bias_vhatIdentity_80:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_81IdentityRestoreV2:tensors:81"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_81AssignVariableOp1assignvariableop_81_adam_dense_output_kernel_vhatIdentity_81:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ?
Identity_82Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_83IdentityIdentity_82:output:0^NoOp_1*
T0*
_output_shapes
: ?
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_83Identity_83:output:0*?
_input_shapes?
?: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
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
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682*
AssignVariableOp_69AssignVariableOp_692(
AssignVariableOp_7AssignVariableOp_72*
AssignVariableOp_70AssignVariableOp_702*
AssignVariableOp_71AssignVariableOp_712*
AssignVariableOp_72AssignVariableOp_722*
AssignVariableOp_73AssignVariableOp_732*
AssignVariableOp_74AssignVariableOp_742*
AssignVariableOp_75AssignVariableOp_752*
AssignVariableOp_76AssignVariableOp_762*
AssignVariableOp_77AssignVariableOp_772*
AssignVariableOp_78AssignVariableOp_782*
AssignVariableOp_79AssignVariableOp_792(
AssignVariableOp_8AssignVariableOp_82*
AssignVariableOp_80AssignVariableOp_802*
AssignVariableOp_81AssignVariableOp_812(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
b
F__inference_MaxPool_1_layer_call_and_return_conditional_losses_6707174

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
D__inference_Dense_1_layer_call_and_return_conditional_losses_6707389

inputs0
matmul_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?+Dense_1/bias/Regularizer/Abs/ReadVariableOp?0Dense_1/kernel/Regularizer/Square/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
0Dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
dtype0?
!Dense_1/kernel/Regularizer/SquareSquare8Dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:@@q
 Dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
Dense_1/kernel/Regularizer/SumSum%Dense_1/kernel/Regularizer/Square:y:0)Dense_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: e
 Dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
Dense_1/kernel/Regularizer/mulMul)Dense_1/kernel/Regularizer/mul/x:output:0'Dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
+Dense_1/bias/Regularizer/Abs/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
Dense_1/bias/Regularizer/AbsAbs3Dense_1/bias/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:@h
Dense_1/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
Dense_1/bias/Regularizer/SumSum Dense_1/bias/Regularizer/Abs:y:0'Dense_1/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: c
Dense_1/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
Dense_1/bias/Regularizer/mulMul'Dense_1/bias/Regularizer/mul/x:output:0%Dense_1/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????@?
NoOpNoOp^BiasAdd/ReadVariableOp,^Dense_1/bias/Regularizer/Abs/ReadVariableOp1^Dense_1/kernel/Regularizer/Square/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2Z
+Dense_1/bias/Regularizer/Abs/ReadVariableOp+Dense_1/bias/Regularizer/Abs/ReadVariableOp2d
0Dense_1/kernel/Regularizer/Square/ReadVariableOp0Dense_1/kernel/Regularizer/Square/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
C
input_88
serving_default_input_8:0?????????@
Dense_output0
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
?
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer-4
layer_with_weights-3
layer-5
layer-6
layer-7
	layer_with_weights-4
	layer-8

layer_with_weights-5

layer-9
layer-10
layer-11
layer_with_weights-6
layer-12
layer_with_weights-7
layer-13
layer-14
layer_with_weights-8
layer-15
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures"
_tf_keras_network
"
_tf_keras_input_layer
?

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
 __call__
*!&call_and_return_all_conditional_losses"
_tf_keras_layer
?
"axis
	#gamma
$beta
%moving_mean
&moving_variance
'	variables
(trainable_variables
)regularization_losses
*	keras_api
+__call__
*,&call_and_return_all_conditional_losses"
_tf_keras_layer
?

-kernel
.bias
/	variables
0trainable_variables
1regularization_losses
2	keras_api
3__call__
*4&call_and_return_all_conditional_losses"
_tf_keras_layer
?
5	variables
6trainable_variables
7regularization_losses
8	keras_api
9__call__
*:&call_and_return_all_conditional_losses"
_tf_keras_layer
?

;kernel
<bias
=	variables
>trainable_variables
?regularization_losses
@	keras_api
A__call__
*B&call_and_return_all_conditional_losses"
_tf_keras_layer
?
C	variables
Dtrainable_variables
Eregularization_losses
F	keras_api
G__call__
*H&call_and_return_all_conditional_losses"
_tf_keras_layer
?
I	variables
Jtrainable_variables
Kregularization_losses
L	keras_api
M__call__
*N&call_and_return_all_conditional_losses"
_tf_keras_layer
?
Oaxis
	Pgamma
Qbeta
Rmoving_mean
Smoving_variance
T	variables
Utrainable_variables
Vregularization_losses
W	keras_api
X__call__
*Y&call_and_return_all_conditional_losses"
_tf_keras_layer
?

Zkernel
[bias
\	variables
]trainable_variables
^regularization_losses
_	keras_api
`__call__
*a&call_and_return_all_conditional_losses"
_tf_keras_layer
?
b	variables
ctrainable_variables
dregularization_losses
e	keras_api
f__call__
*g&call_and_return_all_conditional_losses"
_tf_keras_layer
?
h	variables
itrainable_variables
jregularization_losses
k	keras_api
l_random_generator
m__call__
*n&call_and_return_all_conditional_losses"
_tf_keras_layer
?

okernel
pbias
q	variables
rtrainable_variables
sregularization_losses
t	keras_api
u__call__
*v&call_and_return_all_conditional_losses"
_tf_keras_layer
?

wkernel
xbias
y	variables
ztrainable_variables
{regularization_losses
|	keras_api
}__call__
*~&call_and_return_all_conditional_losses"
_tf_keras_layer
?
	variables
?trainable_variables
?regularization_losses
?	keras_api
?_random_generator
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?kernel
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
	?iter
?beta_1
?beta_2

?decaym?m?#m?$m?-m?.m?;m?<m?Pm?Qm?Zm?[m?om?pm?wm?xm?	?m?v?v?#v?$v?-v?.v?;v?<v?Pv?Qv?Zv?[v?ov?pv?wv?xv?	?v?vhat?vhat?#vhat?$vhat?-vhat?.vhat?;vhat?<vhat?Pvhat?Qvhat?Zvhat?[vhat?ovhat?pvhat?wvhat?xvhat??vhat?"
	optimizer
?
0
1
#2
$3
%4
&5
-6
.7
;8
<9
P10
Q11
R12
S13
Z14
[15
o16
p17
w18
x19
?20"
trackable_list_wrapper
?
0
1
#2
$3
-4
.5
;6
<7
P8
Q9
Z10
[11
o12
p13
w14
x15
?16"
trackable_list_wrapper
@
?0
?1
?2
?3"
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
?2?
)__inference_model_7_layer_call_fn_6706116
)__inference_model_7_layer_call_fn_6706719
)__inference_model_7_layer_call_fn_6706766
)__inference_model_7_layer_call_fn_6706474?
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
D__inference_model_7_layer_call_and_return_conditional_losses_6706869
D__inference_model_7_layer_call_and_return_conditional_losses_6706986
D__inference_model_7_layer_call_and_return_conditional_losses_6706559
D__inference_model_7_layer_call_and_return_conditional_losses_6706644?
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
?B?
"__inference__wrapped_model_6705712input_8"?
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
-
?serving_default"
signature_map
':% 2Conv_1/kernel
: 2Conv_1/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
	variables
trainable_variables
regularization_losses
 __call__
*!&call_and_return_all_conditional_losses
&!"call_and_return_conditional_losses"
_generic_user_object
?2?
(__inference_Conv_1_layer_call_fn_6707044?
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
C__inference_Conv_1_layer_call_and_return_conditional_losses_6707054?
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
 "
trackable_list_wrapper
: 2BatchNorm_1/gamma
: 2BatchNorm_1/beta
':%  (2BatchNorm_1/moving_mean
+:)  (2BatchNorm_1/moving_variance
<
#0
$1
%2
&3"
trackable_list_wrapper
.
#0
$1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
'	variables
(trainable_variables
)regularization_losses
+__call__
*,&call_and_return_all_conditional_losses
&,"call_and_return_conditional_losses"
_generic_user_object
?2?
-__inference_BatchNorm_1_layer_call_fn_6707067
-__inference_BatchNorm_1_layer_call_fn_6707080?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
H__inference_BatchNorm_1_layer_call_and_return_conditional_losses_6707098
H__inference_BatchNorm_1_layer_call_and_return_conditional_losses_6707116?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
':%  2Conv_2/kernel
: 2Conv_2/bias
.
-0
.1"
trackable_list_wrapper
.
-0
.1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
/	variables
0trainable_variables
1regularization_losses
3__call__
*4&call_and_return_all_conditional_losses
&4"call_and_return_conditional_losses"
_generic_user_object
?2?
(__inference_Conv_2_layer_call_fn_6707125?
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
C__inference_Conv_2_layer_call_and_return_conditional_losses_6707135?
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
5	variables
6trainable_variables
7regularization_losses
9__call__
*:&call_and_return_all_conditional_losses
&:"call_and_return_conditional_losses"
_generic_user_object
?2?
.__inference_activation_6_layer_call_fn_6707140?
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
I__inference_activation_6_layer_call_and_return_conditional_losses_6707145?
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
':%  2Conv_3/kernel
: 2Conv_3/bias
.
;0
<1"
trackable_list_wrapper
.
;0
<1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
=	variables
>trainable_variables
?regularization_losses
A__call__
*B&call_and_return_all_conditional_losses
&B"call_and_return_conditional_losses"
_generic_user_object
?2?
(__inference_Conv_3_layer_call_fn_6707154?
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
C__inference_Conv_3_layer_call_and_return_conditional_losses_6707164?
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
C	variables
Dtrainable_variables
Eregularization_losses
G__call__
*H&call_and_return_all_conditional_losses
&H"call_and_return_conditional_losses"
_generic_user_object
?2?
+__inference_MaxPool_1_layer_call_fn_6707169?
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
F__inference_MaxPool_1_layer_call_and_return_conditional_losses_6707174?
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
I	variables
Jtrainable_variables
Kregularization_losses
M__call__
*N&call_and_return_all_conditional_losses
&N"call_and_return_conditional_losses"
_generic_user_object
?2?
.__inference_activation_7_layer_call_fn_6707179?
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
I__inference_activation_7_layer_call_and_return_conditional_losses_6707184?
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
 "
trackable_list_wrapper
: 2BatchNorm_2/gamma
: 2BatchNorm_2/beta
':%  (2BatchNorm_2/moving_mean
+:)  (2BatchNorm_2/moving_variance
<
P0
Q1
R2
S3"
trackable_list_wrapper
.
P0
Q1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
T	variables
Utrainable_variables
Vregularization_losses
X__call__
*Y&call_and_return_all_conditional_losses
&Y"call_and_return_conditional_losses"
_generic_user_object
?2?
-__inference_BatchNorm_2_layer_call_fn_6707197
-__inference_BatchNorm_2_layer_call_fn_6707210?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
H__inference_BatchNorm_2_layer_call_and_return_conditional_losses_6707228
H__inference_BatchNorm_2_layer_call_and_return_conditional_losses_6707246?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
':%  2Conv_4/kernel
: 2Conv_4/bias
.
Z0
[1"
trackable_list_wrapper
.
Z0
[1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
\	variables
]trainable_variables
^regularization_losses
`__call__
*a&call_and_return_all_conditional_losses
&a"call_and_return_conditional_losses"
_generic_user_object
?2?
(__inference_Conv_4_layer_call_fn_6707255?
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
C__inference_Conv_4_layer_call_and_return_conditional_losses_6707265?
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
b	variables
ctrainable_variables
dregularization_losses
f__call__
*g&call_and_return_all_conditional_losses
&g"call_and_return_conditional_losses"
_generic_user_object
?2?
+__inference_flatten_3_layer_call_fn_6707270?
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
F__inference_flatten_3_layer_call_and_return_conditional_losses_6707276?
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
h	variables
itrainable_variables
jregularization_losses
m__call__
*n&call_and_return_all_conditional_losses
&n"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
?2?
+__inference_Dropout_0_layer_call_fn_6707281
+__inference_Dropout_0_layer_call_fn_6707286?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
F__inference_Dropout_0_layer_call_and_return_conditional_losses_6707291
F__inference_Dropout_0_layer_call_and_return_conditional_losses_6707303?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
 : @2Dense_0/kernel
:@2Dense_0/bias
.
o0
p1"
trackable_list_wrapper
.
o0
p1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
q	variables
rtrainable_variables
sregularization_losses
u__call__
*v&call_and_return_all_conditional_losses
&v"call_and_return_conditional_losses"
_generic_user_object
?2?
)__inference_Dense_0_layer_call_fn_6707324?
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
D__inference_Dense_0_layer_call_and_return_conditional_losses_6707346?
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
 :@@2Dense_1/kernel
:@2Dense_1/bias
.
w0
x1"
trackable_list_wrapper
.
w0
x1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
y	variables
ztrainable_variables
{regularization_losses
}__call__
*~&call_and_return_all_conditional_losses
&~"call_and_return_conditional_losses"
_generic_user_object
?2?
)__inference_Dense_1_layer_call_fn_6707367?
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
D__inference_Dense_1_layer_call_and_return_conditional_losses_6707389?
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
?2?
+__inference_Dropout_2_layer_call_fn_6707394
+__inference_Dropout_2_layer_call_fn_6707399?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
F__inference_Dropout_2_layer_call_and_return_conditional_losses_6707404
F__inference_Dropout_2_layer_call_and_return_conditional_losses_6707416?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
%:#@2Dense_output/kernel
(
?0"
trackable_list_wrapper
(
?0"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2?
.__inference_Dense_output_layer_call_fn_6707423?
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
I__inference_Dense_output_layer_call_and_return_conditional_losses_6707431?
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
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
?2?
__inference_loss_fn_0_6707442?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference_loss_fn_1_6707453?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference_loss_fn_2_6707464?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference_loss_fn_3_6707475?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
<
%0
&1
R2
S3"
trackable_list_wrapper
?
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15"
trackable_list_wrapper
8
?0
?1
?2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
%__inference_signature_wrapper_6707035input_8"?
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
.
%0
&1"
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
.
R0
S1"
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
0
?0
?1"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
?0
?1"
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
R

?total

?count
?	variables
?	keras_api"
_tf_keras_metric
c

?total

?count
?
_fn_kwargs
?	variables
?	keras_api"
_tf_keras_metric
c

?total

?count
?
_fn_kwargs
?	variables
?	keras_api"
_tf_keras_metric
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
,:* 2Adam/Conv_1/kernel/m
: 2Adam/Conv_1/bias/m
$:" 2Adam/BatchNorm_1/gamma/m
#:! 2Adam/BatchNorm_1/beta/m
,:*  2Adam/Conv_2/kernel/m
: 2Adam/Conv_2/bias/m
,:*  2Adam/Conv_3/kernel/m
: 2Adam/Conv_3/bias/m
$:" 2Adam/BatchNorm_2/gamma/m
#:! 2Adam/BatchNorm_2/beta/m
,:*  2Adam/Conv_4/kernel/m
: 2Adam/Conv_4/bias/m
%:# @2Adam/Dense_0/kernel/m
:@2Adam/Dense_0/bias/m
%:#@@2Adam/Dense_1/kernel/m
:@2Adam/Dense_1/bias/m
*:(@2Adam/Dense_output/kernel/m
,:* 2Adam/Conv_1/kernel/v
: 2Adam/Conv_1/bias/v
$:" 2Adam/BatchNorm_1/gamma/v
#:! 2Adam/BatchNorm_1/beta/v
,:*  2Adam/Conv_2/kernel/v
: 2Adam/Conv_2/bias/v
,:*  2Adam/Conv_3/kernel/v
: 2Adam/Conv_3/bias/v
$:" 2Adam/BatchNorm_2/gamma/v
#:! 2Adam/BatchNorm_2/beta/v
,:*  2Adam/Conv_4/kernel/v
: 2Adam/Conv_4/bias/v
%:# @2Adam/Dense_0/kernel/v
:@2Adam/Dense_0/bias/v
%:#@@2Adam/Dense_1/kernel/v
:@2Adam/Dense_1/bias/v
*:(@2Adam/Dense_output/kernel/v
/:- 2Adam/Conv_1/kernel/vhat
!: 2Adam/Conv_1/bias/vhat
':% 2Adam/BatchNorm_1/gamma/vhat
&:$ 2Adam/BatchNorm_1/beta/vhat
/:-  2Adam/Conv_2/kernel/vhat
!: 2Adam/Conv_2/bias/vhat
/:-  2Adam/Conv_3/kernel/vhat
!: 2Adam/Conv_3/bias/vhat
':% 2Adam/BatchNorm_2/gamma/vhat
&:$ 2Adam/BatchNorm_2/beta/vhat
/:-  2Adam/Conv_4/kernel/vhat
!: 2Adam/Conv_4/bias/vhat
(:& @2Adam/Dense_0/kernel/vhat
": @2Adam/Dense_0/bias/vhat
(:&@@2Adam/Dense_1/kernel/vhat
": @2Adam/Dense_1/bias/vhat
-:+@2Adam/Dense_output/kernel/vhat?
H__inference_BatchNorm_1_layer_call_and_return_conditional_losses_6707098?#$%&M?J
C?@
:?7
inputs+??????????????????????????? 
p 
? "??<
5?2
0+??????????????????????????? 
? ?
H__inference_BatchNorm_1_layer_call_and_return_conditional_losses_6707116?#$%&M?J
C?@
:?7
inputs+??????????????????????????? 
p
? "??<
5?2
0+??????????????????????????? 
? ?
-__inference_BatchNorm_1_layer_call_fn_6707067?#$%&M?J
C?@
:?7
inputs+??????????????????????????? 
p 
? "2?/+??????????????????????????? ?
-__inference_BatchNorm_1_layer_call_fn_6707080?#$%&M?J
C?@
:?7
inputs+??????????????????????????? 
p
? "2?/+??????????????????????????? ?
H__inference_BatchNorm_2_layer_call_and_return_conditional_losses_6707228?PQRSM?J
C?@
:?7
inputs+??????????????????????????? 
p 
? "??<
5?2
0+??????????????????????????? 
? ?
H__inference_BatchNorm_2_layer_call_and_return_conditional_losses_6707246?PQRSM?J
C?@
:?7
inputs+??????????????????????????? 
p
? "??<
5?2
0+??????????????????????????? 
? ?
-__inference_BatchNorm_2_layer_call_fn_6707197?PQRSM?J
C?@
:?7
inputs+??????????????????????????? 
p 
? "2?/+??????????????????????????? ?
-__inference_BatchNorm_2_layer_call_fn_6707210?PQRSM?J
C?@
:?7
inputs+??????????????????????????? 
p
? "2?/+??????????????????????????? ?
C__inference_Conv_1_layer_call_and_return_conditional_losses_6707054l7?4
-?*
(?%
inputs?????????
? "-?*
#? 
0????????? 
? ?
(__inference_Conv_1_layer_call_fn_6707044_7?4
-?*
(?%
inputs?????????
? " ?????????? ?
C__inference_Conv_2_layer_call_and_return_conditional_losses_6707135l-.7?4
-?*
(?%
inputs????????? 
? "-?*
#? 
0?????????		 
? ?
(__inference_Conv_2_layer_call_fn_6707125_-.7?4
-?*
(?%
inputs????????? 
? " ??????????		 ?
C__inference_Conv_3_layer_call_and_return_conditional_losses_6707164l;<7?4
-?*
(?%
inputs?????????		 
? "-?*
#? 
0????????? 
? ?
(__inference_Conv_3_layer_call_fn_6707154_;<7?4
-?*
(?%
inputs?????????		 
? " ?????????? ?
C__inference_Conv_4_layer_call_and_return_conditional_losses_6707265lZ[7?4
-?*
(?%
inputs????????? 
? "-?*
#? 
0????????? 
? ?
(__inference_Conv_4_layer_call_fn_6707255_Z[7?4
-?*
(?%
inputs????????? 
? " ?????????? ?
D__inference_Dense_0_layer_call_and_return_conditional_losses_6707346\op/?,
%?"
 ?
inputs????????? 
? "%?"
?
0?????????@
? |
)__inference_Dense_0_layer_call_fn_6707324Oop/?,
%?"
 ?
inputs????????? 
? "??????????@?
D__inference_Dense_1_layer_call_and_return_conditional_losses_6707389\wx/?,
%?"
 ?
inputs?????????@
? "%?"
?
0?????????@
? |
)__inference_Dense_1_layer_call_fn_6707367Owx/?,
%?"
 ?
inputs?????????@
? "??????????@?
I__inference_Dense_output_layer_call_and_return_conditional_losses_6707431\?/?,
%?"
 ?
inputs?????????@
? "%?"
?
0?????????
? ?
.__inference_Dense_output_layer_call_fn_6707423O?/?,
%?"
 ?
inputs?????????@
? "???????????
F__inference_Dropout_0_layer_call_and_return_conditional_losses_6707291\3?0
)?&
 ?
inputs????????? 
p 
? "%?"
?
0????????? 
? ?
F__inference_Dropout_0_layer_call_and_return_conditional_losses_6707303\3?0
)?&
 ?
inputs????????? 
p
? "%?"
?
0????????? 
? ~
+__inference_Dropout_0_layer_call_fn_6707281O3?0
)?&
 ?
inputs????????? 
p 
? "?????????? ~
+__inference_Dropout_0_layer_call_fn_6707286O3?0
)?&
 ?
inputs????????? 
p
? "?????????? ?
F__inference_Dropout_2_layer_call_and_return_conditional_losses_6707404\3?0
)?&
 ?
inputs?????????@
p 
? "%?"
?
0?????????@
? ?
F__inference_Dropout_2_layer_call_and_return_conditional_losses_6707416\3?0
)?&
 ?
inputs?????????@
p
? "%?"
?
0?????????@
? ~
+__inference_Dropout_2_layer_call_fn_6707394O3?0
)?&
 ?
inputs?????????@
p 
? "??????????@~
+__inference_Dropout_2_layer_call_fn_6707399O3?0
)?&
 ?
inputs?????????@
p
? "??????????@?
F__inference_MaxPool_1_layer_call_and_return_conditional_losses_6707174?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
+__inference_MaxPool_1_layer_call_fn_6707169?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
"__inference__wrapped_model_6705712?#$%&-.;<PQRSZ[opwx?8?5
.?+
)?&
input_8?????????
? ";?8
6
Dense_output&?#
Dense_output??????????
I__inference_activation_6_layer_call_and_return_conditional_losses_6707145h7?4
-?*
(?%
inputs?????????		 
? "-?*
#? 
0?????????		 
? ?
.__inference_activation_6_layer_call_fn_6707140[7?4
-?*
(?%
inputs?????????		 
? " ??????????		 ?
I__inference_activation_7_layer_call_and_return_conditional_losses_6707184h7?4
-?*
(?%
inputs????????? 
? "-?*
#? 
0????????? 
? ?
.__inference_activation_7_layer_call_fn_6707179[7?4
-?*
(?%
inputs????????? 
? " ?????????? ?
F__inference_flatten_3_layer_call_and_return_conditional_losses_6707276`7?4
-?*
(?%
inputs????????? 
? "%?"
?
0????????? 
? ?
+__inference_flatten_3_layer_call_fn_6707270S7?4
-?*
(?%
inputs????????? 
? "?????????? <
__inference_loss_fn_0_6707442o?

? 
? "? <
__inference_loss_fn_1_6707453p?

? 
? "? <
__inference_loss_fn_2_6707464w?

? 
? "? <
__inference_loss_fn_3_6707475x?

? 
? "? ?
D__inference_model_7_layer_call_and_return_conditional_losses_6706559?#$%&-.;<PQRSZ[opwx?@?=
6?3
)?&
input_8?????????
p 

 
? "%?"
?
0?????????
? ?
D__inference_model_7_layer_call_and_return_conditional_losses_6706644?#$%&-.;<PQRSZ[opwx?@?=
6?3
)?&
input_8?????????
p

 
? "%?"
?
0?????????
? ?
D__inference_model_7_layer_call_and_return_conditional_losses_6706869?#$%&-.;<PQRSZ[opwx???<
5?2
(?%
inputs?????????
p 

 
? "%?"
?
0?????????
? ?
D__inference_model_7_layer_call_and_return_conditional_losses_6706986?#$%&-.;<PQRSZ[opwx???<
5?2
(?%
inputs?????????
p

 
? "%?"
?
0?????????
? ?
)__inference_model_7_layer_call_fn_6706116t#$%&-.;<PQRSZ[opwx?@?=
6?3
)?&
input_8?????????
p 

 
? "???????????
)__inference_model_7_layer_call_fn_6706474t#$%&-.;<PQRSZ[opwx?@?=
6?3
)?&
input_8?????????
p

 
? "???????????
)__inference_model_7_layer_call_fn_6706719s#$%&-.;<PQRSZ[opwx???<
5?2
(?%
inputs?????????
p 

 
? "???????????
)__inference_model_7_layer_call_fn_6706766s#$%&-.;<PQRSZ[opwx???<
5?2
(?%
inputs?????????
p

 
? "???????????
%__inference_signature_wrapper_6707035?#$%&-.;<PQRSZ[opwx?C?@
? 
9?6
4
input_8)?&
input_8?????????";?8
6
Dense_output&?#
Dense_output?????????