Ĺ
??
B
AddV2
x"T
y"T
z"T"
Ttype:
2	??
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
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
?
GatherV2
params"Tparams
indices"Tindices
axis"Taxis
output"Tparams"

batch_dimsint "
Tparamstype"
Tindicestype:
2	"
Taxistype:
2	
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
?
Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
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
?
Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
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
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
;
Sub
x"T
y"T
z"T"
Ttype:
2	
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.4.12v2.4.0-49-g85c8b2a817f8??
?
dddqn/dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*#
shared_namedddqn/dense/kernel
{
&dddqn/dense/kernel/Read/ReadVariableOpReadVariableOpdddqn/dense/kernel* 
_output_shapes
:
??*
dtype0
y
dddqn/dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*!
shared_namedddqn/dense/bias
r
$dddqn/dense/bias/Read/ReadVariableOpReadVariableOpdddqn/dense/bias*
_output_shapes	
:?*
dtype0
?
dddqn/dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*%
shared_namedddqn/dense_1/kernel

(dddqn/dense_1/kernel/Read/ReadVariableOpReadVariableOpdddqn/dense_1/kernel* 
_output_shapes
:
??*
dtype0
}
dddqn/dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*#
shared_namedddqn/dense_1/bias
v
&dddqn/dense_1/bias/Read/ReadVariableOpReadVariableOpdddqn/dense_1/bias*
_output_shapes	
:?*
dtype0
?
dddqn/dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*%
shared_namedddqn/dense_2/kernel
~
(dddqn/dense_2/kernel/Read/ReadVariableOpReadVariableOpdddqn/dense_2/kernel*
_output_shapes
:	?*
dtype0
|
dddqn/dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_namedddqn/dense_2/bias
u
&dddqn/dense_2/bias/Read/ReadVariableOpReadVariableOpdddqn/dense_2/bias*
_output_shapes
:*
dtype0
?
dddqn/dense_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*%
shared_namedddqn/dense_3/kernel
~
(dddqn/dense_3/kernel/Read/ReadVariableOpReadVariableOpdddqn/dense_3/kernel*
_output_shapes
:	?*
dtype0
|
dddqn/dense_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_namedddqn/dense_3/bias
u
&dddqn/dense_3/bias/Read/ReadVariableOpReadVariableOpdddqn/dense_3/bias*
_output_shapes
:*
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
?
Adam/dddqn/dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??**
shared_nameAdam/dddqn/dense/kernel/m
?
-Adam/dddqn/dense/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dddqn/dense/kernel/m* 
_output_shapes
:
??*
dtype0
?
Adam/dddqn/dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*(
shared_nameAdam/dddqn/dense/bias/m
?
+Adam/dddqn/dense/bias/m/Read/ReadVariableOpReadVariableOpAdam/dddqn/dense/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/dddqn/dense_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*,
shared_nameAdam/dddqn/dense_1/kernel/m
?
/Adam/dddqn/dense_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dddqn/dense_1/kernel/m* 
_output_shapes
:
??*
dtype0
?
Adam/dddqn/dense_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?**
shared_nameAdam/dddqn/dense_1/bias/m
?
-Adam/dddqn/dense_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/dddqn/dense_1/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/dddqn/dense_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*,
shared_nameAdam/dddqn/dense_2/kernel/m
?
/Adam/dddqn/dense_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dddqn/dense_2/kernel/m*
_output_shapes
:	?*
dtype0
?
Adam/dddqn/dense_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_nameAdam/dddqn/dense_2/bias/m
?
-Adam/dddqn/dense_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/dddqn/dense_2/bias/m*
_output_shapes
:*
dtype0
?
Adam/dddqn/dense_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*,
shared_nameAdam/dddqn/dense_3/kernel/m
?
/Adam/dddqn/dense_3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dddqn/dense_3/kernel/m*
_output_shapes
:	?*
dtype0
?
Adam/dddqn/dense_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_nameAdam/dddqn/dense_3/bias/m
?
-Adam/dddqn/dense_3/bias/m/Read/ReadVariableOpReadVariableOpAdam/dddqn/dense_3/bias/m*
_output_shapes
:*
dtype0
?
Adam/dddqn/dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??**
shared_nameAdam/dddqn/dense/kernel/v
?
-Adam/dddqn/dense/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dddqn/dense/kernel/v* 
_output_shapes
:
??*
dtype0
?
Adam/dddqn/dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*(
shared_nameAdam/dddqn/dense/bias/v
?
+Adam/dddqn/dense/bias/v/Read/ReadVariableOpReadVariableOpAdam/dddqn/dense/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/dddqn/dense_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*,
shared_nameAdam/dddqn/dense_1/kernel/v
?
/Adam/dddqn/dense_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dddqn/dense_1/kernel/v* 
_output_shapes
:
??*
dtype0
?
Adam/dddqn/dense_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?**
shared_nameAdam/dddqn/dense_1/bias/v
?
-Adam/dddqn/dense_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/dddqn/dense_1/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/dddqn/dense_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*,
shared_nameAdam/dddqn/dense_2/kernel/v
?
/Adam/dddqn/dense_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dddqn/dense_2/kernel/v*
_output_shapes
:	?*
dtype0
?
Adam/dddqn/dense_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_nameAdam/dddqn/dense_2/bias/v
?
-Adam/dddqn/dense_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/dddqn/dense_2/bias/v*
_output_shapes
:*
dtype0
?
Adam/dddqn/dense_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*,
shared_nameAdam/dddqn/dense_3/kernel/v
?
/Adam/dddqn/dense_3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dddqn/dense_3/kernel/v*
_output_shapes
:	?*
dtype0
?
Adam/dddqn/dense_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_nameAdam/dddqn/dense_3/bias/v
?
-Adam/dddqn/dense_3/bias/v/Read/ReadVariableOpReadVariableOpAdam/dddqn/dense_3/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
?(
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?'
value?'B?' B?'
?
d1
d2
v
a
	optimizer
trainable_variables
	variables
regularization_losses
		keras_api


signatures
h

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
h

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
h

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
h

kernel
bias
trainable_variables
 	variables
!regularization_losses
"	keras_api
?
#iter

$beta_1

%beta_2
	&decay
'learning_ratemFmGmHmImJmKmLmMvNvOvPvQvRvSvTvU
8
0
1
2
3
4
5
6
7
8
0
1
2
3
4
5
6
7
 
?
trainable_variables
(layer_regularization_losses
	variables
)layer_metrics
*non_trainable_variables
+metrics
regularization_losses

,layers
 
LJ
VARIABLE_VALUEdddqn/dense/kernel$d1/kernel/.ATTRIBUTES/VARIABLE_VALUE
HF
VARIABLE_VALUEdddqn/dense/bias"d1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
trainable_variables
-layer_regularization_losses
	variables
regularization_losses
.non_trainable_variables
/metrics
0layer_metrics

1layers
NL
VARIABLE_VALUEdddqn/dense_1/kernel$d2/kernel/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEdddqn/dense_1/bias"d2/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
trainable_variables
2layer_regularization_losses
	variables
regularization_losses
3non_trainable_variables
4metrics
5layer_metrics

6layers
MK
VARIABLE_VALUEdddqn/dense_2/kernel#v/kernel/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUEdddqn/dense_2/bias!v/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
trainable_variables
7layer_regularization_losses
	variables
regularization_losses
8non_trainable_variables
9metrics
:layer_metrics

;layers
MK
VARIABLE_VALUEdddqn/dense_3/kernel#a/kernel/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUEdddqn/dense_3/bias!a/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
trainable_variables
<layer_regularization_losses
 	variables
!regularization_losses
=non_trainable_variables
>metrics
?layer_metrics

@layers
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
 
 
 

A0

0
1
2
3
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
4
	Btotal
	Ccount
D	variables
E	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

B0
C1

D	variables
om
VARIABLE_VALUEAdam/dddqn/dense/kernel/m@d1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUEAdam/dddqn/dense/bias/m>d1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUEAdam/dddqn/dense_1/kernel/m@d2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/dddqn/dense_1/bias/m>d2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEAdam/dddqn/dense_2/kernel/m?v/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUEAdam/dddqn/dense_2/bias/m=v/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEAdam/dddqn/dense_3/kernel/m?a/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUEAdam/dddqn/dense_3/bias/m=a/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/dddqn/dense/kernel/v@d1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUEAdam/dddqn/dense/bias/v>d1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUEAdam/dddqn/dense_1/kernel/v@d2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/dddqn/dense_1/bias/v>d2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEAdam/dddqn/dense_2/kernel/v?v/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUEAdam/dddqn/dense_2/bias/v=v/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEAdam/dddqn/dense_3/kernel/v?a/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUEAdam/dddqn/dense_3/bias/v=a/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
serving_default_input_1Placeholder*,
_output_shapes
:??????????*
dtype0*!
shape:??????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1dddqn/dense/kerneldddqn/dense/biasdddqn/dense_1/kerneldddqn/dense_1/biasdddqn/dense_2/kerneldddqn/dense_2/biasdddqn/dense_3/kerneldddqn/dense_3/bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *-
f(R&
$__inference_signature_wrapper_413245
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename&dddqn/dense/kernel/Read/ReadVariableOp$dddqn/dense/bias/Read/ReadVariableOp(dddqn/dense_1/kernel/Read/ReadVariableOp&dddqn/dense_1/bias/Read/ReadVariableOp(dddqn/dense_2/kernel/Read/ReadVariableOp&dddqn/dense_2/bias/Read/ReadVariableOp(dddqn/dense_3/kernel/Read/ReadVariableOp&dddqn/dense_3/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp-Adam/dddqn/dense/kernel/m/Read/ReadVariableOp+Adam/dddqn/dense/bias/m/Read/ReadVariableOp/Adam/dddqn/dense_1/kernel/m/Read/ReadVariableOp-Adam/dddqn/dense_1/bias/m/Read/ReadVariableOp/Adam/dddqn/dense_2/kernel/m/Read/ReadVariableOp-Adam/dddqn/dense_2/bias/m/Read/ReadVariableOp/Adam/dddqn/dense_3/kernel/m/Read/ReadVariableOp-Adam/dddqn/dense_3/bias/m/Read/ReadVariableOp-Adam/dddqn/dense/kernel/v/Read/ReadVariableOp+Adam/dddqn/dense/bias/v/Read/ReadVariableOp/Adam/dddqn/dense_1/kernel/v/Read/ReadVariableOp-Adam/dddqn/dense_1/bias/v/Read/ReadVariableOp/Adam/dddqn/dense_2/kernel/v/Read/ReadVariableOp-Adam/dddqn/dense_2/bias/v/Read/ReadVariableOp/Adam/dddqn/dense_3/kernel/v/Read/ReadVariableOp-Adam/dddqn/dense_3/bias/v/Read/ReadVariableOpConst*,
Tin%
#2!	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *(
f#R!
__inference__traced_save_413519
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedddqn/dense/kerneldddqn/dense/biasdddqn/dense_1/kerneldddqn/dense_1/biasdddqn/dense_2/kerneldddqn/dense_2/biasdddqn/dense_3/kerneldddqn/dense_3/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcountAdam/dddqn/dense/kernel/mAdam/dddqn/dense/bias/mAdam/dddqn/dense_1/kernel/mAdam/dddqn/dense_1/bias/mAdam/dddqn/dense_2/kernel/mAdam/dddqn/dense_2/bias/mAdam/dddqn/dense_3/kernel/mAdam/dddqn/dense_3/bias/mAdam/dddqn/dense/kernel/vAdam/dddqn/dense/bias/vAdam/dddqn/dense_1/kernel/vAdam/dddqn/dense_1/bias/vAdam/dddqn/dense_2/kernel/vAdam/dddqn/dense_2/bias/vAdam/dddqn/dense_3/kernel/vAdam/dddqn/dense_3/bias/v*+
Tin$
"2 *
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *+
f&R$
"__inference__traced_restore_413622??
?C
?
__inference__traced_save_413519
file_prefix1
-savev2_dddqn_dense_kernel_read_readvariableop/
+savev2_dddqn_dense_bias_read_readvariableop3
/savev2_dddqn_dense_1_kernel_read_readvariableop1
-savev2_dddqn_dense_1_bias_read_readvariableop3
/savev2_dddqn_dense_2_kernel_read_readvariableop1
-savev2_dddqn_dense_2_bias_read_readvariableop3
/savev2_dddqn_dense_3_kernel_read_readvariableop1
-savev2_dddqn_dense_3_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop8
4savev2_adam_dddqn_dense_kernel_m_read_readvariableop6
2savev2_adam_dddqn_dense_bias_m_read_readvariableop:
6savev2_adam_dddqn_dense_1_kernel_m_read_readvariableop8
4savev2_adam_dddqn_dense_1_bias_m_read_readvariableop:
6savev2_adam_dddqn_dense_2_kernel_m_read_readvariableop8
4savev2_adam_dddqn_dense_2_bias_m_read_readvariableop:
6savev2_adam_dddqn_dense_3_kernel_m_read_readvariableop8
4savev2_adam_dddqn_dense_3_bias_m_read_readvariableop8
4savev2_adam_dddqn_dense_kernel_v_read_readvariableop6
2savev2_adam_dddqn_dense_bias_v_read_readvariableop:
6savev2_adam_dddqn_dense_1_kernel_v_read_readvariableop8
4savev2_adam_dddqn_dense_1_bias_v_read_readvariableop:
6savev2_adam_dddqn_dense_2_kernel_v_read_readvariableop8
4savev2_adam_dddqn_dense_2_bias_v_read_readvariableop:
6savev2_adam_dddqn_dense_3_kernel_v_read_readvariableop8
4savev2_adam_dddqn_dense_3_bias_v_read_readvariableop
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
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
: *
dtype0*?
value?B? B$d1/kernel/.ATTRIBUTES/VARIABLE_VALUEB"d1/bias/.ATTRIBUTES/VARIABLE_VALUEB$d2/kernel/.ATTRIBUTES/VARIABLE_VALUEB"d2/bias/.ATTRIBUTES/VARIABLE_VALUEB#v/kernel/.ATTRIBUTES/VARIABLE_VALUEB!v/bias/.ATTRIBUTES/VARIABLE_VALUEB#a/kernel/.ATTRIBUTES/VARIABLE_VALUEB!a/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB@d1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB>d1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB@d2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB>d2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB?v/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB=v/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB?a/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB=a/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB@d1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB>d1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB@d2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB>d2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB?v/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB=v/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB?a/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB=a/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
: *
dtype0*S
valueJBH B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0-savev2_dddqn_dense_kernel_read_readvariableop+savev2_dddqn_dense_bias_read_readvariableop/savev2_dddqn_dense_1_kernel_read_readvariableop-savev2_dddqn_dense_1_bias_read_readvariableop/savev2_dddqn_dense_2_kernel_read_readvariableop-savev2_dddqn_dense_2_bias_read_readvariableop/savev2_dddqn_dense_3_kernel_read_readvariableop-savev2_dddqn_dense_3_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop4savev2_adam_dddqn_dense_kernel_m_read_readvariableop2savev2_adam_dddqn_dense_bias_m_read_readvariableop6savev2_adam_dddqn_dense_1_kernel_m_read_readvariableop4savev2_adam_dddqn_dense_1_bias_m_read_readvariableop6savev2_adam_dddqn_dense_2_kernel_m_read_readvariableop4savev2_adam_dddqn_dense_2_bias_m_read_readvariableop6savev2_adam_dddqn_dense_3_kernel_m_read_readvariableop4savev2_adam_dddqn_dense_3_bias_m_read_readvariableop4savev2_adam_dddqn_dense_kernel_v_read_readvariableop2savev2_adam_dddqn_dense_bias_v_read_readvariableop6savev2_adam_dddqn_dense_1_kernel_v_read_readvariableop4savev2_adam_dddqn_dense_1_bias_v_read_readvariableop6savev2_adam_dddqn_dense_2_kernel_v_read_readvariableop4savev2_adam_dddqn_dense_2_bias_v_read_readvariableop6savev2_adam_dddqn_dense_3_kernel_v_read_readvariableop4savev2_adam_dddqn_dense_3_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *.
dtypes$
"2 	2
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

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*?
_input_shapes?
?: :
??:?:
??:?:	?::	?:: : : : : : : :
??:?:
??:?:	?::	?::
??:?:
??:?:	?::	?:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:%!

_output_shapes
:	?: 

_output_shapes
::%!

_output_shapes
:	?: 

_output_shapes
::	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :&"
 
_output_shapes
:
??:!

_output_shapes	
:?:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:%!

_output_shapes
:	?: 

_output_shapes
::%!

_output_shapes
:	?: 

_output_shapes
::&"
 
_output_shapes
:
??:!

_output_shapes	
:?:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:%!

_output_shapes
:	?: 

_output_shapes
::%!

_output_shapes
:	?: 

_output_shapes
:: 

_output_shapes
: 
?
}
(__inference_dense_2_layer_call_fn_413364

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_4131252
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :??????????::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
??
?
"__inference__traced_restore_413622
file_prefix'
#assignvariableop_dddqn_dense_kernel'
#assignvariableop_1_dddqn_dense_bias+
'assignvariableop_2_dddqn_dense_1_kernel)
%assignvariableop_3_dddqn_dense_1_bias+
'assignvariableop_4_dddqn_dense_2_kernel)
%assignvariableop_5_dddqn_dense_2_bias+
'assignvariableop_6_dddqn_dense_3_kernel)
%assignvariableop_7_dddqn_dense_3_bias 
assignvariableop_8_adam_iter"
assignvariableop_9_adam_beta_1#
assignvariableop_10_adam_beta_2"
assignvariableop_11_adam_decay*
&assignvariableop_12_adam_learning_rate
assignvariableop_13_total
assignvariableop_14_count1
-assignvariableop_15_adam_dddqn_dense_kernel_m/
+assignvariableop_16_adam_dddqn_dense_bias_m3
/assignvariableop_17_adam_dddqn_dense_1_kernel_m1
-assignvariableop_18_adam_dddqn_dense_1_bias_m3
/assignvariableop_19_adam_dddqn_dense_2_kernel_m1
-assignvariableop_20_adam_dddqn_dense_2_bias_m3
/assignvariableop_21_adam_dddqn_dense_3_kernel_m1
-assignvariableop_22_adam_dddqn_dense_3_bias_m1
-assignvariableop_23_adam_dddqn_dense_kernel_v/
+assignvariableop_24_adam_dddqn_dense_bias_v3
/assignvariableop_25_adam_dddqn_dense_1_kernel_v1
-assignvariableop_26_adam_dddqn_dense_1_bias_v3
/assignvariableop_27_adam_dddqn_dense_2_kernel_v1
-assignvariableop_28_adam_dddqn_dense_2_bias_v3
/assignvariableop_29_adam_dddqn_dense_3_kernel_v1
-assignvariableop_30_adam_dddqn_dense_3_bias_v
identity_32??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
: *
dtype0*?
value?B? B$d1/kernel/.ATTRIBUTES/VARIABLE_VALUEB"d1/bias/.ATTRIBUTES/VARIABLE_VALUEB$d2/kernel/.ATTRIBUTES/VARIABLE_VALUEB"d2/bias/.ATTRIBUTES/VARIABLE_VALUEB#v/kernel/.ATTRIBUTES/VARIABLE_VALUEB!v/bias/.ATTRIBUTES/VARIABLE_VALUEB#a/kernel/.ATTRIBUTES/VARIABLE_VALUEB!a/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB@d1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB>d1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB@d2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB>d2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB?v/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB=v/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB?a/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB=a/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB@d1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB>d1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB@d2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB>d2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB?v/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB=v/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB?a/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB=a/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
: *
dtype0*S
valueJBH B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::*.
dtypes$
"2 	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp#assignvariableop_dddqn_dense_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp#assignvariableop_1_dddqn_dense_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp'assignvariableop_2_dddqn_dense_1_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp%assignvariableop_3_dddqn_dense_1_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp'assignvariableop_4_dddqn_dense_2_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp%assignvariableop_5_dddqn_dense_2_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp'assignvariableop_6_dddqn_dense_3_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp%assignvariableop_7_dddqn_dense_3_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOpassignvariableop_8_adam_iterIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOpassignvariableop_9_adam_beta_1Identity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOpassignvariableop_10_adam_beta_2Identity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOpassignvariableop_11_adam_decayIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp&assignvariableop_12_adam_learning_rateIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOpassignvariableop_13_totalIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOpassignvariableop_14_countIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOp-assignvariableop_15_adam_dddqn_dense_kernel_mIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOp+assignvariableop_16_adam_dddqn_dense_bias_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOp/assignvariableop_17_adam_dddqn_dense_1_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOp-assignvariableop_18_adam_dddqn_dense_1_bias_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOp/assignvariableop_19_adam_dddqn_dense_2_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp-assignvariableop_20_adam_dddqn_dense_2_bias_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp/assignvariableop_21_adam_dddqn_dense_3_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp-assignvariableop_22_adam_dddqn_dense_3_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp-assignvariableop_23_adam_dddqn_dense_kernel_vIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp+assignvariableop_24_adam_dddqn_dense_bias_vIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOp/assignvariableop_25_adam_dddqn_dense_1_kernel_vIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOp-assignvariableop_26_adam_dddqn_dense_1_bias_vIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOp/assignvariableop_27_adam_dddqn_dense_2_kernel_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOp-assignvariableop_28_adam_dddqn_dense_2_bias_vIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOp/assignvariableop_29_adam_dddqn_dense_3_kernel_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOp-assignvariableop_30_adam_dddqn_dense_3_bias_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_309
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_31Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_31?
Identity_32IdentityIdentity_31:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_32"#
identity_32Identity_32:output:0*?
_input_shapes?
~: :::::::::::::::::::::::::::::::2$
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
AssignVariableOp_30AssignVariableOp_302(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
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
&__inference_dddqn_layer_call_fn_413214
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dddqn_layer_call_and_return_conditional_losses_4131922
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*K
_input_shapes:
8:??????????::::::::22
StatefulPartitionedCallStatefulPartitionedCall:U Q
,
_output_shapes
:??????????
!
_user_specified_name	input_1
? 
?
A__inference_dense_layer_call_and_return_conditional_losses_413276

inputs%
!tensordot_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Tensordot/ReadVariableOp?
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource* 
_output_shapes
:
??*
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axis?
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis?
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const?
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1?
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis?
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat?
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack?
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*,
_output_shapes
:??????????2
Tensordot/transpose?
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
Tensordot/Reshape?
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
Tensordot/MatMulq
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis?
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1?
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????2
	Tensordot?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????2	
BiasAdd]
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
? 
?
C__inference_dense_1_layer_call_and_return_conditional_losses_413316

inputs%
!tensordot_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Tensordot/ReadVariableOp?
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource* 
_output_shapes
:
??*
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axis?
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis?
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const?
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1?
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis?
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat?
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack?
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*,
_output_shapes
:??????????2
Tensordot/transpose?
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
Tensordot/Reshape?
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
Tensordot/MatMulq
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis?
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1?
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????2
	Tensordot?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????2	
BiasAdd]
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
? 
?
A__inference_dense_layer_call_and_return_conditional_losses_413032

inputs%
!tensordot_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Tensordot/ReadVariableOp?
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource* 
_output_shapes
:
??*
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axis?
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis?
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const?
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1?
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis?
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat?
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack?
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*,
_output_shapes
:??????????2
Tensordot/transpose?
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
Tensordot/Reshape?
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
Tensordot/MatMulq
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis?
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1?
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????2
	Tensordot?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????2	
BiasAdd]
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
C__inference_dense_2_layer_call_and_return_conditional_losses_413125

inputs%
!tensordot_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Tensordot/ReadVariableOp?
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes
:	?*
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axis?
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis?
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const?
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1?
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis?
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat?
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack?
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*,
_output_shapes
:??????????2
Tensordot/transpose?
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
Tensordot/Reshape?
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
Tensordot/MatMulp
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis?
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1?
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????2
	Tensordot?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
{
&__inference_dense_layer_call_fn_413285

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_4130322
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :??????????::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
Ϣ
?
!__inference__wrapped_model_412997
input_11
-dddqn_dense_tensordot_readvariableop_resource/
+dddqn_dense_biasadd_readvariableop_resource3
/dddqn_dense_1_tensordot_readvariableop_resource1
-dddqn_dense_1_biasadd_readvariableop_resource3
/dddqn_dense_2_tensordot_readvariableop_resource1
-dddqn_dense_2_biasadd_readvariableop_resource3
/dddqn_dense_3_tensordot_readvariableop_resource1
-dddqn_dense_3_biasadd_readvariableop_resource
identity??"dddqn/dense/BiasAdd/ReadVariableOp?$dddqn/dense/Tensordot/ReadVariableOp?$dddqn/dense_1/BiasAdd/ReadVariableOp?&dddqn/dense_1/Tensordot/ReadVariableOp?$dddqn/dense_2/BiasAdd/ReadVariableOp?&dddqn/dense_2/Tensordot/ReadVariableOp?$dddqn/dense_3/BiasAdd/ReadVariableOp?&dddqn/dense_3/Tensordot/ReadVariableOp?
$dddqn/dense/Tensordot/ReadVariableOpReadVariableOp-dddqn_dense_tensordot_readvariableop_resource* 
_output_shapes
:
??*
dtype02&
$dddqn/dense/Tensordot/ReadVariableOp?
dddqn/dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dddqn/dense/Tensordot/axes?
dddqn/dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dddqn/dense/Tensordot/freeq
dddqn/dense/Tensordot/ShapeShapeinput_1*
T0*
_output_shapes
:2
dddqn/dense/Tensordot/Shape?
#dddqn/dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2%
#dddqn/dense/Tensordot/GatherV2/axis?
dddqn/dense/Tensordot/GatherV2GatherV2$dddqn/dense/Tensordot/Shape:output:0#dddqn/dense/Tensordot/free:output:0,dddqn/dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2 
dddqn/dense/Tensordot/GatherV2?
%dddqn/dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2'
%dddqn/dense/Tensordot/GatherV2_1/axis?
 dddqn/dense/Tensordot/GatherV2_1GatherV2$dddqn/dense/Tensordot/Shape:output:0#dddqn/dense/Tensordot/axes:output:0.dddqn/dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2"
 dddqn/dense/Tensordot/GatherV2_1?
dddqn/dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dddqn/dense/Tensordot/Const?
dddqn/dense/Tensordot/ProdProd'dddqn/dense/Tensordot/GatherV2:output:0$dddqn/dense/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dddqn/dense/Tensordot/Prod?
dddqn/dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dddqn/dense/Tensordot/Const_1?
dddqn/dense/Tensordot/Prod_1Prod)dddqn/dense/Tensordot/GatherV2_1:output:0&dddqn/dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dddqn/dense/Tensordot/Prod_1?
!dddqn/dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dddqn/dense/Tensordot/concat/axis?
dddqn/dense/Tensordot/concatConcatV2#dddqn/dense/Tensordot/free:output:0#dddqn/dense/Tensordot/axes:output:0*dddqn/dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dddqn/dense/Tensordot/concat?
dddqn/dense/Tensordot/stackPack#dddqn/dense/Tensordot/Prod:output:0%dddqn/dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dddqn/dense/Tensordot/stack?
dddqn/dense/Tensordot/transpose	Transposeinput_1%dddqn/dense/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????2!
dddqn/dense/Tensordot/transpose?
dddqn/dense/Tensordot/ReshapeReshape#dddqn/dense/Tensordot/transpose:y:0$dddqn/dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dddqn/dense/Tensordot/Reshape?
dddqn/dense/Tensordot/MatMulMatMul&dddqn/dense/Tensordot/Reshape:output:0,dddqn/dense/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dddqn/dense/Tensordot/MatMul?
dddqn/dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?2
dddqn/dense/Tensordot/Const_2?
#dddqn/dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2%
#dddqn/dense/Tensordot/concat_1/axis?
dddqn/dense/Tensordot/concat_1ConcatV2'dddqn/dense/Tensordot/GatherV2:output:0&dddqn/dense/Tensordot/Const_2:output:0,dddqn/dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2 
dddqn/dense/Tensordot/concat_1?
dddqn/dense/TensordotReshape&dddqn/dense/Tensordot/MatMul:product:0'dddqn/dense/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????2
dddqn/dense/Tensordot?
"dddqn/dense/BiasAdd/ReadVariableOpReadVariableOp+dddqn_dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02$
"dddqn/dense/BiasAdd/ReadVariableOp?
dddqn/dense/BiasAddBiasAdddddqn/dense/Tensordot:output:0*dddqn/dense/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????2
dddqn/dense/BiasAdd?
dddqn/dense/ReluReludddqn/dense/BiasAdd:output:0*
T0*,
_output_shapes
:??????????2
dddqn/dense/Relu?
&dddqn/dense_1/Tensordot/ReadVariableOpReadVariableOp/dddqn_dense_1_tensordot_readvariableop_resource* 
_output_shapes
:
??*
dtype02(
&dddqn/dense_1/Tensordot/ReadVariableOp?
dddqn/dense_1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dddqn/dense_1/Tensordot/axes?
dddqn/dense_1/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dddqn/dense_1/Tensordot/free?
dddqn/dense_1/Tensordot/ShapeShapedddqn/dense/Relu:activations:0*
T0*
_output_shapes
:2
dddqn/dense_1/Tensordot/Shape?
%dddqn/dense_1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2'
%dddqn/dense_1/Tensordot/GatherV2/axis?
 dddqn/dense_1/Tensordot/GatherV2GatherV2&dddqn/dense_1/Tensordot/Shape:output:0%dddqn/dense_1/Tensordot/free:output:0.dddqn/dense_1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2"
 dddqn/dense_1/Tensordot/GatherV2?
'dddqn/dense_1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2)
'dddqn/dense_1/Tensordot/GatherV2_1/axis?
"dddqn/dense_1/Tensordot/GatherV2_1GatherV2&dddqn/dense_1/Tensordot/Shape:output:0%dddqn/dense_1/Tensordot/axes:output:00dddqn/dense_1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2$
"dddqn/dense_1/Tensordot/GatherV2_1?
dddqn/dense_1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dddqn/dense_1/Tensordot/Const?
dddqn/dense_1/Tensordot/ProdProd)dddqn/dense_1/Tensordot/GatherV2:output:0&dddqn/dense_1/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dddqn/dense_1/Tensordot/Prod?
dddqn/dense_1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2!
dddqn/dense_1/Tensordot/Const_1?
dddqn/dense_1/Tensordot/Prod_1Prod+dddqn/dense_1/Tensordot/GatherV2_1:output:0(dddqn/dense_1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2 
dddqn/dense_1/Tensordot/Prod_1?
#dddqn/dense_1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2%
#dddqn/dense_1/Tensordot/concat/axis?
dddqn/dense_1/Tensordot/concatConcatV2%dddqn/dense_1/Tensordot/free:output:0%dddqn/dense_1/Tensordot/axes:output:0,dddqn/dense_1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2 
dddqn/dense_1/Tensordot/concat?
dddqn/dense_1/Tensordot/stackPack%dddqn/dense_1/Tensordot/Prod:output:0'dddqn/dense_1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dddqn/dense_1/Tensordot/stack?
!dddqn/dense_1/Tensordot/transpose	Transposedddqn/dense/Relu:activations:0'dddqn/dense_1/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????2#
!dddqn/dense_1/Tensordot/transpose?
dddqn/dense_1/Tensordot/ReshapeReshape%dddqn/dense_1/Tensordot/transpose:y:0&dddqn/dense_1/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2!
dddqn/dense_1/Tensordot/Reshape?
dddqn/dense_1/Tensordot/MatMulMatMul(dddqn/dense_1/Tensordot/Reshape:output:0.dddqn/dense_1/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2 
dddqn/dense_1/Tensordot/MatMul?
dddqn/dense_1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?2!
dddqn/dense_1/Tensordot/Const_2?
%dddqn/dense_1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2'
%dddqn/dense_1/Tensordot/concat_1/axis?
 dddqn/dense_1/Tensordot/concat_1ConcatV2)dddqn/dense_1/Tensordot/GatherV2:output:0(dddqn/dense_1/Tensordot/Const_2:output:0.dddqn/dense_1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2"
 dddqn/dense_1/Tensordot/concat_1?
dddqn/dense_1/TensordotReshape(dddqn/dense_1/Tensordot/MatMul:product:0)dddqn/dense_1/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????2
dddqn/dense_1/Tensordot?
$dddqn/dense_1/BiasAdd/ReadVariableOpReadVariableOp-dddqn_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02&
$dddqn/dense_1/BiasAdd/ReadVariableOp?
dddqn/dense_1/BiasAddBiasAdd dddqn/dense_1/Tensordot:output:0,dddqn/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????2
dddqn/dense_1/BiasAdd?
dddqn/dense_1/ReluReludddqn/dense_1/BiasAdd:output:0*
T0*,
_output_shapes
:??????????2
dddqn/dense_1/Relu?
&dddqn/dense_2/Tensordot/ReadVariableOpReadVariableOp/dddqn_dense_2_tensordot_readvariableop_resource*
_output_shapes
:	?*
dtype02(
&dddqn/dense_2/Tensordot/ReadVariableOp?
dddqn/dense_2/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dddqn/dense_2/Tensordot/axes?
dddqn/dense_2/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dddqn/dense_2/Tensordot/free?
dddqn/dense_2/Tensordot/ShapeShape dddqn/dense_1/Relu:activations:0*
T0*
_output_shapes
:2
dddqn/dense_2/Tensordot/Shape?
%dddqn/dense_2/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2'
%dddqn/dense_2/Tensordot/GatherV2/axis?
 dddqn/dense_2/Tensordot/GatherV2GatherV2&dddqn/dense_2/Tensordot/Shape:output:0%dddqn/dense_2/Tensordot/free:output:0.dddqn/dense_2/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2"
 dddqn/dense_2/Tensordot/GatherV2?
'dddqn/dense_2/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2)
'dddqn/dense_2/Tensordot/GatherV2_1/axis?
"dddqn/dense_2/Tensordot/GatherV2_1GatherV2&dddqn/dense_2/Tensordot/Shape:output:0%dddqn/dense_2/Tensordot/axes:output:00dddqn/dense_2/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2$
"dddqn/dense_2/Tensordot/GatherV2_1?
dddqn/dense_2/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dddqn/dense_2/Tensordot/Const?
dddqn/dense_2/Tensordot/ProdProd)dddqn/dense_2/Tensordot/GatherV2:output:0&dddqn/dense_2/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dddqn/dense_2/Tensordot/Prod?
dddqn/dense_2/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2!
dddqn/dense_2/Tensordot/Const_1?
dddqn/dense_2/Tensordot/Prod_1Prod+dddqn/dense_2/Tensordot/GatherV2_1:output:0(dddqn/dense_2/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2 
dddqn/dense_2/Tensordot/Prod_1?
#dddqn/dense_2/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2%
#dddqn/dense_2/Tensordot/concat/axis?
dddqn/dense_2/Tensordot/concatConcatV2%dddqn/dense_2/Tensordot/free:output:0%dddqn/dense_2/Tensordot/axes:output:0,dddqn/dense_2/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2 
dddqn/dense_2/Tensordot/concat?
dddqn/dense_2/Tensordot/stackPack%dddqn/dense_2/Tensordot/Prod:output:0'dddqn/dense_2/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dddqn/dense_2/Tensordot/stack?
!dddqn/dense_2/Tensordot/transpose	Transpose dddqn/dense_1/Relu:activations:0'dddqn/dense_2/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????2#
!dddqn/dense_2/Tensordot/transpose?
dddqn/dense_2/Tensordot/ReshapeReshape%dddqn/dense_2/Tensordot/transpose:y:0&dddqn/dense_2/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2!
dddqn/dense_2/Tensordot/Reshape?
dddqn/dense_2/Tensordot/MatMulMatMul(dddqn/dense_2/Tensordot/Reshape:output:0.dddqn/dense_2/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2 
dddqn/dense_2/Tensordot/MatMul?
dddqn/dense_2/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2!
dddqn/dense_2/Tensordot/Const_2?
%dddqn/dense_2/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2'
%dddqn/dense_2/Tensordot/concat_1/axis?
 dddqn/dense_2/Tensordot/concat_1ConcatV2)dddqn/dense_2/Tensordot/GatherV2:output:0(dddqn/dense_2/Tensordot/Const_2:output:0.dddqn/dense_2/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2"
 dddqn/dense_2/Tensordot/concat_1?
dddqn/dense_2/TensordotReshape(dddqn/dense_2/Tensordot/MatMul:product:0)dddqn/dense_2/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????2
dddqn/dense_2/Tensordot?
$dddqn/dense_2/BiasAdd/ReadVariableOpReadVariableOp-dddqn_dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02&
$dddqn/dense_2/BiasAdd/ReadVariableOp?
dddqn/dense_2/BiasAddBiasAdd dddqn/dense_2/Tensordot:output:0,dddqn/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????2
dddqn/dense_2/BiasAdd?
&dddqn/dense_3/Tensordot/ReadVariableOpReadVariableOp/dddqn_dense_3_tensordot_readvariableop_resource*
_output_shapes
:	?*
dtype02(
&dddqn/dense_3/Tensordot/ReadVariableOp?
dddqn/dense_3/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dddqn/dense_3/Tensordot/axes?
dddqn/dense_3/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dddqn/dense_3/Tensordot/free?
dddqn/dense_3/Tensordot/ShapeShape dddqn/dense_1/Relu:activations:0*
T0*
_output_shapes
:2
dddqn/dense_3/Tensordot/Shape?
%dddqn/dense_3/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2'
%dddqn/dense_3/Tensordot/GatherV2/axis?
 dddqn/dense_3/Tensordot/GatherV2GatherV2&dddqn/dense_3/Tensordot/Shape:output:0%dddqn/dense_3/Tensordot/free:output:0.dddqn/dense_3/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2"
 dddqn/dense_3/Tensordot/GatherV2?
'dddqn/dense_3/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2)
'dddqn/dense_3/Tensordot/GatherV2_1/axis?
"dddqn/dense_3/Tensordot/GatherV2_1GatherV2&dddqn/dense_3/Tensordot/Shape:output:0%dddqn/dense_3/Tensordot/axes:output:00dddqn/dense_3/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2$
"dddqn/dense_3/Tensordot/GatherV2_1?
dddqn/dense_3/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dddqn/dense_3/Tensordot/Const?
dddqn/dense_3/Tensordot/ProdProd)dddqn/dense_3/Tensordot/GatherV2:output:0&dddqn/dense_3/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dddqn/dense_3/Tensordot/Prod?
dddqn/dense_3/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2!
dddqn/dense_3/Tensordot/Const_1?
dddqn/dense_3/Tensordot/Prod_1Prod+dddqn/dense_3/Tensordot/GatherV2_1:output:0(dddqn/dense_3/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2 
dddqn/dense_3/Tensordot/Prod_1?
#dddqn/dense_3/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2%
#dddqn/dense_3/Tensordot/concat/axis?
dddqn/dense_3/Tensordot/concatConcatV2%dddqn/dense_3/Tensordot/free:output:0%dddqn/dense_3/Tensordot/axes:output:0,dddqn/dense_3/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2 
dddqn/dense_3/Tensordot/concat?
dddqn/dense_3/Tensordot/stackPack%dddqn/dense_3/Tensordot/Prod:output:0'dddqn/dense_3/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dddqn/dense_3/Tensordot/stack?
!dddqn/dense_3/Tensordot/transpose	Transpose dddqn/dense_1/Relu:activations:0'dddqn/dense_3/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????2#
!dddqn/dense_3/Tensordot/transpose?
dddqn/dense_3/Tensordot/ReshapeReshape%dddqn/dense_3/Tensordot/transpose:y:0&dddqn/dense_3/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2!
dddqn/dense_3/Tensordot/Reshape?
dddqn/dense_3/Tensordot/MatMulMatMul(dddqn/dense_3/Tensordot/Reshape:output:0.dddqn/dense_3/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2 
dddqn/dense_3/Tensordot/MatMul?
dddqn/dense_3/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2!
dddqn/dense_3/Tensordot/Const_2?
%dddqn/dense_3/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2'
%dddqn/dense_3/Tensordot/concat_1/axis?
 dddqn/dense_3/Tensordot/concat_1ConcatV2)dddqn/dense_3/Tensordot/GatherV2:output:0(dddqn/dense_3/Tensordot/Const_2:output:0.dddqn/dense_3/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2"
 dddqn/dense_3/Tensordot/concat_1?
dddqn/dense_3/TensordotReshape(dddqn/dense_3/Tensordot/MatMul:product:0)dddqn/dense_3/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????2
dddqn/dense_3/Tensordot?
$dddqn/dense_3/BiasAdd/ReadVariableOpReadVariableOp-dddqn_dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02&
$dddqn/dense_3/BiasAdd/ReadVariableOp?
dddqn/dense_3/BiasAddBiasAdd dddqn/dense_3/Tensordot:output:0,dddqn/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????2
dddqn/dense_3/BiasAdd~
dddqn/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
dddqn/Mean/reduction_indices?

dddqn/MeanMeandddqn/dense_3/BiasAdd:output:0%dddqn/Mean/reduction_indices:output:0*
T0*+
_output_shapes
:?????????*
	keep_dims(2

dddqn/Mean?
	dddqn/subSubdddqn/dense_3/BiasAdd:output:0dddqn/Mean:output:0*
T0*+
_output_shapes
:?????????2
	dddqn/sub?
	dddqn/addAddV2dddqn/dense_2/BiasAdd:output:0dddqn/sub:z:0*
T0*+
_output_shapes
:?????????2
	dddqn/add?
IdentityIdentitydddqn/add:z:0#^dddqn/dense/BiasAdd/ReadVariableOp%^dddqn/dense/Tensordot/ReadVariableOp%^dddqn/dense_1/BiasAdd/ReadVariableOp'^dddqn/dense_1/Tensordot/ReadVariableOp%^dddqn/dense_2/BiasAdd/ReadVariableOp'^dddqn/dense_2/Tensordot/ReadVariableOp%^dddqn/dense_3/BiasAdd/ReadVariableOp'^dddqn/dense_3/Tensordot/ReadVariableOp*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*K
_input_shapes:
8:??????????::::::::2H
"dddqn/dense/BiasAdd/ReadVariableOp"dddqn/dense/BiasAdd/ReadVariableOp2L
$dddqn/dense/Tensordot/ReadVariableOp$dddqn/dense/Tensordot/ReadVariableOp2L
$dddqn/dense_1/BiasAdd/ReadVariableOp$dddqn/dense_1/BiasAdd/ReadVariableOp2P
&dddqn/dense_1/Tensordot/ReadVariableOp&dddqn/dense_1/Tensordot/ReadVariableOp2L
$dddqn/dense_2/BiasAdd/ReadVariableOp$dddqn/dense_2/BiasAdd/ReadVariableOp2P
&dddqn/dense_2/Tensordot/ReadVariableOp&dddqn/dense_2/Tensordot/ReadVariableOp2L
$dddqn/dense_3/BiasAdd/ReadVariableOp$dddqn/dense_3/BiasAdd/ReadVariableOp2P
&dddqn/dense_3/Tensordot/ReadVariableOp&dddqn/dense_3/Tensordot/ReadVariableOp:U Q
,
_output_shapes
:??????????
!
_user_specified_name	input_1
?
?
C__inference_dense_3_layer_call_and_return_conditional_losses_413171

inputs%
!tensordot_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Tensordot/ReadVariableOp?
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes
:	?*
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axis?
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis?
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const?
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1?
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis?
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat?
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack?
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*,
_output_shapes
:??????????2
Tensordot/transpose?
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
Tensordot/Reshape?
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
Tensordot/MatMulp
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis?
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1?
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????2
	Tensordot?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
}
(__inference_dense_3_layer_call_fn_413403

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_3_layer_call_and_return_conditional_losses_4131712
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :??????????::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
C__inference_dense_2_layer_call_and_return_conditional_losses_413355

inputs%
!tensordot_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Tensordot/ReadVariableOp?
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes
:	?*
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axis?
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis?
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const?
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1?
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis?
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat?
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack?
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*,
_output_shapes
:??????????2
Tensordot/transpose?
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
Tensordot/Reshape?
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
Tensordot/MatMulp
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis?
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1?
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????2
	Tensordot?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
? 
?
C__inference_dense_1_layer_call_and_return_conditional_losses_413079

inputs%
!tensordot_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Tensordot/ReadVariableOp?
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource* 
_output_shapes
:
??*
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axis?
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis?
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const?
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1?
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis?
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat?
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack?
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*,
_output_shapes
:??????????2
Tensordot/transpose?
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
Tensordot/Reshape?
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
Tensordot/MatMulq
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis?
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1?
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????2
	Tensordot?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????2	
BiasAdd]
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
$__inference_signature_wrapper_413245
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? **
f%R#
!__inference__wrapped_model_4129972
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*K
_input_shapes:
8:??????????::::::::22
StatefulPartitionedCallStatefulPartitionedCall:U Q
,
_output_shapes
:??????????
!
_user_specified_name	input_1
?
}
(__inference_dense_1_layer_call_fn_413325

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_4130792
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :??????????::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
A__inference_dddqn_layer_call_and_return_conditional_losses_413192
input_1
dense_413043
dense_413045
dense_1_413090
dense_1_413092
dense_2_413136
dense_2_413138
dense_3_413182
dense_3_413184
identity??dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?dense_2/StatefulPartitionedCall?dense_3/StatefulPartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCallinput_1dense_413043dense_413045*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_4130322
dense/StatefulPartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_413090dense_1_413092*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_4130792!
dense_1/StatefulPartitionedCall?
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_413136dense_2_413138*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_4131252!
dense_2/StatefulPartitionedCall?
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_3_413182dense_3_413184*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_3_layer_call_and_return_conditional_losses_4131712!
dense_3/StatefulPartitionedCallr
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Mean/reduction_indices?
MeanMean(dense_3/StatefulPartitionedCall:output:0Mean/reduction_indices:output:0*
T0*+
_output_shapes
:?????????*
	keep_dims(2
Mean?
subSub(dense_3/StatefulPartitionedCall:output:0Mean:output:0*
T0*+
_output_shapes
:?????????2
sub|
addAddV2(dense_2/StatefulPartitionedCall:output:0sub:z:0*
T0*+
_output_shapes
:?????????2
add?
IdentityIdentityadd:z:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*K
_input_shapes:
8:??????????::::::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall:U Q
,
_output_shapes
:??????????
!
_user_specified_name	input_1
?
?
C__inference_dense_3_layer_call_and_return_conditional_losses_413394

inputs%
!tensordot_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Tensordot/ReadVariableOp?
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes
:	?*
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axis?
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis?
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const?
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1?
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis?
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat?
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack?
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*,
_output_shapes
:??????????2
Tensordot/transpose?
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
Tensordot/Reshape?
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
Tensordot/MatMulp
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis?
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1?
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????2
	Tensordot?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
@
input_15
serving_default_input_1:0??????????@
output_14
StatefulPartitionedCall:0?????????tensorflow/serving/predict:?q
?
d1
d2
v
a
	optimizer
trainable_variables
	variables
regularization_losses
		keras_api


signatures
V__call__
W_default_save_signature
*X&call_and_return_all_conditional_losses"?
_tf_keras_model?{"class_name": "DDDQN", "name": "dddqn", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"layer was saved without config": true}, "is_graph_network": false, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "DDDQN"}, "training_config": {"loss": "mse", "metrics": null, "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 9.999999747378752e-06, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
?

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
Y__call__
*Z&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 289}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1, 289]}}
?

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
[__call__
*\&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1, 128]}}
?

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
]__call__
*^&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1, 128]}}
?

kernel
bias
trainable_variables
 	variables
!regularization_losses
"	keras_api
___call__
*`&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_3", "trainable": true, "dtype": "float32", "units": 6, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1, 128]}}
?
#iter

$beta_1

%beta_2
	&decay
'learning_ratemFmGmHmImJmKmLmMvNvOvPvQvRvSvTvU"
	optimizer
X
0
1
2
3
4
5
6
7"
trackable_list_wrapper
X
0
1
2
3
4
5
6
7"
trackable_list_wrapper
 "
trackable_list_wrapper
?
trainable_variables
(layer_regularization_losses
	variables
)layer_metrics
*non_trainable_variables
+metrics
regularization_losses

,layers
V__call__
W_default_save_signature
*X&call_and_return_all_conditional_losses
&X"call_and_return_conditional_losses"
_generic_user_object
,
aserving_default"
signature_map
&:$
??2dddqn/dense/kernel
:?2dddqn/dense/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
trainable_variables
-layer_regularization_losses
	variables
regularization_losses
.non_trainable_variables
/metrics
0layer_metrics

1layers
Y__call__
*Z&call_and_return_all_conditional_losses
&Z"call_and_return_conditional_losses"
_generic_user_object
(:&
??2dddqn/dense_1/kernel
!:?2dddqn/dense_1/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
trainable_variables
2layer_regularization_losses
	variables
regularization_losses
3non_trainable_variables
4metrics
5layer_metrics

6layers
[__call__
*\&call_and_return_all_conditional_losses
&\"call_and_return_conditional_losses"
_generic_user_object
':%	?2dddqn/dense_2/kernel
 :2dddqn/dense_2/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
trainable_variables
7layer_regularization_losses
	variables
regularization_losses
8non_trainable_variables
9metrics
:layer_metrics

;layers
]__call__
*^&call_and_return_all_conditional_losses
&^"call_and_return_conditional_losses"
_generic_user_object
':%	?2dddqn/dense_3/kernel
 :2dddqn/dense_3/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
trainable_variables
<layer_regularization_losses
 	variables
!regularization_losses
=non_trainable_variables
>metrics
?layer_metrics

@layers
___call__
*`&call_and_return_all_conditional_losses
&`"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
'
A0"
trackable_list_wrapper
<
0
1
2
3"
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
?
	Btotal
	Ccount
D	variables
E	keras_api"?
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
:  (2total
:  (2count
.
B0
C1"
trackable_list_wrapper
-
D	variables"
_generic_user_object
+:)
??2Adam/dddqn/dense/kernel/m
$:"?2Adam/dddqn/dense/bias/m
-:+
??2Adam/dddqn/dense_1/kernel/m
&:$?2Adam/dddqn/dense_1/bias/m
,:*	?2Adam/dddqn/dense_2/kernel/m
%:#2Adam/dddqn/dense_2/bias/m
,:*	?2Adam/dddqn/dense_3/kernel/m
%:#2Adam/dddqn/dense_3/bias/m
+:)
??2Adam/dddqn/dense/kernel/v
$:"?2Adam/dddqn/dense/bias/v
-:+
??2Adam/dddqn/dense_1/kernel/v
&:$?2Adam/dddqn/dense_1/bias/v
,:*	?2Adam/dddqn/dense_2/kernel/v
%:#2Adam/dddqn/dense_2/bias/v
,:*	?2Adam/dddqn/dense_3/kernel/v
%:#2Adam/dddqn/dense_3/bias/v
?2?
&__inference_dddqn_layer_call_fn_413214?
???
FullArgSpec!
args?
jself
j
input_data
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *+?(
&?#
input_1??????????
?2?
!__inference__wrapped_model_412997?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *+?(
&?#
input_1??????????
?2?
A__inference_dddqn_layer_call_and_return_conditional_losses_413192?
???
FullArgSpec!
args?
jself
j
input_data
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *+?(
&?#
input_1??????????
?2?
&__inference_dense_layer_call_fn_413285?
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
A__inference_dense_layer_call_and_return_conditional_losses_413276?
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
(__inference_dense_1_layer_call_fn_413325?
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
C__inference_dense_1_layer_call_and_return_conditional_losses_413316?
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
(__inference_dense_2_layer_call_fn_413364?
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
C__inference_dense_2_layer_call_and_return_conditional_losses_413355?
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
(__inference_dense_3_layer_call_fn_413403?
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
C__inference_dense_3_layer_call_and_return_conditional_losses_413394?
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
?B?
$__inference_signature_wrapper_413245input_1"?
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
 ?
!__inference__wrapped_model_412997z5?2
+?(
&?#
input_1??????????
? "7?4
2
output_1&?#
output_1??????????
A__inference_dddqn_layer_call_and_return_conditional_losses_413192l5?2
+?(
&?#
input_1??????????
? ")?&
?
0?????????
? ?
&__inference_dddqn_layer_call_fn_413214_5?2
+?(
&?#
input_1??????????
? "???????????
C__inference_dense_1_layer_call_and_return_conditional_losses_413316f4?1
*?'
%?"
inputs??????????
? "*?'
 ?
0??????????
? ?
(__inference_dense_1_layer_call_fn_413325Y4?1
*?'
%?"
inputs??????????
? "????????????
C__inference_dense_2_layer_call_and_return_conditional_losses_413355e4?1
*?'
%?"
inputs??????????
? ")?&
?
0?????????
? ?
(__inference_dense_2_layer_call_fn_413364X4?1
*?'
%?"
inputs??????????
? "???????????
C__inference_dense_3_layer_call_and_return_conditional_losses_413394e4?1
*?'
%?"
inputs??????????
? ")?&
?
0?????????
? ?
(__inference_dense_3_layer_call_fn_413403X4?1
*?'
%?"
inputs??????????
? "???????????
A__inference_dense_layer_call_and_return_conditional_losses_413276f4?1
*?'
%?"
inputs??????????
? "*?'
 ?
0??????????
? ?
&__inference_dense_layer_call_fn_413285Y4?1
*?'
%?"
inputs??????????
? "????????????
$__inference_signature_wrapper_413245?@?=
? 
6?3
1
input_1&?#
input_1??????????"7?4
2
output_1&?#
output_1?????????