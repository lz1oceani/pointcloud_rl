# SAPIEN-RL YAML Documentation

## Include
Specified by `_include`, the value is a `str` of relative SAPIEN-RL YAML file path.
A sibling `_override` key is used to override loaded values.

## Variants
Specified by `_variants`, the value is a `dict` containing `type` and `global_id`, and other type-related fields.

## Scoped variable
Start with a `$`. Takes a **constant** expression as value. Used in any variable context.

## Expression
For any string, the following evaluation process applies
1. If the string starts with `eval`, the content inside `eval(*)` it is treated as an expression.
2. If the string starts with `Uniform`, `quat`, or anything indicating it is a function, it is treated as an expression
3. If the string contains `$`, it is treated as an expression.
4. Otherwise, it is treated as an literal

For any expression, it is `eval`ed in the variable processor. Before `eval` is
called, the `$` tokens will be replaced by a serialized version of the variable
value.

## Process order
Preprocessor:
1. Load YAML file as raw string
2. Convert all file paths to absolute
3. For each include, use the preprocessor to load it

Variable processor:
1. Take RNG, evaluate expression for each scoped variable (No scoped variable is allowed in the expressions)
2. Evaluate any expression with current scope

Variant processor:
1. Take user config and RNG and determine all variants


## Guide
- _variants is used to realize randomization
- If something needs to be modified online, it should be directly passed to env, instead of writing it in yaml