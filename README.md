InboundsArrays gives a convenient way to remove bounds-checks on array accesses, without
needing to using `@inbounds` in many places or use `--check-bounds=no`. The effect is to
apply `@inbounds` to all `getindex()` and `setindex!()` calls.

An InboundsArray can wrap any other array type. For example for `Array`/`Vector`/`Matrix`:
```julia
a = InboundsArray(rand(3,4,5))
b = similar(a)
for i ∈ 1:5, j ∈ 1:4, k ∈ 1:3
    b[k,j,i] = a[k,j,i] + 1
end
c = a .* 2

v = InboundsVector(rand(6))
M = InboundsMatrix(rand(7,8))
```
Broadcasting is implemented so that the result of broadcasting an InboundsArray with any
other array types is an InboundsArray (if the result is an array and not a scalar). So `c`
in the example above is an `InboundsArray`.

To get the wrapped, non-inbounds array back from an `InboundsArray` use
```julia
a_noninbounds = get_noninbounds(a)
```
`get_noninbouds()` is a no-op on anything other than an `InboundsArray`.


Testing - this is very IMPORTANT!
---------------------------------

As bounds checks (on array accesses) are disabled by default when using `InboundsArray`,
you should make sure to test your package using `--check-bounds=yes`, which will restore
the bounds checks.

`MethodError` no method matching `InboundsArray`/`InboundsVector`/`InboundsMatrix`
----------------------------------------------------------------------------------

Many packages (e.g. LinearAlgebra, etc.) support `AbstractArray` objects but
have specialized, more optimized methods for specific array types (e.g.
`Array`, `SparseMatrixCSC`, etc.). An `InboundsArray` is a wrapper around an
array of some type `TArray`. To achieve best performance, we should avoid the
`AbstractArray` interface and instead pass the wrapped `TArray` to the package
(the package authors should take care of using `@inbounds` anywhere it is
needed, so there should be no performance benefit to passing an `InboundsArray`
to a dependency package - unless perhaps that dependency explicitly supports or
requires `InboundsArray`). Doing this pass-through requires a method to be
defined that takes an `InboundsArray` argument or arguments, and calls the
standard version of the function using the wrapped array, passing through any
extra non-`InboundsArray` arguments. `InboundsArrays` provides many of these
pass-through methods for commonly used packages/functions (if you need more,
please open a PR - they are generally short to write!).

In order to guarantee the best performance possible, by default
`InboundsArrays` is defined so that there will be an error if an
`InboundsArray` is passed to an unsupported function. This is achieved by not
making `AbstractInboundsArray` a subtype of `AbstractArray`, so that a
`MethodError()` will be throw if an `InboundsArray` is passed to a function
that accepts an `AbstractArray`. In other words, we error rather than risking
poor performance. If you know that all performance-critical functions are
supported correctly, but there are some other functions that are not supported,
or if you know that the `AbstractArray` interface provides optimal performance
for any unsupported functions, it might be convenient for `InboundsArray` to be
a subtype of `AbstractArray`. This can be achieved by calling
```julia
InboundsArrays.set_inherit_from_AbstractArray(true)
```
This setting will be saved in `LocalPreferences.toml` and so will persist. Call
again with no argument (or `false`) to reset to the default. After changing the
setting you must restart Julia and recompile any system images that include
`InboundsArrays` in order for the change to take effect.

If you use `set_inherit_from_AbstractArray = true`, you are responsible for
ensuring that there are no significant slow-downs from using `InboundsArray`
instead of the array type that you would otherwise have used! It is preferred
to add support for whatever functions you need to `InboundsArrays.jl`.

Status and development
----------------------

This package should be considered experimental. It can improve performance, but
when it interacts with packages that support `AbstractArray`s, but have
specialised, optimised implementations for `Array` we most likely have to
include a wrapper in this package to make sure the `Array` implementation is
used (see the [Coverage section below](#Coverage) for the current status).
Therefore if an `InboundsArray` interacts with an unsupported (feature of a)
package, it can dramatically decrease performance. Ideally you should benchmark
each performance-critical function that you want to use, comparing `Array` and
`InboundsArray` (or in general, the array type you would otherwise use, and
`InboundsArray` wrapping that array type).

Contributions are very, very welcome to extend support/coverage. This is often
as simple as defining an `@inline` wrapper to pass the `a` field of the
`InboundsArray` arguments (the wrapped array) to the standard implementation,
for example
```julia
@inline function ldiv!(x::InboundsVector, Alu::Factorization, b::InboundsVector)
    ldiv!(x.a, Alu, b.a)
    return x
end
```
and possibly re-wrapping the result when the function is not in-place
```julia
@inline function ldiv(Alu::Factorization, b::InboundsVector)
    return InboundsArray(ldiv(Alu, b.a))
end
```

Coverage
--------

At present, `InboundsArray` supports:
* `Base` functionality
    * `getindex()`, `setindex!()`, `length()`, `size()`, `axes()`, `similar()`.
    * Broadcasting
    * `reshape()` and `vec()`
    * If inheriting from `AbstractArray` is enabled:
        * The `AbstractArray` interface and broadcasting (returning an `InboundsArray`)
        * Also any package that only requires the generic `AbstractArray` interface
* `LinearAlgebra`
    * `mul!`, `lu`, `lu!`, `ldiv`, `ldiv!`, `*`
* `SparseArrays` and `SparseMatricesCSR`
    * `sparse`/`sparsecsr`, `convert`, `mul!`, `lu`, `lu!`, `ldiv`, `ldiv!`, `*`
* `MPI` is intended to support all functions by wrapping those listed, but has
  not been comprehensively tested
    * `Buffer`, `UBuffer`, `VBuffer`
* `FFTW`
    * `plan_fft!`, `plan_ifft!`, `*`
