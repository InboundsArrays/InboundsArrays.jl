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


Testing
-------

As bounds checks (on array accesses) are disabled by default when using `InboundsArray`,
you should make sure to test your package using `--check-bounds=yes`, which will restore
the bounds checks.
