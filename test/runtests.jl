module InboundsArraysTests

using InboundsArrays
using Test

using LinearAlgebra

# Import packages for extensions if possible
try
    using MPI
catch
end

function isequal(args...)
    return isapprox(args...; rtol=0.0, atol=0.0)
end

function isclose(args...)
    return isapprox(args...; rtol=1.0e-14, atol=0.0)
end

function runtests()

    @testset "InboundsArrays" begin
        @testset "InboundsVector" begin
            a = InboundsVector([1.0, 2.0, 3.0, 4.0])
            b = InboundsVector([5.0, 6.0, 7.0, 8.0])

            @test size(a) == (4,)
            @test length(a) == 4
            @test a[2] == 2.0
            a[2] = 42.0
            @test a[2] == 42.0
            a[2] = 2.0

            for i ∈ 1:length(a)
                a[i] += b[i]
            end

            @test isequal(a, [6.0, 8.0, 10.0, 12.0])

            c = a .+ b
            @test c isa InboundsVector
            @test isequal(c, [11.0, 14.0, 17.0, 20.0])

            c .= a .* b
            @test c isa InboundsVector
            @test isequal(c, [30.0, 48.0, 70.0, 96.0])

            d = similar(a)
            @test d isa InboundsVector{Float64, Vector{Float64}}
            @test size(d) == (4,)

            d = similar(a, Int64)
            @test d isa InboundsVector{Int64, Vector{Int64}}
            @test size(d) == (4,)

            d = similar(a, (3, 5))
            @test d isa InboundsArray{Float64, 2, Array{Float64, 2}}
            @test size(d) == (3, 5)

            d = similar(a, Int64, (3, 5))
            @test d isa InboundsArray{Int64, 2, Array{Int64, 2}}
            @test size(d) == (3, 5)

            @test axes(a) == (1:4,)
        end

        @testset "InboundsMatrix" begin
            a = InboundsMatrix([1.0 2.0; 3.0 4.0])
            b = InboundsMatrix([5.0 6.0; 7.0 8.0])

            @test size(a) == (2, 2)
            @test length(a) == 4
            @test a[1, 2] == 2.0
            a[1, 2] = 42.0
            @test a[1, 2] == 42.0
            a[1, 2] = 2.0

            for i ∈ 1:length(a)
                a[i] += b[i]
            end

            @test isequal(a, [6.0 8.0; 10.0 12.0])

            a .= [1.0 2.0; 3.0 4.0]

            for i ∈ 1:size(a, 2), j ∈ 1:size(a, 1)
                a[j, i] += b[j, i]
            end

            @test isequal(a, [6.0 8.0; 10.0 12.0])

            c = a .+ b
            @test c isa InboundsMatrix
            @test isequal(c, [11.0 14.0; 17.0 20.0])

            c .= a .* b
            @test c isa InboundsMatrix
            @test isequal(c, [30.0 48.0; 70.0 96.0])

            d = similar(a)
            @test d isa InboundsMatrix{Float64, Matrix{Float64}}
            @test size(d) == (2, 2)

            d = similar(a, Int64)
            @test d isa InboundsMatrix{Int64, Matrix{Int64}}
            @test size(d) == (2, 2)

            d = similar(a, (3, 5, 6))
            @test d isa InboundsArray{Float64, 3, Array{Float64, 3}}
            @test size(d) == (3, 5, 6)

            d = similar(a, Int64, (3, 5, 6))
            @test d isa InboundsArray{Int64, 3, Array{Int64, 3}}
            @test size(d) == (3, 5, 6)

            @test axes(a) == (1:2, 1:2)
        end

        @testset "InboundsArray" begin
            a = InboundsArray([1.0 2.0; 3.0 4.0;;; 1.0 2.0; 3.0 4.0])
            b = InboundsArray([5.0 6.0; 7.0 8.0;;; 5.0 6.0; 7.0 8.0])

            @test size(a) == (2, 2, 2)
            @test length(a) == 8
            @test a[1, 2, 1] == 2.0
            a[1, 2, 1] = 42.0
            @test a[1, 2, 1] == 42.0
            a[1, 2, 1] = 2.0

            for i ∈ 1:length(a)
                a[i] += b[i]
            end

            @test isequal(a, [6.0 8.0; 10.0 12.0;;; 6.0 8.0; 10.0 12.0])

            a .= [1.0 2.0; 3.0 4.0;;; 1.0 2.0; 3.0 4.0]

            for i ∈ 1:size(a, 3), j ∈ 1:size(a, 2), k ∈ 1:size(a, 1)
                a[k, j, i] += b[k, j, i]
            end

            @test isequal(a, [6.0 8.0; 10.0 12.0;;; 6.0 8.0; 10.0 12.0])

            c = a .+ b
            @test c isa InboundsArray
            @test isequal(c, [11.0 14.0; 17.0 20.0;;; 11.0 14.0; 17.0 20.0])

            c .= a .* b
            @test c isa InboundsArray
            @test isequal(c, [30.0 48.0; 70.0 96.0;;; 30.0 48.0; 70.0 96.0])

            c = a .+ 2
            @test c isa InboundsArray
            @test isequal(c, [8.0 10.0; 12.0 14.0;;; 8.0 10.0; 12.0 14.0])

            d = similar(a)
            @test d isa InboundsArray{Float64, 3, Array{Float64, 3}}
            @test size(d) == (2, 2, 2)

            d = similar(a, Int64)
            @test d isa InboundsArray{Int64, 3, Array{Int64, 3}}
            @test size(d) == (2, 2, 2)

            d = similar(a, (3, 5))
            @test d isa InboundsArray{Float64, 2, Array{Float64, 2}}
            @test size(d) == (3, 5)

            d = similar(a, Int64, (3, 5))
            @test d isa InboundsArray{Int64, 2, Array{Int64, 2}}
            @test size(d) == (3, 5)

            @test axes(a) == (1:2, 1:2, 1:2)
        end

        @testset "LinearAlgebra interface" begin
            A = InboundsArray([1.0 2.0; 3.0 4.0])
            B = InboundsArray([5.0 6.0; 7.0 8.0])
            C = InboundsArray([9.0 10.0; 11.0 12.0])
            x = InboundsArray([5.0, 7.0])
            y = InboundsArray([6.0, 8.0])

            mul!(y, A, x)
            @test isequal(y, [19.0, 43.0])
            @test y isa InboundsVector

            z = A * x
            @test isequal(z, [19.0, 43.0])
            @test z isa InboundsVector

            y .= [6.0, 8.0]
            mul!(y, A, x, 2.0, 3.0)
            @test isequal(y, [56.0, 110.0])
            @test y isa InboundsVector

            mul!(C, A, B)
            @test isequal(C, [19.0 22.0; 43.0 50.0])
            @test C isa InboundsMatrix

            D = A * B
            @test isequal(D, [19.0 22.0; 43.0 50.0])
            @test D isa InboundsMatrix

            C .= [9.0 10.0; 11.0 12.0]
            mul!(C, A, B, 2.0, 3.0)
            @test isequal(C, [65.0 74.0; 119.0 136.0])
            @test C isa InboundsMatrix

            Ax = InboundsArray([19.0, 43.0])
            Alu = lu(A)
            ldiv!(y, Alu, Ax)
            @test isclose(y, x)
            @test y isa InboundsVector

            z = Alu \ Ax
            @test isclose(z, x)
            @test z isa InboundsVector

            z = A \ Ax
            @test isclose(z, x)
            @test z isa InboundsVector

            AB = InboundsArray([19.0 22.0; 43.0 50.0])
            ldiv!(C, Alu, AB)
            @test isclose(C, B)
            @test C isa InboundsMatrix

            D = Alu \ AB
            @test isclose(D, B)
            @test D isa InboundsMatrix

            D = A \ AB
            @test isclose(D, B)
            @test D isa InboundsMatrix
        end

        mpiext = Base.get_extension(InboundsArrays, :MPIExt)
        if mpiext !== nothing
            @testset "MPIExt" begin
                a = InboundsArray(ones(3, 4, 5))
                b = similar(a)

                MPI.Init()

                MPI.Allgather!(a, b, MPI.COMM_WORLD)
                @test isequal(b, ones(3, 4, 5))

                b .= 0.0
                vb = MPI.VBuffer(b, [3 * 4 * 5])
                MPI.Allgatherv!(a, vb, MPI.COMM_WORLD)
                @test isequal(b, ones(3, 4, 5))

                b .= 0.0
                MPI.Bcast!(a, 0, MPI.COMM_WORLD)
                @test isequal(a, ones(3, 4, 5))
            end
        end
    end

end

end # module InboundsArraysTests

using .InboundsArraysTests
InboundsArraysTests.runtests()
