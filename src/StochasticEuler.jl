module StochasticEuler

export ieuler, CumulativeNormal, Ito, Stratonovich,
  ieuler_sde, ieuler_mayurama, ieuler_heun



"""
## ieuler()

Solve an ode using the implicit euler method. The ode is passed as a function
`ode!(t, x, xdot)` that writes the time derivative of `x`
at time `t` into the vector `xdot`.

The initial state is `x0`. The time grid at which the states are returned is `ts`,
the maximum stepsize to be used is `hmax`.

At each time step the method solves the following non-linear system of equations
for x(t+h):

    x(t+h) == x(t) + [(1-ν) f(t, x(t)) + ν f(t+h, x(t+h))] h


This is achieved by a damped pseudo-Newton iteration with the initial guess
provided by the standard forward Euler step.
Optional keyword parameters are:

  - ν: relative weight of start / end-point combination (default ν=.5)
  - κ: damping factor for pseudo-Newton iteration (default κ=.8)
  - ϵ_rel: relative error tol. in pseudo-Newton iteration (default ϵ_rel=1e-3)
  - max_iter: maximum number of Newton iterations (default max_iter=5)
  - verbose: print messages (default = true)
"""
function ieuler{T}(ode!, x0::Vector{T}, ts, hmax;
    κ=.8, ν=.5, ϵ_rel=1e-3, max_iter=5, verbose=true)
    0 <= ν <= 1 || error("Please specify ν ∈ [0,1].")
    0 < κ <= 1 || error("Please specify κ ∈ (0,1].")
    hmax = float(hmax)
    nsteps = length(ts)-1
    nsteps >=1 || error("Please specify at least two different times")
    
    if verbose
      status = msg -> println(msg)
    else
      status = msg -> nothing
    end
    

    ts = collect(Float64, ts)

#     nsteps = round(Int, ceil((t1-t0)/hmax))
#     ts = linspace(t0, t1, nsteps+1)
    neq = size(x0, 1)

    xs = zeros(T, neq, nsteps + 1)
    xs[:,1] = x0

    x_aux = zeros(T, neq)
    xdot0 = zeros(T, neq)
    xdot_ν = zeros(T, neq)
    xdot_aux_ν = zeros(T, neq)
    const feps = eps(Float64)
    kkincr = round(Int, nsteps/10)

    status("[ ")
    for kk=1:nsteps
        x::SubArray{Float64,1,Array{Float64,2},Tuple{UnitRange{Int64},Int64},2} = sub(xs,1:neq,kk+1)
        x[:] = sub(xs,1:neq,kk)
        t = ts[kk]
        while t < ts[kk+1]

            h = min(ts[kk+1]-t, hmax)
            if h < feps
                break
            end
            ode!(t, x, xdot0)
            t+=h

            nrm_rel = Inf64
            x_aux[:] = x + h*xdot0
            fill!(xdot_ν, 0)
            ii = 0
            if ν > 0
                while nrm_rel > ϵ_rel && ii < max_iter
                    x[:] = x_aux + (h * ν) * xdot_ν
                    ode!(t, x, xdot_aux_ν)
                    xdot_aux_ν -= xdot0 + xdot_ν
                    xdot_ν[:] +=  κ*xdot_aux_ν
                    nrm_rel = norm(xdot_aux_ν)/(feps + norm(xdot_ν))
                    ii += 1
                end
            else
                x[:] = x_aux
            end


        end
        if kk % kkincr == 0
            status(". ")
        end
    end
    status("]\n")
    ts, xs
end


import Base: srand, randn!, size

"""
## CumulativeNormal

Helper type for sampling coarse grained realizations of a d-dimensional Brownian motion.
It takes as an existing `rng` whose samples corresponds to the minimum grid size
and yields sums over `N` variables sampled from `rng` spaced at a dimension `D`
and renormalizes them.

This type is not optimized for efficiency and should only be used to verify
stochastic convergence properties.

For convenience, complex random normals are supported as well.
A `CumulativeNormal` rng should only be used to generate fixed length vectors of
real or complex normal variables `[x1,...,xD]` at a time.

When the type parameter is `N`, it sums `N` random variables for every output
variable such that if the underlying rng samples

    [ x11 x12 ... x1N  
      x21 x22 ... x2N  
      ...  
      xD1 xD2 ... XDN ]  

then the `CumulativeNormal` rng sums over each row and renormalizes by `1/sqrt(N)`
to produce a `D` dimensional vector of real/complex normal variables.

"""
type CumulativeNormal{N} <: AbstractRNG
    rng::AbstractRNG
    _tmp::Array{Float64,1}
end

"Return (N,D), the coarse-graining number N and the random vector dimension D."
size{N}(cn::CumulativeNormal{N}) = (N, length(cn._tmp/2))

CumulativeNormal(rng::AbstractRNG, N, D) = CumulativeNormal{N}(rng, zeros(2D))
CumulativeNormal(N, D) = CumulativeNormal(MersenneTwister(), N, D)

srand(rng::CumulativeNormal, seed) = srand(rng.rng, seed)

function randn!{N}(rng::CumulativeNormal{N}, A::Vector{Float64})
    size(A,1) == size(rng._tmp,1)/2 || error("Wrong dimension")
    fill!(A, 0)
    sub_tmp = sub(rng._tmp, 1:size(A,1))
    for kk=1:N
        A[:] += randn!(rng.rng, sub_tmp)
    end
    scale!(A, 1/sqrt(N))
    A
end


function randn!{N}(rng::CumulativeNormal{N}, A::Vector{Complex128})
    size(A,1) == size(rng._tmp,1)/2 || error("Wrong dimension")
    fill!(A, 0)
    Ar = reinterpret(Float64, A)
    for kk=1:N
        Ar[:] += randn!(rng.rng, rng._tmp)
    end
    scale!(A, 1/sqrt(N))
    A
end

"Generate complex random normals (with total variance = 2.)"
function randn!(rng::AbstractRNG, A::Array{Complex128})
    n = length(A)
    Ar = reinterpret(Float64, A, (2n,))
    randn!(Ar)
    A
end



abstract SDEType
type Ito <: SDEType
end
type Stratonovich <: SDEType
end

# sde!(t, x, f, gdW, dW, compute_f, compute_gdW) in-place sde function
"""
## ieuler_sde()

Solve an SDE using the implicit stochastic euler method and return the states `xs`
at each point in the time grid `ts`.
The method also returns the noise increments `dWs` for each integration interval.
The SDE is assumed to be given by

\\begin{align}
  dx(t) = f(t,x(t)) dt + g(t,x(t)) dW
\\end{align}

where the type of the SDE (Ito/Stratonovich) must be specified as the the first
function argument.

The sde is passed as a function
`sde!(t, x, f, gdW, dW, compute_f, compute_gdW)` that writes the
drift term of `x` at time `t` into a vector `f` and the diffusion term into
a vector `gdW`. The noise increments at time `t` are passed
as a vector `dW`. Two additional boolean flags `compute_f` and `compute_gdW`
specify whether to compute `f` or `gdW` in a given `sde!()` call. This
allows to save work. The solver also calls `sde!` after each success integration step
with both `compute_f=compute_gdW=false`.
The `sde!` function can use this as a callback mechanism
to carry out useful cleanup work on `x` itself before `x` is appended to the result.

The initial state is `x0`. The time grid at which the states are returned is `ts`,
the maximum stepsize to be used is `hmax`.

At each time step the method solves the following non-linear system of equations
for x(t+h):

\\begin{align}
  x(t+h) == x(t) + [(1-ν) f(t, x(t)) + ν f(t+h, x(t+h))] h + ΔB(x(t),ΔW(t))
\\end{align}

where the diffusion term is given by:

    ΔB(x(t),ΔW) := g(t, x(t)) ΔW                         (for Ito SDEs)
    ΔB(x(t),ΔW) := (1/2)[g(t, x(t)) + g(t, y(t))] ΔW     (for Stratonovich SDEs)
      where y(t) := x(t) + g(t,x(t)) ΔW

This is achieved by a damped pseudo-Newton iteration with the initial guess
provided by the standard forward Euler step.
Optional keyword parameters are:

  - ν: relative weight of start / end-point combination (default ν=.5)
  - κ: damping factor for pseudo-Newton iteration (default κ=.8)
  - ϵ_rel: relative error tol. in pseudo-Newton iteration (default ϵ_rel=1e-3)
  - max_iter: maximum number of Newton iterations (default max_iter=5)
  - verbose: print messages (default = true)
  - rng: An AbstractRNG instance to sample the noise increments from
  - seed: A seed for the RNG.
  - verbose: Print status messages (default = false)
"""
function ieuler_sde{S<:SDEType,T}(::Type{S}, sde!, x0::Vector{T}, ts, hmax, ndW;
    κ=.8, ν=.5, ϵ_rel=1e-3, max_iter=5, rng=nothing, seed=0, verbose=false)
    
    if rng === nothing
        rng = MersenneTwister()
    end
    if verbose
      status = msg -> println(msg)
    else
      status = msg -> nothing
    end
    
    srand(rng, seed)

    hmax = float(hmax)
    ts = collect(Float64, ts)
    0 <= ν <= 1 || error("Please specify ν ∈ [0,1].")
    0 < κ <= 1 || error("Please specify κ ∈ (0,1].")


    nsteps = length(ts)-1
    nsteps >=1 || error("Please specify at least two different times")


    neq = size(x0, 1)

    xs = zeros(T, neq, nsteps + 1)
    xs[:,1] = x0

    x_aux = zeros(T, neq)
    xdot0 = zeros(T, neq)
    xdot_ν = zeros(T, neq)
    xdot_aux_ν = zeros(T, neq)
    gdW0 = zeros(T, neq)
    gdW1 = zeros(T, neq)

    dWs = zeros(T, ndW, nsteps)
    dW = zeros(T, ndW)

    const feps = eps(Float64)
    kkincr = round(Int, nsteps/10)

    if S <: Stratonovich
        status("Stratonovich mode")
    elseif S <: Ito
        status("Ito mode")
    else
      error("Unsupported SDE type: $S")
    end

    status("[ ")
    for kk=1:nsteps

        x::SubArray{T,1,Array{T,2},Tuple{UnitRange{Int64},Int64},2} = sub(xs,1:neq,kk+1)
        x[:] = sub(xs,1:neq,kk)
        t = ts[kk]
        while t < ts[kk+1]

            h = min(ts[kk+1]-t, hmax)
            if h < feps
                break
            end
            randn!(rng, dW)

            if T <: Complex
                scale!(dW, sqrt(h/2))
            else
                scale!(dW, sqrt(h))
            end

            sde!(t, x, xdot0, gdW0, dW, true, true)
            x_aux[:] = x + gdW0


            if S <: Stratonovich
                sde!(ts[kk], x_aux, nothing, gdW1, dW, false, true)
                x_aux[:] = x + (gdW1+gdW0)/2
            end

            x_aux += xdot0*h
            t += h

            x[:] = x_aux


            nrm_rel = Inf64
            fill!(xdot_ν, 0)
            ii = 0
            if ν > 0
                while nrm_rel > ϵ_rel && ii < max_iter

                    sde!(t, x, xdot_aux_ν, nothing, nothing, true, false)
                    xdot_aux_ν -= xdot0 + xdot_ν
                    xdot_ν[:] +=  κ*xdot_aux_ν
                    nrm_rel = norm(xdot_aux_ν)/(feps + norm(xdot_ν))
                    x[:] = x_aux +  (h * ν) * xdot_ν
                    ii += 1
                end
            end

            # allow sde! function to act on x to normalize, etc.
            sde!(t, x, nothing, nothing, nothing, false, false)
            dWs[:,kk] += dW
        end

        if kk % kkincr == 0
            status(". ")
        end
    end
    status("]\n")
    ts, xs, dWs
end

"""
Implicit stochastic Euler Mayurama scheme for integrating Ito SDEs.
See `?ieuler_sde` for more information.
"""
ieuler_mayurama(sde!, x0, ts, hmax, ndW;  kwargs...) = ieuler_sde(
  Ito, sde!, x0, ts, hmax, ndW;
  kwargs...)

"""
Implicit stochastic Euler Heun scheme for integrating Stratonovich SDEs.
See `?ieuler_sde` for more information.
"""
ieuler_heun(sde!, x0, ts, hmax, ndW; kwargs...) = ieuler_sde(
  Stratonovich, sde!, x0, ts, hmax, ndW;
  kwargs...)

end
