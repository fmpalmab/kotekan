#!/usr/bin/env julia
# CHARTS Radio Telescope F-Engine Simulation Runner
# =================================================

using Pkg
Pkg.activate(normpath(joinpath(@__DIR__, "..")))

using RadioTelescopeFEngine

println("======================================================================")
println(" CHARTS F-Engine Simulation (RadioTelescopeFEngine.jl)")
println("======================================================================")

# Default configuration: 64 antennas (8x8)
num_antennas = 64
if length(ARGS) >= 1
    num_antennas = parse(Int, ARGS[1])
end

filename = length(ARGS) >= 2 ? ARGS[2] : "voltage_charts_$(num_antennas)ant.h5"

println(" Configuration:")
println("   Antennas    : $(num_antennas)")
println("   Output File : $(filename)")

T = Float64
adc_frequency = 4.9152e+9       # 4.9152 GHz ADC sampling rate
pfb_nsamples = 16384
Δf = adc_frequency / pfb_nsamples # 300 kHz channel spacing

# Noise level
noise = Noise{T}(sqrt(1.0 * pfb_nsamples))

# Injected monochromatic point sources
sources = [
    MonochromaticSource{T}(1000 * Δf, (1.0, 0.0), 0.0, 0.0),        # 300.0 MHz on-axis (zenith)
    MonochromaticSource{T}(1168 * Δf, (1.0, 0.0), 0.08, -0.04),     # 350.4 MHz off-zenith steered
]

frb_sources = FRBSource{T}[]

dishgrid = DishGrid{T}(0.6, 0.6) # 0.6m physical spacing
dishes = Dish[]

if num_antennas <= 64
    for y in 0:7, x in 0:7
        push!(dishes, Dish(x, y))
    end
else
    for y in 0:15, x in 0:15
        push!(dishes, Dish(x, y))
    end
end

adc = ADC{T}(0, inv(adc_frequency))
# 336 frequency channels (from 300.0 MHz to 400.5 MHz)
pfb = PFB(4, pfb_nsamples, collect(1000:1335))

# Time samples (1 chunk of 15360 samples = ~50 ms)
buffersize = 15360
ntimes = buffersize

fengine(filename, noise, sources, frb_sources, dishgrid, dishes, adc, pfb, ntimes, buffersize)

println("======================================================================")
println(" F-Engine Simulation Completed: $(filename)")
println("======================================================================")
