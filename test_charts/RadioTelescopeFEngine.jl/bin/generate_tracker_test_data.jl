using RadioTelescopeFEngine

# Simulation parameter configuration
T = Float64
adc_frequency = 4.9152e+9       # 4.9152 GHz ADC
pfb_nsamples = 16384
Δf = adc_frequency / pfb_nsamples # 300 kHz channel width

# Parse antenna mode (64 or 256)
num_antennas = 64
if length(ARGS) >= 1 && ARGS[1] == "256"
    num_antennas = 256
end

filename = length(ARGS) >= 2 ? ARGS[2] : "fengine_sim_$(num_antennas)ant.h5"

println("Generating F-Engine simulation for $(num_antennas) antennas -> $(filename)")

# Noise
noise = Noise{T}(sqrt(1.0 * pfb_nsamples))

# Sources: Monochromatic point sources on sky
sources = [
    MonochromaticSource{T}(1000 * Δf, (1.0, 0.0), 0.0, 0.0), # Zenith on-axis
    MonochromaticSource{T}(1168 * Δf, (1.0, 0.0), 0.08, -0.04), # Off-zenith steered
]

frb_sources = FRBSource{T}[]

dishgrid = DishGrid{T}(0.6, 0.6)
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
pfb = PFB(4, pfb_nsamples, collect(1000:1335)) # 336 channels (300 MHz ... 400.8 MHz)

buffersize = 15360
ntimes = buffersize

fengine(filename, noise, sources, frb_sources, dishgrid, dishes, adc, pfb, ntimes, buffersize)
println("Done generating $(filename)")
