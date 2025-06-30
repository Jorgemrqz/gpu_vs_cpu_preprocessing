all:

	g++ superres_cpu.cpp --std=c++17 `pkg-config --cflags --libs opencv4` -o bin/cpu_bin
	g++ superres_gpu.cpp --std=c++17 `pkg-config --cflags --libs opencv4` -o bin/gpu_bin

run_cpu:
	./bin/cpu_bin

run_gpu:
	./bin/gpu_bin

run_both:
	./bin/cpu_bin & ./bin/gpu_bin
