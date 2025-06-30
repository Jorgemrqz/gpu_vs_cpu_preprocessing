all:
	g++ superres_cpu.cpp --std=c++17 `pkg-config --cflags --libs opencv4` -o cpu_app
	g++ superres_gpu.cpp --std=c++17 `pkg-config --cflags --libs opencv4` -o gpu_app

run_cpu:
	./cpu_app

run_gpu:
	./gpu_app

run_both:
	./cpu_app & ./gpu_app
