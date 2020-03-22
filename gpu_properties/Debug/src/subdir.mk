################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../src/deviceQuery.cpp 

OBJS += \
./src/deviceQuery.o 

CPP_DEPS += \
./src/deviceQuery.d 


# Each subdirectory must supply rules for building sources it contributes
src/%.o: ../src/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-10.1/bin/nvcc -I"/usr/local/cuda-10.1/samples/1_Utilities" -I"/usr/local/cuda-10.1/samples/common/inc" -I"/home/cuda-lab07/cuda-workspace/gpu_properties" -G -g -O0 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_75,code=sm_75  -odir "src" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-10.1/bin/nvcc -I"/usr/local/cuda-10.1/samples/1_Utilities" -I"/usr/local/cuda-10.1/samples/common/inc" -I"/home/cuda-lab07/cuda-workspace/gpu_properties" -G -g -O0 --compile  -x c++ -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


