# ${KERNEL_FILES} are used to compile library, push files written by ascendc in ${KERNEL_FILES}.
# ref to cmake/npu.cmake ascendc_library, cmake/cpu.cmake add_library
file(GLOB KERNEL_FILES kernels/*.cpp)

message(STATUS "kernel files: ${KERNEL_FILES}")

include(npu_lib.cmake)

target_include_directories(cache_kernels
  PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
)