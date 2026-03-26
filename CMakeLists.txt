cmake_minimum_required(VERSION 3.29)
project(HelloTriangle)

option(ENABLE_CPP20_MODULE "Use Vulkan C++20 module (import vulkan_hpp) instead of headers" OFF)

if(ENABLE_CPP20_MODULE)
    set(CMAKE_CXX_SCAN_FOR_MODULES ON)
endif()

find_package(glfw3 REQUIRED)
find_package(glm REQUIRED)
find_package(Vulkan 1.4.335 REQUIRED)
find_path(STB_INCLUDEDIR stb_image.h PATH_SUFFIXES stb REQUIRED)
find_path(TINYOBJ_INCLUDEDIR tiny_obj_loader.h REQUIRED)
find_program(SLANGC_EXECUTABLE slangc HINTS $ENV{VULKAN_SDK}/bin REQUIRED)

# Vulkan C++20 module wrapper (or a plain interface when the option is off)
if(ENABLE_CPP20_MODULE)
    add_library(VulkanCppModule)
    set_target_properties(VulkanCppModule PROPERTIES CXX_STANDARD 20)
    target_compile_definitions(VulkanCppModule PUBLIC
        VULKAN_HPP_DISPATCH_LOADER_DYNAMIC=1
        VULKAN_HPP_NO_STRUCT_CONSTRUCTORS=1
    )
    target_include_directories(VulkanCppModule PUBLIC ${Vulkan_INCLUDE_DIR})
    target_link_libraries(VulkanCppModule PUBLIC Vulkan::Vulkan)
    target_sources(VulkanCppModule PUBLIC
        FILE_SET cxx_modules TYPE CXX_MODULES
        BASE_DIRS ${Vulkan_INCLUDE_DIR}
        FILES ${Vulkan_INCLUDE_DIR}/vulkan/vulkan.cppm
    )
else()
    add_library(VulkanCppModule INTERFACE)
    target_compile_definitions(VulkanCppModule INTERFACE
        VULKAN_HPP_DISPATCH_LOADER_DYNAMIC=1
        VULKAN_HPP_NO_STRUCT_CONSTRUCTORS=1
    )
    target_link_libraries(VulkanCppModule INTERFACE Vulkan::Vulkan)
endif()
add_library(Vulkan::cppm ALIAS VulkanCppModule)

# Compile Slang shaders -> SPIR-V at build time
add_custom_command(
    OUTPUT ${CMAKE_BINARY_DIR}/shaders/slang.spv
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/shaders
    COMMAND ${SLANGC_EXECUTABLE}
        ${CMAKE_SOURCE_DIR}/shaders/shader.slang
        -target spirv -profile spirv_1_4 -emit-spirv-directly -fvk-use-entrypoint-name
        -entry vertMain -entry fragMain
        -o ${CMAKE_BINARY_DIR}/shaders/slang.spv
    DEPENDS ${CMAKE_SOURCE_DIR}/shaders/shader.slang
    COMMENT "Compiling Slang shaders"
    VERBATIM
)
add_custom_target(shaders DEPENDS ${CMAKE_BINARY_DIR}/shaders/slang.spv)

# Copy textures and models to build directory at configure time
file(COPY ${CMAKE_SOURCE_DIR}/textures DESTINATION ${CMAKE_BINARY_DIR})
file(COPY ${CMAKE_SOURCE_DIR}/models DESTINATION ${CMAKE_BINARY_DIR})

add_executable(main main.cpp)
set_target_properties(main PROPERTIES
    CXX_STANDARD 20
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)
target_link_libraries(main PRIVATE Vulkan::cppm glfw)
target_include_directories(main PRIVATE ${STB_INCLUDEDIR} ${TINYOBJ_INCLUDEDIR})
target_compile_definitions(main PRIVATE VULKAN_HPP_HANDLE_ERROR_OUT_OF_DATE_AS_SUCCESS)
add_dependencies(main shaders)

if(ENABLE_CPP20_MODULE)
    target_compile_definitions(main PRIVATE USE_CPP20_MODULES=1)
endif()
