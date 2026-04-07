# Sparse notes on Vulkan and Computer Graphics

This repository documents my process of studying the Vulkan API and my efforts in getting to know low level graphical APIs to eventually being able to create cool graphic projects or implement complex graphics workflows in my own projects.
It also documents an effort to better understand low level concepts as a whole and to make my C++ up to speed with modern standards. 
So watch how I fail and this repository gets forgotten in a month ;`)

After reaching the Compute Shaders chapter of the [Khronos Group's Vulkan Tutorial](https://docs.vulkan.org/tutorial/latest/00_Introduction.html), I've decided to diverge from the tutorial itself to implement classical CG effects, like Shadow Mapping, and to implement a GUI which can help me debug and change the state of my renderer, since most of the backbone of how a basic Vulkan application has been dealt with.
I'm integrating this study with well known resources, like learn-opengl.com, for a reference implementation of most of the effect. After I'll have implemeneted a complete renderer with a couple of interesting effects, I'll proceed with the rest of the tutorial.

My objective is to have a commit a day, and see where it goes. So far I've been able to exceed this during weekends, but I failed during intense weekdays at my daily job.

I'll eventually clean up this readme, but for the time being relevant notes on the API itself will be in the [notes](./notes) folder.

## Building

### Prerequisites

- CMake 3.29+
- Vulkan SDK 1.4.335+
- GLFW3, GLM, STB
- `slangc` (Slang shader compiler)

The Slang shared libraries (`libslang-compiler.so` etc.) must be on the system library path. If you installed `slangc` manually and only copied the binary, copy the accompanying `.so` files to `/usr/local/lib/` and run `sudo ldconfig`.

### Clone

This repository uses git submodules (Dear ImGui). Clone with:

```bash
git clone --recurse-submodules <repo-url>
```

If you already cloned without submodules:

```bash
git submodule update --init
```

### Compile and run

```bash
./run.sh              # debug build (default, validation layers active)
./run.sh release      # release build (optimized, no validation)
./run.sh relwithdebinfo  # optimized + debug symbols + validation
```

### What the run script does (step by step)

```bash
# 1. Configure: reads CMakeLists.txt, generates build files in _build/,
#    copies textures/ and models/ into _build/
cmake -S . -B _build -DCMAKE_BUILD_TYPE=<build-type>

# 2. Build: compiles shaders (shader.slang -> _build/shaders/slang.spv),
#    compiles ImGui sources and the C++ executable to _build/main
cmake --build _build

# 3. Run from _build/ so that relative paths in the code
#    ("shaders/slang.spv", "textures/texture.jpg") resolve correctly
cd _build && ./main
```


## Current state

<div align="center">
  <img src="./media/shadow_maps.gif" alt="Shadow Maps" width="80%" />
</div>

- JSON scene descriptor (color, size per object)
- Multisampling, mipmapping
- Shadow mapping (depth pass with a dedicated pipeline) 
- Diffuse and ambient lighting with normals
- Light orbiting via keyboard commands
- Dear ImGui integration
- Will try to abstract and reorganize a bit more, currently only abstracted Device and Swapchain.

---

## On the graphics pipeline

<div align="center">
  <img src="./media/screenshots//vulkan_simplified_pipeline.svg" alt="Graphics Pipeline" width="60%" />
</div>

Messy notes about general concepts:

- The pipeline is a **sequence of operations** that transform vertex/texture data into rendered pixels
- There are two kind of stages: **fixed-function** and **programmable**
    - Fixed function are **Input Assembler, Rasterization and Color Blending**
- Some programmable stages are optional, based on your intent.
    - Geometry or tessellation can be disabled for simple geometry, or the fragment shader can be disabled for shadow map generation

- Compared to other APIs, the graphics pipeline in Vulkan is **immutable**, and must be **recreated from scratch** if a change in shaders, blending function or a different framebuffer bind is needed.
    - Less ergonomics, more performance

- Detail on what every stage of the pipeline does are absent for now to avoid clutter, here are some highlights:
    - Vertex shader is used to apply transformation to every vertex
        - Vertices are simply points in a 3D space, bundled with certain additional attributes (like normals, colors etc)
    - Rasterization stage breaks primitives (triangles, lines, point) into **fragments** and here you can discard fragments based on their position relative to other fragments or the camera
    - Fragment shader is invoked for every surviving fragment and determines in which framebuffers the fragments are written to and with which color and depth
    - Color blending stage mixes different fragments that map to the same pixel.
        - For example if a transparent red glass has a yellow wall behind it, you'll mix the colors based on this information
    
## On the Vertex Shader
- Written in Slang, compiled to SPIR-V.
    - SPIR-V is a bytecode format and Vulkan has released a platform independent compiler to avoid GPU vendor specific oddities
    - Slang is a shading language with C-style syntax, similar with HLSL, with built-in vector and matrix primitives
- Input: World position, color, normal and texture coordinates of the incoming vertex
- Output: Final position in clip coordinates and the attributes that need to get passed on to the fragment shader
- **Clip coordinates** are four-dimensional vectors (x, y, z, w)
    - The 4th coordinate is what makes perspective work
    - Objects further away get a larger w
- Clip coordinates turn into **normalized device coordinates**, which are 3-dimensional vectors where every coordinate is divided by w.
    - (x/w, y/w, z/w)
    - They map the framebuffer into a [-1, 1] [-1, 1] coordinate system.
    - Vulkan flips the sign of the Y coordinate compared to OpenGL
        - It better reflects how image/texture memory layout actually works
        - Be mindful of backface culling and windling order when porting from OpenGL
