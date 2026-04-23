# Sparse notes on Vulkan and Computer Graphics

This repository documents my learning path in the field of Computer Graphics and a personal effort in improving my knowledge of C++, CMake, the Vulkan API and CG Rasterization techniques. The scope of this project is ever-evolving, based on the features I stumble upon my study.

It is in no way a complete product nor it is well-desgined but more like a personal laboratory, a tour if you will, to explore features and design principles.

My goal is to keep this repository well-maintained: trying to push for a commit a day keeps the doctor away and helps with remaining focused. So watch how I fail and this repository gets forgotten in a month ;`) (So far I've been able to keep it up, but I'm writing this to make myself accountable)

Currently, after reaching the Compute Shaders chapter of the [Khronos Group's Vulkan Tutorial](https://docs.vulkan.org/tutorial/latest/00_Introduction.html), I've decided to diverge from the tutorial. Since the bulk of what makes a functioning Vulkan program was dealt with, my next goals were to create simple scenes with geometrical primitives, implementing effects like Shadow Mapping, Bump Mapping, Blinn-Phong lighting and a simple debug GUI to move objects, lights, change properties and enable/disable/tweak various scene properties like skybox color, fog, shadow map bias and light positioning and color.

I'm integrating this study with well known resources, like learn-opengl.com, for a reference implementation of most of the effects. I'm also reading books like C++ Primer and Real Time Rendering 4th edition as I go, trying to make sense of some of the topics.

The next steps, as of mid April 2026, are to write a comprehensive set of notes on the API itself in the notes folder to document the whole process. [notes](./notes) folder.

## Building

> This project is being developed inside a distrobox Fedora container and is not being currently tested elsewhere. If you want to test it on other platforms and add some issues, you're welcome!

### Prerequisites

- CMake 3.29+
- [LunarG Vulkan SDK](https://vulkan.lunarg.com/sdk/home) 1.4.335+ — must be sourced before building (`source ~/VulkanSDK/<version>/setup-env.sh`)
- GLFW3, GLM, STB
- `slangc` (Slang shader compiler)

The Slang shared libraries (`libslang-compiler.so` etc.) must be on the system library path. If you installed `slangc` manually and only copied the binary, copy the accompanying `.so` files to `/usr/local/lib/` and run `sudo ldconfig`.

### Clone

This repository uses git submodules (Dear ImGui, fastgltf). Clone with:

```bash
git clone --recurse-submodules <repo-url>
```

If you already cloned without submodules:

```bash
git submodule update --init
```

### Compile and run

The repo ships a `CMakePresets.json` with three named configurations. The `run.sh` script is a thin wrapper that invokes the preset and then runs the binary:

```bash
./run.sh              # debug build (default, validation layers active)
./run.sh release      # release build (optimized, no validation)
./run.sh relwithdebinfo  # optimized + debug symbols + validation
```

You can also drive the build directly without the script:

```bash
cmake --workflow --preset debug   # configure + build
./_build/main                     # run
```

Or pick a preset from the CMake Tools status bar if you're using VSCode.

### What the build does

```
cmake --workflow --preset <name>
  │
  ├── Configure  (cmake -S . -B _build -DCMAKE_BUILD_TYPE=...)
  │     Reads CMakeLists.txt, generates Makefiles in _build/,
  │     copies assets/ (textures, models, scenes) into _build/,
  │     bakes ASSET_DIR=<absolute path to _build/> into the binary
  │     so the executable can locate assets regardless of working directory.
  │
  └── Build  (cmake --build _build)
        Compiles scene.slang, shadow.slang, sky.slang → _build/shaders/*.spv,
        compiles ImGui sources and the C++ renderer → _build/main.
```


## Current state

<div align="center">
  <img src="./media/screenshots/sponza_loaded.png" alt="Sponza loaded via GLTF" width="80%" />
</div>

<div align="center">
  <img src="./media/screenshots/normal+specular+pom+tonemapping+skybox.png" alt="Normal mapping, POM, tonemapping and skybox" width="80%" />
</div>

<div align="center">
  <img src="./media/shadow_maps.gif" alt="Shadow Maps" width="80%" />
</div>


### Rendering

- **Cook-Torrance BRDF** — physically-based direct lighting with GGX normal distribution, Smith geometry term, and Fresnel-Schlick; metallic-roughness workflow using maps extracted by the GLTF loader; replaces Blinn-Phong
- **Shadow mapping** — dedicated depth pre-pass, 9-tap PCF filtering, slope-scale bias (tweakable min/max via ImGui to balance acne vs. peter panning)
- **Specular maps** — per-texel tinted specular intensity (RGB), used as a roughness override fallback
- **Normal mapping** — tangent-space normal maps; tangent vectors computed analytically for procedural meshes and via Gram-Schmidt for OBJ models, stored as `float4` with bitangent sign in `w`
- **Parallax Occlusion Mapping (POM)** — adaptive linear raymarch (8–32 steps) + 5-step binary refinement in tangent space; depth scale and step counts tunable in ImGui; activated per-object via `"heightMap"` in scene JSON
- **Point lights** — up to 4, polynomial `(1-(d/r)²)²` falloff, no shadow casting (by design); add/remove at runtime via ImGui
- **Procedural skybox** — fullscreen-triangle pass reconstructing view rays from `invProj`/`invViewRot`; three-zone gradient (ground/horizon/zenith) + sun disk; colors configurable in ImGui and scene JSON
- **ACES filmic tonemapping** — exposure control; toggle on/off at runtime
- **MSAA** — configurable sample count up to hardware max, live-switchable in ImGui
- **Exponential height fog** — density, height falloff, and max-opacity controls in ImGui; color optionally locked to the sky horizon color for seamless blending

### Scene

- **JSON scene descriptor** — objects with mesh (`cube`, `plane`, `sphere`, or file path), position, rotation, scale, texture, specularMap, normalMap, heightMap, vertex color; skybox colors and point lights also declared in JSON
- **Bindless texture array** — all textures in a single descriptor binding (`PARTIALLY_BOUND`), indexed via push constants; `0xFFFF` sentinel for "no texture"; up to 2048 slots
- **Per-object transform editing** — position, rotation (XYZ Euler), uniform scale via ImGui drag sliders, model matrix rebuilt on change
- **OBJ loading** — via tinyobjloader; tangent vectors computed per-triangle from UV deltas and accumulated + orthogonalised per vertex
- **GLTF/GLB loading** — via fastgltf; reads all primitives with their per-mesh transforms; extracts baseColor, normalMap, and metallicRoughness textures; handles both external image files and embedded GLB buffers; optional Y-up to Z-up axis remap

### Camera & controls

- **Free-fly camera** — hold RMB to enter fly mode (cursor captured), WASD + QE for movement, mouse for look; Z-up spherical coordinates
- **Camera parameters** — position, yaw, pitch, speed, sensitivity editable in ImGui

### Infrastructure

- **Vulkan Profiles** (`VP_KHR_ROADMAP_2022`) — device selection and logical device creation use the profiles API; `vpGetPhysicalDeviceProfileSupport` replaces manual feature checks, `vpCreateDevice` ensures the full profile feature set is enabled; `vulkan_profiles.hpp` comes from the LunarG Vulkan SDK
- **Dynamic rendering** (`VK_KHR_dynamic_rendering`) — no render pass objects; shadow, scene, sky, and ImGui each in their own `beginRendering`/`endRendering` block
- **Vulkan 1.3 features** — `dynamicRendering`, `synchronization2`, `runtimeDescriptorArray`, `descriptorBindingPartiallyBound`, `shaderSampledImageArrayNonUniformIndexing` all guaranteed via `VP_KHR_ROADMAP_2022`
- **Swapchain** and **Device** abstracted into their own classes
- **Dear ImGui** integration with dynamic rendering backend
- **CMakePresets** — `CMakePresets.json` encodes debug/release/relwithdebinfo configurations; `run.sh` is a thin wrapper around `cmake --workflow`; IDE preset selection works out of the box with VSCode CMake Tools

### Future developments

- **SSAO** — normal buffer MRT output is already in place (WIP); next step is the screen-space occlusion pass and blur
- **Mesh instancing / shared buffers** — currently each renderable owns its own vertex and index buffers; duplicate meshes should share GPU buffers and use instanced draw calls
- Better code organisation — the renderer class has grown large; splitting it into more focused subsystems or exploring render graph concepts
- Revisit implemented techniques to fix edge cases (shadow map coverage for large scenes, POM artifacts at silhouettes)

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
