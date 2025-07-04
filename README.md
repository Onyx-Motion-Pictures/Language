// MIT License

// Copyright © 2025 Onyx Motion Pictures

// Permission is hereby granted, free of charge, to any person obtaining a copy
// of these MATERIALS and associated documentation files (the "MATERIALS"), to deal
// in the MATERIALS without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the MATERIALS, and to permit persons to whom the MATERIALS is
// furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the MATERIALS.

// THE MATERIALS ARE PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE MATERIALS OR THE USE OR OTHER DEALINGS IN THE
// MATERIALS.

fn show_license() {
    print("This Onyx source file is an example of code licensed under the MIT License.");
}

// SECTION 1: INTRODUCTION TO ONYX

// Onyx is a modern, statically-typed, and compiled language conceived for high-performance computing, data analysis, and, most notably, 3D graphics and video processing.
// Its design prioritizes:
// Performance by Default: Ahead-of-time (AOT) compiled via an LLVM backend with first-class parallelism.
// Expressive & Concise Syntax: A clean, readable syntax inspired by Rust, Swift, and F# to maximize developer productivity.
// Built for Modern Hardware: Native support for `async/await` and `parallel map` to easily leverage multi-core CPUs and asynchronous I/O.
// Domain-Specific Tooling: Built-in types like vectors, matrices, and video streams with hardware-accelerated operations.

// | Feature              | Description                                                                             | Example                                           |
// |  |  | - |
// | Variables            | `let` for immutable and `var` for mutable bindings. Type inference is used extensively. | `let gravity = 9.8;`<br>`var velocity = 0.0;`     |
// | Functions            | Declared with `fn`. Clear type annotations for parameters and return values.            | `fn add(a: i32, b: i32) -> i32 { return a + b; }` |
// | Structs & Impl       | `struct` defines custom data types. `impl` blocks add methods and operator overloads.   | `struct Vec3 { ... }`<br>`impl Vec3 { ... }`      |
// | Operator Overloading | The `operator` keyword provides a clean syntax for defining custom operator behavior.   | `operator + (other: Vec3) -> Vec3 { ... }`        |
// | Concurrency          | `async/await` for I/O-bound tasks and `parallel map` for CPU-bound data parallelism.    | `await File::read("data.bin");`                   |
// | Pipeline Operator    | The `|>` operator allows for chaining function calls in a readable, linear sequence.    | `data \|> filter(...) \|> map(...) \|> collect()` |
// | GPU Kernels          | A hypothetical `@gpu` attribute and `kernel` function type for GPGPU programming.       | `@gpu kernel update(...) { ... }`                 |

// SECTION 2: CORE LANGUAGE CONCEPTS - A VEC3 EXAMPLE

// This demonstrates a complete, ergonomic `Vec3` type, crucial for any 3D application.
// Import standard formatting traits.
use onyx::core::fmt::{Display, Formatter, Result};

// Define a 3D vector struct.
struct Vec3 {
    x: f64,
    y: f64,
    z: f64,
}

// Implement methods, traits, and operator overloads for Vec3.
impl Vec3 {
    // Constructor-like function to create a new Vec3.
    fn new(x: f64, y: f64, z: f64) -> Vec3 {
        Vec3 { x, y, z }
    }

    // Calculate the magnitude (length) of the vector.
    fn magnitude(&self) -> f64 {
        self.dot(self).sqrt()
    }

    // Returns a new vector with the same direction but a magnitude of 1.
    fn normalized(&self) -> Vec3 {
        let mag = self.magnitude();
        if mag > 0.0 {
            return self / mag;
        }
        Vec3::new(0.0, 0.0, 0.0)
    }

    // Calculate the dot product with another Vec3.
    fn dot(&self, other: &Vec3) -> f64 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    // Calculate the cross product with another Vec3.
    fn cross(&self, other: &Vec3) -> Vec3 {
        Vec3::new(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )
    }

    // Calculate the distance to another vector.
    fn distance_to(&self, other: &Vec3) -> f64 {
        Vec3::new(self.x - other.x, self.y - other.y, self.z - other.z).magnitude()
    }

    // Operator Overloads for expressive math.
    operator + (other: Vec3) -> Vec3 { Vec3::new(self.x + other.x, self.y + other.y, self.z + other.z) }
    operator - (other: Vec3) -> Vec3 { Vec3::new(self.x - other.x, self.y - other.y, self.z - other.z) }
    operator * (scalar: f64) -> Vec3 { Vec3::new(self.x * scalar, self.y * scalar, self.z * scalar) }
    operator / (scalar: f64) -> Vec3 { Vec3::new(self.x / scalar, self.y / scalar, self.z / scalar) }
}

// Implement the Display trait to define how Vec3 should be printed.
impl Display for Vec3 {
    // This function tells the formatter how to represent a Vec3 as a string.
    fn fmt(&self, f: &mut Formatter) -> Result {
        f.write_str("({self.x}, {self.y}, {self.z})")
    }
}

// A function to demonstrate the Vec3 struct and its capabilities.
fn main_vector_demo() {
    let v1 = Vec3::new(1.0, 2.0, 3.0);
    let v2 = Vec3::new(4.0, 5.0, 6.0);

    print(" Vector Math Demo ");
    // With the Display trait implemented, we can print Vec3 directly.
    print("Vector 1: {v1}");
    print("Vector 2: {v2}");

    // Operators
    print("v1 + v2 = {v1 + v2}");
    print("v2 - v1 = {v2 - v1}");
    print("v1 * 2.0 = {v1 * 2.0}");
    print("v2 / 2.0 = {v2 / 2.0}");

    // Methods
    print("v1 dot v2 = {v1.dot(&v2)}");
    print("v1 cross v2 = {v1.cross(&v2)}");
    print("Magnitude of v1: {v1.magnitude()}");
    print("Normalized v1: {v1.normalized()}");
}

// SECTION 3: THE ONYX STANDARD LIBRARY

// The standard library is organized into a hierarchy of modules under the top-level Onyx namespace.

// onyx::core - The Core Module
// Purpose: Contains fundamental types, traits, and functions that are essential for almost all Onyx programs. This module is often implicitly imported (part of the "prelude").
// Key Components:
// Primitives: i32, f64, string, bool, char.
// Collections: [T] (dynamic array), Map<K, V>, Set<T>.
// Control Flow: Option<T>, Result<T, E> for robust error and optional value handling.
// Core Traits: Display (for string formatting), Iterable, Equatable.
// Functions: print(), len(), range().

// onyx::io - Input/Output
// Purpose: Provides tools for interacting with the outside world, from the file system to the network.
// Key Components:
// File: Asynchronous and synchronous file reading/writing.
// Path: Platform-agnostic path manipulation.
// process: For creating and managing subprocesses.
// net: Networking primitives like TcpStream, UdpSocket, and a simple HttpClient.

// onyx::math - Mathematics & Linear Algebra
// Purpose: The foundation for scientific computing, data analysis, and graphics.
// Key Components:
// Vectors: Vec2, Vec3, Vec4 with built-in, hardware-accelerated operations.
// Matrices: Matrix3, Matrix4 for transformations.
// Quaternions: Quaternion for efficient 3D rotations.
// stats: Statistical functions (mean, stddev, correlation).
// rand: Advanced random number generation.

// onyx::media - Video, Image, and Audio Processing
// Purpose: A high-level, domain-specific module that makes Onyx a premier language for media manipulation.
// Key Components:
// Image: A rich image type supporting various pixel formats and transformations (resize, crop, apply_filter).
// VideoStream: For decoding, processing, and encoding video files.
// AudioStream: For handling audio data.
// Color: A robust color type with conversions (e.g., RGBA, HSB).
// Filter: A collection of pre-built, high-performance media filters (Grayscale, Blur, Sharpen).
// Codec: Enums representing media formats (PNG, JPEG, H264, MP3).

// onyx::concurrent - Concurrency & Parallelism
// Purpose: Provides the tools to unlock the performance of modern multi-core hardware.
// Key Components:
// task: For spawning and managing asynchronous tasks (async/await is built on this).
// channel: For safe communication between tasks.
// The implementation for parallel map and other data-parallel constructs resides here.

// onyx::data - Data Analysis & Manipulation
// Purpose: Provides powerful, expressive tools for working with structured data, inspired by libraries like Pandas.
// Key Components:
// DataFrame: The core 2D, labeled data structure for analysis.
// Series: A 1D labeled array, the building block of a DataFrame.
// csv: A highly optimized reader and writer for CSV files.
// json: Tools for parsing and serializing JSON data.

// SECTION 4: ADVANCED FEATURES IN ACTION

// 4.1: Asynchronous I/O and Data Pipelines
// This function processes a video file asynchronously.
// It loads a video, applies a grayscale filter to each frame in parallel, and saves the result.
use onyx::media::{VideoStream, Image, Codec, Filter};

async fn process_video(input_path: string, output_path: string) {
    print("Starting video processing for {input_path}...");

    // Load a video stream. This is an async operation.
    let video_stream = await VideoStream::from_file(input_path);

    // Create a processing pipeline using the pipe operator '|>'.
    // 'parallel map' distributes the filter application across CPU cores.
    let processed_frames: [Image] = video_stream.frames()
        |> parallel map |frame| => frame.apply_filter(Filter::Grayscale)
        |> collect();

    print("All frames processed. Encoding and saving to {output_path}...");
    await VideoStream::save(processed_frames, output_path, Codec::H264);
    print("Video processing complete.");
}

// 4.2: Data Analysis with DataFrame
// This example demonstrates how different parts of the standard library (io, data, math) work together for a data analysis task.
use onyx::data::{DataFrame, csv};
use onyx::io::File;

fn main_data_analysis_demo() {
    // 1. Load Data
    // Use the 'csv' submodule from 'onyx::data' to read a file.
    print("Loading point data from points.csv...");
    let df = csv::read("points.csv")?;

    // 2. Process Data
    // Use a data pipeline to transform the DataFrame into Vec3 objects.
    let points: [Vec3] = df.rows()
        |> map |row| => Vec3::new(
            row.get("x")?.to_f64(),
            row.get("y")?.to_f64(),
            row.get("z")?.to_f64()
        )
        |> collect();

    print("Successfully loaded {points.len()} points.");

    // 3. Analyze Data
    // Calculate the center of mass (average position).
    // Assumes Vec3 supports `sum()` on an iterator, which is reasonable given operator overloading.
    let total = points.iter().sum();
    let center_of_mass = total / points.len() as f64;

    print("Center of Mass: {center_of_mass}");

    // Find the point furthest from the center using the method we defined on Vec3.
    let furthest_point = points.iter()
        .max_by_key |p| => p.distance_to(&center_of_mass);

    print("Furthest point from center: {furthest_point}");
}

// 4.3: First-Class Parallelism
// 'parallel map' is a core language construct, not just a library function.
// This makes the intent to parallelize exceptionally clear and allows for deep compiler optimizations.
// The '^' operator provides intuitive syntax for exponentiation.

fn main_parallelism_demo() {
    let positions = [
        Vec3::new(1.0, 2.0, 3.0),
        Vec3::new(4.0, 1.0, -2.0),
        Vec3::new(-3.0, 5.0, 0.0),
        // ... Potentially millions of other vectors
    ];

    print("Calculating squared distances in parallel...");

    // This pipeline showcases Onyx's unique flavor.
    let squared_distances: [f64] = positions
        |> parallel map |p| => p.x^2 + p.y^2 + p.z^2
        |> collect();

    print("Calculation complete. Results: {squared_distances}");
}

// SECTION 5: SYSTEM-LEVEL & HYPOTHETICAL FEATURES

// 5.1: GPU Acceleration
// Onyx introduces a 'kernel' function type and a '@gpu' attribute to compile and execute code on a compatible GPU (e.g., via CUDA or WebGPU).
use onyx::gpu;
use onyx::math::Matrix4; // Assuming Matrix4 is needed elsewhere or for context.
// A kernel to update particle positions on the GPU.
@gpu
kernel update_particles(positions: &mut [Vec3], velocities: &[Vec3], delta_time: f32) {
    // 'gpu::thread_id()' is a special variable available inside kernels.
    let i = gpu::thread_id();
    positions[i] = positions[i] + velocities[i] * delta_time;
}

fn run_gpu_simulation() {
    let mut positions_gpu = gpu::buffer_from(positions); // `positions` from an outer scope.
    let velocities_gpu = gpu::buffer_from(velocities); // `velocities` from an outer scope.
    // 'launch' provides a clean syntax for dispatching the kernel.
    launch update_particles(positions_gpu, velocities_gpu, 0.016) with 1_000_000 threads;
    print("GPU kernel launched.");
}


// 5.2: Foreign Function Interface (FFI)
// Onyx has a clean FFI to call into C libraries, allowing it to leverage vast existing ecosystems like physics engines or codecs.
extern "C" {
    // Declare an external C function.
    fn c_cosine(input: f64) -> f64;
}

fn call_c_library() {
    let angle = 1.57;
    // The 'unsafe' block is required, as Onyx cannot verify the safety of external C code.
    let result = unsafe { c_cosine(angle) };
    print("Result from C library: c_cosine({angle}) = {result}");
}

// 5.3: Command-Line Applications
// As a compiled language, Onyx produces self-contained native executables.
// The standard library provides an `onyx::env` module for CLI development.
use onyx::env;

fn main_cli() {
    let args = env::args();
    print(" Onyx CLI Tool ");
    print("Arguments received: {args.len()}");

    if args.len() > 1 {
        let input_file = args[1];
        print("Processing file: {input_file}");
        // Logic to process the file goes here.
    } else {
        print("Usage: my_tool <input_file>");
    }
}

// SECTION 6: PRACTICAL APPLICATION - BATCH VIDEO GENERATION

// This script automates generating videos from a list of image URLs by calling an external API concurrently.
use onyx::net::{HttpClient, HttpRequest, Json};
use onyx::time::Duration;
use onyx::concurrent::Task;
use onyx::core::{Ok, Err};

// Configuration - We will use Dream Machine as the API for generating videos.
let GENERATE_API_URL = "https://api.lumalabs.ai/dream-machine/v1/generations";
let API_KEY = env::var("LUMA_API_KEY")?;
let POLLING_INTERVAL = Duration::from_secs(15);

// A struct to hold the ID of a job that has been successfully started.
struct GenerationJob {
    id: string,
    source_url: string,
}

// Step 1: Function to START a single generation job.
async fn start_generation(client: &HttpClient, image_url: &string) -> Result<GenerationJob, string> {
    let payload = Json::object([
        ("generation_type", Json::from("video")),
        ("prompt", Json::from("An artistic and cinematic animation.")),
        ("model", Json::from("ray-flash-2")),
        ("aspect_ratio", Json::from("16:9")),
        ("resolution", Json::from("4k")),
        ("duration", Json::from("9s")),
        ("loop", Json::from("False")),
        ("callback_url", Json::from("https://onyxmotionpictures.com/generations")),
        ("keyframes", Json::from("frame0", "type", "image", "url", image_url.clone())),
    ]);

    // Build and send the HTTP POST request asynchronously.
    let request = HttpRequest::post(GENERATE_API_URL)
        .header("Authorization", "Bearer {API_KEY}")
        .body(payload);

    let response = await client.send(request)?;

    if response.is_success() {
        let job_id = response.json()?["id"].as_string()?;
        print("-> Started job ID: {job_id} for {image_url}");
        return Ok(GenerationJob { id: job_id, source_url: image_url.clone() });
    } else {
        return Err(format!("Failed to start job for {image_url}: {response.status_text()}"));
    }
}

// Step 2: Function to POLL a single job until completion.
async fn poll_for_completion(client: &HttpClient, job: &GenerationJob) -> Result<string, string> {
    let status_url = "{GENERATE_API_URL}/{job.id}";
    loop {
        let request = HttpRequest::get(&status_url).header("Authorization", "Bearer {API_KEY}");
        let response = await client.send(request)?;
        let state = response.json()?["state"].as_string()?;

        match state {
            "completed" => {
                let video_url = response.json()?["video"]["url"].as_string()?;
                print("-> Job {job.id} COMPLETED: {video_url}");
                return Ok(video_url);
            },
            "processing" | "pending" => {
                await Task::sleep(POLLING_INTERVAL);
            },
            _ => {
                print("-> Job {job.id} FAILED with state: {state}");
                return Err(format!("Job for {job.source_url} failed."));
            }
        }
    }
}

// Wrapper function to handle the full generation process for one URL.
async fn generate_video_from_url(client: &HttpClient, url: &string, index: i32) -> Result<string, string> {
    print("[{index}] Starting process for URL: {url}");
    let job = await start_generation(client, url)?;
    let video_url = await poll_for_completion(client, &job)?;
    return Ok(video_url);
}

async fn main_batch_video_generation() {
    let image_urls = await File::read_lines("onyx.links-website_logos.txt")?;
    print("Found {image_urls.len()} URLs. Starting batch video generation...");

    let client = HttpClient::new();

    // This pipeline processes all URLs in parallel.
    let successful_urls: [string] = image_urls.iter().enumerate()
        |> parallel map |(i, url)| => generate_video_from_url(&client, url, i + 1)
        |> await all
        |> filter_map |result| => result.ok()
        |> collect();

    print("\n Batch Processing Complete ");
    print("Successfully generated {successful_urls.len()} videos.");
}

// SECTION 7: MAIN ENTRY POINT

// The main function serves as the entry point for the program.
// Here we can call the various demo functions defined above.
fn main() {
    main_vector_demo();
    print("\n-\n");

    // To run the video processing demo, you would call it like this:
    // await process_video("input.mp4", "output_grayscale.mp4");

    // To run the data analysis demo:
    // main_data_analysis_demo();

    // To run the batch generation script:
    // await main_batch_video_generation();

    // To run other demos:
    // main_parallelism_demo();
    // main_cli();
    // call_c_library();
    // run_gpu_simulation();
}

// SECTION 8: DESIGN PHILOSOPHY & CONCLUSION

// 8.1: Design Philosophy
// While the foundation is very Rust-like, Onyx introduces its own flavor by borrowing from other languages to achieve its goal of being expressive and concise:
// - From Swift/JavaScript: The `let`/`var` distinction for immutability/mutability.
// - From F#/Elixir: The pipeline operator `|>` for readable data transformation chains.
// - Unique to Onyx: The dedicated `operator` keyword simplifies operator overloading for the domain-specific math Onyx is designed for.

// 8.2: Conclusion
// Onyx represents a forward-thinking approach to development in the media and entertainment industry.
// By combining performance, safety, and an expressive, domain-specific syntax, it demonstrates a clear path toward more efficient and powerful creative workflows.
