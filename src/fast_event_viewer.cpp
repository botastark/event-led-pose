#include "fast_event_viewer.hpp"

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <thread>

namespace event_led_pose {
namespace {

GLuint compile_shader(GLenum type, const char *src) {
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, nullptr);
    glCompileShader(s);

    GLint ok = GL_FALSE;
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (ok == GL_TRUE)
        return s;

    GLint len = 0;
    glGetShaderiv(s, GL_INFO_LOG_LENGTH, &len);
    std::string log(static_cast<std::size_t>(std::max(len, 1)), '\0');
    glGetShaderInfoLog(s, len, nullptr, log.data());
    glDeleteShader(s);
    throw std::runtime_error("shader compile failed:\n" + log);
}

GLuint make_program(const char *vs_src, const char *fs_src) {
    GLuint vs = compile_shader(GL_VERTEX_SHADER, vs_src);
    GLuint fs = compile_shader(GL_FRAGMENT_SHADER, fs_src);
    GLuint p = glCreateProgram();
    glAttachShader(p, vs);
    glAttachShader(p, fs);
    glLinkProgram(p);
    glDeleteShader(vs);
    glDeleteShader(fs);

    GLint ok = GL_FALSE;
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (ok == GL_TRUE)
        return p;

    GLint len = 0;
    glGetProgramiv(p, GL_INFO_LOG_LENGTH, &len);
    std::string log(static_cast<std::size_t>(std::max(len, 1)), '\0');
    glGetProgramInfoLog(p, len, nullptr, log.data());
    glDeleteProgram(p);
    throw std::runtime_error("program link failed:\n" + log);
}

inline bool newer_ts(std::uint32_t a, std::uint32_t b) noexcept {
    return static_cast<std::int32_t>(a - b) > 0;
}

} // namespace


FastEventViewer::FastEventViewer(const Config &cfg)
    : cfg_(cfg),
      raw_ring_(cfg.raw_ring_capacity),
      freq_ring_(cfg.freq_ring_capacity) {
    raw_points_.reserve(cfg_.max_raw_points_per_render);
    freq_points_.reserve(cfg_.max_freq_points_per_render);
    init_gl();
}

FastEventViewer::~FastEventViewer() {
    destroy_gl();
}


// ---------------- producer APIs ----------------

void FastEventViewer::raw_begin_batch() noexcept {
    raw_batch_submitted_ = 0;
    raw_batch_dropped_ = 0;
    raw_ring_.producer_begin();
}

bool FastEventViewer::raw_push(const PackedEvent &e) noexcept {
    ++raw_batch_submitted_;
    if (!raw_ring_.producer_push(e)) {
        ++raw_batch_dropped_;
        return false;
    }
    return true;
}

void FastEventViewer::raw_end_batch() noexcept {
    raw_ring_.producer_end();
    raw_submitted_.fetch_add(raw_batch_submitted_, std::memory_order_relaxed);
    raw_ring_drops_.fetch_add(raw_batch_dropped_, std::memory_order_relaxed);
}

void FastEventViewer::freq_begin_batch() noexcept {
    freq_batch_submitted_ = 0;
    freq_batch_dropped_ = 0;
    freq_ring_.producer_begin();
}

bool FastEventViewer::freq_push(std::uint16_t x,
                                std::uint16_t y,
                                std::uint32_t t,
                                std::uint8_t id) noexcept {
    ++freq_batch_submitted_;

    Point p{};
    p.x = x;
    p.y = y;
    p.t = t;

    switch (id) {
    case 0: p.r = 0;   p.g = 255; p.b = 0;   break; // 165
    case 1: p.r = 255; p.g = 255; p.b = 0;   break; // 366
    case 2: p.r = 0;   p.g = 128; p.b = 255; break; // 596
    default:p.r = 255; p.g = 255; p.b = 255; break;
    }

    if (!freq_ring_.producer_push(p)) {
        ++freq_batch_dropped_;
        return false;
    }
    return true;
}

void FastEventViewer::freq_end_batch() noexcept {
    freq_ring_.producer_end();
    freq_submitted_.fetch_add(freq_batch_submitted_, std::memory_order_relaxed);
    freq_ring_drops_.fetch_add(freq_batch_dropped_, std::memory_order_relaxed);
}


// ---------------- GL ----------------

void FastEventViewer::init_gl() {
    if (!glfwInit())
        throw std::runtime_error("glfwInit failed");

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    window_ = glfwCreateWindow(
        cfg_.sensor_width, cfg_.sensor_height,
        cfg_.title.c_str(), nullptr, nullptr);

    if (!window_) {
        glfwTerminate();
        throw std::runtime_error("glfwCreateWindow failed");
    }

    glfwMakeContextCurrent(window_);
    glfwSwapInterval(cfg_.vsync ? 1 : 0);

    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK)
        throw std::runtime_error("glewInit failed");

    while (glGetError() != GL_NO_ERROR) {}

    const char *event_vs = R"GLSL(
        #version 330 core
        layout(location=0) in vec2 in_xy;
        layout(location=1) in vec3 in_rgb;
        uniform vec2 sensor_size;
        uniform float point_size;
        out vec3 vcolor;
        void main() {
            vec2 uv = in_xy / sensor_size;
            gl_Position = vec4(uv.x*2.0-1.0, 1.0-uv.y*2.0, 0.0, 1.0);
            gl_PointSize = point_size;
            vcolor = in_rgb;
        }
    )GLSL";

    const char *event_fs = R"GLSL(
        #version 330 core
        in vec3 vcolor;
        out vec4 frag;
        void main() { frag = vec4(vcolor, 1.0); }
    )GLSL";

    const char *screen_vs = R"GLSL(
        #version 330 core
        layout(location=0) in vec2 in_pos;
        out vec2 uv;
        void main() {
            gl_Position = vec4(in_pos, 0.0, 1.0);
            uv = (in_pos + vec2(1.0)) * 0.5;
        }
    )GLSL";

    const char *screen_fs = R"GLSL(
        #version 330 core
        uniform sampler2D canvas;
        in vec2 uv;
        out vec4 frag;
        void main() { frag = texture(canvas, uv); }
    )GLSL";

    event_program_ = make_program(event_vs, event_fs);
    screen_program_ = make_program(screen_vs, screen_fs);

    glGenVertexArrays(1, &event_vao_);
    glBindVertexArray(event_vao_);
    glGenBuffers(1, &event_vbo_);
    glBindBuffer(GL_ARRAY_BUFFER, event_vbo_);

    const std::size_t max_points =
        std::max(cfg_.max_raw_points_per_render,
                 cfg_.max_freq_points_per_render);

    glBufferData(GL_ARRAY_BUFFER,
                 static_cast<GLsizeiptr>(max_points * sizeof(Point)),
                 nullptr, GL_STREAM_DRAW);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(
        0, 2, GL_UNSIGNED_SHORT, GL_FALSE, sizeof(Point),
        reinterpret_cast<void *>(offsetof(Point, x)));

    glEnableVertexAttribArray(1);
    glVertexAttribPointer(
        1, 3, GL_UNSIGNED_BYTE, GL_TRUE, sizeof(Point),
        reinterpret_cast<void *>(offsetof(Point, r)));

    static constexpr float quad[] = {
        -1.f,-1.f,  1.f,-1.f,  1.f, 1.f,
        -1.f,-1.f,  1.f, 1.f, -1.f, 1.f
    };

    glGenVertexArrays(1, &quad_vao_);
    glBindVertexArray(quad_vao_);
    glGenBuffers(1, &quad_vbo_);
    glBindBuffer(GL_ARRAY_BUFFER, quad_vbo_);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quad), quad, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE,
                          2*sizeof(float), nullptr);

    glGenTextures(1, &canvas_texture_);
    glBindTexture(GL_TEXTURE_2D, canvas_texture_);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8,
                 cfg_.sensor_width, cfg_.sensor_height,
                 0, GL_RGB, GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    glGenFramebuffers(1, &canvas_fbo_);
    glBindFramebuffer(GL_FRAMEBUFFER, canvas_fbo_);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                           GL_TEXTURE_2D, canvas_texture_, 0);

    if (glCheckFramebufferStatus(GL_FRAMEBUFFER)
        != GL_FRAMEBUFFER_COMPLETE)
        throw std::runtime_error("canvas framebuffer incomplete");

    glViewport(0, 0, cfg_.sensor_width, cfg_.sensor_height);
    glClearColor(0.f,0.f,0.f,1.f);
    glClear(GL_COLOR_BUFFER_BIT);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    glUseProgram(event_program_);
    glUniform2f(glGetUniformLocation(event_program_, "sensor_size"),
                static_cast<float>(cfg_.sensor_width),
                static_cast<float>(cfg_.sensor_height));
    glUniform1f(glGetUniformLocation(event_program_, "point_size"),
                cfg_.point_size);

    glDisable(GL_DEPTH_TEST);
    glDisable(GL_BLEND);
}

void FastEventViewer::destroy_gl() noexcept {
    if (window_)
        glfwMakeContextCurrent(window_);
    if (canvas_fbo_) glDeleteFramebuffers(1, &canvas_fbo_);
    if (canvas_texture_) glDeleteTextures(1, &canvas_texture_);
    if (event_vbo_) glDeleteBuffers(1, &event_vbo_);
    if (event_vao_) glDeleteVertexArrays(1, &event_vao_);
    if (quad_vbo_) glDeleteBuffers(1, &quad_vbo_);
    if (quad_vao_) glDeleteVertexArrays(1, &quad_vao_);
    if (event_program_) glDeleteProgram(event_program_);
    if (screen_program_) glDeleteProgram(screen_program_);
    if (window_) glfwDestroyWindow(window_);
    window_ = nullptr;
    glfwTerminate();
}


// ---------------- drain + render ----------------

bool FastEventViewer::drain_raw() {
    std::uint64_t tail = raw_ring_.consumer_tail();
    const std::uint64_t head = raw_ring_.consumer_head();
    if (head == tail) {
        raw_points_.clear();
        return false;
    }

    std::uint64_t available = head - tail;
    if (available > cfg_.max_raw_points_per_render) {
        const std::uint64_t drop =
            available - cfg_.max_raw_points_per_render;
        tail += drop;
        available = cfg_.max_raw_points_per_render;
        raw_stale_drops_.fetch_add(drop, std::memory_order_relaxed);
    }

    raw_points_.resize(static_cast<std::size_t>(available));

    for (std::uint64_t i = 0; i < available; ++i) {
        const PackedEvent &e = raw_ring_.consumer_at(tail + i);
        Point &p = raw_points_[static_cast<std::size_t>(i)];
        p.x = event_x(e);
        p.y = event_y(e);
        p.t = e.t;
        const std::uint8_t g = event_p(e) ? 105 : 35;
        p.r = g; p.g = g; p.b = g; p.pad = 0;
    }

    raw_ring_.consumer_commit(head);
    return true;
}

bool FastEventViewer::drain_freq() {
    std::uint64_t tail = freq_ring_.consumer_tail();
    const std::uint64_t head = freq_ring_.consumer_head();
    if (head == tail) {
        freq_points_.clear();
        return false;
    }

    std::uint64_t available = head - tail;
    if (available > cfg_.max_freq_points_per_render) {
        const std::uint64_t drop =
            available - cfg_.max_freq_points_per_render;
        tail += drop;
        available = cfg_.max_freq_points_per_render;
        freq_stale_drops_.fetch_add(drop, std::memory_order_relaxed);
    }

    freq_points_.resize(static_cast<std::size_t>(available));
    for (std::uint64_t i = 0; i < available; ++i)
        freq_points_[static_cast<std::size_t>(i)] =
            freq_ring_.consumer_at(tail + i);

    freq_ring_.consumer_commit(head);
    return true;
}

void FastEventViewer::draw_points(const std::vector<Point> &points) {
    if (points.empty())
        return;

    glUseProgram(event_program_);
    glBindVertexArray(event_vao_);
    glBindBuffer(GL_ARRAY_BUFFER, event_vbo_);

    glBufferData(GL_ARRAY_BUFFER,
                 static_cast<GLsizeiptr>(points.size()*sizeof(Point)),
                 nullptr, GL_STREAM_DRAW);
    glBufferSubData(GL_ARRAY_BUFFER, 0,
                    static_cast<GLsizeiptr>(points.size()*sizeof(Point)),
                    points.data());

    glDrawArrays(GL_POINTS, 0,
                 static_cast<GLsizei>(points.size()));
}

bool FastEventViewer::render_once() {
    const bool have_raw = drain_raw();
    const bool have_freq = drain_freq();

    if (!have_raw && !have_freq)
        return false;

    std::uint32_t newest_ts = 0;
    bool have_ts = false;

    if (!raw_points_.empty()) {
        newest_ts = raw_points_.back().t;
        have_ts = true;
    }
    if (!freq_points_.empty()) {
        const std::uint32_t t = freq_points_.back().t;
        if (!have_ts || newer_ts(t, newest_ts))
            newest_ts = t;
        have_ts = true;
    }

    glBindFramebuffer(GL_FRAMEBUFFER, canvas_fbo_);
    glViewport(0, 0, cfg_.sensor_width, cfg_.sensor_height);

    if (!canvas_initialized_) {
        glClear(GL_COLOR_BUFFER_BIT);
        canvas_initialized_ = true;
        last_clear_ts_ = newest_ts;
    } else if (static_cast<std::uint32_t>(newest_ts - last_clear_ts_)
               >= cfg_.persistence_us) {
        glClear(GL_COLOR_BUFFER_BIT);
        last_clear_ts_ = newest_ts;
    }

    // Raw first, colored classifications second.
    draw_points(raw_points_);
    draw_points(freq_points_);

    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    int fbw=0, fbh=0;
    glfwGetFramebufferSize(window_, &fbw, &fbh);
    glViewport(0, 0, fbw, fbh);
    glClear(GL_COLOR_BUFFER_BIT);

    glUseProgram(screen_program_);
    glBindVertexArray(quad_vao_);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, canvas_texture_);
    glUniform1i(glGetUniformLocation(screen_program_, "canvas"), 0);
    glDrawArrays(GL_TRIANGLES, 0, 6);

    glfwSwapBuffers(window_);
    rendered_frames_.fetch_add(1, std::memory_order_relaxed);
    return true;
}

void FastEventViewer::run() {
    using Clock = std::chrono::steady_clock;

    const bool limit = cfg_.max_fps != 0;
    const auto period = limit
        ? std::chrono::duration_cast<Clock::duration>(
              std::chrono::duration<double>(1.0 / cfg_.max_fps))
        : Clock::duration::zero();

    auto next = Clock::now();

    while (!stop_requested_.load(std::memory_order_acquire)
           && !glfwWindowShouldClose(window_)) {
        glfwPollEvents();

        if (glfwGetKey(window_, GLFW_KEY_Q) == GLFW_PRESS
            || glfwGetKey(window_, GLFW_KEY_ESCAPE) == GLFW_PRESS)
            break;

        const auto now = Clock::now();
        if (limit && now < next) {
            const auto rem = next - now;
            if (rem > std::chrono::microseconds(500))
                std::this_thread::sleep_for(rem - std::chrono::microseconds(250));
            else
                std::this_thread::yield();
            continue;
        }

        if (render_once()) {
            if (limit)
                next = Clock::now() + period;
        } else {
            std::this_thread::sleep_for(std::chrono::microseconds(200));
        }
    }

    stop_requested_.store(true, std::memory_order_release);
}

FastEventViewer::Stats FastEventViewer::stats() const noexcept {
    return Stats{
        raw_submitted_.load(std::memory_order_relaxed),
        raw_ring_drops_.load(std::memory_order_relaxed),
        raw_stale_drops_.load(std::memory_order_relaxed),
        freq_submitted_.load(std::memory_order_relaxed),
        freq_ring_drops_.load(std::memory_order_relaxed),
        freq_stale_drops_.load(std::memory_order_relaxed),
        rendered_frames_.load(std::memory_order_relaxed)
    };
}

} // namespace event_led_pose
