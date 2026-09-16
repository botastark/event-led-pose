#include "fast_event_viewer.hpp"

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>

namespace event_led_pose {
namespace {

GLuint compile_shader(GLenum type, const char *src) {
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &src, nullptr);
    glCompileShader(shader);

    GLint ok = GL_FALSE;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &ok);

    if (ok == GL_TRUE)
        return shader;

    GLint length = 0;
    glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &length);

    std::string log(
        static_cast<std::size_t>(std::max(length, 1)),
        '\0');

    glGetShaderInfoLog(
        shader,
        length,
        nullptr,
        log.data());

    glDeleteShader(shader);

    throw std::runtime_error(
        "Shader compilation failed:\n" + log);
}

GLuint make_program(const char *vs_src, const char *fs_src) {
    const GLuint vs = compile_shader(GL_VERTEX_SHADER, vs_src);
    const GLuint fs = compile_shader(GL_FRAGMENT_SHADER, fs_src);

    const GLuint program = glCreateProgram();

    glAttachShader(program, vs);
    glAttachShader(program, fs);
    glLinkProgram(program);

    glDeleteShader(vs);
    glDeleteShader(fs);

    GLint ok = GL_FALSE;
    glGetProgramiv(program, GL_LINK_STATUS, &ok);

    if (ok == GL_TRUE)
        return program;

    GLint length = 0;
    glGetProgramiv(program, GL_INFO_LOG_LENGTH, &length);

    std::string log(
        static_cast<std::size_t>(std::max(length, 1)),
        '\0');

    glGetProgramInfoLog(
        program,
        length,
        nullptr,
        log.data());

    glDeleteProgram(program);

    throw std::runtime_error(
        "Program link failed:\n" + log);
}

inline bool newer_ts(std::uint32_t a, std::uint32_t b) noexcept {
    return static_cast<std::int32_t>(a - b) > 0;
}

inline void color_for_frequency(std::size_t id,
                                std::uint8_t &r,
                                std::uint8_t &g,
                                std::uint8_t &b) noexcept {
    switch (id) {
    case 0:
        r = 0; g = 255; b = 0;       // 165 Hz
        break;
    case 1:
        r = 255; g = 255; b = 0;     // 366 Hz
        break;
    case 2:
        r = 0; g = 128; b = 255;     // 596 Hz
        break;
    default:
        r = g = b = 255;
        break;
    }
}

inline void color_for_overlay(std::size_t id,
                              std::uint8_t &r,
                              std::uint8_t &g,
                              std::uint8_t &b) noexcept {
    // Darker shades of the classified-event colors, used only for
    // center crosses and radial-statistics circles.
    switch (id) {
    case 0:
        r = 0; g = 150; b = 0;       // dark green, 165 Hz
        break;
    case 1:
        r = 165; g = 165; b = 0;     // dark yellow/olive, 366 Hz
        break;
    case 2:
        r = 0; g = 75; b = 155;      // dark blue, 596 Hz
        break;
    default:
        r = g = b = 150;
        break;
    }
}

} // namespace


FastEventViewer::FastEventViewer(const Config &cfg,
                                 CenterStore *center_store)
    : cfg_(cfg),
      center_store_(center_store),
      raw_ring_(cfg.raw_ring_capacity),
      freq_ring_(cfg.freq_ring_capacity) {
    raw_points_.reserve(cfg_.max_raw_points_per_render);
    freq_points_.reserve(cfg_.max_freq_points_per_render);
    center_lines_.reserve(12);
    stat_circle_lines_.reserve(
        static_cast<std::size_t>(
            std::max(8, cfg_.stat_circle_segments))
        * 4u
        * CENTER_FREQ_COUNT);

    plot_vertices_.reserve(
        2u * RADIAL_HISTOGRAM_BINS + 16u);

    init_gl();
}

FastEventViewer::~FastEventViewer() {
    destroy_gl();
}


// ============================================================
// PRODUCER APIs
// ============================================================

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

    raw_submitted_.fetch_add(
        raw_batch_submitted_,
        std::memory_order_relaxed);

    raw_ring_drops_.fetch_add(
        raw_batch_dropped_,
        std::memory_order_relaxed);
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

    color_for_frequency(id, p.r, p.g, p.b);

    if (!freq_ring_.producer_push(p)) {
        ++freq_batch_dropped_;
        return false;
    }

    return true;
}

void FastEventViewer::freq_end_batch() noexcept {
    freq_ring_.producer_end();

    freq_submitted_.fetch_add(
        freq_batch_submitted_,
        std::memory_order_relaxed);

    freq_ring_drops_.fetch_add(
        freq_batch_dropped_,
        std::memory_order_relaxed);
}


// ============================================================
// OPENGL
// ============================================================

void FastEventViewer::init_gl() {
    if (!glfwInit())
        throw std::runtime_error("glfwInit failed");

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    const int initial_window_height =
        cfg_.sensor_height
        +
        (
            cfg_.show_histograms
            ? std::max(80, cfg_.histogram_panel_height)
            : 0
        );

    window_ = glfwCreateWindow(
        cfg_.sensor_width,
        initial_window_height,
        cfg_.title.c_str(),
        nullptr,
        nullptr);

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

            gl_Position = vec4(
                uv.x * 2.0 - 1.0,
                1.0 - uv.y * 2.0,
                0.0,
                1.0);

            gl_PointSize = point_size;
            vcolor = in_rgb;
        }
    )GLSL";

    const char *event_fs = R"GLSL(
        #version 330 core

        in vec3 vcolor;
        out vec4 frag;

        void main() {
            frag = vec4(vcolor, 1.0);
        }
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

        void main() {
            frag = texture(canvas, uv);
        }
    )GLSL";

    const char *plot_vs = R"GLSL(
        #version 330 core

        layout(location=0) in vec2 in_pos;
        layout(location=1) in vec3 in_color;

        out vec3 vcolor;

        void main() {
            gl_Position = vec4(in_pos, 0.0, 1.0);
            vcolor = in_color;
        }
    )GLSL";

    const char *plot_fs = R"GLSL(
        #version 330 core

        in vec3 vcolor;
        out vec4 frag;

        void main() {
            frag = vec4(vcolor, 1.0);
        }
    )GLSL";

    event_program_ = make_program(event_vs, event_fs);
    screen_program_ = make_program(screen_vs, screen_fs);
    plot_program_ = make_program(plot_vs, plot_fs);

    glGenVertexArrays(1, &event_vao_);
    glBindVertexArray(event_vao_);

    glGenBuffers(1, &event_vbo_);
    glBindBuffer(GL_ARRAY_BUFFER, event_vbo_);

    const std::size_t max_points =
        std::max(
            cfg_.max_raw_points_per_render,
            cfg_.max_freq_points_per_render);

    glBufferData(
        GL_ARRAY_BUFFER,
        static_cast<GLsizeiptr>(
            max_points * sizeof(Point)),
        nullptr,
        GL_STREAM_DRAW);

    glEnableVertexAttribArray(0);

    glVertexAttribPointer(
        0,
        2,
        GL_UNSIGNED_SHORT,
        GL_FALSE,
        sizeof(Point),
        reinterpret_cast<void *>(offsetof(Point, x)));

    glEnableVertexAttribArray(1);

    glVertexAttribPointer(
        1,
        3,
        GL_UNSIGNED_BYTE,
        GL_TRUE,
        sizeof(Point),
        reinterpret_cast<void *>(offsetof(Point, r)));

    static constexpr float quad[] = {
        -1.f,-1.f,  1.f,-1.f,  1.f, 1.f,
        -1.f,-1.f,  1.f, 1.f, -1.f, 1.f
    };

    glGenVertexArrays(1, &quad_vao_);
    glBindVertexArray(quad_vao_);

    glGenBuffers(1, &quad_vbo_);
    glBindBuffer(GL_ARRAY_BUFFER, quad_vbo_);

    glBufferData(
        GL_ARRAY_BUFFER,
        sizeof(quad),
        quad,
        GL_STATIC_DRAW);

    glEnableVertexAttribArray(0);

    glVertexAttribPointer(
        0,
        2,
        GL_FLOAT,
        GL_FALSE,
        2 * sizeof(float),
        nullptr);

    // Histogram line renderer.
    glGenVertexArrays(1, &plot_vao_);
    glBindVertexArray(plot_vao_);

    glGenBuffers(1, &plot_vbo_);
    glBindBuffer(GL_ARRAY_BUFFER, plot_vbo_);

    glBufferData(
        GL_ARRAY_BUFFER,
        static_cast<GLsizeiptr>(
            (2u * RADIAL_HISTOGRAM_BINS + 32u)
            * sizeof(PlotVertex)),
        nullptr,
        GL_STREAM_DRAW);

    glEnableVertexAttribArray(0);

    glVertexAttribPointer(
        0,
        2,
        GL_FLOAT,
        GL_FALSE,
        sizeof(PlotVertex),
        reinterpret_cast<void *>(
            offsetof(PlotVertex, x)));

    glEnableVertexAttribArray(1);

    glVertexAttribPointer(
        1,
        3,
        GL_FLOAT,
        GL_FALSE,
        sizeof(PlotVertex),
        reinterpret_cast<void *>(
            offsetof(PlotVertex, r)));

    glGenTextures(1, &canvas_texture_);
    glBindTexture(GL_TEXTURE_2D, canvas_texture_);

    glTexImage2D(
        GL_TEXTURE_2D,
        0,
        GL_RGB8,
        cfg_.sensor_width,
        cfg_.sensor_height,
        0,
        GL_RGB,
        GL_UNSIGNED_BYTE,
        nullptr);

    glTexParameteri(
        GL_TEXTURE_2D,
        GL_TEXTURE_MIN_FILTER,
        GL_NEAREST);

    glTexParameteri(
        GL_TEXTURE_2D,
        GL_TEXTURE_MAG_FILTER,
        GL_NEAREST);

    glGenFramebuffers(1, &canvas_fbo_);
    glBindFramebuffer(GL_FRAMEBUFFER, canvas_fbo_);

    glFramebufferTexture2D(
        GL_FRAMEBUFFER,
        GL_COLOR_ATTACHMENT0,
        GL_TEXTURE_2D,
        canvas_texture_,
        0);

    if (glCheckFramebufferStatus(GL_FRAMEBUFFER)
        != GL_FRAMEBUFFER_COMPLETE) {
        throw std::runtime_error(
            "canvas framebuffer incomplete");
    }

    glViewport(
        0,
        0,
        cfg_.sensor_width,
        cfg_.sensor_height);

    glClearColor(0.f, 0.f, 0.f, 1.f);
    glClear(GL_COLOR_BUFFER_BIT);

    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    glUseProgram(event_program_);

    glUniform2f(
        glGetUniformLocation(
            event_program_,
            "sensor_size"),
        static_cast<float>(cfg_.sensor_width),
        static_cast<float>(cfg_.sensor_height));

    glUniform1f(
        glGetUniformLocation(
            event_program_,
            "point_size"),
        cfg_.point_size);

    glDisable(GL_DEPTH_TEST);
    glDisable(GL_BLEND);
}

void FastEventViewer::destroy_gl() noexcept {
    if (window_)
        glfwMakeContextCurrent(window_);

    if (canvas_fbo_)
        glDeleteFramebuffers(1, &canvas_fbo_);

    if (canvas_texture_)
        glDeleteTextures(1, &canvas_texture_);

    if (plot_vbo_)
        glDeleteBuffers(1, &plot_vbo_);

    if (plot_vao_)
        glDeleteVertexArrays(1, &plot_vao_);

    if (event_vbo_)
        glDeleteBuffers(1, &event_vbo_);

    if (event_vao_)
        glDeleteVertexArrays(1, &event_vao_);

    if (quad_vbo_)
        glDeleteBuffers(1, &quad_vbo_);

    if (quad_vao_)
        glDeleteVertexArrays(1, &quad_vao_);

    if (event_program_)
        glDeleteProgram(event_program_);

    if (screen_program_)
        glDeleteProgram(screen_program_);

    if (plot_program_)
        glDeleteProgram(plot_program_);

    if (window_)
        glfwDestroyWindow(window_);

    window_ = nullptr;
    glfwTerminate();
}


// ============================================================
// DRAIN
// ============================================================

bool FastEventViewer::drain_raw() {
    std::uint64_t tail = raw_ring_.consumer_tail();
    const std::uint64_t head = raw_ring_.consumer_head();

    if (head == tail) {
        raw_points_.clear();
        return false;
    }

    std::uint64_t available = head - tail;

    if (available >
        cfg_.max_raw_points_per_render) {
        const std::uint64_t drop =
            available -
            cfg_.max_raw_points_per_render;

        tail += drop;
        available =
            cfg_.max_raw_points_per_render;

        raw_stale_drops_.fetch_add(
            drop,
            std::memory_order_relaxed);
    }

    raw_points_.resize(
        static_cast<std::size_t>(available));

    for (std::uint64_t i = 0;
         i < available;
         ++i) {
        const PackedEvent &e =
            raw_ring_.consumer_at(tail + i);

        Point &p =
            raw_points_[
                static_cast<std::size_t>(i)];

        p.x = event_x(e);
        p.y = event_y(e);
        p.t = e.t;

        const std::uint8_t gray =
            event_p(e) ? 105 : 35;

        p.r = gray;
        p.g = gray;
        p.b = gray;
        p.pad = 0;
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

    if (available >
        cfg_.max_freq_points_per_render) {
        const std::uint64_t drop =
            available -
            cfg_.max_freq_points_per_render;

        tail += drop;
        available =
            cfg_.max_freq_points_per_render;

        freq_stale_drops_.fetch_add(
            drop,
            std::memory_order_relaxed);
    }

    freq_points_.resize(
        static_cast<std::size_t>(available));

    for (std::uint64_t i = 0;
         i < available;
         ++i) {
        freq_points_[
            static_cast<std::size_t>(i)]
            =
            freq_ring_.consumer_at(
                tail + i);
    }

    freq_ring_.consumer_commit(head);
    return true;
}


// ============================================================
// DRAW
// ============================================================

void FastEventViewer::draw_points(
    const std::vector<Point> &points,
    unsigned int primitive) {
    if (points.empty())
        return;

    glUseProgram(event_program_);
    glBindVertexArray(event_vao_);
    glBindBuffer(GL_ARRAY_BUFFER, event_vbo_);

    glBufferData(
        GL_ARRAY_BUFFER,
        static_cast<GLsizeiptr>(
            points.size() * sizeof(Point)),
        nullptr,
        GL_STREAM_DRAW);

    glBufferSubData(
        GL_ARRAY_BUFFER,
        0,
        static_cast<GLsizeiptr>(
            points.size() * sizeof(Point)),
        points.data());

    glDrawArrays(
        static_cast<GLenum>(primitive),
        0,
        static_cast<GLsizei>(
            points.size()));
}

void FastEventViewer::draw_centers() {
    center_lines_.clear();
    stat_circle_lines_.clear();

    const int half =
        std::max(
            2,
            static_cast<int>(
                std::lround(
                    cfg_.center_marker_half_size_px)));

    const int segments =
        std::max(
            8,
            cfg_.stat_circle_segments);

    for (std::size_t id = 0;
         id < CENTER_FREQ_COUNT;
         ++id)
    {
        const auto &center =
            center_snapshot_.frequency[id];

        if (!center.valid)
            continue;

        const int cx =
            static_cast<int>(
                std::lround(center.x));

        const int cy =
            static_cast<int>(
                std::lround(center.y));

        const auto clamp_x =
            [&](int x) {
                return static_cast<std::uint16_t>(
                    std::clamp(
                        x,
                        0,
                        cfg_.sensor_width - 1));
            };

        const auto clamp_y =
            [&](int y) {
                return static_cast<std::uint16_t>(
                    std::clamp(
                        y,
                        0,
                        cfg_.sensor_height - 1));
            };

        std::uint8_t r, g, b;

        color_for_overlay(
            id,
            r, g, b);

        // Center cross.
        center_lines_.push_back(
            Point{
                clamp_x(cx - half),
                clamp_y(cy),
                center.newest_t,
                r,g,b,0});

        center_lines_.push_back(
            Point{
                clamp_x(cx + half),
                clamp_y(cy),
                center.newest_t,
                r,g,b,0});

        center_lines_.push_back(
            Point{
                clamp_x(cx),
                clamp_y(cy - half),
                center.newest_t,
                r,g,b,0});

        center_lines_.push_back(
            Point{
                clamp_x(cx),
                clamp_y(cy + half),
                center.newest_t,
                r,g,b,0});

        if (!cfg_.show_stat_circles)
            continue;

        const auto add_circle =
            [&](float radius)
            {
                if (radius <= 0.5f)
                    return;

                constexpr double TWO_PI =
                    6.28318530717958647692;

                for (int i = 0;
                     i < segments;
                     ++i)
                {
                    const double a0 =
                        TWO_PI
                        *
                        static_cast<double>(i)
                        /
                        static_cast<double>(segments);

                    const double a1 =
                        TWO_PI
                        *
                        static_cast<double>(i + 1)
                        /
                        static_cast<double>(segments);

                    const int x0 =
                        static_cast<int>(
                            std::lround(
                                center.x
                                +
                                static_cast<double>(radius)
                                *
                                std::cos(a0)));

                    const int y0 =
                        static_cast<int>(
                            std::lround(
                                center.y
                                +
                                static_cast<double>(radius)
                                *
                                std::sin(a0)));

                    const int x1 =
                        static_cast<int>(
                            std::lround(
                                center.x
                                +
                                static_cast<double>(radius)
                                *
                                std::cos(a1)));

                    const int y1 =
                        static_cast<int>(
                            std::lround(
                                center.y
                                +
                                static_cast<double>(radius)
                                *
                                std::sin(a1)));

                    stat_circle_lines_.push_back(
                        Point{
                            clamp_x(x0),
                            clamp_y(y0),
                            center.newest_t,
                            r,g,b,0});

                    stat_circle_lines_.push_back(
                        Point{
                            clamp_x(x1),
                            clamp_y(y1),
                            center.newest_t,
                            r,g,b,0});
                }
            };

        // Inner ring = mean radial distance.
        add_circle(
            center.mean_radius);

        // Outer ring = 95th percentile radial distance.
        add_circle(
            center.p95_radius);
    }

    if (!stat_circle_lines_.empty()) {
        glLineWidth(
            cfg_.stat_circle_line_width);

        draw_points(
            stat_circle_lines_,
            GL_LINES);
    }

    if (!center_lines_.empty()) {
        glLineWidth(
            cfg_.center_marker_line_width);

        draw_points(
            center_lines_,
            GL_LINES);
    }

    glLineWidth(1.0f);
}


void FastEventViewer::draw_histograms(
    int framebuffer_width,
    int histogram_height)
{
    if (!cfg_.show_histograms ||
        !center_store_ ||
        histogram_height <= 0)
    {
        return;
    }

    std::uint32_t shared_y_max = 1;

    for (std::size_t id = 0;
         id < CENTER_FREQ_COUNT;
         ++id)
    {
        const auto &s =
            center_snapshot_.frequency[id];

        if (!s.valid)
            continue;

        for (std::uint32_t count :
             s.radial_histogram)
        {
            shared_y_max =
                std::max(
                    shared_y_max,
                    count);
        }
    }

    const int panel_width =
        std::max(
            1,
            framebuffer_width
            /
            static_cast<int>(
                CENTER_FREQ_COUNT));

    const float x_left = -0.88f;
    const float x_right = 0.95f;

    const float y_bottom = -0.78f;
    const float y_top = 0.88f;

    glUseProgram(
        plot_program_);

    glBindVertexArray(
        plot_vao_);

    glBindBuffer(
        GL_ARRAY_BUFFER,
        plot_vbo_);

    for (std::size_t id = 0;
         id < CENTER_FREQ_COUNT;
         ++id)
    {
        const int viewport_x =
            static_cast<int>(id)
            *
            panel_width;

        const int viewport_width =
            (
                id ==
                CENTER_FREQ_COUNT - 1
            )
            ?
            framebuffer_width -
            viewport_x
            :
            panel_width;

        glViewport(
            viewport_x,
            0,
            viewport_width,
            histogram_height);

        float r = 1.0f;
        float g = 1.0f;
        float b = 1.0f;

        if (id == 0) {
            r = 0.0f;
            g = 1.0f;
            b = 0.0f;
        }
        else if (id == 1) {
            r = 1.0f;
            g = 1.0f;
            b = 0.0f;
        }
        else {
            r = 0.0f;
            g = 0.5f;
            b = 1.0f;
        }

        // Axes.
        plot_vertices_.clear();

        auto add_line =
            [&](float x0,
                float y0,
                float x1,
                float y1,
                float lr,
                float lg,
                float lb)
            {
                plot_vertices_.push_back(
                    PlotVertex{
                        x0,y0,
                        lr,lg,lb});

                plot_vertices_.push_back(
                    PlotVertex{
                        x1,y1,
                        lr,lg,lb});
            };

        add_line(
            x_left,
            y_bottom,
            x_right,
            y_bottom,
            0.32f,0.32f,0.36f);

        add_line(
            x_left,
            y_bottom,
            x_left,
            y_top,
            0.32f,0.32f,0.36f);

        // 25%, 50%, 75% shared-Y reference lines.
        for (int q = 1;
             q <= 3;
             ++q)
        {
            const float fraction =
                static_cast<float>(q)
                /
                4.0f;

            const float y =
                y_bottom
                +
                fraction
                *
                (y_top - y_bottom);

            add_line(
                x_left,
                y,
                x_right,
                y,
                0.12f,0.12f,0.14f);
        }

        glBufferData(
            GL_ARRAY_BUFFER,
            static_cast<GLsizeiptr>(
                plot_vertices_.size()
                *
                sizeof(PlotVertex)),
            plot_vertices_.data(),
            GL_STREAM_DRAW);

        glDrawArrays(
            GL_LINES,
            0,
            static_cast<GLsizei>(
                plot_vertices_.size()));

        const auto &stats =
            center_snapshot_.frequency[id];

        if (!stats.valid)
            continue;

        // One vertical line per radial bin.
        plot_vertices_.clear();

        for (std::size_t bin = 0;
             bin < RADIAL_HISTOGRAM_BINS;
             ++bin)
        {
            const float u =
                (
                    static_cast<float>(bin)
                    +
                    0.5f
                )
                /
                static_cast<float>(
                    RADIAL_HISTOGRAM_BINS);

            const float x =
                x_left
                +
                u
                *
                (x_right - x_left);

            const float fraction =
                static_cast<float>(
                    stats.radial_histogram[bin])
                /
                static_cast<float>(
                    shared_y_max);

            const float y =
                y_bottom
                +
                fraction
                *
                (y_top - y_bottom);

            add_line(
                x,
                y_bottom,
                x,
                y,
                r,g,b);
        }

        // Mean radius marker.
        const float x_max =
            std::max(
                1.0f,
                center_snapshot_
                    .histogram_max_radius_px);

        const auto radius_to_x =
            [&](float radius)
            {
                const float u =
                    std::clamp(
                        radius / x_max,
                        0.0f,
                        1.0f);

                return
                    x_left
                    +
                    u
                    *
                    (x_right - x_left);
            };

        const float mean_x =
            radius_to_x(
                stats.mean_radius);

        const float p95_x =
            radius_to_x(
                stats.p95_radius);

        // Mean = full-height marker.
        add_line(
            mean_x,
            y_bottom,
            mean_x,
            y_top,
            r,g,b);

        // p95 = shorter marker.
        add_line(
            p95_x,
            y_bottom,
            p95_x,
            y_bottom
                +
                0.55f
                *
                (y_top - y_bottom),
            r,g,b);

        glBufferData(
            GL_ARRAY_BUFFER,
            static_cast<GLsizeiptr>(
                plot_vertices_.size()
                *
                sizeof(PlotVertex)),
            plot_vertices_.data(),
            GL_STREAM_DRAW);

        glDrawArrays(
            GL_LINES,
            0,
            static_cast<GLsizei>(
                plot_vertices_.size()));
    }
}

void FastEventViewer::update_window_title() {
    if (!center_store_)
        return;

    std::ostringstream title;

    title
        << cfg_.title
        << " | hist[0.."
        << center_snapshot_.histogram_max_radius_px
        << "px]"
        << " | radial stats ";

    static constexpr const char *names[3] = {
        "165", "366", "596"
    };

    title
        << std::fixed
        << std::setprecision(1);

    for (std::size_t id = 0;
         id < 3;
         ++id)
    {
        if (id != 0)
            title << " | ";

        const auto &s =
            center_snapshot_.frequency[id];

        title << names[id] << ":";

        if (!s.valid) {
            title << "--";
        }
        else {
            title
                << "c("
                << s.x
                << ","
                << s.y
                << ")"
                << " mu="
                << s.mean_radius
                << " sd="
                << s.std_radius
                << " p95="
                << s.p95_radius
                << " N="
                << s.sample_count;
        }
    }

    glfwSetWindowTitle(
        window_,
        title.str().c_str());
}


// ============================================================
// RENDER LOOP
// ============================================================

bool FastEventViewer::render_once() {
    const bool have_raw =
        drain_raw();

    const bool have_freq =
        drain_freq();

    bool have_center_update = false;

    if (center_store_) {
        have_center_update =
            center_store_->copy_if_new(
                center_version_,
                center_snapshot_);

        if (have_center_update)
            update_window_title();
    }

    if (!have_raw &&
        !have_freq &&
        !have_center_update) {
        return false;
    }

    std::uint32_t newest_ts = 0;
    bool have_ts = false;

    if (!raw_points_.empty()) {
        newest_ts =
            raw_points_.back().t;
        have_ts = true;
    }

    if (!freq_points_.empty()) {
        const std::uint32_t t =
            freq_points_.back().t;

        if (!have_ts ||
            newer_ts(t, newest_ts)) {
            newest_ts = t;
        }

        have_ts = true;
    }

    // Update persistent event canvas.
    if (have_raw ||
        have_freq) {
        glBindFramebuffer(
            GL_FRAMEBUFFER,
            canvas_fbo_);

        glViewport(
            0,
            0,
            cfg_.sensor_width,
            cfg_.sensor_height);

        if (!canvas_initialized_) {
            glClear(GL_COLOR_BUFFER_BIT);

            canvas_initialized_ = true;
            last_clear_ts_ =
                newest_ts;
        }
        else if (
            have_ts
            &&
            static_cast<std::uint32_t>(
                newest_ts -
                last_clear_ts_)
            >= cfg_.persistence_us) {
            glClear(GL_COLOR_BUFFER_BIT);

            last_clear_ts_ =
                newest_ts;
        }

        draw_points(
            raw_points_,
            GL_POINTS);

        draw_points(
            freq_points_,
            GL_POINTS);
    }

    // Present sensor canvas.
    glBindFramebuffer(
        GL_FRAMEBUFFER,
        0);

    int framebuffer_width = 0;
    int framebuffer_height = 0;

    glfwGetFramebufferSize(
        window_,
        &framebuffer_width,
        &framebuffer_height);

    glViewport(
        0,
        0,
        framebuffer_width,
        framebuffer_height);

    glClear(GL_COLOR_BUFFER_BIT);

    const int histogram_height =
        cfg_.show_histograms
        ?
        std::clamp(
            cfg_.histogram_panel_height,
            80,
            std::max(
                80,
                framebuffer_height / 2))
        :
        0;

    const int sensor_view_height =
        std::max(
            1,
            framebuffer_height
            -
            histogram_height);

    // Sensor image occupies the upper viewport.
    glViewport(
        0,
        histogram_height,
        framebuffer_width,
        sensor_view_height);

    glUseProgram(
        screen_program_);

    glBindVertexArray(
        quad_vao_);

    glActiveTexture(
        GL_TEXTURE0);

    glBindTexture(
        GL_TEXTURE_2D,
        canvas_texture_);

    glUniform1i(
        glGetUniformLocation(
            screen_program_,
            "canvas"),
        0);

    glDrawArrays(
        GL_TRIANGLES,
        0,
        6);

    // Centers/radius circles use sensor pixel coordinates and therefore
    // must be drawn while the sensor viewport is active.
    draw_centers();

    // Three radial-distance histograms occupy the lower strip.
    // Their Y scale is shared across all three frequencies.
    draw_histograms(
        framebuffer_width,
        histogram_height);

    glfwSwapBuffers(window_);

    rendered_frames_.fetch_add(
        1,
        std::memory_order_relaxed);

    return true;
}

void FastEventViewer::run() {
    using Clock =
        std::chrono::steady_clock;

    const bool limited =
        cfg_.max_fps != 0;

    const auto period =
        limited
        ?
        std::chrono::duration_cast<
            Clock::duration>(
            std::chrono::duration<double>(
                1.0 /
                static_cast<double>(
                    cfg_.max_fps)))
        :
        Clock::duration::zero();

    auto next =
        Clock::now();

    while (
        !stop_requested_.load(
            std::memory_order_acquire)
        &&
        !glfwWindowShouldClose(window_)) {
        glfwPollEvents();

        if (
            glfwGetKey(
                window_,
                GLFW_KEY_Q)
                == GLFW_PRESS
            ||
            glfwGetKey(
                window_,
                GLFW_KEY_ESCAPE)
                == GLFW_PRESS) {
            break;
        }

        const auto now =
            Clock::now();

        if (limited &&
            now < next) {
            const auto remaining =
                next - now;

            if (remaining >
                std::chrono::microseconds(500)) {
                std::this_thread::sleep_for(
                    remaining -
                    std::chrono::microseconds(250));
            }
            else {
                std::this_thread::yield();
            }

            continue;
        }

        if (render_once()) {
            if (limited)
                next =
                    Clock::now() +
                    period;
        }
        else {
            std::this_thread::sleep_for(
                std::chrono::microseconds(200));
        }
    }

    stop_requested_.store(
        true,
        std::memory_order_release);
}

FastEventViewer::Stats
FastEventViewer::stats() const noexcept {
    return Stats{
        raw_submitted_.load(
            std::memory_order_relaxed),

        raw_ring_drops_.load(
            std::memory_order_relaxed),

        raw_stale_drops_.load(
            std::memory_order_relaxed),

        freq_submitted_.load(
            std::memory_order_relaxed),

        freq_ring_drops_.load(
            std::memory_order_relaxed),

        freq_stale_drops_.load(
            std::memory_order_relaxed),

        rendered_frames_.load(
            std::memory_order_relaxed)
    };
}

} // namespace event_led_pose
