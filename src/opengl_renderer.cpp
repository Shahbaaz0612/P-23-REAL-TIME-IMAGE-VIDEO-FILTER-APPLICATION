#include "opengl_renderer.h"
#include <iostream>
using namespace std;

//========================= openGL window and context creation =============================

bool OpenGLRenderer::initialize( int width, int height, int channels)
{
    if (width <= 0 || height <= 0)
    {
        cout << "Invalid image dimensions!" << endl;
        return false;
    }
    if (channels != 1 && channels != 3)
    {
        cout << "Invalid channel count!" << endl;
        return false;
    }

    this->width = width;
    this->height = height;
    this->channels = channels;

    // Initialize GLFW
    if (!glfwInit())
    {
        cout << "GLFW initialization failed!" << endl;
        return false;
    }
    glfwInitialized = true;

    // Request OpenGL 3.3
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(
        GLFW_OPENGL_PROFILE,
        GLFW_OPENGL_CORE_PROFILE
    );

    // Create window
    window = glfwCreateWindow(
        width,
        height,
        "CUDA Image Filter",
        nullptr,
        nullptr
    );

    if (!window)
    {
        cout << "OpenGL window creation failed!" << endl;
        cleanup();
        return false;
    }

    // Make OpenGL context current
    glfwMakeContextCurrent(window);

    // Initialize GLEW
    if (glewInit() != GLEW_OK)
    {
        cout << "GLEW initialization failed!" << endl;
        cleanup();
        return false;
    }

    cout << "OpenGL initialized successfully!" << endl;

  /*  // Create texture
    if (!createTexture())
    {
        cleanup();
        return false;
    }
*/
    return true;
}

//======================================= creating openGL Textures ================================

bool OpenGLRenderer::createTexture()
{
    // Generate texture ID
    glGenTextures(1, &texture);

    // Bind texture
    glBindTexture(GL_TEXTURE_2D, texture);

    // Texture filtering
    glTexParameteri(
        GL_TEXTURE_2D,
        GL_TEXTURE_MIN_FILTER,
        GL_LINEAR
    );

    glTexParameteri(
        GL_TEXTURE_2D,
        GL_TEXTURE_MAG_FILTER,
        GL_LINEAR
    );

    // Texture wrapping
    glTexParameteri(
        GL_TEXTURE_2D,
        GL_TEXTURE_WRAP_S,
        GL_CLAMP_TO_EDGE
    );

    glTexParameteri(
        GL_TEXTURE_2D,
        GL_TEXTURE_WRAP_T,
        GL_CLAMP_TO_EDGE
    );

    // Set pixel alignment
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

    // Allocate texture storage based on channels
    if (channels == 1)
    {
        // Grayscale texture
        glTexImage2D(
            GL_TEXTURE_2D,
            0,
            GL_R8,
            width,
            height,
            0,
            GL_RED,
            GL_UNSIGNED_BYTE,
            nullptr
        );
    }
    else
    {
        // RGB texture for inversion and blur
        glTexImage2D(
            GL_TEXTURE_2D,
            0,
            GL_RGB8,
            width,
            height,
            0,
            GL_RGB,
            GL_UNSIGNED_BYTE,
            nullptr
        );
    }

    // Unbind texture
    glBindTexture(GL_TEXTURE_2D, 0);

    cout << "OpenGL texture created successfully!" << endl;

    return true;
}

//============================ updating Textures using PBO ===========================


bool OpenGLRenderer::updateTextureFromPBO(GLuint pbo)
{
    if (pbo == 0 || texture == 0)
    {
        cout << "Invalid PBO or texture!" << endl;
        return false;
    }

    // Make OpenGL context current
    makeContextCurrent();

    // Bind PBO
    glBindBuffer(
        GL_PIXEL_UNPACK_BUFFER,
        pbo
    );

    // Bind texture
    glBindTexture(
        GL_TEXTURE_2D,
        texture
    );

    // Set alignment to 1 byte
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

    // Select pixel format
    GLenum format;
    if (channels == 1)
    {
        format = GL_RED;
    }
    else
    {
        format = GL_BGR;
    }

    // Update texture using PBO data
    glTexSubImage2D(
        GL_TEXTURE_2D,
        0,
        0,
        0,
        width,
        height,
        format,
        GL_UNSIGNED_BYTE,
        nullptr
    );

    // Unbind PBO
    glBindBuffer(
        GL_PIXEL_UNPACK_BUFFER,
        0
    );

    // Unbind texture
    glBindTexture(
        GL_TEXTURE_2D,
        0
    );

    GLenum err = glGetError();

    if (err != GL_NO_ERROR)
    {
        cout << "Texture update failed. OpenGL error: " << err << endl;
        return false;
    }

    return true;
}

//==================================================

// Check window status
bool OpenGLRenderer::shouldClose()
{
    return (window == nullptr ||
            glfwWindowShouldClose(window));
}


// Process window events
void OpenGLRenderer::pollEvents()
{
    glfwPollEvents();
}

// Make OpenGL context current
void OpenGLRenderer::makeContextCurrent()
{
    if (window != nullptr)
    {
        glfwMakeContextCurrent(window);
    }
}

// Release OpenGL resources
void OpenGLRenderer::cleanup()
{
    if (window != nullptr)
    {
        makeContextCurrent();

        // Delete texture
        if (texture != 0)
        {
            glDeleteTextures(1, &texture);
            texture = 0;
        }

        // Destroy window
        glfwDestroyWindow(window);

        window = nullptr;
    }

    if (glfwInitialized)
    {
        glfwTerminate();
        glfwInitialized = false;
    }

    cout << "OpenGL resources released!" << endl;
}

// Destructor

OpenGLRenderer::~OpenGLRenderer()
{
    cleanup();
}
