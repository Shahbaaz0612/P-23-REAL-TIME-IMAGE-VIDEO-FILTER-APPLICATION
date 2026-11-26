#ifndef OPENGL_RENDERER_H
#define OPENGL_RENDERER_H

#include <GL/glew.h>
#include <GLFW/glfw3.h>


class OpenGLRenderer
{
public:

    // Initialize OpenGL window and context
    bool initialize(int width, int height, int channels);

    // Make OpenGL context current
    void makeContextCurrent();

    // Update texture using PBO
    bool updateTextureFromPBO(GLuint pbo);

    // Render texture
 //   void render();

    // Check window status
    bool shouldClose();

    // Release resources
    void cleanup();

    // Process window events
	void pollEvents();

    // Destructor
    ~OpenGLRenderer();


private:

    int width = 0;
    int height = 0;
    int channels = 0;

    // GLFW window
    GLFWwindow* window = nullptr;

    // OpenGL texture
    GLuint texture = 0;

    // Shader program
    GLuint shaderProgram = 0;

    bool glfwInitialized = false;
    
    // Internal functions
    bool createTexture();

    /*    // VAO and VBO
    GLuint VAO = 0;
    GLuint VBO = 0;

    // Shader uniform locations
    GLint textureLocation = -1;
    GLint grayLocation = -1;

    bool createShaderProgram();

    bool createQuad();  */
};

#endif
