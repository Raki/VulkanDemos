#include "Utility.h"


//#define STB_IMAGE_WRITE_IMPLEMENTATION
//#include "stb_image_write.h"
namespace Utility
{
    std::string readTxtFileContents(std::string filePath)
    {
        std::ifstream t(filePath);
        std::stringstream buffer;
        buffer << t.rdbuf();
        return buffer.str();
    }

    std::vector<unsigned char> readBinaryFileContents(const std::string filename)
    {
        // binary mode is only for switching off newline translation
        std::ifstream file(filename, std::ios::binary);

        if (!file.good())
            return std::vector<unsigned char>();

        file.unsetf(std::ios::skipws);

        std::streampos file_size;
        file.seekg(0, std::ios::end);
        file_size = file.tellg();
        file.seekg(0, std::ios::beg);

        std::vector<unsigned char> vec;
        vec.reserve(file_size);
        vec.insert(vec.begin(),
            std::istream_iterator<unsigned char>(file),
            std::istream_iterator<unsigned char>());
        return (vec);
    }
    
    //void savePngFile(std::string filename,int w, int h, int comp, unsigned char *data)
    //{
    //    //int stbi_write_png(char const *filename, int w, int h, int comp, const void *data, int stride_in_bytes);
    //    stbi_flip_vertically_on_write(1);
    //    stbi_write_png("test.png",w,h,comp,data,w*comp);
    //}
    
    std::vector<std::string> split(std::string str, std::string delim)
    {
        if (str.length() <= 1)
            return std::vector<std::string>();

        std::vector<std::string> resultArray;
        auto found = str.find_first_of(delim);
        decltype(found) prevPos = 0;

        while (found!=std::string::npos)
        {
            resultArray.push_back(str.substr(prevPos, found - prevPos));
            prevPos = found + 1;
            found = str.find_first_of(delim,prevPos);
        }

       


        return resultArray;
    }
    float getReflectionAngle(float cAngle, float normal)
    {
        auto flip = (cAngle+ 180);
        if (flip > 360)flip -= 360;
        //incident angle
        auto iAngle = flip - normal;
        //reflected angle
        auto rAngle = normal - iAngle;
        return rAngle;
    }
}



