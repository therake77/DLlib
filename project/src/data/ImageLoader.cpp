#include <data/ImageLoader.hpp>

#include <algorithm>
#include <cctype>
#include <stdexcept>
#include <string>

#include <external/stb_image.h>

ImageLoader::ImageLoader(std::shared_ptr<Device> dev)
    : _dev(std::move(dev))
{
    if(!_dev){
        throw std::runtime_error("Error: Invalid device");
    }
}

bool ImageLoader::is_supported_extension(const std::filesystem::path& path) {
    std::string ext = path.extension().string();

    std::transform(
        ext.begin(),
        ext.end(),
        ext.begin(),
        [](unsigned char c){
            return static_cast<char>(std::tolower(c));
        }
    );

    return ext == ".png"  ||
           ext == ".jpg"  ||
           ext == ".jpeg" ||
           ext == ".bmp"  ||
           ext == ".tga"  ||
           ext == ".gif"  ||
           ext == ".psd"  ||
           ext == ".pic"  ||
           ext == ".pnm"  ||
           ext == ".ppm"  ||
           ext == ".pgm";
}

Tensor ImageLoader::_load_image_cpu_impl(const std::filesystem::path& path, bool grayscale, bool normalize) {
    if(!std::filesystem::exists(path)){
        throw std::runtime_error("Error: Image file does not exist");
    }

    if(!std::filesystem::is_regular_file(path)){
        throw std::runtime_error("Error: Image path is not a regular file");
    }

    if(!is_supported_extension(path)){
        throw std::runtime_error("Error: Unsupported image extension");
    }

    int width = 0;
    int height = 0;
    int file_channels = 0;
    int desired_channels = grayscale ? 1 : 3;

    unsigned char* pixels = stbi_load(
        path.string().c_str(),
        &width,
        &height,
        &file_channels,
        desired_channels
    );

    if(pixels == nullptr){
        throw std::runtime_error(std::string("Error: Failed to load image: ") + stbi_failure_reason());
    }

    Tensor out(
        {1, static_cast<std::size_t>(desired_channels), static_cast<std::size_t>(height), static_cast<std::size_t>(width)},
        DeviceManager::get_cpu_device(),
        0.0f,
        true
    );

    float* dst = out.data();
    std::size_t w = static_cast<std::size_t>(width);
    std::size_t h = static_cast<std::size_t>(height);

    std::size_t c_count = static_cast<std::size_t>(desired_channels);
    std::size_t spatial_size = w * h;

    for(std::size_t c = 0; c < c_count; ++c) {
        for(std::size_t y = 0; y < h; ++y) {
            for(std::size_t x = 0; x < w; ++x) {
                std::size_t src_idx = (y * w + x) * c_count + c;
                std::size_t dst_idx = c * spatial_size + y * w + x;

                float value = static_cast<float>(pixels[src_idx]);
                if(normalize){
                    value /= 255.0f;
                }
                dst[dst_idx] = value;
            }
        }
    }

    stbi_image_free(pixels);
    return out;
}

Tensor ImageLoader::load_image_cpu(const std::filesystem::path& path, bool grayscale, bool normalize) const{
    return _load_image_cpu_impl(
        path,
        grayscale,
        normalize
    );
}

Tensor ImageLoader::load_image(const std::filesystem::path& path, bool grayscale, bool normalize) const{
    Tensor out = _load_image_cpu_impl(path, grayscale, normalize);

    if(_dev->get_type() != DeviceType::CPU){
        out.to(_dev);
    }

    return out;
}