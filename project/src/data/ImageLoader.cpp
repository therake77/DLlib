#include <data/ImageLoader.hpp>

#include <algorithm>
#include <cctype>
#include <cstring>
#include <stdexcept>
#include <vector>

#include <external/stb_image.h>

ImageLoader::ImageLoader(std::shared_ptr<Device> dev)
    : _dev(std::move(dev)) {
    if(!_dev){
        throw std::runtime_error("Error: Invalid device");
    }
}

bool ImageLoader::_is_supported_extension(const std::filesystem::path& path){
    std::string ext = path.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(),
        [](unsigned char c){ return static_cast<char>(std::tolower(c)); });

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

Tensor ImageLoader::_load_image_cpu(const std::filesystem::path& path, bool grayscale, bool normalize){
    if(!std::filesystem::exists(path)){
        throw std::runtime_error("Error: Image file does not exist");
    }

    if(!std::filesystem::is_regular_file(path)){
        throw std::runtime_error("Error: Image path is not a regular file");
    }

    if(!_is_supported_extension(path)){
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
    std::size_t spatial_size = static_cast<std::size_t>(width) * static_cast<std::size_t>(height);

    for(int c = 0; c < desired_channels; ++c){
        for(int y = 0; y < height; ++y){
            for(int x = 0; x < width; ++x){
                std::size_t src_idx = (static_cast<std::size_t>(y) * static_cast<std::size_t>(width) + static_cast<std::size_t>(x)) * desired_channels + static_cast<std::size_t>(c);
                std::size_t dst_idx = static_cast<std::size_t>(c) * spatial_size + static_cast<std::size_t>(y) * static_cast<std::size_t>(width) + static_cast<std::size_t>(x);

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

Tensor ImageLoader::load_image(const std::filesystem::path& path, bool grayscale, bool normalize) const{
    Tensor out = _load_image_cpu(path, grayscale, normalize);

    if(_dev->get_type() != DeviceType::CPU){
        out.to(_dev);
    }

    return out;
}

Tensor ImageLoader::load_folder(const std::filesystem::path& folder, bool grayscale, bool normalize) const{
    if(!std::filesystem::exists(folder)){
        throw std::runtime_error("Error: Folder does not exist");
    }

    if(!std::filesystem::is_directory(folder)){
        throw std::runtime_error("Error: Provided path is not a folder");
    }

    std::vector<std::filesystem::path> files;
    for(const auto& entry : std::filesystem::directory_iterator(folder)){
        if(entry.is_regular_file() && _is_supported_extension(entry.path())){
            files.push_back(entry.path());
        }
    }

    if(files.empty()){
        throw std::runtime_error("Error: No supported images found in folder");
    }

    std::sort(files.begin(), files.end());

    std::vector<Tensor> samples;
    samples.reserve(files.size());

    for(const auto& file : files){
        samples.push_back(_load_image_cpu(file, grayscale, normalize));
    }

    const Shape& sample_shape = samples.front().shape;
    std::size_t batch_size = samples.size();
    std::size_t channels = sample_shape[1];
    std::size_t height = sample_shape[2];
    std::size_t width = sample_shape[3];
    std::size_t sample_size = channels * height * width;

    for(const auto& sample : samples){
        if(sample.shape != sample_shape){
            throw std::runtime_error("Error: All images in the folder must have the same shape");
        }
    }

    Tensor batch(
        {batch_size, channels, height, width},
        DeviceManager::get_cpu_device(),
        0.0f,
        true
    );

    float* batch_data = batch.data();
    for(std::size_t i = 0; i < batch_size; ++i){
        std::memcpy(
            batch_data + i * sample_size,
            samples[i].data(),
            sample_size * sizeof(float)
        );
    }

    if(_dev->get_type() != DeviceType::CPU){
        batch.to(_dev);
    }

    return batch;
}