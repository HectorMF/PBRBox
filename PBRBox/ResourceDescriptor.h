#pragma once
#include <string>
#include "tinyxml2.h"

template<typename T>
class ResourceDescriptor
{
public:
	std::string fileName;

	
	//public final Class<T> type;
	/*public final AssetLoaderParameters params;

	public FileHandle file;

	public AssetDescriptor(String fileName, Class<T> assetType) {
		this(fileName, assetType, null);
	}


	public AssetDescriptor(FileHandle file, Class<T> assetType) {
		this(file, assetType, null);
	}

	public AssetDescriptor(String fileName, Class<T> assetType, AssetLoaderParameters<T> params) {
		this.fileName = fileName.replaceAll("\\\\", "/");
		this.type = assetType;
		this.params = params;
	}


	public AssetDescriptor(FileHandle file, Class<T> assetType, AssetLoaderParameters<T> params) {
		this.fileName = file.path().replaceAll("\\\\", "/");
		this.file = file;
		this.type = assetType;
		this.params = params;
	}

	@Override
		public String toString() {
		StringBuffer buffer = new StringBuffer();
		buffer.append(fileName);
		buffer.append(", ");
		buffer.append(type.getName());
		return buffer.toString();
	}*/
};