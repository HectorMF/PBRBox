
class TextureData
{
	unsigned int width, height;
	void* data;
	unsigned int dataType;
	unsigned int format;
public:
	void* getData() { return data; }

	void setData(unsigned int width, unsigned int height, void* data)
	{
		this->data = data;
		this->width = width;
		this->height = height;
	}

	void setDataType(unsigned int r)
	{
		dataType = r;
	}

	void setFormat(unsigned int r)
	{
		format = r;
	}

	unsigned int getDataType()
	{
		return dataType;
	}

	unsigned int getWidth()
	{
		return width;
	}

	unsigned int getHeight()
	{
		return height;
	}

	unsigned int getFormat()
	{
		return format;
	}
};