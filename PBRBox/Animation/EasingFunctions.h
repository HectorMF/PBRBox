#pragma once
#include "..\MathUtil.h"

namespace Easing
{
	static float Linear(float t)
	{
		return t;
	}

	static float QuadraticIn(float t)
	{
		return t * t;
	}

	static float QuadraticOut(float t)
	{
		return t * (2 - t);
	}

	static float QuadraticInOut(float t)
	{
		return (t < .5f) ? 2 * t * t : -1 + (4 - 2 * t) * t;
	}

	static float CubicIn(float t)
	{
		return t * t * t;
	}

	static float CubicOut(float t)
	{
		float time = 1 - t;
		time = powf(time, 3.0f);
		return 1 - time;
	}

	static float CubicInOut(float t)
	{
		float time;
		if (t < 0.5f)
			time = 4 * t * t * t;
		else
			time = (t - 1) * (2 * t - 2) * (2 * t - 2) + 1;

		return time;
	}

	static float QuarticIn(float t)
	{
		return t * t * t * t;
	}

	static float QuarticOut(float t)
	{
		float time = (t - 1);
		time = time * time * time * (1 - t) + 1;
		return time;
	}

	static float QuarticInOut(float t)
	{
		float time;
		if (t < 0.5)
		{
			time = 8 * t * t * t * t;
		}
		else
		{
			time = (t - 1);
			time = -8 * time * time * time * time + 1;
		}

		return time;
	}

	static float QuinticIn(float t)
	{
		return t * t * t * t * t;
	}

	static float QuinticOut(float t)
	{
		float time = t - 1;
		time = powf(time, 5);
		return time;
	}

	static float QuinticInOut(float t)
	{
		float time;
		if (t < 0.5f)
		{
			time = 16 * t * t * t * t * t;
		}
		else
		{
			time = ((2 * t) - 2);
			time = 0.5f * time * time * time * time * time + 1;
		}
		return time;
	}

	static float SineIn(float t)
	{
		return sinf((t - 1) * (float)M_PI_2) + 1;
	}

	static float SineOut(float t)
	{
		return sinf(t * (float)M_PI_2);
	}

	static float SineInOut(float t)
	{
		return 0.5f * (1.0f - cosf(t * (float)M_PI));
	}

	static float CircularIn(float t)
	{
		return 1 - sqrtf(1 - (t * t));
	}

	static float CircularOut(float t)
	{
		return sqrtf((2 - t) * t);
	}

	static float CircularInOut(float t)
	{
		float time;
		if (t < 0.5f)
			time = 0.5f * (1 - sqrtf(1 - 4 * (t * t)));
		else
			time = 0.5f * (sqrtf(-((2 * t) - 3) * ((2 * t) - 1)) + 1);

		return time;
	}

	static float ExtonentialIn(float t)
	{
		return (t == 0.0f) ? t : powf(2, 10 * (t - 1));
	}

	static float ExtonentialOut(float t)
	{
		return (t == 1.0f) ? t : 1 - powf(2, -10 * t);
	}

	static float ExtonentialInOut(float t)
	{
		float time;
		if (t < 0.5f)
		{
			time = 0.5f * powf(2, (20 * t) - 10);
		}
		else
		{
			time = -0.5f * powf(2, (-20 * t) + 10) + 1;
		}

		if (t == 0.0f || t == 1.0f)
			time = t;

		return time;
	}

	static float ElasticIn(float t)
	{
		return sinf(13 * (float)M_PI_2 * t) * powf(2, 10 * (t - 1));
	}

	static float ElasticOut(float t)
	{
		return sinf(-13 * (float)M_PI_2 * (t + 1)) * powf(2, -10 * t) + 1;
	}

	static float ElasticInOut(float t)
	{
		float time;
		if (t < 0.5f)
			time = 0.5f * sinf(13 * (float)M_PI_2 * (2 * t)) * powf(2, 10 * ((2 * t) - 1));
		else
			time = 0.5f * (sinf(-13 * (float)M_PI_2 * ((2 * t - 1) + 1)) * powf(2, -10 * (2 * t - 1)) + 2);

		return time;
	}

	static float BackIn(float t)
	{
		return t * t * t - t * sinf(t * (float)M_PI);
	}

	static float BackOut(float t)
	{
		float time = (1 - t);
		time = 1 - (time * time * time - time * sinf(time * (float)M_PI));
		return time;
	}

	static float BackInOut(float t)
	{
		float time;
		if (t < 0.5f)
		{
			time = 2 * t;
			time = 0.5f * (time * time * time - time * sinf(time * (float)M_PI));
		}
		else
		{
			time = (1 - (2 * t - 1));
			time = 0.5f * (1 - (time * time * time - time * sinf(time * (float)M_PI))) + 0.5f;
		}

		return time;
	}

	static float BounceIn(float t)
	{
		float time;
		float f = 1 - t;
		if (f < 4 / 11.0f)
		{
			time = (121 * f * f) / 16.0f;
		}
		else if (f < 8 / 11.0f)
		{
			time = (363 / 40.0f * f * f) - (99 / 10.0f * f) + 17 / 5.0f;
		}
		else if (f < 9 / 10.0f)
		{
			time = (4356 / 361.0f * f * f) - (35442 / 1805.0f * f) + 16061 / 1805.0f;
		}
		else
		{
			time = (54 / 5.0f * f * f) - (513 / 25.0f * f) + 268 / 25.0f;
		}

		return 1 - time;
	}

	static float BounceOut(float t)
	{
		float time;
		if (t < 4 / 11.0f)
		{
			time = (121 * t * t) / 16.0f;
		}
		else if (t < 8 / 11.0f)
		{
			time = (363 / 40.0f * t * t) - (99 / 10.0f * t) + 17 / 5.0f;
		}
		else if (t < 9 / 10.0)
		{
			time = (4356 / 361.0f * t * t) - (35442 / 1805.0f * t) + 16061 / 1805.0f;
		}
		else
		{
			time = (54 / 5.0f * t * t) - (513 / 25.0f * t) + 268 / 25.0f;
		}

		return time;
	}

	static float BounceInOut(float t)
	{
		float time;
		if (t < 0.5f)
		{
			float f = 1 - (t * 2);
			if (f < 4 / 11.0f)
			{
				time = (121 * f * f) / 16.0f;
			}
			else if (f < 8 / 11.0f)
			{
				time = (363 / 40.0f * f * f) - (99 / 10.0f * f) + 17 / 5.0f;
			}
			else if (f < 9 / 10.0)
			{
				time = (4356 / 361.0f * f * f) - (35442 / 1805.0f * f) + 16061 / 1805.0f;
			}
			else
			{
				time = (54 / 5.0f * f * f) - (513 / 25.0f * f) + 268 / 25.0f;
			}
			time = 0.5f * (1 - time);
		}
		else
		{
			float f = t * 2 - 1;
			if (f < 4 / 11.0f)
			{
				time = (121 * f * f) / 16.0f;
			}
			else if (f < 8 / 11.0f)
			{
				time = (363 / 40.0f * f * f) - (99 / 10.0f * f) + 17 / 5.0f;
			}
			else if (f < 9 / 10.0f)
			{
				time = (4356 / 361.0f * f * f) - (35442 / 1805.0f * f) + 16061 / 1805.0f;
			}
			else
			{
				time = (54 / 5.0f * f * f) - (513 / 25.0f * f) + 268 / 25.0f;
			}
			time = 0.5f * time + 0.5f;
		}

		return time;
	}
}
