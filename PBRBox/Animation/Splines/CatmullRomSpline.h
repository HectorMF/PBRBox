/***************************************************************
*  Copyright (C) 2013 Ohio Supercomputer Center, Ohio State University
*
* This file and its content is protected by a software license.
* You should have received a copy of this license with this file.
* If not, please contact the Ohio Supercomputer Center immediately:
* Attn: Brad Hittle Re: 1224 Kinnear Rd, Columbus, Ohio 43212
*            bhittle@osc.edu
***************************************************************/
#ifndef _CatmullRom_Spline_h_
#define _CatmullRom_Spline_h_

#include "Animation/Spline.h"

namespace gearbox{

	template <class T>
	class CatmullRomSpline : public Spline<T>
	{
	public:
		CatmullRomSpline() : Spline() { }

		CatmullRomSpline* clone() const
		{
			CatmullRomSpline<T>* spline = new CatmullRomSpline<T>(*this);
			return spline; 
		}

	protected:
		T compute(const std::vector<T> points, const float t)
		{
			int segment = (int)((points.size() - 1) * t);
			segment = segment < (points.size() - 2) ? segment : points.size() - 2;

			float dt = t * (points.size() - 1) - segment;

			if (segment == 0)
				return computeSpline(points[0], points[0], points[1], points[2], dt);
			else if (segment == points.size() - 2)
				return computeSpline(points[segment - 1], points[segment], points[segment + 1], points[segment + 1], dt);
			else
				return computeSpline(points[segment - 1], points[segment], points[segment + 1], points[segment + 2], dt);
		}

		T computeSpline(T a, T b, T c, T d, float t)
		{
			T t1 = (c - a) * .5f;
			T t2 = (d - b) * .5f;

			float h1 = +2 * t * t * t - 3 * t * t + 1;
			float h2 = -2 * t * t * t + 3 * t * t;
			float h3 = t * t * t - 2 * t * t + t;
			float h4 = t * t * t - t * t;

			return b * h1 + c * h2 + t1 * h3 + t2 * h4;
		}
	};
}
#endif