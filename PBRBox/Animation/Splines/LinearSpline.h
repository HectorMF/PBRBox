/***************************************************************
*  Copyright (C) 2013 Ohio Supercomputer Center, Ohio State University
*
* This file and its content is protected by a software license.
* You should have received a copy of this license with this file.
* If not, please contact the Ohio Supercomputer Center immediately:
* Attn: Brad Hittle Re: 1224 Kinnear Rd, Columbus, Ohio 43212
*            bhittle@osc.edu
***************************************************************/
#ifndef _Linear_Spline_h_
#define _Linear_Spline_h_

#include "Animation/Spline.h"

template <class T>
class LinearSpline : public Spline<T>
{
public:
	LinearSpline() : Spline() { }

	LinearSpline* clone() const
	{
		LinearSpline<T>* spline = new LinearSpline<T>(*this);
		return spline;
	}

protected:
	T compute(const std::vector<T> points, const float t)
	{
		int segment = (int)((points.size() - 1) * t);
		segment = MAX(segment, 0);
		segment = MIN(segment, points.size() - 2);

		float dt = t * (points.size() - 1) - segment;

		return points[segment] + dt * (points[segment + 1] - points[segment]);
	}
};