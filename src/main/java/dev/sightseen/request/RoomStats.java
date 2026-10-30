package dev.sightseen.request;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;

/**
 * What one MapAnything run produced, as the Python helper reports it (worker/reconstruct.py).
 *
 * @param frames      frames taken from the video
 * @param seconds     time MapAnything took, model loading not included
 * @param peakVramGb  the most GPU memory the run used
 * @param scaleFactor MapAnything's own metric scale for this run
 * @param points      points in the saved 3D room
 * @param sizeX       rough extent of the 3D along the first frame's left-right axis, in metres
 * @param sizeY       ... up-down axis
 * @param sizeZ       ... forward-back axis
 * @param ply         the point cloud's file name in the request folder
 */
@JsonIgnoreProperties(ignoreUnknown = true)
public record RoomStats(Integer frames, Double seconds, Double peakVramGb, Double scaleFactor, Long points,
                        Double sizeX, Double sizeY, Double sizeZ, String ply) {
}
