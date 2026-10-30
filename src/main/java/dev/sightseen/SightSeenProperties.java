package dev.sightseen;

import org.springframework.boot.context.properties.ConfigurationProperties;

/**
 * The "sightseen.*" settings in application.properties.
 *
 * @param dataDir where requests, videos and 3D rooms are kept
 * @param worker  the Python helper that runs MapAnything
 */
@ConfigurationProperties("sightseen")
public record SightSeenProperties(String dataDir, Worker worker) {

    /**
     * @param url       where the helper listens
     * @param model     which MapAnything checkpoint a new video gets: "apache" or "default"
     * @param fps       frames taken per second of video
     * @param maxFrames the most frames in one run
     */
    public record Worker(String url, String model, double fps, int maxFrames) {
    }
}
