package dev.sightseen.request;

import dev.sightseen.SightSeenProperties;
import org.springframework.stereotype.Component;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Locale;
import java.util.Set;

/**
 * Each request gets a folder, data/requests/{id}/, shared with the Python helper: the video goes in, and the helper
 * writes frames/, room-splats.ply, room-surface.ply, room.ply and result.json next to it. File names are always ours,
 * never the uploader's.
 */
@Component
public class RequestFiles {

    private static final Set<String> VIDEO_EXTENSIONS = Set.of("mp4", "mov", "m4v", "webm", "mkv", "avi", "3gp");

    private final Path root;

    public RequestFiles(SightSeenProperties props) {
        this.root = Path.of(props.dataDir(), "requests").toAbsolutePath().normalize();
    }

    public Path folder(String requestId) {
        Path folder = root.resolve(requestId);
        try {
            Files.createDirectories(folder);
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
        return folder;
    }

    public static boolean looksLikeVideo(MultipartFile file) {
        String type = file.getContentType();
        return (type != null && type.startsWith("video/")) || VIDEO_EXTENSIONS.contains(extension(file.getOriginalFilename()));
    }

    /** Saves the upload as video.{ext} and returns that file name. */
    public String saveVideo(String requestId, MultipartFile file) throws IOException {
        String ext = extension(file.getOriginalFilename());
        String name = "video." + (VIDEO_EXTENSIONS.contains(ext) ? ext : "mp4");
        file.transferTo(folder(requestId).resolve(name));
        return name;
    }

    public Path roomFile(String requestId) {
        return folder(requestId).resolve("room.ply");
    }

    /** The room as a solid surface (a triangle mesh). Rooms built before the surface existed don't have one. */
    public Path surfaceFile(String requestId) {
        return folder(requestId).resolve("room-surface.ply");
    }

    /** The room as Gaussian splats, trained to look like the video. Rooms built before splats existed don't have one. */
    public Path splatsFile(String requestId) {
        return folder(requestId).resolve("room-splats.ply");
    }

    /** Where the phone stood for each frame, so the viewer can walk the video's path. Older rooms don't have one. */
    public Path camerasFile(String requestId) {
        return folder(requestId).resolve("cameras.json");
    }

    /** Which way is up and where the floor is, so the viewer stands the room upright. Older rooms don't have one. */
    public Path roomFrameFile(String requestId) {
        return folder(requestId).resolve("room-frame.json");
    }

    /** The room's walls, floor and ceiling as a box, drawn clean behind the splats. Older rooms don't have one. */
    public Path roomShellFile(String requestId) {
        return folder(requestId).resolve("room-shell.json");
    }

    /** One of the room's own generated files by name (wall textures, furniture); the controller checks the name. */
    public Path assetFile(String requestId, String name) {
        return folder(requestId).resolve(name);
    }

    /** The furniture SAM 3.1 found, as blocks for the Walk look. Rooms built before it don't have one. */
    public Path roomObjectsFile(String requestId) {
        return folder(requestId).resolve("room-objects.json");
    }

    private static String extension(String fileName) {
        if (fileName == null || !fileName.contains(".")) {
            return "";
        }
        return fileName.substring(fileName.lastIndexOf('.') + 1).toLowerCase(Locale.ROOT);
    }
}
