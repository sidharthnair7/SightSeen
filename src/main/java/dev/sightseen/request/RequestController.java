package dev.sightseen.request;

import org.springframework.core.io.FileSystemResource;
import org.springframework.core.io.Resource;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;
import org.springframework.web.server.ResponseStatusException;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/** The student's side: start a request, watch it, and open the 3D room. */
@RestController
@RequestMapping("/api/requests")
class RequestController {

    record CreateRequest(String label) {
    }

    private final RequestService service;
    private final RequestFiles files;

    RequestController(RequestService service, RequestFiles files) {
        this.service = service;
        this.files = files;
    }

    @PostMapping
    RequestView create(@RequestBody CreateRequest body) {
        return RequestView.of(service.create(body.label()));
    }

    @GetMapping("/{id}")
    RequestView get(@PathVariable String id) {
        return RequestView.of(service.byId(id));
    }

    /** The backup path: the landlord already sent the student a video some other way. */
    @PostMapping(path = "/{id}/video", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    RequestView upload(@PathVariable String id, @RequestParam("video") MultipartFile video) {
        return RequestView.of(service.receiveVideo(service.byId(id), video, VideoSource.STUDENT_UPLOAD));
    }

    @PostMapping("/{id}/rerun")
    RequestView rerun(@PathVariable String id, @RequestParam(defaultValue = "") String model) {
        return RequestView.of(service.rerun(service.byId(id), model));
    }

    @GetMapping("/{id}/room.ply")
    ResponseEntity<Resource> room(@PathVariable String id) {
        RoomRequest r = service.byId(id);
        return ply(files.roomFile(r.getId()), "The 3D room isn't ready yet.");
    }

    @GetMapping("/{id}/room-surface.ply")
    ResponseEntity<Resource> surface(@PathVariable String id) {
        RoomRequest r = service.byId(id);
        return ply(files.surfaceFile(r.getId()), "This room has no solid surface. Build it again to make one.");
    }

    @GetMapping("/{id}/room-splats.ply")
    ResponseEntity<Resource> splats(@PathVariable String id) {
        RoomRequest r = service.byId(id);
        return ply(files.splatsFile(r.getId()), "This room has no splats. Build it again to make them.");
    }

    @GetMapping("/{id}/cameras.json")
    ResponseEntity<Resource> cameras(@PathVariable String id) {
        RoomRequest r = service.byId(id);
        return json(files.camerasFile(r.getId()), "This room has no camera path. Build it again to make one.");
    }

    @GetMapping("/{id}/room-frame.json")
    ResponseEntity<Resource> roomFrame(@PathVariable String id) {
        RoomRequest r = service.byId(id);
        return json(files.roomFrameFile(r.getId()), "This room doesn't know where its floor is. Build it again to find it.");
    }

    @GetMapping("/{id}/room-shell.json")
    ResponseEntity<Resource> roomShell(@PathVariable String id) {
        RoomRequest r = service.byId(id);
        return json(files.roomShellFile(r.getId()), "This room has no walls found yet. Build it again to find them.");
    }

    @GetMapping("/{id}/room-objects.json")
    ResponseEntity<Resource> roomObjects(@PathVariable String id) {
        RoomRequest r = service.byId(id);
        return json(files.roomObjectsFile(r.getId()), "This room's furniture hasn't been found yet.");
    }

    /**
     * The room's other generated files (wall textures, furniture) by name, so new ones need no new endpoint. A name is
     * lowercase letters, digits and dashes with one known extension: no folders, and never the landlord's video.
     */
    @GetMapping("/{id}/assets/{name:.+}")
    ResponseEntity<Resource> asset(@PathVariable String id, @PathVariable String name) {
        RoomRequest r = service.byId(id);
        Matcher match = ASSET.matcher(name);
        Path file = files.assetFile(r.getId(), name);
        if (!match.matches() || !Files.exists(file)) {
            throw new ResponseStatusException(HttpStatus.NOT_FOUND, "This room has no file called " + name + ".");
        }
        MediaType type = switch (match.group(1)) {
            case "json" -> MediaType.APPLICATION_JSON;
            case "png" -> MediaType.IMAGE_PNG;
            case "jpg" -> MediaType.IMAGE_JPEG;
            default -> MediaType.APPLICATION_OCTET_STREAM;
        };
        return ResponseEntity.ok().contentType(type).body(new FileSystemResource(file));
    }

    private static final Pattern ASSET = Pattern.compile("[a-z0-9][a-z0-9-]*\\.(json|png|jpg|ply)");

    private static ResponseEntity<Resource> json(Path file, String missing) {
        if (!Files.exists(file)) {
            throw new ResponseStatusException(HttpStatus.NOT_FOUND, missing);
        }
        return ResponseEntity.ok().contentType(MediaType.APPLICATION_JSON).body(new FileSystemResource(file));
    }

    private static ResponseEntity<Resource> ply(Path file, String missing) {
        if (!Files.exists(file)) {
            throw new ResponseStatusException(HttpStatus.NOT_FOUND, missing);
        }
        return ResponseEntity.ok()
                .contentType(MediaType.APPLICATION_OCTET_STREAM)
                .body(new FileSystemResource(file));
    }
}
