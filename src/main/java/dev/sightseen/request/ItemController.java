package dev.sightseen.request;

import dev.sightseen.reconstruction.WorkerClient;
import org.springframework.core.io.FileSystemResource;
import org.springframework.core.io.Resource;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.client.RestClientException;
import org.springframework.web.multipart.MultipartFile;
import org.springframework.web.server.ResponseStatusException;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.SecureRandom;
import java.time.Instant;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Set;
import java.util.regex.Pattern;
import java.util.stream.Stream;

/**
 * The student's own things: a photo of one, what it is and its real width in; a 3D model of it at that size out, to
 * place in the rebuilt room (worker/items.py builds it). Each lives in items/{itemId}/ in the room's folder, where
 * item.json says how it's going (queued, outlining, building, done, failed) and the helper keeps it up to date.
 */
@RestController
@RequestMapping("/api/requests/{id}/items")
class ItemController {

    private static final Set<String> PHOTO_EXTENSIONS = Set.of("jpg", "jpeg", "png", "webp");
    private static final Pattern ITEM_ID = Pattern.compile("[a-z0-9]{10}");
    private static final String LETTERS = "abcdefghijklmnopqrstuvwxyz0123456789";
    private static final SecureRandom RANDOM = new SecureRandom();

    private final RequestService service;
    private final RequestFiles files;
    private final WorkerClient worker;

    ItemController(RequestService service, RequestFiles files, WorkerClient worker) {
        this.service = service;
        this.files = files;
        this.worker = worker;
    }

    @PostMapping(consumes = MediaType.MULTIPART_FORM_DATA_VALUE, produces = MediaType.APPLICATION_JSON_VALUE)
    String add(@PathVariable String id, @RequestParam("photo") MultipartFile photo, @RequestParam("name") String name,
               @RequestParam("widthCm") double widthCm) throws IOException {
        RoomRequest request = service.byId(id);
        String what = name.toLowerCase(Locale.ROOT).replaceAll("[^a-z0-9 ]", " ").trim().replaceAll(" +", " ");
        if (what.isEmpty() || what.length() > 40) {
            throw new ResponseStatusException(HttpStatus.BAD_REQUEST, "Say what it is in a few words: desk, chair, monitor.");
        }
        if (!(widthCm >= 5 && widthCm <= 500)) {
            throw new ResponseStatusException(HttpStatus.BAD_REQUEST, "Its width should be between 5 and 500 cm.");
        }
        String ext = extension(photo.getOriginalFilename());
        if (!PHOTO_EXTENSIONS.contains(ext)) {
            throw new ResponseStatusException(HttpStatus.BAD_REQUEST, "That isn't a photo this can read: use a JPG or PNG.");
        }
        String itemId = newId();
        Path folder = files.folder(request.getId());
        Path dir = folder.resolve("items").resolve(itemId);
        Files.createDirectories(dir);
        String photoName = "upload." + (ext.equals("jpeg") ? "jpg" : ext);
        photo.transferTo(dir.resolve(photoName));
        String meta = """
                {"id": "%s", "name": "%s", "widthCm": %s, "photo": "%s", "status": "queued", "createdAt": "%s"}
                """.formatted(itemId, what, Double.toString(widthCm), photoName, Instant.now()).strip();
        Files.writeString(dir.resolve("item.json"), meta, StandardCharsets.UTF_8);
        try {
            worker.submitItem(new WorkerClient.ItemJob(itemId, folder.toString()));
        } catch (RestClientException e) {
            meta = meta.replace("\"status\": \"queued\"",
                    "\"status\": \"failed\", \"why\": \"The 3D helper isn't running (start worker/run.ps1).\"");
            Files.writeString(dir.resolve("item.json"), meta, StandardCharsets.UTF_8);
        }
        return meta;
    }

    /** Every item of this room, oldest first, as the helper last wrote it. */
    @GetMapping(produces = MediaType.APPLICATION_JSON_VALUE)
    String list(@PathVariable String id) throws IOException {
        Path items = files.folder(service.byId(id).getId()).resolve("items");
        List<String> found = new ArrayList<>();
        if (Files.isDirectory(items)) {
            try (Stream<Path> dirs = Files.list(items)) {
                for (Path dir : dirs.sorted().toList()) {
                    Path meta = dir.resolve("item.json");
                    if (ITEM_ID.matcher(dir.getFileName().toString()).matches() && Files.exists(meta)) {
                        found.add(Files.readString(meta, StandardCharsets.UTF_8).strip());
                    }
                }
            }
        }
        found.sort((a, b) -> createdAt(a).compareTo(createdAt(b)));
        return "[" + String.join(",", found) + "]";
    }

    @GetMapping("/{itemId}/item.ply")
    ResponseEntity<Resource> model(@PathVariable String id, @PathVariable String itemId) {
        Path file = files.folder(service.byId(id).getId()).resolve("items").resolve(itemId).resolve("item.ply");
        if (!ITEM_ID.matcher(itemId).matches() || !Files.exists(file)) {
            throw new ResponseStatusException(HttpStatus.NOT_FOUND, "That thing isn't built yet.");
        }
        return ResponseEntity.ok().contentType(MediaType.APPLICATION_OCTET_STREAM).body(new FileSystemResource(file));
    }

    private static String createdAt(String meta) {
        int at = meta.indexOf("\"createdAt\"");
        return at < 0 ? "" : meta.substring(at);
    }

    private static String newId() {
        StringBuilder id = new StringBuilder();
        for (int i = 0; i < 10; i++) {
            id.append(LETTERS.charAt(RANDOM.nextInt(LETTERS.length())));
        }
        return id.toString();
    }

    private static String extension(String filename) {
        if (filename == null || filename.lastIndexOf('.') < 0) {
            return "";
        }
        return filename.substring(filename.lastIndexOf('.') + 1).toLowerCase(Locale.ROOT);
    }
}
