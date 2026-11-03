package dev.sightseen.request;

import dev.sightseen.reconstruction.WorkerClient;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.client.HttpClientErrorException;
import org.springframework.web.client.RestClientException;
import org.springframework.web.server.ResponseStatusException;

import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * The room's real size from one real length: the viewer measures two points (in the room's own units) and the person
 * says what a tape measure gives between them. The helper sets the scale (worker/room_scale.py) and makes again what
 * it sizes: the walls, the student's things, the issues' areas.
 */
@RestController
@RequestMapping("/api/requests/{id}/scale")
class ScaleController {

    /** What the viewer sends: the two points' distance in room units, and the real one in metres. */
    record ScaleIn(double units, double metres) {
    }

    private static final Pattern DETAIL = Pattern.compile("\"detail\"\\s*:\\s*\"([^\"]*)\""); // the helper's reason

    private final RequestService service;
    private final RequestFiles files;
    private final WorkerClient worker;

    ScaleController(RequestService service, RequestFiles files, WorkerClient worker) {
        this.service = service;
        this.files = files;
        this.worker = worker;
    }

    @PostMapping(consumes = MediaType.APPLICATION_JSON_VALUE, produces = MediaType.APPLICATION_JSON_VALUE)
    String set(@PathVariable String id, @RequestBody ScaleIn body) {
        RoomRequest request = service.byId(id);
        if (!(body.units() > 0) || !(body.metres() >= 0.05 && body.metres() <= 20)) {
            throw new ResponseStatusException(HttpStatus.BAD_REQUEST, "Give a real length between 5 cm and 20 m.");
        }
        String folder = files.folder(request.getId()).toString();
        try {
            return worker.setScale(new WorkerClient.ScaleRequest(folder, body.units(), body.metres()));
        } catch (HttpClientErrorException.BadRequest e) {
            Matcher why = DETAIL.matcher(e.getResponseBodyAsString());
            throw new ResponseStatusException(HttpStatus.BAD_REQUEST, why.find() ? why.group(1) : "That length doesn't fit this room.");
        } catch (RestClientException e) {
            throw new ResponseStatusException(HttpStatus.SERVICE_UNAVAILABLE,
                    "The 3D helper isn't running, so the room's size can't change now.");
        }
    }
}
