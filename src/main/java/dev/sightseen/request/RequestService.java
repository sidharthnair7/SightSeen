package dev.sightseen.request;

import dev.sightseen.reconstruction.ReconstructionService;
import org.springframework.http.HttpStatus;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;
import org.springframework.web.server.ResponseStatusException;

import java.io.IOException;

@Service
public class RequestService {

    private final RoomRequestRepository requests;
    private final RequestFiles files;
    private final ReconstructionService reconstruction;

    public RequestService(RoomRequestRepository requests, RequestFiles files, ReconstructionService reconstruction) {
        this.requests = requests;
        this.files = files;
        this.reconstruction = reconstruction;
    }

    public RoomRequest create(String label) {
        String clean = label == null ? "" : label.strip();
        if (clean.isEmpty()) {
            throw new ResponseStatusException(HttpStatus.BAD_REQUEST, "Give the place a name, like the street.");
        }
        if (clean.length() > 120) {
            clean = clean.substring(0, 120);
        }
        return requests.save(RoomRequest.create(clean));
    }

    public RoomRequest byId(String id) {
        return requests.findById(id)
                .orElseThrow(() -> new ResponseStatusException(HttpStatus.NOT_FOUND, "No request with that link."));
    }

    public RoomRequest byLandlordToken(String token) {
        return requests.findByLandlordToken(token)
                .orElseThrow(() -> new ResponseStatusException(HttpStatus.NOT_FOUND, "This filming link doesn't work. Ask the student for a new one."));
    }

    /** Saves the video and hands it straight to the 3D helper. */
    public RoomRequest receiveVideo(RoomRequest r, MultipartFile video, VideoSource source) {
        return receiveVideo(r, video, source, false);
    }

    /**
     * {@code recordedLive}: filmed in SightSeen's own page (the landlord link's recorder), not a file picked from a
     * phone. The issues mode's badge says which, from video-live.json beside the video.
     */
    public RoomRequest receiveVideo(RoomRequest r, MultipartFile video, VideoSource source, boolean recordedLive) {
        if (video == null || video.isEmpty()) {
            throw new ResponseStatusException(HttpStatus.BAD_REQUEST, "The video didn't arrive. Try sending it again.");
        }
        if (!RequestFiles.looksLikeVideo(video)) {
            throw new ResponseStatusException(HttpStatus.BAD_REQUEST, "That file isn't a video.");
        }
        try {
            r.videoArrived(source, files.saveVideo(r.getId(), video));
        } catch (IOException e) {
            throw new ResponseStatusException(HttpStatus.INTERNAL_SERVER_ERROR, "Couldn't save the video.", e);
        }
        files.markRecordedLive(r.getId(), recordedLive);
        requests.save(r);
        reconstruction.start(r, null);
        return r;
    }

    /** Builds the room again from the same video, with the given model ("apache" or "default"), or the usual one. */
    public RoomRequest rerun(RoomRequest r, String model) {
        if (!r.hasVideo()) {
            throw new ResponseStatusException(HttpStatus.CONFLICT, "There's no video yet.");
        }
        if (r.getStatus() == RequestStatus.RECONSTRUCTING) {
            throw new ResponseStatusException(HttpStatus.CONFLICT, "It's already being built.");
        }
        reconstruction.start(r, model);
        return r;
    }
}
