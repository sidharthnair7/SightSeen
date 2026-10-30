package dev.sightseen.request;

import org.springframework.data.jpa.repository.JpaRepository;

import java.util.List;
import java.util.Optional;

public interface RoomRequestRepository extends JpaRepository<RoomRequest, String> {

    Optional<RoomRequest> findByLandlordToken(String landlordToken);

    List<RoomRequest> findByStatus(RequestStatus status);
}
